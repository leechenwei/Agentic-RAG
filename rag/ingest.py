"""Document ingestion with parent-child chunking + recursive splitting.

Chunking is the #1 quality lever in RAG. This module uses two production-grade
techniques together:

1. **Recursive Character Splitter** (LangChain): tries to split on natural
   boundaries in priority order — paragraph → newline → sentence → word →
   character. Chunks respect document structure when possible.

2. **Parent-Child** chunking: small CHILD chunks are embedded for precise
   retrieval, but at query time we return the larger PARENT block to the LLM
   so the answer has full surrounding context. Best of both worlds —
   precision of small chunks, coherence of large blocks.

Each child chunk is stored in ChromaDB with metadata pointing to its parent:
  - source        : filename
  - chunk_index   : position of this CHILD in the document
  - parent_index  : id of the parent block this child belongs to
  - parent_text   : the full parent block (stored alongside child for O(1)
                    lookup at query time — fine at demo scale)
"""
from __future__ import annotations

import hashlib
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

from .session import get_collection

try:
    from docx import Document as DocxDocument
except ImportError:  # python-docx not installed; .docx ingestion will fail loudly
    DocxDocument = None  # type: ignore

# Parent-child chunking parameters.
# Children are small for precise embedding-based retrieval.
# Parents are larger so the LLM gets enough context to answer well.
PARENT_CHUNK_SIZE = 1024     # characters (~256 tokens for English text)
PARENT_CHUNK_OVERLAP = 100
CHILD_CHUNK_SIZE = 256       # characters (~64 tokens)
CHILD_CHUNK_OVERLAP = 40

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md", ".docx"}


# ---------------------------------------------------------------------------
# File readers
# ---------------------------------------------------------------------------

def _read_pdf(path: Path) -> str:
    return "\n".join(page.extract_text() or "" for page in PdfReader(str(path)).pages)


def _read_docx(path: Path) -> str:
    if DocxDocument is None:
        raise RuntimeError(
            "python-docx is not installed. Install it with `pip install python-docx`."
        )
    doc = DocxDocument(str(path))
    return "\n".join(p.text for p in doc.paragraphs if p.text)


def _read_file(path: Path) -> str:
    """Dispatch to the right reader based on file extension."""
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return _read_pdf(path)
    if suffix == ".docx":
        return _read_docx(path)
    # TXT and MD treated as plain text
    return path.read_text(encoding="utf-8", errors="ignore")


# ---------------------------------------------------------------------------
# Splitters (LangChain RecursiveCharacterTextSplitter)
# ---------------------------------------------------------------------------
# Separator priority: try paragraph breaks first, then newlines, then sentences,
# then words. Falls back to character splitting only as a last resort. This
# beats raw character chunking because most splits land on natural boundaries.

_parent_splitter = RecursiveCharacterTextSplitter(
    chunk_size=PARENT_CHUNK_SIZE,
    chunk_overlap=PARENT_CHUNK_OVERLAP,
    separators=["\n\n", "\n", ". ", " ", ""],
)

_child_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHILD_CHUNK_SIZE,
    chunk_overlap=CHILD_CHUNK_OVERLAP,
    separators=["\n\n", "\n", ". ", " ", ""],
)


# ---------------------------------------------------------------------------
# Parent-child split + index
# ---------------------------------------------------------------------------

# Backwards-compat alias for older imports (rag.bm25_store, rag.retriever)
_get_collection = get_collection

def _build_parent_child_chunks(text: str) -> list[dict]:
    """Return a list of child-chunk records, each with a reference to its parent.

    Algorithm:
      1. Split text into PARENT blocks (large, semantic boundaries respected).
      2. For each parent, split it into CHILD chunks (small, for retrieval).
      3. Emit a record per child that includes the parent text inline so we
         can serve the parent to the LLM at query time with no extra lookup.
    """
    records: list[dict] = []
    text = text.strip()
    if not text:
        return records

    parents = _parent_splitter.split_text(text)
    for parent_idx, parent_text in enumerate(parents):
        children = _child_splitter.split_text(parent_text)
        for child in children:
            records.append({
                "child_text": child,
                "parent_text": parent_text,
                "parent_index": parent_idx,
            })
    return records


def ingest_file(path: Path) -> int:
    """Ingest one file. Returns number of CHILD chunks added.

    Same file re-ingested overwrites previous chunks (stable IDs).
    """
    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file type: {path.suffix}")

    collection = _get_collection()
    text = _read_file(path)
    records = _build_parent_child_chunks(text)
    if not records:
        return 0

    ids = [
        hashlib.md5(
            f"{path.name}:{i}:{r['child_text'][:32]}".encode()
        ).hexdigest()
        for i, r in enumerate(records)
    ]
    documents = [r["child_text"] for r in records]
    metadatas = [
        {
            "source": path.name,
            "chunk_index": i,
            "parent_index": r["parent_index"],
            "parent_text": r["parent_text"],
        }
        for i, r in enumerate(records)
    ]
    collection.upsert(ids=ids, documents=documents, metadatas=metadatas)

    # Invalidate the BM25 cache so it rebuilds on the next query.
    try:
        from .bm25_store import invalidate as _invalidate_bm25
        _invalidate_bm25()
    except ImportError:
        pass

    return len(records)


def ingest_directory(data_dir: str = "data") -> int:
    """Ingest every supported file in a directory. Returns total child chunks added."""
    total = 0
    for path in Path(data_dir).glob("**/*"):
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        n = ingest_file(path)
        total += n
        print(f"  ingested {path.name}: {n} child chunks")
    return total


if __name__ == "__main__":
    n = ingest_directory()
    print(f"Done. {n} total child chunks in collection '{COLLECTION_NAME}'.")
