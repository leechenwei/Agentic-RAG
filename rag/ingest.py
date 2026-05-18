"""Document ingestion: load, chunk, embed, and store in ChromaDB."""
from __future__ import annotations

import hashlib
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions
from pypdf import PdfReader

CHROMA_PATH = "chroma_db"
COLLECTION_NAME = "docs"
EMBED_MODEL = "all-MiniLM-L6-v2"

# Chunking parameters — these are the *most impactful* knobs in RAG quality.
# Too small: chunks lose context. Too large: retrieval gets noisy and the LLM
# wastes its context window on irrelevant text.
CHUNK_SIZE = 500       # characters
CHUNK_OVERLAP = 80     # characters of overlap between adjacent chunks


def _read_file(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        return "\n".join(page.extract_text() or "" for page in PdfReader(str(path)).pages)
    return path.read_text(encoding="utf-8", errors="ignore")


def _chunk(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Sliding-window chunking on character count.

    Note: char-based chunking is naive — production systems chunk on sentence or
    semantic boundaries (e.g. langchain RecursiveCharacterTextSplitter). Char-based
    is intentionally chosen here for clarity in the tech-test discussion.
    """
    text = " ".join(text.split())  # collapse whitespace
    if not text:
        return []
    chunks, start = [], 0
    while start < len(text):
        end = min(start + size, len(text))
        chunks.append(text[start:end])
        if end == len(text):
            break
        start = end - overlap
    return chunks


def _get_collection():
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBED_MODEL
    )
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embed_fn,
        metadata={"hnsw:space": "cosine"},
    )


def ingest_file(path: Path) -> int:
    """Ingest a single file. Returns number of chunks added. Idempotent."""
    if path.suffix.lower() not in {".pdf", ".txt", ".md"}:
        raise ValueError(f"Unsupported file type: {path.suffix}")
    collection = _get_collection()
    text = _read_file(path)
    chunks = _chunk(text)
    if not chunks:
        return 0
    ids = [
        hashlib.md5(f"{path.name}:{i}:{c[:32]}".encode()).hexdigest()
        for i, c in enumerate(chunks)
    ]
    metadatas = [
        {"source": path.name, "chunk_index": i} for i in range(len(chunks))
    ]
    collection.upsert(ids=ids, documents=chunks, metadatas=metadatas)
    return len(chunks)


def ingest_directory(data_dir: str = "data") -> int:
    """Ingest every supported file in a directory. Returns total chunks added."""
    collection = _get_collection()
    total = 0
    for path in Path(data_dir).glob("**/*"):
        if path.suffix.lower() not in {".pdf", ".txt", ".md"}:
            continue
        text = _read_file(path)
        chunks = _chunk(text)
        if not chunks:
            continue
        # Stable IDs so re-ingesting the same file overwrites instead of duplicating.
        ids = [
            hashlib.md5(f"{path.name}:{i}:{c[:32]}".encode()).hexdigest()
            for i, c in enumerate(chunks)
        ]
        metadatas = [
            {"source": path.name, "chunk_index": i} for i in range(len(chunks))
        ]
        collection.upsert(ids=ids, documents=chunks, metadatas=metadatas)
        total += len(chunks)
        print(f"  ingested {path.name}: {len(chunks)} chunks")
    return total


if __name__ == "__main__":
    n = ingest_directory()
    print(f"Done. {n} chunks total in collection '{COLLECTION_NAME}'.")
