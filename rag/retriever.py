"""Retrieval pipeline: dense + BM25 + RRF fusion + cross-encoder rerank.

Stages (each independently testable):

    [query]
       │
       ├──► dense (Chroma cosine)  ──┐
       │                              ├──► RRF fuse  ──► rerank (BGE)
       └──► sparse (BM25)  ───────────┘

Production entry point: `retrieve_hybrid_reranked`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .session import get_collection as _get_collection


@dataclass
class RetrievedChunk:
    """A retrieved CHILD chunk + its PARENT context.

    `score` meaning depends on the stage that produced it:
      Dense:  cosine similarity in [0, 1]
      BM25:   raw BM25 score (unbounded)
      RRF:    fused rank score (~0.01-0.03)
      Rerank: cross-encoder logit (~-10..10)
    """
    text: str
    parent_text: str
    source: str
    chunk_index: int
    parent_index: int
    score: float

    def cite(self) -> str:
        return f"[{self.source}#chunk{self.chunk_index}]"


# ---------------------------------------------------------------------------
# KB management
# ---------------------------------------------------------------------------

def list_sources() -> dict[str, int]:
    """Return {source_filename: child_chunk_count}."""
    res = _get_collection().get(include=["metadatas"])
    counts: dict[str, int] = {}
    for meta in res["metadatas"]:
        src = meta["source"]
        counts[src] = counts.get(src, 0) + 1
    return dict(sorted(counts.items()))


def delete_source(source: str) -> int:
    """Remove every chunk for a source; invalidate BM25 cache."""
    collection = _get_collection()
    res = collection.get(where={"source": source})
    ids = res["ids"]
    if ids:
        collection.delete(ids=ids)
        from .bm25_store import invalidate as _invalidate_bm25
        _invalidate_bm25()
    return len(ids)


# ---------------------------------------------------------------------------
# Stage 1 — dense retrieval
# ---------------------------------------------------------------------------

def retrieve_dense(query: str, k: int = 4) -> list[RetrievedChunk]:
    """Semantic retrieval via Chroma cosine. Top-k child chunks."""
    res = _get_collection().query(query_texts=[query], n_results=k)
    if not res["documents"] or not res["documents"][0]:
        return []
    return [
        RetrievedChunk(
            text=doc,
            parent_text=meta.get("parent_text", doc),
            source=meta["source"],
            chunk_index=int(meta.get("chunk_index", 0)),
            parent_index=int(meta.get("parent_index", 0)),
            score=1.0 - float(dist),
        )
        for doc, meta, dist in zip(
            res["documents"][0], res["metadatas"][0], res["distances"][0]
        )
    ]


# ---------------------------------------------------------------------------
# Stage 2 — RRF fusion of dense + BM25
# ---------------------------------------------------------------------------

def _rrf_fuse(
    dense: list[RetrievedChunk],
    sparse: list[RetrievedChunk],
    k_rrf: int = 60,
) -> list[RetrievedChunk]:
    """Reciprocal Rank Fusion. Combines two rankings using only ranks (not
    raw scores) so dense/BM25 score scales don't have to be reconciled.
    Constant k=60 from the original RRF paper. Both inputs are already
    RetrievedChunk objects with parent_text populated."""
    fused: dict[tuple[str, int], dict] = {}
    for source_list in (dense, sparse):
        for rank, hit in enumerate(source_list):
            key = (hit.source, hit.chunk_index)
            if key not in fused:
                fused[key] = {"chunk": hit, "score": 0.0}
            fused[key]["score"] += 1.0 / (k_rrf + rank)
    out: list[RetrievedChunk] = []
    for item in sorted(fused.values(), key=lambda x: x["score"], reverse=True):
        item["chunk"].score = item["score"]
        out.append(item["chunk"])
    return out


def retrieve_hybrid(
    query: str,
    k: int = 5,
    candidate_k: int = 20,
) -> list[RetrievedChunk]:
    """Dense + BM25 → RRF. Returns top-k after fusion."""
    from .bm25_store import search_bm25

    dense = retrieve_dense(query, k=candidate_k)
    sparse = search_bm25(query, k=candidate_k)
    return _rrf_fuse(dense, sparse)[:k]


# ---------------------------------------------------------------------------
# Stage 3 — rerank with BGE cross-encoder
# ---------------------------------------------------------------------------

def retrieve_hybrid_reranked(
    query: str,
    k: int = 3,
    candidate_k: int = 20,
) -> list[RetrievedChunk]:
    """Production pipeline: dense + BM25 → RRF → rerank → top-k."""
    from .reranker import rerank

    candidates = retrieve_hybrid(query, k=candidate_k, candidate_k=candidate_k)
    return rerank(query, candidates, top_k=k)


# ---------------------------------------------------------------------------
# Parent-context serialization for the LLM
# ---------------------------------------------------------------------------

def to_llm_context(chunks: Iterable[RetrievedChunk]) -> str:
    """Format chunks for the LLM. Uses PARENT text (not child), deduplicated
    so the same parent block isn't repeated when multiple children of it
    appear in the same retrieval."""
    seen: set[tuple[str, int]] = set()
    blocks: list[str] = []
    for c in chunks:
        key = (c.source, c.parent_index)
        if key in seen:
            continue
        seen.add(key)
        blocks.append(f"{c.cite()} (score={c.score:.3f})\n{c.parent_text}")
    return "\n\n".join(blocks)
