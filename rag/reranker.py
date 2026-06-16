"""Cross-encoder reranker using BGE Reranker base.

A reranker reads (query, doc) JOINTLY through a single transformer pass.
Attention layers let query tokens attend to doc tokens directly — far more
accurate than bi-encoder cosine similarity, which embeds query and doc
independently. The price is latency, which is why we only rerank the top-K
candidates from broad retrieval, not the whole corpus.

Model choice: BAAI/bge-reranker-base
  - ~278MB, multilingual (handles English/Chinese well)
  - Loaded lazily on first call so import is cheap
  - Free and runs locally; for higher quality at the cost of API calls,
    swap in Cohere Rerank or BGE Reranker v2 m3.
"""
from __future__ import annotations

from sentence_transformers import CrossEncoder

from .retriever import RetrievedChunk

# Module-level singleton so we don't reload the model on every query.
_model: CrossEncoder | None = None


def _get_model() -> CrossEncoder:
    """Lazy singleton — first call downloads ~278MB and warms up."""
    global _model
    if _model is None:
        _model = CrossEncoder("BAAI/bge-reranker-base", max_length=512)
    return _model


def rerank(
    query: str,
    chunks: list[RetrievedChunk],
    top_k: int = 5,
) -> list[RetrievedChunk]:
    """Re-score a list of chunks by reading (query, chunk_text) jointly.

    Mutates the input chunks' `score` field to hold the rerank score (which
    is more meaningful for the UI than the previous RRF / cosine score).
    Returns the top_k highest-scoring chunks in descending order.
    """
    if not chunks:
        return []
    model = _get_model()
    pairs = [(query, c.text) for c in chunks]
    scores = model.predict(pairs)
    scored = list(zip(chunks, scores))
    scored.sort(key=lambda pair: float(pair[1]), reverse=True)
    out: list[RetrievedChunk] = []
    for chunk, score in scored[:top_k]:
        chunk.score = float(score)   # overwrite with the rerank confidence
        out.append(chunk)
    return out


def confidence_badge(score: float) -> str:
    """Map a BGE rerank score to a UI badge.
    BGE Reranker scores are raw model outputs: >5 strong, 1-5 medium, <1 weak.
    """
    if score >= 5.0:
        return "🟢"
    if score >= 1.0:
        return "🟡"
    return "🔴"
