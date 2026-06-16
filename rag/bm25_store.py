"""BM25 lexical index built from the documents already in this session's Chroma.

The cache lives in the session, so different users don't share each other's
index. Rebuilt when corpus size changes (cheap at demo scale).

BM25 complements dense vector retrieval by catching exact-token matches that
embeddings blur (IDs, code identifiers, acronyms, version numbers).
"""
from __future__ import annotations

import re

from rank_bm25 import BM25Okapi

from .session import get_collection, get_session


def _tokenize(text: str) -> list[str]:
    """Word-level: lowercase, split on non-alphanumeric."""
    return [t for t in re.split(r"[^A-Za-z0-9]+", text.lower()) if t]


def _build_index() -> dict:
    collection = get_collection()
    res = collection.get(include=["documents", "metadatas"])
    tokenized = [_tokenize(d) for d in res["documents"]]
    return {
        "bm25": BM25Okapi(tokenized) if tokenized else None,
        "docs": res["documents"],
        "metas": res["metadatas"],
        "size": len(res["documents"]),
    }


def _ensure_index() -> dict:
    """Return the session's cached BM25 index, rebuilding if stale."""
    session = get_session()
    cache = session.get("bm25_cache")
    current_size = get_collection().count()
    if cache is None or cache["size"] != current_size:
        cache = _build_index()
        session["bm25_cache"] = cache
    return cache


def invalidate() -> None:
    """Force rebuild on next call. Use after ingest/delete."""
    get_session()["bm25_cache"] = None


def search_bm25(query: str, k: int = 20):
    """Return top-k BM25 results as RetrievedChunk objects (with parent_text)."""
    from .retriever import RetrievedChunk   # lazy: breaks the import cycle

    idx = _ensure_index()
    if idx["bm25"] is None or idx["size"] == 0:
        return []
    scores = idx["bm25"].get_scores(_tokenize(query))
    top = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    hits = []
    for i in top:
        if scores[i] <= 0:
            continue
        meta = idx["metas"][i]
        hits.append(RetrievedChunk(
            text=idx["docs"][i],
            parent_text=meta.get("parent_text", idx["docs"][i]),
            source=meta["source"],
            chunk_index=int(meta.get("chunk_index", 0)),
            parent_index=int(meta.get("parent_index", 0)),
            score=float(scores[i]),
        ))
    return hits
