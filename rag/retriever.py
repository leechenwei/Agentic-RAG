"""Retrieval interface over the ChromaDB collection."""
from __future__ import annotations

from dataclasses import dataclass

from .ingest import _get_collection


@dataclass
class RetrievedChunk:
    text: str
    source: str
    chunk_index: int
    score: float  # cosine similarity in [0, 1] — higher is better

    def cite(self) -> str:
        return f"[{self.source}#chunk{self.chunk_index}]"


def list_sources() -> dict[str, int]:
    """Return {source_filename: chunk_count} for everything in the index."""
    collection = _get_collection()
    # ChromaDB has no direct "group by" — fetch all metadatas and tally.
    # Fine for demo-scale (<100k chunks). For production, store source counts
    # separately or use the upcoming Chroma aggregation API.
    res = collection.get(include=["metadatas"])
    counts: dict[str, int] = {}
    for meta in res["metadatas"]:
        src = meta["source"]
        counts[src] = counts.get(src, 0) + 1
    return dict(sorted(counts.items()))


def delete_source(source: str) -> int:
    """Remove every chunk belonging to a given source filename. Returns count deleted."""
    collection = _get_collection()
    res = collection.get(where={"source": source})
    ids = res["ids"]
    if ids:
        collection.delete(ids=ids)
    return len(ids)


def retrieve(query: str, k: int = 4) -> list[RetrievedChunk]:
    """Semantic retrieval. Returns top-k chunks ordered by similarity."""
    collection = _get_collection()
    res = collection.query(query_texts=[query], n_results=k)
    # ChromaDB returns cosine *distance* (lower = better). Convert to similarity.
    return [
        RetrievedChunk(
            text=doc,
            source=meta["source"],
            chunk_index=int(meta["chunk_index"]),
            score=1.0 - float(dist),
        )
        for doc, meta, dist in zip(
            res["documents"][0], res["metadatas"][0], res["distances"][0]
        )
    ]
