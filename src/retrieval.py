"""Similarity search over the Chroma collection."""

from __future__ import annotations

from dataclasses import dataclass

from src.ingestion import embed_texts, get_collection

DEFAULT_TOP_K = 4


@dataclass(frozen=True)
class RetrievedChunk:
    id: str
    text: str
    source: str
    chunk_index: int
    score: float

    @property
    def label(self) -> str:
        return f"{self.source} (chunk {self.chunk_index})"


def retrieve(
    query: str, k: int = DEFAULT_TOP_K, collection=None, client=None
) -> list[RetrievedChunk]:
    """Return the top-``k`` chunks most similar to ``query`` (cosine), best first."""
    collection = collection if collection is not None else get_collection()
    count = collection.count()
    if count == 0 or not query.strip():
        return []
    [query_vector] = embed_texts([query], client=client)
    result = collection.query(
        query_embeddings=[query_vector],
        n_results=min(k, count),
        include=["documents", "metadatas", "distances"],
    )
    return [
        RetrievedChunk(
            id=chunk_id,
            text=doc,
            source=str(meta.get("source", "unknown")),
            chunk_index=int(meta.get("chunk_index", -1)),
            score=1.0 - float(dist),
        )
        for chunk_id, doc, meta, dist in zip(
            result["ids"][0],
            result["documents"][0],
            result["metadatas"][0],
            result["distances"][0],
            strict=True,
        )
    ]
