"""In-memory KnowledgeStore for testing and ephemeral use."""

from __future__ import annotations

import math

from syrin.knowledge._chunker import Chunk
from syrin.knowledge._store import MetadataFilter, SearchResult, chunk_id


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity; assumes vectors may not be normalized. Returns value in [-1, 1]."""
    if len(a) != len(b):
        raise ValueError("Vectors must have same dimension")
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(x * x for x in b)) or 1.0
    raw = dot / (na * nb)
    # Clamp to [-1, 1] for numerical stability
    raw = max(-1.0, min(1.0, raw))
    # Map [-1, 1] to [0, 1] for score
    return (raw + 1.0) / 2.0


def _metadata_matches(chunk: Chunk, filter_spec: MetadataFilter) -> bool:
    """True if chunk metadata satisfies all filter key-value equalities."""
    for key, expected in filter_spec.items():
        actual = chunk.metadata.get(key)
        if expected != actual:
            return False
    return True


class InMemoryKnowledgeStore:
    """In-memory vector store for chunks.

    No external dependencies. Suitable for testing and ephemeral use.
    Search uses brute-force cosine similarity.
    """

    def __init__(self, embedding_dimensions: int = 1536) -> None:
        """Initialize in-memory store.

        Args:
            embedding_dimensions: Expected vector size for embeddings.
        """
        if embedding_dimensions < 1:
            raise ValueError("embedding_dimensions must be >= 1")
        self._embedding_dimensions = embedding_dimensions
        self._chunks: dict[str, Chunk] = {}
        self._embeddings: dict[str, list[float]] = {}

    async def upsert(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        """Insert or update chunks with their embeddings."""
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"chunks and embeddings length mismatch: {len(chunks)} vs {len(embeddings)}"
            )
        for chunk, emb in zip(chunks, embeddings, strict=True):
            if len(emb) != self._embedding_dimensions:
                raise ValueError(f"embedding dimension {len(emb)} != {self._embedding_dimensions}")
            cid = chunk_id(chunk)
            self._chunks[cid] = chunk
            self._embeddings[cid] = list(emb)

    async def search(
        self,
        query_embedding: list[float],
        *,
        top_k: int = 5,
        filter: MetadataFilter | None = None,
        score_threshold: float = 0.0,
    ) -> list[SearchResult]:
        """Semantic search over stored chunks."""
        if top_k <= 0:
            return []
        if len(query_embedding) != self._embedding_dimensions:
            raise ValueError(
                f"query_embedding dimension {len(query_embedding)} != {self._embedding_dimensions}"
            )
        if not 0 <= score_threshold <= 1:
            raise ValueError("score_threshold must be in [0, 1]")

        candidates: list[tuple[Chunk, float]] = []
        for cid, chunk in self._chunks.items():
            if filter is not None and not _metadata_matches(chunk, filter):
                continue
            emb = self._embeddings[cid]
            score = _cosine_similarity(query_embedding, emb)
            if score >= score_threshold:
                candidates.append((chunk, score))

        candidates.sort(key=lambda x: -x[1])
        results = [
            SearchResult(chunk=c, score=s, rank=i + 1)
            for i, (c, s) in enumerate(candidates[:top_k])
        ]
        return results

    async def delete(
        self,
        *,
        source: str | None = None,
        document_id: str | None = None,
    ) -> int:
        """Delete chunks by source or document_id. Both refer to document identifier."""
        if source is None and document_id is None:
            raise ValueError("Must provide source or document_id")
        target = source if source is not None else document_id
        assert target is not None

        to_delete = [cid for cid, c in self._chunks.items() if c.document_id == target]
        for cid in to_delete:
            del self._chunks[cid]
            del self._embeddings[cid]
        return len(to_delete)

    async def count(self) -> int:
        """Return total chunks stored."""
        return len(self._chunks)
