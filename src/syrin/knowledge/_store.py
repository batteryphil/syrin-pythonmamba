"""KnowledgeStore protocol, SearchResult, and chunk identity for vector backends."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, TypeAlias, runtime_checkable

from syrin.knowledge._chunker import Chunk

# Metadata filter for search: equality on key-value pairs. Same value types as ChunkMetadata.
MetadataFilter: TypeAlias = dict[
    str,
    str | int | float | bool | None | list[str],
]


def chunk_id(chunk: Chunk) -> str:
    """Generate a stable ID for a chunk. Used by all KnowledgeStore backends for upsert/delete.

    Args:
        chunk: The chunk to identify.

    Returns:
        Stable string ID: "{document_id}::{chunk_index}".

    Example:
        >>> chunk_id(Chunk(..., document_id="resume.pdf", chunk_index=0))
        "resume.pdf::0"
    """
    return f"{chunk.document_id}::{chunk.chunk_index}"


@dataclass
class SearchResult:
    """A single search result from the knowledge store.

    Attributes:
        chunk: The retrieved chunk.
        score: Similarity score in [0, 1] (1 = identical).
        rank: Position in results (1-based).
    """

    chunk: Chunk
    score: float
    rank: int

    def __post_init__(self) -> None:
        if not 0 <= self.score <= 1:
            raise ValueError("SearchResult.score must be in [0, 1]")
        if self.rank < 1:
            raise ValueError("SearchResult.rank must be >= 1")


@runtime_checkable
class KnowledgeStore(Protocol):
    """Protocol for vector storage backends.

    Implementations store chunks with embeddings and support semantic search.
    All methods are async.
    """

    async def upsert(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        """Insert or update chunks with their embeddings.

        Args:
            chunks: Chunks to store.
            embeddings: Embedding vectors, one per chunk. Length must match chunks.

        Raises:
            ValueError: If lengths of chunks and embeddings differ.
        """
        ...

    async def search(
        self,
        query_embedding: list[float],
        *,
        top_k: int = 5,
        filter: MetadataFilter | None = None,
        score_threshold: float = 0.0,
    ) -> list[SearchResult]:
        """Semantic search over stored chunks.

        Args:
            query_embedding: Query vector.
            top_k: Maximum number of results to return.
            filter: Optional metadata equality filter (e.g. {"source_type": "pdf"}).
            score_threshold: Minimum similarity score [0, 1]. Results below are excluded.

        Returns:
            Search results, ordered by score descending, with rank 1-based.
        """
        ...

    async def delete(
        self,
        *,
        source: str | None = None,
        document_id: str | None = None,
    ) -> int:
        """Delete chunks by source or document ID.

        Args:
            source: Delete chunks with this source (same as document_id).
            document_id: Delete chunks with this document_id.

        Returns:
            Number of chunks deleted. Must pass at least one of source or document_id.
        """
        ...

    async def count(self) -> int:
        """Return total number of chunks stored."""
        ...
