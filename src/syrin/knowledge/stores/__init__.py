"""Knowledge store backends for vector storage."""

from __future__ import annotations

from typing import TYPE_CHECKING

from syrin.enums import KnowledgeBackend

from ._memory import InMemoryKnowledgeStore

__all__ = [
    "InMemoryKnowledgeStore",
    "get_knowledge_store",
]

if TYPE_CHECKING:
    from syrin.knowledge._store import KnowledgeStore


def get_knowledge_store(
    backend: KnowledgeBackend,
    *,
    embedding_dimensions: int = 1536,
    connection_url: str | None = None,
    path: str | None = None,
    table_name: str = "syrin_knowledge",
    collection: str = "syrin_knowledge",
) -> KnowledgeStore:
    """Create a KnowledgeStore for the given backend.

    Args:
        backend: Which backend to use.
        embedding_dimensions: Vector size (required for MEMORY, POSTGRES, etc.).
        connection_url: Postgres connection URL (for POSTGRES).
        path: File path for SQLite or embedded Qdrant.
        table_name: Postgres table name.
        collection: Qdrant/Chroma collection name.

    Returns:
        KnowledgeStore instance.

    Raises:
        ImportError: If optional deps for backend not installed.
        ValueError: If required kwargs missing for backend.
    """
    if backend == KnowledgeBackend.MEMORY:
        return InMemoryKnowledgeStore(embedding_dimensions=embedding_dimensions)

    if backend == KnowledgeBackend.POSTGRES:
        if not connection_url:
            raise ValueError("connection_url required for POSTGRES backend")
        from syrin.knowledge.stores._postgres import PostgresKnowledgeStore

        return PostgresKnowledgeStore(
            connection_url=connection_url,
            table_name=table_name,
            embedding_dimensions=embedding_dimensions,
        )

    if backend == KnowledgeBackend.QDRANT:
        from syrin.knowledge.stores._qdrant import QdrantKnowledgeStore

        return QdrantKnowledgeStore(
            embedding_dimensions=embedding_dimensions,
            collection=collection,
            path=path,
        )

    if backend == KnowledgeBackend.CHROMA:
        from syrin.knowledge.stores._chroma import ChromaKnowledgeStore

        return ChromaKnowledgeStore(
            embedding_dimensions=embedding_dimensions,
            collection_name=collection,
            path=path,
        )

    if backend == KnowledgeBackend.SQLITE:
        if not path:
            raise ValueError("path required for SQLITE backend")
        from syrin.knowledge.stores._sqlite import SQLiteKnowledgeStore

        return SQLiteKnowledgeStore(
            path=path,
            embedding_dimensions=embedding_dimensions,
        )

    raise ValueError(f"Unknown backend: {backend}")
