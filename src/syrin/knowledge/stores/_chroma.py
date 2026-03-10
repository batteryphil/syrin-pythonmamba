"""Chroma KnowledgeStore for local dev and lightweight vector search."""

from __future__ import annotations

import asyncio
import json
from typing import TYPE_CHECKING

from syrin.knowledge._chunker import Chunk
from syrin.knowledge._store import MetadataFilter, SearchResult, chunk_id

if TYPE_CHECKING:
    pass

_CHROMA_AVAILABLE = False
chromadb = None

try:
    import chromadb
    from chromadb.config import Settings

    _CHROMA_AVAILABLE = True
except ImportError:
    chromadb = None
    Settings = None


def _chunk_to_metadata(chunk: Chunk) -> dict[str, str | int | float | bool]:
    """Convert chunk metadata to Chroma-compatible (str, int, float, bool only)."""
    out: dict[str, str | int | float | bool] = {
        "document_id": chunk.document_id,
        "chunk_index": chunk.chunk_index,
        "token_count": chunk.token_count,
        "source": chunk.document_id,
    }
    for k, v in chunk.metadata.items():
        if v is None:
            continue
        if isinstance(v, (str, int, float, bool)):
            out[k] = v
        elif isinstance(v, list):
            out[k] = json.dumps(v)
        else:
            out[k] = str(v)
    return out


def _chunk_from_metadata(
    content: str,
    metadata: dict[str, object],
) -> Chunk:
    """Reconstruct Chunk from Chroma document and metadata."""
    meta: dict[str, str | int | float | bool | None | list[str]] = {}
    skip = {"document_id", "chunk_index", "token_count", "source"}
    for k, v in metadata.items():
        if k in skip:
            continue
        if isinstance(v, str) and v.startswith("["):
            try:
                meta[k] = json.loads(v)
            except json.JSONDecodeError:
                meta[k] = v
        elif isinstance(v, (str, int, float, bool, type(None))):
            meta[k] = v
    return Chunk(
        content=content,
        metadata=meta,
        document_id=str(metadata.get("document_id", metadata.get("source", ""))),
        chunk_index=int(metadata.get("chunk_index", 0)),
        token_count=int(metadata.get("token_count", 0)),
    )


class ChromaKnowledgeStore:
    """Chroma-based knowledge store for local development.

    Requires: pip install chromadb
    Chroma uses cosine similarity; returns distance. We convert to score: 1 - distance.
    """

    def __init__(
        self,
        embedding_dimensions: int = 1536,
        collection_name: str = "syrin_knowledge",
        path: str | None = None,
    ) -> None:
        """Initialize Chroma store.

        Args:
            embedding_dimensions: Vector size (Chroma infers from first upsert).
            collection_name: Collection name.
            path: Directory for persistent storage; None for ephemeral.
        """
        if not _CHROMA_AVAILABLE or chromadb is None:
            raise ImportError("chromadb is not installed. Install with: pip install chromadb")
        if embedding_dimensions < 1:
            raise ValueError("embedding_dimensions must be >= 1")

        self._embedding_dimensions = embedding_dimensions
        self._collection_name = collection_name
        settings = Settings(anonymized_telemetry=False)
        if path:
            self._client = chromadb.PersistentClient(path=path, settings=settings)
        else:
            self._client = chromadb.Client(settings)
        try:
            self._collection = self._client.get_collection(name=collection_name)
        except Exception:
            self._collection = self._client.create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"},
            )

    def _upsert_sync(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        ids = [chunk_id(c) for c in chunks]
        documents = [c.content for c in chunks]
        metadatas = [_chunk_to_metadata(c) for c in chunks]
        self._collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )

    async def upsert(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        """Insert or update chunks with their embeddings."""
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"chunks and embeddings length mismatch: {len(chunks)} vs {len(embeddings)}"
            )
        for _chunk, emb in zip(chunks, embeddings, strict=True):
            if len(emb) != self._embedding_dimensions:
                raise ValueError(f"embedding dimension {len(emb)} != {self._embedding_dimensions}")
        await asyncio.to_thread(
            self._upsert_sync,
            chunks,
            embeddings,
        )

    def _search_sync(
        self,
        query_embedding: list[float],
        top_k: int,
        filter: MetadataFilter | None,
        score_threshold: float,
    ) -> list[SearchResult]:
        where: dict[str, object] | None = None
        if filter:
            if len(filter) == 1:
                k, v = next(iter(filter.items()))
                where = {k: v}
            else:
                where = {"$and": [{k: v} for k, v in filter.items()]}
        result = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )
        docs = result.get("documents") or [[]]
        metas = result.get("metadatas") or [[]]
        dists = result.get("distances") or [[]]
        out: list[SearchResult] = []
        for i, doc in enumerate(docs[0] if docs else []):
            meta_row = metas[0][i] if metas and metas[0] and i < len(metas[0]) else {}
            dist = dists[0][i] if dists and dists[0] and i < len(dists[0]) else 1.0
            # Chroma cosine distance: 0 = identical, 2 = opposite. Score = 1 - distance, clamp to [0,1]
            score = max(0.0, min(1.0, 1.0 - dist))
            if score < score_threshold:
                continue
            chunk = _chunk_from_metadata(str(doc), meta_row or {})
            out.append(SearchResult(chunk=chunk, score=score, rank=len(out) + 1))
        return out

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

        return await asyncio.to_thread(
            self._search_sync,
            query_embedding,
            top_k,
            filter,
            score_threshold,
        )

    def _delete_sync(self, target: str) -> int:
        get_result = self._collection.get(
            where={"document_id": target},
            include=[],
        )
        ids = get_result.get("ids") or []
        if ids:
            self._collection.delete(ids=ids)
        return len(ids)

    async def delete(
        self,
        *,
        source: str | None = None,
        document_id: str | None = None,
    ) -> int:
        """Delete chunks by source or document_id."""
        if source is None and document_id is None:
            raise ValueError("Must provide source or document_id")
        target = source if source is not None else document_id
        assert target is not None
        return await asyncio.to_thread(self._delete_sync, target)

    def _count_sync(self) -> int:
        return self._collection.count()

    async def count(self) -> int:
        """Return total chunks stored."""
        return await asyncio.to_thread(self._count_sync)
