"""Qdrant KnowledgeStore for high-performance vector search."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from syrin.knowledge._chunker import Chunk
from syrin.knowledge._store import MetadataFilter, SearchResult, chunk_id

if TYPE_CHECKING:
    pass

_QDRANT_AVAILABLE = False
AsyncQdrantClient = None
Distance = None
PointStruct = None
VectorParams = None
Filter = None
FieldCondition = None
MatchValue = None
FilterSelector = None

try:
    from qdrant_client import AsyncQdrantClient as _AsyncQdrantClient
    from qdrant_client.models import (
        Distance as _Distance,
    )
    from qdrant_client.models import (
        FieldCondition as _FieldCondition,
    )
    from qdrant_client.models import (
        Filter as _Filter,
    )
    from qdrant_client.models import (
        FilterSelector as _FilterSelector,
    )
    from qdrant_client.models import (
        MatchValue as _MatchValue,
    )
    from qdrant_client.models import (
        PointStruct as _PointStruct,
    )
    from qdrant_client.models import (
        VectorParams as _VectorParams,
    )

    _QDRANT_AVAILABLE = True
    AsyncQdrantClient = _AsyncQdrantClient
    Distance = _Distance
    PointStruct = _PointStruct
    VectorParams = _VectorParams
    Filter = _Filter
    FieldCondition = _FieldCondition
    MatchValue = _MatchValue
    FilterSelector = _FilterSelector
except ImportError:
    pass


def _payload_from_chunk(chunk: Chunk) -> dict[str, str | int | float | bool | None]:
    """Convert chunk to JSON-serializable payload for Qdrant."""
    payload: dict[str, str | int | float | bool | None] = {
        "content": chunk.content,
        "document_id": chunk.document_id,
        "chunk_index": chunk.chunk_index,
        "token_count": chunk.token_count,
        "source": chunk.document_id,
        "source_type": str(chunk.metadata.get("source_type", "unknown")),
    }
    for k, v in chunk.metadata.items():
        if v is None or isinstance(v, (str, int, float, bool)):
            payload[k] = v
        elif isinstance(v, list):
            payload[k] = json.dumps(v)
        else:
            payload[k] = str(v)
    return payload


def _chunk_from_payload(payload: dict[str, object]) -> Chunk:
    """Reconstruct Chunk from Qdrant payload."""
    meta: dict[str, str | int | float | bool | None | list[str]] = {}
    skip = {"content", "document_id", "chunk_index", "token_count", "source", "source_type"}
    for k, v in payload.items():
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
        content=str(payload.get("content", "")),
        metadata=meta,
        document_id=str(payload.get("document_id", payload.get("source", ""))),
        chunk_index=int(payload.get("chunk_index", 0)),
        token_count=int(payload.get("token_count", 0)),
    )


class QdrantKnowledgeStore:
    """Qdrant-based knowledge store for high-performance vector search.

    Requires: pip install qdrant-client
    """

    def __init__(
        self,
        embedding_dimensions: int = 1536,
        collection: str = "syrin_knowledge",
        path: str | None = None,
        url: str | None = None,
        api_key: str | None = None,
        host: str = "localhost",
        port: int = 6333,
    ) -> None:
        """Initialize Qdrant store.

        Args:
            embedding_dimensions: Vector size.
            collection: Collection name.
            path: Local path for embedded Qdrant (or :memory:).
            url: Qdrant Cloud or remote URL.
            api_key: API key for Qdrant Cloud.
            host: Server host (when url/path not set).
            port: Server port.
        """
        if not _QDRANT_AVAILABLE:
            raise ImportError(
                "qdrant-client is not installed. Install with: pip install qdrant-client"
            )
        if embedding_dimensions < 1:
            raise ValueError("embedding_dimensions must be >= 1")

        self._embedding_dimensions = embedding_dimensions
        self._collection = collection

        if path is not None:
            self._client = AsyncQdrantClient(path=path)
        elif url is not None:
            self._client = AsyncQdrantClient(url=url, api_key=api_key)
        else:
            self._client = AsyncQdrantClient(host=host, port=port)

        self._ensured = False

    async def _ensure_collection(self) -> None:
        if self._ensured:
            return
        collections = await self._client.get_collections()
        names = [c.name for c in collections.collections]
        if self._collection not in names:
            await self._client.create_collection(
                collection_name=self._collection,
                vectors_config=VectorParams(
                    size=self._embedding_dimensions,
                    distance=Distance.COSINE,
                ),
            )
        self._ensured = True

    async def upsert(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        """Insert or update chunks with their embeddings."""
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"chunks and embeddings length mismatch: {len(chunks)} vs {len(embeddings)}"
            )
        await self._ensure_collection()

        points = []
        for chunk, emb in zip(chunks, embeddings, strict=True):
            if len(emb) != self._embedding_dimensions:
                raise ValueError(f"embedding dimension {len(emb)} != {self._embedding_dimensions}")
            cid = chunk_id(chunk)
            points.append(
                PointStruct(
                    id=cid,
                    vector=emb,
                    payload=_payload_from_chunk(chunk),
                )
            )
        await self._client.upsert(
            collection_name=self._collection,
            points=points,
        )

    async def search(
        self,
        query_embedding: list[float],
        *,
        top_k: int = 5,
        filter: MetadataFilter | None = None,
        score_threshold: float = 0.0,
    ) -> list[SearchResult]:
        """Semantic search over stored chunks. Qdrant returns cosine similarity in [0,1]."""
        if top_k <= 0:
            return []
        if len(query_embedding) != self._embedding_dimensions:
            raise ValueError(
                f"query_embedding dimension {len(query_embedding)} != {self._embedding_dimensions}"
            )
        if not 0 <= score_threshold <= 1:
            raise ValueError("score_threshold must be in [0, 1]")

        await self._ensure_collection()

        query_filter = None
        if filter:
            conditions = [
                FieldCondition(key=k, match=MatchValue(value=v)) for k, v in filter.items()
            ]
            if conditions:
                query_filter = Filter(must=conditions)

        results_raw = await self._client.query_points(
            collection_name=self._collection,
            query=query_embedding,
            limit=top_k,
            score_threshold=score_threshold,
            query_filter=query_filter,
        )

        out: list[SearchResult] = []
        for i, pt in enumerate(results_raw.points):
            score = pt.score or 0.0
            score = max(0.0, min(1.0, score))
            payload = pt.payload or {}
            chunk = _chunk_from_payload(dict(payload.items()))
            out.append(SearchResult(chunk=chunk, score=score, rank=i + 1))
        return out

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

        await self._ensure_collection()

        filter_obj = Filter(
            must=[
                FieldCondition(key="document_id", match=MatchValue(value=target)),
            ]
        )
        scroll_result = await self._client.scroll(
            collection_name=self._collection,
            scroll_filter=filter_obj,
            limit=10000,
            with_payload=False,
        )
        deleted = len(scroll_result[0])
        if deleted > 0:
            await self._client.delete(
                collection_name=self._collection,
                points_selector=FilterSelector(filter=filter_obj),
            )
        return deleted

    async def count(self) -> int:
        """Return total chunks stored."""
        await self._ensure_collection()
        result = await self._client.count(
            collection_name=self._collection,
            exact=True,
        )
        return result.count
