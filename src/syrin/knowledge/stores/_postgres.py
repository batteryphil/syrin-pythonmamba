"""PostgreSQL + pgvector KnowledgeStore for production use."""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING

from syrin.knowledge._chunker import Chunk
from syrin.knowledge._store import MetadataFilter, SearchResult, chunk_id

if TYPE_CHECKING:
    pass

_POSTGRES_AVAILABLE = False
_pgvector_available = False
asyncpg = None
register_vector = None

try:
    import asyncpg as _asyncpg

    _POSTGRES_AVAILABLE = True
    asyncpg = _asyncpg
except ImportError:
    pass

if _POSTGRES_AVAILABLE:
    try:
        from pgvector.asyncpg import register_vector as _register_vector

        _pgvector_available = True
        register_vector = _register_vector
    except ImportError:
        pass

_TABLE_NAME_RE = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")


def _json_serializable(obj: object) -> object:
    """Convert to JSON-serializable form (handles list[str] etc)."""
    if isinstance(obj, list):
        return [_json_serializable(x) for x in obj]
    if isinstance(obj, (str, int, float, bool, type(None))):
        return obj
    return str(obj)


class PostgresKnowledgeStore:
    """Production knowledge store using PostgreSQL + pgvector.

    Requires: pip install asyncpg pgvector
    """

    def __init__(
        self,
        connection_url: str,
        table_name: str = "syrin_knowledge",
        embedding_dimensions: int = 1536,
    ) -> None:
        """Initialize Postgres store.

        Args:
            connection_url: PostgreSQL connection URL (e.g. postgresql://user:pass@host:5432/db).
            table_name: Table name (valid identifier).
            embedding_dimensions: Vector size.
        """
        if not _POSTGRES_AVAILABLE:
            raise ImportError("asyncpg is not installed. Install with: pip install asyncpg")
        if not _pgvector_available:
            raise ImportError("pgvector is not installed. Install with: pip install pgvector")
        if not _TABLE_NAME_RE.match(table_name):
            raise ValueError(
                f"Invalid table_name {table_name!r}. Must be valid PostgreSQL identifier."
            )
        if embedding_dimensions < 1:
            raise ValueError("embedding_dimensions must be >= 1")

        self._connection_url = connection_url
        self._table_name = table_name
        self._embedding_dimensions = embedding_dimensions
        self._pool: asyncpg.Pool | None = None

    async def _get_pool(self) -> asyncpg.Pool:
        """Create or return connection pool with pgvector registered."""
        if self._pool is None:
            self._pool = await asyncpg.create_pool(
                self._connection_url,
                min_size=1,
                max_size=10,
                init=self._init_conn,
            )
        return self._pool

    async def _init_conn(self, conn: object) -> None:
        await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        await register_vector(conn)
        await self._ensure_schema(conn)

    async def _ensure_schema(self, conn: object) -> None:
        t = self._table_name
        dim = self._embedding_dimensions
        await conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {self._quote_ident(t)} (
                id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                embedding vector({dim}),
                metadata JSONB,
                source TEXT NOT NULL,
                source_type TEXT NOT NULL,
                document_id TEXT,
                chunk_index INTEGER,
                token_count INTEGER,
                created_at TIMESTAMPTZ DEFAULT NOW(),
                updated_at TIMESTAMPTZ DEFAULT NOW()
            )
            """
        )
        # ivfflat index for cosine (vector_cosine_ops)
        idx_name = f"{t}_embedding_idx"
        await conn.execute(
            f"""
            CREATE INDEX IF NOT EXISTS {self._quote_ident(idx_name)}
            ON {self._quote_ident(t)} USING ivfflat (embedding vector_cosine_ops)
            WITH (lists = 100)
            """
        )
        await conn.execute(
            f"CREATE INDEX IF NOT EXISTS {self._quote_ident(t + '_source_idx')} ON {self._quote_ident(t)} (source)"
        )
        await conn.execute(
            f"CREATE INDEX IF NOT EXISTS {self._quote_ident(t + '_metadata_idx')} ON {self._quote_ident(t)} USING GIN (metadata)"
        )

    def _quote_ident(self, name: str) -> str:
        """Quote identifier. Name must be validated as identifier."""
        return f'"{name}"'

    async def upsert(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        """Insert or update chunks with their embeddings."""
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"chunks and embeddings length mismatch: {len(chunks)} vs {len(embeddings)}"
            )
        pool = await self._get_pool()
        t = self._quote_ident(self._table_name)
        async with pool.acquire() as conn:
            for chunk, emb in zip(chunks, embeddings, strict=True):
                if len(emb) != self._embedding_dimensions:
                    raise ValueError(
                        f"embedding dimension {len(emb)} != {self._embedding_dimensions}"
                    )
                cid = chunk_id(chunk)
                meta = {k: _json_serializable(v) for k, v in chunk.metadata.items()}
                source = chunk.document_id
                source_type = chunk.metadata.get("source_type")
                if isinstance(source_type, str):
                    pass
                else:
                    source_type = "unknown"

                await conn.execute(
                    f"""
                    INSERT INTO {t} (id, content, embedding, metadata, source, source_type, document_id, chunk_index, token_count, updated_at)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, NOW())
                    ON CONFLICT (id) DO UPDATE SET
                        content = EXCLUDED.content,
                        embedding = EXCLUDED.embedding,
                        metadata = EXCLUDED.metadata,
                        updated_at = NOW()
                    """,
                    cid,
                    chunk.content,
                    emb,
                    json.dumps(meta),
                    source,
                    source_type,
                    chunk.document_id,
                    chunk.chunk_index,
                    chunk.token_count,
                )

    async def search(
        self,
        query_embedding: list[float],
        *,
        top_k: int = 5,
        filter: MetadataFilter | None = None,
        score_threshold: float = 0.0,
    ) -> list[SearchResult]:
        """Semantic search over stored chunks. Uses cosine distance (<=>)."""
        if top_k <= 0:
            return []
        if len(query_embedding) != self._embedding_dimensions:
            raise ValueError(
                f"query_embedding dimension {len(query_embedding)} != {self._embedding_dimensions}"
            )
        if not 0 <= score_threshold <= 1:
            raise ValueError("score_threshold must be in [0, 1]")

        pool = await self._get_pool()
        t = self._quote_ident(self._table_name)
        # pgvector: <=> is cosine distance (1 - cosine_sim). Score = 1 - distance.
        if filter:
            filter_json = json.dumps({k: _json_serializable(v) for k, v in filter.items()})
            sql = f"""
                SELECT id, content, metadata, source, source_type, document_id, chunk_index, token_count,
                       (1 - (embedding <=> $1)) AS score
                FROM {t}
                WHERE (1 - (embedding <=> $1)) >= $2 AND metadata @> $4::jsonb
                ORDER BY embedding <=> $1
                LIMIT $3
            """
            args: list[object] = [
                query_embedding,
                score_threshold,
                top_k,
                filter_json,
            ]
        else:
            sql = f"""
                SELECT id, content, metadata, source, source_type, document_id, chunk_index, token_count,
                       (1 - (embedding <=> $1)) AS score
                FROM {t}
                WHERE (1 - (embedding <=> $1)) >= $2
                ORDER BY embedding <=> $1
                LIMIT $3
            """
            args = [query_embedding, score_threshold, top_k]

        async with pool.acquire() as conn:
            rows = await conn.fetch(sql, *args)

        results: list[SearchResult] = []
        for i, row in enumerate(rows):
            score = float(row["score"])
            score = max(0.0, min(1.0, score))
            meta = row["metadata"]
            if isinstance(meta, str):
                meta = json.loads(meta)
            chunk = Chunk(
                content=str(row["content"]),
                metadata=meta,
                document_id=str(row["document_id"] or row["source"]),
                chunk_index=int(row["chunk_index"] or 0),
                token_count=int(row["token_count"] or 0),
            )
            results.append(SearchResult(chunk=chunk, score=score, rank=i + 1))
        return results

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

        pool = await self._get_pool()
        t = self._quote_ident(self._table_name)
        result = await pool.execute(
            f"DELETE FROM {t} WHERE document_id = $1 OR source = $1",
            target,
        )
        # Result format: "DELETE N"
        parts = result.split()
        return int(parts[1]) if len(parts) >= 2 else 0

    async def count(self) -> int:
        """Return total chunks stored."""
        pool = await self._get_pool()
        t = self._quote_ident(self._table_name)
        row = await pool.fetchrow(f"SELECT COUNT(*) AS n FROM {t}")
        return int(row["n"]) if row else 0
