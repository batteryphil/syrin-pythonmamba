"""SQLite + sqlite-vec KnowledgeStore for zero-config, single-file storage."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from syrin.knowledge._chunker import Chunk
from syrin.knowledge._store import MetadataFilter, SearchResult, chunk_id

_SQLITE_VEC_AVAILABLE = False
sqlite_vec: object = None

try:
    import sqlite_vec as _sqlite_vec_mod

    _SQLITE_VEC_AVAILABLE = True
    sqlite_vec = _sqlite_vec_mod
except ImportError:
    sqlite_vec = None


def _json_serializable(obj: object) -> object:
    if isinstance(obj, list):
        return [_json_serializable(x) for x in obj]
    if isinstance(obj, (str, int, float, bool, type(None))):
        return obj
    return str(obj)


class SQLiteKnowledgeStore:
    """SQLite knowledge store with sqlite-vec for vector search.

    Requires: pip install sqlite-vec
    """

    def __init__(
        self,
        path: str,
        embedding_dimensions: int = 1536,
    ) -> None:
        """Initialize SQLite store.

        Args:
            path: Path to SQLite database file.
            embedding_dimensions: Vector size.
        """
        if not _SQLITE_VEC_AVAILABLE or sqlite_vec is None:
            raise ImportError("sqlite-vec is not installed. Install with: pip install sqlite-vec")
        if embedding_dimensions < 1:
            raise ValueError("embedding_dimensions must be >= 1")

        self._path = str(Path(path).resolve())
        self._embedding_dimensions = embedding_dimensions

    def _get_conn(self) -> object:
        import sqlite3

        conn = sqlite3.connect(self._path)
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.enable_load_extension(False)
        conn.execute(
            "CREATE TABLE IF NOT EXISTS syrin_knowledge ("
            "id TEXT PRIMARY KEY, "
            "content TEXT NOT NULL, "
            "embedding BLOB, "
            "metadata TEXT, "
            "document_id TEXT NOT NULL, "
            "chunk_index INTEGER, "
            "token_count INTEGER)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_syrin_knowledge_document_id "
            "ON syrin_knowledge(document_id)"
        )
        conn.commit()
        return conn

    def _upsert_sync(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        conn = self._get_conn()
        try:
            for chunk, emb in zip(chunks, embeddings, strict=True):
                blob = sqlite_vec.serialize_float32(emb)
                meta_json = json.dumps(
                    {k: _json_serializable(v) for k, v in chunk.metadata.items()}
                )
                conn.execute(
                    "INSERT OR REPLACE INTO syrin_knowledge "
                    "(id, content, embedding, metadata, document_id, chunk_index, token_count) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (
                        chunk_id(chunk),
                        chunk.content,
                        blob,
                        meta_json,
                        chunk.document_id,
                        chunk.chunk_index,
                        chunk.token_count,
                    ),
                )
            conn.commit()
        finally:
            conn.close()

    async def upsert(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        """Insert or update chunks with their embeddings."""
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"chunks and embeddings length mismatch: {len(chunks)} vs {len(embeddings)}"
            )
        for _chunk, emb in zip(chunks, embeddings, strict=True):
            if len(emb) != self._embedding_dimensions:
                raise ValueError(f"embedding dimension {len(emb)} != {self._embedding_dimensions}")
        await asyncio.to_thread(self._upsert_sync, chunks, embeddings)

    def _search_sync(
        self,
        query_embedding: list[float],
        top_k: int,
        filter: MetadataFilter | None,
        score_threshold: float,
    ) -> list[SearchResult]:
        conn = self._get_conn()
        try:
            query_blob = sqlite_vec.serialize_float32(query_embedding)
            rows = conn.execute(
                "SELECT id, content, metadata, document_id, chunk_index, token_count, "
                "vec_distance_cosine(embedding, ?) AS dist FROM syrin_knowledge "
                "ORDER BY dist LIMIT ?",
                (query_blob, top_k),
            ).fetchall()

            out: list[SearchResult] = []
            for _i, row in enumerate(rows):
                dist = float(row[6])
                score = max(0.0, min(1.0, 1.0 - dist))
                if score < score_threshold:
                    continue
                meta = json.loads(row[2]) if row[2] else {}
                if filter and any(meta.get(k) != v for k, v in filter.items()):
                    continue
                chunk = Chunk(
                    content=str(row[1]),
                    metadata=meta,
                    document_id=str(row[3]),
                    chunk_index=int(row[4] or 0),
                    token_count=int(row[5] or 0),
                )
                out.append(SearchResult(chunk=chunk, score=score, rank=len(out) + 1))
            return out
        finally:
            conn.close()

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
        conn = self._get_conn()
        try:
            cur = conn.execute(
                "DELETE FROM syrin_knowledge WHERE document_id = ?",
                (target,),
            )
            n = cur.rowcount
            conn.commit()
            return n or 0
        finally:
            conn.close()

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
        conn = self._get_conn()
        try:
            row = conn.execute("SELECT COUNT(*) FROM syrin_knowledge").fetchone()
            return int(row[0]) if row else 0
        finally:
            conn.close()

    async def count(self) -> int:
        """Return total chunks stored."""
        return await asyncio.to_thread(self._count_sync)
