"""PostgreSQL backend for persistent memory storage."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import TYPE_CHECKING, Any

from syrin.enums import MemoryScope, MemoryType

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    import psycopg2

try:
    import psycopg2

    POSTGRES_AVAILABLE = True
except ImportError:
    POSTGRES_AVAILABLE = False
    psycopg2 = None

from syrin.memory.config import MemoryEntry


class PostgresBackend:
    """PostgreSQL-based storage for memories.

    Requires: pip install psycopg2-binary

    Features:
    - Persistent storage
    - SQL queries
    - Connection pooling
    - Can support vector search with pgvector extension

    Note: For vector/semantic search, ensure pgvector extension is installed
    and use the vector dimension parameter.
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 5432,
        database: str = "syrin",
        user: str = "postgres",
        password: str = "",
        table: str = "memories",
        vector_size: int = 0,
    ) -> None:
        """Initialize Postgres backend.

        Args:
            host: PostgreSQL host.
            port: PostgreSQL port.
            database: Database name.
            user: Database user.
            password: Database password.
            table: Table name for memories.
            vector_size: If > 0, enable pgvector; requires pgvector extension.
        """
        if not POSTGRES_AVAILABLE:
            raise ImportError(
                "psycopg2-binary is not installed. Install with: pip install psycopg2-binary"
            )

        self._host = host
        self._port = port
        self._database = database
        self._user = user
        self._password = password
        self._table = table
        self._vector_size = vector_size

        self._conn = psycopg2.connect(
            host=host,
            port=port,
            database=database,
            user=user,
            password=password,
        )
        self._conn.autocommit = True
        self._create_table()

    def _create_table(self) -> None:
        """Create the memories table if it doesn't exist."""
        if self._vector_size > 0:
            # Try to enable pgvector
            try:
                self._conn.cursor().execute("CREATE EXTENSION IF NOT EXISTS vector")
            except Exception:
                logger.warning("pgvector not available, vector search disabled")

        # Create table
        vector_col = f"embedding vector({self._vector_size})," if self._vector_size > 0 else ""

        self._conn.cursor().execute(f"""
            CREATE TABLE IF NOT EXISTS {self._table} (
                id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                type TEXT NOT NULL,
                importance REAL DEFAULT 1.0,
                scope TEXT DEFAULT 'user',
                source TEXT,
                created_at TIMESTAMP NOT NULL,
                last_accessed TIMESTAMP,
                access_count INTEGER DEFAULT 0,
                valid_from TIMESTAMP,
                valid_until TIMESTAMP,
                keywords TEXT DEFAULT '[]',
                related_ids TEXT DEFAULT '[]',
                supersedes TEXT,
                metadata TEXT DEFAULT '{{}}',
                {vector_col}
                importance_idx REAL GENERATED ALWAYS AS (importance) STORED
            )
        """)
        self._conn.commit()

        # Create indexes
        self._conn.cursor().execute(f"""
            CREATE INDEX IF NOT EXISTS idx_{self._table}_type ON {self._table}(type)
        """)
        self._conn.cursor().execute(f"""
            CREATE INDEX IF NOT EXISTS idx_{self._table}_importance ON {self._table}(importance_idx)
        """)

    def _row_to_entry(self, row: tuple[Any, ...]) -> MemoryEntry:
        """Convert a database row to a MemoryEntry.

        Column order: id, content, type, importance, scope, source, created_at,
        last_accessed, access_count, valid_from, valid_until, keywords, related_ids,
        supersedes, metadata, [embedding if vector_size>0], importance_idx (generated).
        """
        n = len(row)
        meta = row[14] if n > 14 else "{}"
        meta_val = json.loads(meta) if isinstance(meta, (str, bytes)) else {}
        return MemoryEntry(
            id=row[0],
            content=row[1],
            type=MemoryType(row[2]),
            importance=row[3],
            scope=MemoryScope(row[4]),
            source=row[5],
            created_at=row[6] if n > 6 else datetime.now(),
            last_accessed=row[7] if n > 7 and row[7] else None,
            access_count=row[8] if n > 8 else 0,
            keywords=json.loads(row[11]) if n > 11 and row[11] else [],
            related_ids=json.loads(row[12]) if n > 12 and row[12] else [],
            supersedes=row[13] if n > 13 else None,
            metadata=meta_val,
        )

    def add(self, memory: MemoryEntry) -> None:
        """Add a memory to PostgreSQL."""
        cursor = self._conn.cursor()
        cursor.execute(
            f"""
            INSERT INTO {self._table}
            (id, content, type, importance, scope, source, created_at, last_accessed,
             access_count, valid_from, valid_until, keywords, related_ids, supersedes, metadata)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (id) DO UPDATE SET
                content = EXCLUDED.content,
                importance = EXCLUDED.importance,
                access_count = EXCLUDED.access_count,
                last_accessed = EXCLUDED.last_accessed
            """,
            (
                memory.id,
                memory.content,
                memory.type.value,
                memory.importance,
                memory.scope.value,
                memory.source,
                memory.created_at,
                memory.last_accessed,
                memory.access_count,
                memory.valid_from,
                memory.valid_until,
                json.dumps(memory.keywords),
                json.dumps(memory.related_ids),
                memory.supersedes,
                json.dumps(memory.metadata),
            ),
        )
        self._conn.commit()

    def get(self, memory_id: str) -> MemoryEntry | None:
        """Get a memory by ID."""
        cursor = self._conn.cursor()
        cursor.execute(
            f"SELECT * FROM {self._table} WHERE id = %s",
            (memory_id,),
        )
        row = cursor.fetchone()
        return self._row_to_entry(row) if row else None

    def search(
        self,
        query: str,
        memory_type: MemoryType | None = None,
        top_k: int = 10,
    ) -> list[MemoryEntry]:
        """Search memories by content (simple text search)."""
        cursor = self._conn.cursor()

        sql = f"SELECT * FROM {self._table} WHERE content LIKE %s"
        params: list[Any] = [f"%{query}%"]

        if memory_type:
            sql += " AND type = %s"
            params.append(memory_type.value)

        sql += " ORDER BY importance DESC LIMIT %s"
        params.append(top_k)

        cursor.execute(sql, params)
        return [self._row_to_entry(row) for row in cursor.fetchall()]

    def list(
        self,
        memory_type: MemoryType | None = None,
        scope: MemoryScope | None = None,
        limit: int = 100,
    ) -> list[MemoryEntry]:
        """List all memories."""
        cursor = self._conn.cursor()

        conditions = []
        params = []

        if memory_type:
            conditions.append("type = %s")
            params.append(memory_type.value)
        if scope:
            conditions.append("scope = %s")
            params.append(scope.value)

        sql = f"SELECT * FROM {self._table}"
        if conditions:
            sql += " WHERE " + " AND ".join(conditions)

        sql += f" ORDER BY importance DESC LIMIT {limit}"

        cursor.execute(sql, params)
        return [self._row_to_entry(row) for row in cursor.fetchall()]

    def update(self, memory: MemoryEntry) -> None:
        """Update a memory."""
        self.add(memory)

    def delete(self, memory_id: str) -> None:
        """Delete a memory by ID."""
        cursor = self._conn.cursor()
        cursor.execute(f"DELETE FROM {self._table} WHERE id = %s", (memory_id,))
        self._conn.commit()

    def clear(self) -> None:
        """Clear all memories."""
        cursor = self._conn.cursor()
        cursor.execute(f"DELETE FROM {self._table}")
        self._conn.commit()

    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()


__all__ = ["PostgresBackend"]
