"""Vector memory backend configs: QdrantConfig, ChromaConfig, EmbeddingConfig."""

from __future__ import annotations

from dataclasses import dataclass

from syrin.memory.embedding import EmbeddingConfig


@dataclass(frozen=True)
class QdrantConfig:
    """Configuration for Qdrant vector backend.

    Use url for Qdrant Cloud or remote; use path for local embedded; use host/port for local server.
    Provide exactly one of: url, path, or (host, port).
    """

    url: str | None = None
    api_key: str | None = None
    collection: str = "syrin_memory"
    namespace: str | None = None
    vector_size: int = 384
    path: str | None = None
    host: str = "localhost"
    port: int = 6333
    embedding_config: EmbeddingConfig | None = None


@dataclass(frozen=True)
class ChromaConfig:
    """Configuration for Chroma vector backend.

    Use path for persistent local storage; None for ephemeral in-memory.
    """

    path: str | None = None
    collection: str = "syrin_memory"
    namespace: str | None = None
    embedding_config: EmbeddingConfig | None = None


@dataclass(frozen=True)
class RedisConfig:
    """Configuration for Redis memory backend.

    Ultra-fast, distributed cache. Use for high-throughput agents.
    Requires: pip install redis

    Attributes:
        host: Redis host. Default: localhost.
        port: Redis port. Default: 6379.
        db: Redis database number (0-15). Default: 0.
        password: Optional password for auth. Default: None.
        prefix: Key prefix for all memory keys. Default: syrin:memory:.
        ttl: Optional TTL in seconds for expiring memories. None = no expiry.

    Example:
        >>> from syrin.memory import Memory, RedisConfig
        >>> from syrin.enums import MemoryBackend
        >>>
        >>> mem = Memory(
        ...     backend=MemoryBackend.REDIS,
        ...     redis=RedisConfig(host="localhost", port=6380, prefix="syrin:demo:"),
        ... )
    """

    host: str = "localhost"
    port: int = 6379
    db: int = 0
    password: str | None = None
    prefix: str = "syrin:memory:"
    ttl: int | None = None


@dataclass(frozen=True)
class PostgresConfig:
    """Configuration for PostgreSQL memory backend.

    Enterprise-grade, SQL support. Use for production.
    Requires: pip install psycopg2-binary

    Attributes:
        host: PostgreSQL host. Default: localhost.
        port: PostgreSQL port. Default: 5432.
        database: Database name. Default: syrin.
        user: Database user. Default: postgres.
        password: Database password. Default: empty string.
        table: Table name for memories. Default: memories.
        vector_size: If > 0, enable pgvector for semantic search. Default: 0.

    Example:
        >>> from syrin.memory import Memory, PostgresConfig
        >>> from syrin.enums import MemoryBackend
        >>>
        >>> mem = Memory(
        ...     backend=MemoryBackend.POSTGRES,
        ...     postgres=PostgresConfig(database="syrin_lib_testdb", table="memories"),
        ... )
    """

    host: str = "localhost"
    port: int = 5432
    database: str = "syrin"
    user: str = "postgres"
    password: str = ""
    table: str = "memories"
    vector_size: int = 0
