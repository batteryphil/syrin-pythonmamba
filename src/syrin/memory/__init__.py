"""Memory module — first-class persistent memory for agents.

Supports four memory types (Facts, History, Knowledge, Instructions),
pluggable backends, automatic extraction, forgetting curves, budget
integration, and position-aware context injection.
"""

from .backends import (
    BACKENDS,
    ChromaBackend,
    InMemoryBackend,
    PostgresBackend,
    QdrantBackend,
    RedisBackend,
    SQLiteBackend,
    get_backend,
)
from .config import (
    Decay,
    Memory,
    MemoryEntry,
)
from .embedding import EmbeddingConfig
from .snapshot import MemorySnapshot, MemorySnapshotEntry
from .store import MemoryStore
from .types import (
    FactsMemory,
    HistoryMemory,
    InstructionsMemory,
    KnowledgeMemory,
    create_memory,
)
from .vector_configs import ChromaConfig, PostgresConfig, QdrantConfig, RedisConfig

__all__ = [
    "ChromaConfig",
    "EmbeddingConfig",
    "MemorySnapshot",
    "MemorySnapshotEntry",
    "Memory",
    "Decay",
    "MemoryEntry",
    # Backends
    "InMemoryBackend",
    "SQLiteBackend",
    "QdrantBackend",
    "ChromaBackend",
    "RedisBackend",
    "PostgresBackend",
    "get_backend",
    "BACKENDS",
    # Storage
    "MemoryStore",
    # Memory types
    "FactsMemory",
    "HistoryMemory",
    "KnowledgeMemory",
    "InstructionsMemory",
    "create_memory",
    "PostgresConfig",
    "QdrantConfig",
    "RedisConfig",
]
