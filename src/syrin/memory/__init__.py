"""Memory module — first-class persistent memory for agents.

Supports four memory types (Core, Episodic, Semantic, Procedural),
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
    Consolidation,
    Decay,
    Memory,
    MemoryBudget,
    MemoryEntry,
)
from .conversation import BufferMemory, ConversationMemory, WindowMemory
from .embedding import EmbeddingConfig
from .snapshot import MemorySnapshot, MemorySnapshotEntry
from .store import MemoryStore
from .types import (
    CoreMemory,
    EpisodicMemory,
    ProceduralMemory,
    SemanticMemory,
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
    "MemoryBudget",
    "Consolidation",
    "MemoryEntry",
    "ConversationMemory",
    "BufferMemory",
    "WindowMemory",
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
    "CoreMemory",
    "EpisodicMemory",
    "SemanticMemory",
    "ProceduralMemory",
    "create_memory",
    "PostgresConfig",
    "QdrantConfig",
    "RedisConfig",
]
