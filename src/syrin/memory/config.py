"""Memory configuration models."""

from __future__ import annotations

import contextlib
import logging
import math
import threading
import uuid
from collections.abc import Callable
from datetime import datetime
from typing import TYPE_CHECKING, Any, Protocol, cast

from pydantic import BaseModel, Field, PrivateAttr, model_validator

from syrin.budget import BudgetExceededContext
from syrin.enums import (
    DecayStrategy,
    InjectionStrategy,
    MemoryBackend,
    MemoryScope,
    MemoryType,
    WriteMode,
)
from syrin.memory.vector_configs import (
    ChromaConfig,
    PostgresConfig,
    QdrantConfig,
    RedisConfig,
)

if TYPE_CHECKING:
    from syrin.context.store import ContextSegment
    from syrin.memory.snapshot import MemorySnapshot
    from syrin.memory.store import MemoryStore

_logger = logging.getLogger(__name__)


class Decay(BaseModel):
    """Ebbinghaus-inspired forgetting curve.

    Memories lose importance over time unless reinforced by access.
    Use either ``rate`` (per-hour decay multiplier) or ``half_life_hours``
    (hours until importance halves); if both are set, ``half_life_hours`` wins.

    Rate semantics differ by strategy:
    - EXPONENTIAL: rate = multiplier per hour (0.995 ≈ 0.5% loss/hour).
    - LINEAR: rate = fraction lost per 24h (0.1 = 10% lost/day; 0.995 = 99.5% lost/day).
    - LOGARITHMIC: rate controls decay curve steepness.
    """

    strategy: DecayStrategy = DecayStrategy.EXPONENTIAL
    rate: float = Field(
        0.995,
        gt=0.0,
        le=1.0,
        description="Per-strategy: EXPONENTIAL=per-hour multiplier; LINEAR=fraction lost per 24h; LOGARITHMIC=steepness.",
    )
    half_life_hours: float | None = Field(
        None,
        gt=0,
        description="Hours until importance halves (exponential). If set, rate is derived as 0.5**(1/half_life_hours).",
    )
    reinforce_on_access: bool = True
    min_importance: float = Field(0.1, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _apply_half_life(self) -> Decay:
        if self.half_life_hours is not None and self.strategy == DecayStrategy.EXPONENTIAL:
            # importance halves every half_life_hours: 0.5 = rate^half_life => rate = 0.5^(1/half_life_hours)
            object.__setattr__(self, "rate", 0.5 ** (1.0 / self.half_life_hours))
        return self

    def apply(self, entry: MemoryEntry) -> None:
        """Apply decay to a memory entry based on age.

        Args:
            entry: The memory entry to decay
        """
        if self.strategy == DecayStrategy.NONE:
            return

        age_hours: float = (datetime.now() - entry.created_at).total_seconds() / 3600

        if age_hours < 0:
            return

        decay_factor: float
        if self.strategy == DecayStrategy.EXPONENTIAL:
            decay_factor = math.pow(self.rate, age_hours)
        elif self.strategy == DecayStrategy.LINEAR:
            decay_factor = max(0.0, 1.0 - (self.rate * age_hours / 24))
        elif self.strategy == DecayStrategy.LOGARITHMIC:
            decay_factor = 1.0 / (1.0 + self.rate * (age_hours / 24))
        else:
            decay_factor = 1.0

        entry.importance = max(self.min_importance, entry.importance * decay_factor)

    def on_access(self, entry: MemoryEntry) -> None:
        """Called when a memory is accessed - reinforces it.

        Args:
            entry: The memory entry that was accessed
        """
        if self.reinforce_on_access:
            entry.access_count += 1
            entry.last_accessed = datetime.now()
            boost = min(0.1, 0.01 * entry.access_count)
            entry.importance = min(1.0, entry.importance + boost)


class MemoryBudget(BaseModel):
    """Budget constraints for memory operations.

    When budget is low, memory ops degrade gracefully.
    on_exceeded: If set, called when budget would be exceeded. Raise to reject the op; return to allow.
    """

    model_config = {"arbitrary_types_allowed": True}

    extraction_budget: float | None = Field(None, gt=0)
    consolidation_budget: float | None = Field(None, gt=0)
    on_exceeded: Callable[[BudgetExceededContext], None] | None = Field(
        default=None,
        description="Called when memory budget exceeded. Raise to reject store; return to allow.",
    )


class Consolidation(BaseModel):
    """Background memory consolidation — analogous to human memory during sleep.

    Runs periodically to deduplicate, compress, and resolve contradictions.
    """

    interval: str = "1h"
    deduplicate: bool = True
    compress_older_than: str | None = "7d"
    resolve_contradictions: bool = True
    model: str | None = None


class MemoryEntry(BaseModel):
    """A single memory stored by the agent. Carries full provenance metadata."""

    id: str
    content: str
    type: MemoryType
    importance: float = Field(1.0, ge=0.0, le=1.0)
    scope: MemoryScope = MemoryScope.USER

    source: str | None = None
    created_at: datetime = Field(default_factory=datetime.now)
    last_accessed: datetime | None = None
    access_count: int = 0

    valid_from: datetime | None = None
    valid_until: datetime | None = None

    keywords: list[str] = Field(default_factory=list)
    related_ids: list[str] = Field(default_factory=list)
    supersedes: str | None = None

    metadata: dict[str, object] = Field(default_factory=dict)


class MemoryBackendProtocol(Protocol):
    """Protocol for memory backends (search, list, add, delete)."""

    def search(
        self,
        query: str,
        memory_type: MemoryType | None = None,
        top_k: int = 10,
    ) -> list[MemoryEntry]: ...

    def list(
        self,
        memory_type: MemoryType | None = None,
        scope: MemoryScope | None = None,
        limit: int = 100,
    ) -> list[MemoryEntry]: ...

    def add(self, memory: MemoryEntry) -> None: ...

    def delete(self, memory_id: str) -> None: ...


class Memory(BaseModel):
    """Declarative memory configuration for an agent.

    Supports four memory types (Core, Episodic, Semantic, Procedural),
    pluggable backends, automatic extraction, forgetting curves, budget
    integration, and position-aware context injection. Acts as a facade
    to MemoryStore: use remember(), recall(), forget().

    Parameters:
        backend: Storage backend. One of: memory, sqlite, postgres, qdrant,
            chroma, redis. Default: memory.
        path: Path for file-based backends (e.g. sqlite DB path). Used when
            backend is sqlite or for embedded qdrant/chroma.
        qdrant: Qdrant-specific config when backend=QDRANT. Use url for cloud,
            path for local embedded, host/port for server.
        chroma: Chroma-specific config when backend=CHROMA.
        redis: Redis-specific config when backend=REDIS. Host, port, prefix, ttl.
        postgres: Postgres-specific config when backend=POSTGRES. Host, database,
            user, password, table.
        types: Memory types to enable. Default: all four (CORE, EPISODIC,
            SEMANTIC, PROCEDURAL).
        auto_extract: Extract facts from turns into semantic memory (when implemented).
        extraction_model: Model for extraction. None = use agent's model.
        top_k: Max memories to recall per query. Higher = more context, higher cost.
        relevance_threshold: Min similarity (0-1) for recall. Filter out low-relevance.
        injection_strategy: How to inject recalled memories into context.
        auto_store: Auto-store user+assistant turns as episodic. No remember tool needed.
        decay: Forgetting curve. Memories lose importance over time unless reinforced.
        memory_budget: Cost limits for memory ops. None = no limit.
        consolidation: Background deduplication/compression. None = disabled.
        scope: USER or SESSION. Affects isolation.
        redact_pii: Redact PII before storage.
        retention_days: Max age in days. Older memories pruned. None = no limit.
        write_mode: SYNC blocks until complete; ASYNC fire-and-forget for remember/forget.

    Example:
        >>> from syrin.memory import Memory, RedisConfig, PostgresConfig
        >>> from syrin.enums import MemoryBackend, MemoryType
        >>>
        >>> # In-memory (ephemeral)
        >>> mem = Memory()
        >>>
        >>> # Redis (fast, distributed)
        >>> mem = Memory(
        ...     backend=MemoryBackend.REDIS,
        ...     redis=RedisConfig(host="localhost", port=6379, prefix="syrin:demo:"),
        ... )
        >>>
        >>> # Postgres (production)
        >>> mem = Memory(
        ...     backend=MemoryBackend.POSTGRES,
        ...     postgres=PostgresConfig(database="syrin", table="memories"),
        ... )
    """

    _store: MemoryStore | None = PrivateAttr(default=None)
    _backend: MemoryBackendProtocol | None = PrivateAttr(default=None)
    _segment_store: Any = PrivateAttr(default=None)  # InMemoryContextStore, lazy-init

    backend: MemoryBackend = Field(
        default=MemoryBackend.MEMORY,
        description="Storage backend: memory, sqlite, postgres, qdrant, chroma, redis",
    )
    path: str | None = Field(
        default=None,
        description="Path for file-based backends (e.g. sqlite DB path)",
    )
    qdrant: QdrantConfig | None = Field(
        default=None,
        description="Qdrant-specific config when backend=QDRANT. Use url for cloud, path for local.",
    )
    chroma: ChromaConfig | None = Field(
        default=None,
        description="Chroma-specific config when backend=CHROMA.",
    )
    redis: RedisConfig | None = Field(
        default=None,
        description="Redis-specific config when backend=REDIS.",
    )
    postgres: PostgresConfig | None = Field(
        default=None,
        description="Postgres-specific config when backend=POSTGRES.",
    )

    types: list[MemoryType] = Field(
        default=[
            MemoryType.CORE,
            MemoryType.EPISODIC,
            MemoryType.SEMANTIC,
            MemoryType.PROCEDURAL,
        ],
        description="Memory types to enable. Default: all four.",
    )

    auto_extract: bool = Field(
        default=True,
        description="Extract facts from turns into semantic memory (when implemented)",
    )
    extraction_model: str | None = Field(
        default=None,
        description="Model for extraction. None = use agent's model",
    )

    top_k: int = Field(
        10,
        gt=0,
        description="Max memories to recall per query. Higher = more context, higher cost.",
    )
    relevance_threshold: float = Field(
        0.7,
        ge=0.0,
        le=1.0,
        description="Min similarity (0–1) for recall. Filter out low-relevance.",
    )
    injection_strategy: InjectionStrategy = Field(
        default=InjectionStrategy.ATTENTION_OPTIMIZED,
        description="How to inject recalled memories into context",
    )

    auto_store: bool = Field(
        default=False,
        description="Auto-store user+assistant turns as episodic. No remember tool needed.",
    )

    decay: Decay | None = Field(
        default=None,
        description="Forgetting curve. Memories lose importance over time.",
    )

    memory_budget: MemoryBudget | None = Field(
        default=None,
        description="Cost limits for memory ops. None = no limit.",
    )

    consolidation: Consolidation | None = Field(
        default=None,
        description="Background deduplication/compression. None = disabled.",
    )

    scope: MemoryScope = Field(
        default=MemoryScope.USER,
        description="USER or SESSION. Affects isolation.",
    )

    redact_pii: bool = Field(
        default=False,
        description="Redact PII before storage",
    )
    retention_days: int | None = Field(
        default=None,
        description="Max age in days. Older memories pruned. None = no limit.",
    )

    write_mode: WriteMode = Field(
        default=WriteMode.ASYNC,
        description="SYNC blocks until complete; ASYNC fire-and-forget for remember/forget.",
    )

    def __init__(
        self,
        *,
        backend: MemoryBackend = MemoryBackend.MEMORY,
        path: str | None = None,
        qdrant: QdrantConfig | None = None,
        chroma: ChromaConfig | None = None,
        redis: RedisConfig | None = None,
        postgres: PostgresConfig | None = None,
        types: list[MemoryType] | None = None,
        auto_extract: bool = True,
        extraction_model: str | None = None,
        top_k: int = 10,
        relevance_threshold: float = 0.7,
        injection_strategy: InjectionStrategy = InjectionStrategy.ATTENTION_OPTIMIZED,
        auto_store: bool = False,
        decay: Decay | None = None,
        memory_budget: MemoryBudget | None = None,
        consolidation: Consolidation | None = None,
        scope: MemoryScope = MemoryScope.USER,
        redact_pii: bool = False,
        retention_days: int | None = None,
        write_mode: WriteMode = WriteMode.ASYNC,
    ) -> None:
        """Initialize Memory. All parameters have defaults; pass only what you need."""
        super().__init__(
            backend=backend,
            path=path,
            qdrant=qdrant,
            chroma=chroma,
            redis=redis,
            postgres=postgres,
            types=types
            if types is not None
            else [MemoryType.CORE, MemoryType.EPISODIC, MemoryType.SEMANTIC, MemoryType.PROCEDURAL],
            auto_extract=auto_extract,
            extraction_model=extraction_model,
            top_k=top_k,
            relevance_threshold=relevance_threshold,
            injection_strategy=injection_strategy,
            auto_store=auto_store,
            decay=decay,
            memory_budget=memory_budget,
            consolidation=consolidation,
            scope=scope,
            redact_pii=redact_pii,
            retention_days=retention_days,
            write_mode=write_mode,
        )
        self._init_store()

    def model_post_init(self, __context: object) -> None:
        """Post-init: set up internal store or backend."""
        pass  # _init_store called from __init__ to avoid double init

    def _init_store(self) -> None:
        """Initialize MemoryStore (for MEMORY backend) or leave for lazy backend init."""
        if self.backend == MemoryBackend.MEMORY:
            from syrin.memory.store import MemoryStore

            try:
                self._store = MemoryStore(
                    decay=self.decay,
                    budget=self.memory_budget,
                )
            except Exception as e:
                import logging

                logger = logging.getLogger(__name__)
                logger.warning(f"Failed to initialize memory store: {e}")
                self._store = None

    def _backend_kwargs(self) -> dict[str, object]:
        """Build kwargs for get_backend from Memory config."""
        kwargs: dict[str, object] = {}
        if self.backend == MemoryBackend.QDRANT and self.qdrant is not None:
            qdrant_cfg = self.qdrant
            if qdrant_cfg.url is not None:
                kwargs["url"] = qdrant_cfg.url
                if qdrant_cfg.api_key is not None:
                    kwargs["api_key"] = qdrant_cfg.api_key
            elif qdrant_cfg.path is not None:
                kwargs["path"] = qdrant_cfg.path
            else:
                kwargs["host"] = qdrant_cfg.host
                kwargs["port"] = qdrant_cfg.port
            kwargs["collection"] = qdrant_cfg.collection
            kwargs["vector_size"] = qdrant_cfg.vector_size
            if qdrant_cfg.namespace is not None:
                kwargs["namespace"] = qdrant_cfg.namespace
            if qdrant_cfg.embedding_config is not None:
                kwargs["embedding_config"] = qdrant_cfg.embedding_config
                kwargs["vector_size"] = qdrant_cfg.embedding_config.dimensions
        elif self.backend == MemoryBackend.CHROMA and self.chroma is not None:
            chroma_cfg = self.chroma
            if chroma_cfg.path is not None:
                kwargs["path"] = chroma_cfg.path
            kwargs["collection_name"] = chroma_cfg.collection
            if chroma_cfg.namespace is not None:
                kwargs["namespace"] = chroma_cfg.namespace
            if chroma_cfg.embedding_config is not None:
                kwargs["embedding_config"] = chroma_cfg.embedding_config
        elif self.backend == MemoryBackend.REDIS and self.redis is not None:
            r = self.redis
            kwargs["host"] = r.host
            kwargs["port"] = r.port
            kwargs["db"] = r.db
            if r.password is not None:
                kwargs["password"] = r.password
            kwargs["prefix"] = r.prefix
            if r.ttl is not None:
                kwargs["ttl"] = r.ttl
        elif self.backend == MemoryBackend.POSTGRES and self.postgres is not None:
            p = self.postgres
            kwargs["host"] = p.host
            kwargs["port"] = p.port
            kwargs["database"] = p.database
            kwargs["user"] = p.user
            kwargs["password"] = p.password
            kwargs["table"] = p.table
            kwargs["vector_size"] = p.vector_size
        else:
            if self.path is not None:
                kwargs["path"] = self.path
        return kwargs

    def _get_backend(self) -> MemoryBackendProtocol | None:
        """Lazy-init and return backend when backend != MEMORY."""
        if self.backend == MemoryBackend.MEMORY:
            return None
        if self._backend is None:
            from syrin.memory.backends import get_backend

            kwargs = self._backend_kwargs()
            try:
                self._backend = get_backend(self.backend, **kwargs)
            except Exception as e:
                import logging

                logger = logging.getLogger(__name__)
                logger.warning(f"Failed to initialize memory backend {self.backend}: {e}")
                return None
        return self._backend

    def recall(
        self,
        query: str = "",
        memory_type: MemoryType | None = None,
        count: int = 10,
    ) -> list[MemoryEntry]:
        """Recall memories matching query or type.

        Args:
            query: Search query. Empty string lists all (up to count).
            memory_type: Filter by memory type. None = all types.
            count: Maximum results to return. Default: 10.

        Returns:
            List of matching MemoryEntries, sorted by importance.

        Example:
            >>> mem = Memory()
            >>> mem.remember("User prefers Python", memory_type=MemoryType.CORE)
            >>> entries = mem.recall(query="Python", count=5)
            >>> [e.content for e in entries]
            ['User prefers Python']
        """
        if self.backend == MemoryBackend.MEMORY:
            if self._store is None:
                return []
            return self._store.recall(query=query, memory_type=memory_type, limit=count)
        backend = self._get_backend()
        if backend is None:
            return []
        if query:
            return backend.search(query, memory_type, top_k=count)
        return backend.list(memory_type=memory_type, scope=None, limit=count)

    def _remember_sync(
        self,
        content: str,
        memory_type: MemoryType,
        importance_val: float,
        metadata: dict[str, object] | None,
    ) -> bool:
        """Synchronous remember implementation."""
        if self.backend == MemoryBackend.MEMORY:
            if self._store is None:
                return False
            entry = MemoryEntry(
                id="",
                content=content,
                type=memory_type,
                importance=importance_val,
                scope=self.scope,
                metadata=metadata or {},
            )
            return self._store.add(entry=entry)
        entry = MemoryEntry(
            id=str(uuid.uuid4()),
            content=content,
            type=memory_type,
            importance=importance_val,
            scope=self.scope,
            metadata=metadata or {},
        )
        backend = self._get_backend()
        if backend is None:
            return False
        try:
            backend.add(entry)
            return True
        except Exception:
            return False

    def remember(
        self,
        content: str,
        memory_type: MemoryType = MemoryType.EPISODIC,
        importance: float = 1.0,
        metadata: dict[str, object] | None = None,
    ) -> bool:
        """Store a memory.

        Args:
            content: Memory content (e.g. "User prefers Python").
            memory_type: Type of memory. Default: EPISODIC.
                CORE=identity/prefs, EPISODIC=events, SEMANTIC=facts, PROCEDURAL=how-to.
            importance: Importance score (0.0-1.0). Default: 1.0.
            metadata: Optional metadata dict. Default: None.

        Returns:
            True if stored successfully (or accepted for async write), False otherwise.

        Example:
            >>> mem = Memory()
            >>> mem.remember("User name is Alice", memory_type=MemoryType.CORE)
            True
        """
        importance_val = min(1.0, max(0.0, importance))
        if self.write_mode == WriteMode.ASYNC:

            def _do() -> None:
                self._remember_sync(content, memory_type, importance_val, metadata)

            threading.Thread(target=_do, daemon=True).start()
            return True
        return self._remember_sync(content, memory_type, importance_val, metadata)

    def _forget_sync(
        self,
        query: str | None,
        memory_type: MemoryType | None,
        memory_id: str | None,
    ) -> int:
        """Synchronous forget implementation."""
        if self.backend == MemoryBackend.MEMORY:
            if self._store is None:
                return 0
            return self._store.forget(
                memory_id=memory_id,
                memory_type=memory_type,
                query=query,
            )
        backend = self._get_backend()
        if backend is None:
            return 0
        deleted = 0
        if memory_id:
            backend.delete(memory_id)
            return 1
        memories = backend.list(memory_type=memory_type, scope=None, limit=1000)
        for mem in memories:
            if query and query.lower() not in (mem.content or "").lower():
                continue
            backend.delete(mem.id)
            deleted += 1
        return deleted

    def forget(
        self,
        query: str | None = None,
        memory_type: MemoryType | None = None,
        memory_id: str | None = None,
    ) -> int:
        """Forget memories by query, type, or ID.

        Args:
            query: Search query to match memories for deletion. None = not used.
            memory_type: Delete all memories of this type. None = not used.
            memory_id: Delete specific memory by ID. None = not used.
                Provide at most one of query, memory_type, memory_id.

        Returns:
            Number of memories deleted (or 1/0 for async when ASYNC write_mode).

        Example:
            >>> mem = Memory()
            >>> mem.remember("Old fact", memory_type=MemoryType.EPISODIC)
            True
            >>> mem.forget(query="Old fact")
            1
        """
        if self.write_mode == WriteMode.ASYNC:

            def _do() -> None:
                try:
                    self._forget_sync(query, memory_type, memory_id)
                except Exception as e:  # noqa: BLE001
                    _logger.debug(
                        "Async forget failed (backend may be torn down): %s",
                        e,
                        exc_info=True,
                    )

            threading.Thread(target=_do, daemon=True).start()
            return 1 if memory_id else 0
        return self._forget_sync(query, memory_type, memory_id)

    def export(self) -> MemorySnapshot:
        """Export memories as a MemorySnapshot for serialization or backup.

        Returns:
            MemorySnapshot with version, memories, and metadata.
        """
        from syrin.memory.snapshot import MemorySnapshot, MemorySnapshotEntry

        entries = self.entries(limit=10000)
        memories = [
            MemorySnapshotEntry(
                id=e.id,
                content=e.content,
                type=e.type.value,
                importance=e.importance,
                scope=e.scope.value,
                created_at=e.created_at.isoformat() if e.created_at else None,
                metadata=dict(e.metadata),
            )
            for e in entries
        ]
        meta: dict[str, object] = {"exported_at": datetime.now().isoformat()}
        if self.qdrant and self.qdrant.namespace is not None:
            meta["namespace"] = self.qdrant.namespace
        elif self.chroma and self.chroma.namespace is not None:
            meta["namespace"] = self.chroma.namespace
        return MemorySnapshot(version=1, memories=memories, metadata=meta)

    def import_from(self, snapshot: MemorySnapshot) -> int:
        """Import memories from a MemorySnapshot. Appends; does not clear existing.

        Args:
            snapshot: MemorySnapshot to import.

        Returns:
            Number of memories imported.
        """
        from syrin.enums import MemoryType

        count = 0
        for m in snapshot.memories:
            try:
                mem_type = (
                    MemoryType(m.type)
                    if m.type in ("core", "episodic", "semantic", "procedural")
                    else MemoryType.EPISODIC
                )
                ok = self._remember_sync(
                    content=m.content,
                    memory_type=mem_type,
                    importance_val=m.importance,
                    metadata=dict(m.metadata),
                )
                if ok:
                    count += 1
            except Exception:
                pass
        return count

    def consolidate(
        self,
        *,
        deduplicate: bool | None = None,
        consolidation_budget: float | None = None,
    ) -> int:
        """Run memory consolidation (deduplicate by content). Optional, budget-aware.

        When consolidation is configured, uses its deduplicate setting; otherwise
        defaults to True. Respects memory_budget.consolidation_budget when set.
        Only MEMORY backend supports consolidation; vector backends return 0.

        Returns:
            Number of duplicate entries removed.
        """
        if self.backend != MemoryBackend.MEMORY:
            return 0
        if self._store is None:
            return 0
        dedup = (
            deduplicate
            if deduplicate is not None
            else getattr(self.consolidation, "deduplicate", True)
        )
        budget = consolidation_budget
        if budget is None and self.memory_budget is not None:
            budget = self.memory_budget.consolidation_budget
        return self._store.consolidate(
            deduplicate=dedup,
            consolidation_budget=budget,
        )

    def _get_segment_store(self) -> Any:
        """Lazy-init internal segment store for pull-based context."""
        if self._segment_store is None:
            from syrin.context.store import InMemoryContextStore

            object.__setattr__(self, "_segment_store", InMemoryContextStore())
        return self._segment_store

    def add_conversation_segment(self, content: str, role: str = "user") -> None:
        """Store a conversation segment for pull-based context retrieval.

        Used when formation_mode=PULL. Segments are retrieved by relevance to
        the current prompt. Memory is the single long-term storage for both
        facts (remember/recall) and conversation segments.
        """
        from syrin.context.store import ContextSegment

        store = self._get_segment_store()
        store.add_segment(ContextSegment(content=content, role=role))

    def get_relevant_segments(
        self,
        query: str,
        top_k: int = 10,
        threshold: float = 0.0,
    ) -> list[tuple[ContextSegment, float]]:
        """Return conversation segments most relevant to query, for pull-based context."""
        from syrin.context.store import ContextSegment as _CS

        store = self._get_segment_store()
        result = store.get_relevant(query=query, top_k=top_k, threshold=threshold)
        return cast(list[tuple[_CS, float]], result)

    def get_conversation_messages(self, limit: int = 10000) -> list[Any]:
        """Return conversation segments as messages in order, for push-based context.

        Used when formation_mode=PUSH. Converts stored segments to Message objects.
        """
        from syrin.enums import MessageRole
        from syrin.types import Message

        store = self._get_segment_store()
        segments = store.list_recent(n=limit)
        return [
            Message(
                role=MessageRole(s.role)
                if s.role in ("user", "assistant", "system")
                else MessageRole.USER,
                content=s.content,
            )
            for s in segments
        ]

    def load_conversation_messages(self, messages: list[Any]) -> None:
        """Replace conversation segments with the given messages. For checkpoint restore."""
        from syrin.context.store import ContextSegment

        store = self._get_segment_store()
        store.clear()
        for m in messages:
            if isinstance(m, dict):
                role_str = str(m.get("role", "user"))
                content = str(m.get("content", ""))
            else:
                role = getattr(m, "role", "user")
                role_str = role.value if hasattr(role, "value") else str(role)
                content = getattr(m, "content", "") or ""
            store.add_segment(ContextSegment(content=content, role=role_str))

    def entries(
        self,
        memory_type: MemoryType | None = None,
        scope: MemoryScope | None = None,
        limit: int = 100,
    ) -> list[MemoryEntry]:
        """Return memories, optionally filtered by type or scope.

        Args:
            memory_type: Filter by memory type.
            scope: Filter by scope.
            limit: Max results (default 100).

        Returns:
            List of MemoryEntry. Empty if store not initialized.

        Example:
            >>> entries = memory.entries(limit=20)
        """
        if self.backend == MemoryBackend.MEMORY:
            if self._store is None:
                return []
            return self._store.list(
                memory_type=memory_type,
                scope=scope,
                limit=limit,
            )
        backend = self._get_backend()
        if backend is None:
            return []
        return backend.list(memory_type=memory_type, scope=scope, limit=limit)

    def close(self) -> None:
        """Close backend connections (SQLite, Redis, Postgres). Idempotent."""
        backend = self._backend
        if backend is not None:
            close_fn = getattr(backend, "close", None)
            if callable(close_fn):
                with contextlib.suppress(Exception):
                    close_fn()
                self._backend = None

    def __del__(self) -> None:
        """Close backend on GC to avoid unclosed connection warnings."""
        with contextlib.suppress(Exception):
            self.close()

    def get_remote_config_schema(self, section_key: str) -> tuple[Any, dict[str, object]]:
        """RemoteConfigurable: return (schema, current_values) for the memory section."""
        from syrin.remote._schema import build_section_schema_from_obj
        from syrin.remote._types import ConfigSchema

        if section_key != "memory":
            return (ConfigSchema(section="memory", class_name="Memory", fields=[]), {})
        return build_section_schema_from_obj(self, "memory", "Memory")

    def apply_remote_overrides(
        self,
        agent: Any,
        pairs: list[tuple[str, object]],
        section_schema: Any,
    ) -> None:
        """RemoteConfigurable: apply memory overrides to agent._persistent_memory."""
        from syrin.remote._resolver_helpers import build_nested_update, merge_nested_update

        update = build_nested_update(section_schema, pairs, "memory")
        if not update:
            return
        current = getattr(agent, "_persistent_memory", None)
        if current is None:
            return
        object.__setattr__(
            agent, "_persistent_memory", merge_nested_update(current, update, Memory)
        )

    model_config = {"arbitrary_types_allowed": True}


__all__ = [
    "Memory",
    "Decay",
    "MemoryBudget",
    "Consolidation",
    "MemoryEntry",
]
