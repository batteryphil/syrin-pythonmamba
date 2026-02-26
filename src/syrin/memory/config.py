"""Memory configuration models."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field, model_validator

from syrin.budget import BudgetExceededContext
from syrin.enums import (
    DecayStrategy,
    InjectionStrategy,
    MemoryBackend,
    MemoryScope,
    MemoryType,
)

if TYPE_CHECKING:
    from syrin.memory.store import MemoryStore


class Decay(BaseModel):
    """Ebbinghaus-inspired forgetting curve.

    Memories lose importance over time unless reinforced by access.
    Use either ``rate`` (per-hour decay multiplier) or ``half_life_hours``
    (hours until importance halves); if both are set, ``half_life_hours`` wins.
    """

    strategy: DecayStrategy = DecayStrategy.EXPONENTIAL
    rate: float = Field(0.995, gt=0.0, le=1.0)
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
            return self.model_copy(update={"rate": 0.5 ** (1.0 / self.half_life_hours)})
        return self

    def apply(self, entry: MemoryEntry) -> None:
        """Apply decay to a memory entry based on age.

        Args:
            entry: The memory entry to decay
        """
        if self.strategy == DecayStrategy.NONE:
            return

        age_hours = (datetime.now() - entry.created_at).total_seconds() / 3600

        if age_hours < 0:
            return

        if self.strategy == DecayStrategy.EXPONENTIAL:
            decay_factor = self.rate**age_hours
        elif self.strategy == DecayStrategy.LINEAR:
            decay_factor = max(0, 1 - (self.rate * age_hours / 24))
        elif self.strategy == DecayStrategy.LOGARITHMIC:
            decay_factor = 1 / (1 + self.rate * (age_hours / 24))
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

    metadata: dict[str, Any] = Field(default_factory=dict)


class Memory(BaseModel):
    """Declarative memory configuration for an agent.

    Supports four memory types (Core, Episodic, Semantic, Procedural),
    pluggable backends, automatic extraction, forgetting curves, budget
    integration, and position-aware context injection.

    Also acts as a facade to MemoryStore for working with memories.
    """

    backend: MemoryBackend = MemoryBackend.MEMORY
    path: str | None = None

    types: list[MemoryType] = Field(
        default=[
            MemoryType.CORE,
            MemoryType.EPISODIC,
            MemoryType.SEMANTIC,
            MemoryType.PROCEDURAL,
        ]
    )

    auto_extract: bool = True
    extraction_model: str | None = None

    top_k: int = Field(
        10,
        gt=0,
        description="Number of top memories to recall. Must be int >= 1. Pydantic coerces numeric strings.",
    )
    relevance_threshold: float = Field(0.7, ge=0.0, le=1.0)
    injection_strategy: InjectionStrategy = InjectionStrategy.ATTENTION_OPTIMIZED

    auto_store: bool = Field(
        default=False,
        description="Automatically store user inputs and agent responses as EPISODIC memories",
    )

    decay: Decay | None = None

    memory_budget: MemoryBudget | None = None

    consolidation: Consolidation | None = None

    scope: MemoryScope = MemoryScope.USER

    redact_pii: bool = False
    retention_days: int | None = None

    def __init__(self, **data: Any) -> None:
        """Initialize Memory and set up internal store."""
        super().__init__(**data)
        self._store: MemoryStore | None = None
        self._init_store()

    def _init_store(self) -> None:
        """Initialize the underlying MemoryStore."""
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

    def recall(
        self,
        query: str = "",
        memory_type: MemoryType | None = None,
        count: int = 10,
    ) -> list[MemoryEntry]:
        """Recall memories matching query or type.

        Args:
            query: Search query
            memory_type: Filter by memory type
            count: Maximum results to return

        Returns:
            List of matching MemoryEntries, sorted by importance
        """
        if self._store is None:
            return []
        return self._store.recall(query=query, memory_type=memory_type, limit=count)

    def remember(
        self,
        content: str,
        memory_type: MemoryType = MemoryType.EPISODIC,
        importance: float = 1.0,
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        """Store a memory.

        Args:
            content: Memory content
            memory_type: Type of memory (episodic, semantic, etc.)
            importance: Importance score (0.0-1.0)
            metadata: Optional metadata

        Returns:
            True if stored successfully, False otherwise
        """
        if self._store is None:
            return False

        entry = MemoryEntry(
            id="",  # Will be generated
            content=content,
            type=memory_type,
            importance=min(1.0, max(0.0, importance)),
            scope=self.scope,
            metadata=metadata or {},
        )
        return self._store.add(entry=entry)

    def forget(
        self,
        query: str | None = None,
        memory_type: MemoryType | None = None,
        memory_id: str | None = None,
    ) -> int:
        """Forget memories by query, type, or ID.

        Args:
            query: Search query to match memories for deletion
            memory_type: Delete all memories of this type
            memory_id: Delete specific memory by ID

        Returns:
            Number of memories deleted
        """
        if self._store is None:
            return 0

        return self._store.forget(
            memory_id=memory_id,
            memory_type=memory_type,
            query=query,
        )

    def consolidate(
        self,
        *,
        deduplicate: bool | None = None,
        consolidation_budget: float | None = None,
    ) -> int:
        """Run memory consolidation (deduplicate by content). Optional, budget-aware.

        When consolidation is configured, uses its deduplicate setting; otherwise
        defaults to True. Respects memory_budget.consolidation_budget when set.

        Returns:
            Number of duplicate entries removed.
        """
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
        if self._store is None:
            return []
        return self._store.list(
            memory_type=memory_type,
            scope=scope,
            limit=limit,
        )

    model_config = {"arbitrary_types_allowed": True}


__all__ = [
    "Memory",
    "Decay",
    "MemoryBudget",
    "Consolidation",
    "MemoryEntry",
]
