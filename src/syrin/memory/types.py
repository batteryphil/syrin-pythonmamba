"""Memory type classes for Facts, History, Knowledge, and Instructions memory."""

from __future__ import annotations

from datetime import datetime
from typing import TypedDict, Unpack

from syrin.enums import MemoryScope, MemoryType
from syrin.memory.config import MemoryEntry


class MemoryEntryKwargs(TypedDict, total=False):
    """Optional kwargs for MemoryEntry subclasses."""

    source: str | None
    created_at: datetime
    last_accessed: datetime | None
    access_count: int
    valid_from: datetime | None
    valid_until: datetime | None
    keywords: list[str]
    related_ids: list[str]
    supersedes: str | None
    metadata: dict[str, object]


class FactsMemory(MemoryEntry):  # type: ignore[explicit-any]
    """Facts memory - persistent facts about the agent/user.

    Facts memories are high-importance, long-lasting facts that should
    rarely decay. Examples: user name, preferences, identity.
    """

    def __init__(
        self,
        id: str,
        content: str,
        importance: float = 0.9,
        scope: MemoryScope = MemoryScope.USER,
        **kwargs: Unpack[MemoryEntryKwargs],
    ) -> None:
        super().__init__(
            id=id,
            content=content,
            type=MemoryType.FACTS,
            importance=min(importance, 0.9),  # Facts defaults high
            scope=scope,
            **kwargs,
        )


class HistoryMemory(MemoryEntry):  # type: ignore[explicit-any]
    """History memory - specific events and experiences.

    History memories capture specific moments, conversations, or events.
    They decay over time unless reinforced. Examples: what happened yesterday.
    """

    def __init__(
        self,
        id: str,
        content: str,
        importance: float = 0.7,
        scope: MemoryScope = MemoryScope.USER,
        **kwargs: Unpack[MemoryEntryKwargs],
    ) -> None:
        super().__init__(
            id=id,
            content=content,
            type=MemoryType.HISTORY,
            importance=importance,
            scope=scope,
            **kwargs,
        )


class KnowledgeMemory(MemoryEntry):  # type: ignore[explicit-any]
    """Knowledge memory - facts and knowledge.

    Knowledge memories store factual knowledge that can be recalled
    regardless of when it was learned. Examples: facts, definitions.
    """

    def __init__(
        self,
        id: str,
        content: str,
        importance: float = 0.8,
        scope: MemoryScope = MemoryScope.USER,
        **kwargs: Unpack[MemoryEntryKwargs],
    ) -> None:
        super().__init__(
            id=id,
            content=content,
            type=MemoryType.KNOWLEDGE,
            importance=importance,
            scope=scope,
            **kwargs,
        )


class InstructionsMemory(MemoryEntry):  # type: ignore[explicit-any]
    """Instructions memory - how-to knowledge and skills.

    Instructions memories store instructions and procedures. They should
    decay slowly as they represent learned skills. Examples: how to make coffee.
    """

    def __init__(
        self,
        id: str,
        content: str,
        importance: float = 0.85,
        scope: MemoryScope = MemoryScope.USER,
        **kwargs: Unpack[MemoryEntryKwargs],
    ) -> None:
        super().__init__(
            id=id,
            content=content,
            type=MemoryType.INSTRUCTIONS,
            importance=importance,
            scope=scope,
            **kwargs,
        )


def create_memory(
    memory_type: MemoryType,
    id: str,
    content: str,
    importance: float | None = None,
    **kwargs: Unpack[MemoryEntryKwargs],
) -> MemoryEntry:
    """Factory function to create memory entries by type.

    Args:
        memory_type: The type of memory to create
        id: Unique identifier for the memory
        content: The memory content
        importance: Optional importance (type-specific default if not provided)
        **kwargs: Additional fields for MemoryEntry

    Returns:
        A MemoryEntry of the appropriate type

    Example:
        >>> mem = create_memory(MemoryType.FACTS, "user-name", "My name is John")
        >>> assert mem.type == MemoryType.FACTS
    """
    defaults = {
        MemoryType.FACTS: 0.9,
        MemoryType.HISTORY: 0.7,
        MemoryType.KNOWLEDGE: 0.8,
        MemoryType.INSTRUCTIONS: 0.85,
    }

    imp = importance if importance is not None else defaults.get(memory_type, 0.5)

    if memory_type == MemoryType.FACTS:
        return FactsMemory(id=id, content=content, importance=imp, **kwargs)
    elif memory_type == MemoryType.HISTORY:
        return HistoryMemory(id=id, content=content, importance=imp, **kwargs)
    elif memory_type == MemoryType.KNOWLEDGE:
        return KnowledgeMemory(id=id, content=content, importance=imp, **kwargs)
    elif memory_type == MemoryType.INSTRUCTIONS:
        return InstructionsMemory(id=id, content=content, importance=imp, **kwargs)
    else:
        return MemoryEntry(id=id, content=content, type=memory_type, importance=imp, **kwargs)


__all__ = [
    "FactsMemory",
    "HistoryMemory",
    "InstructionsMemory",
    "KnowledgeMemory",
    "MemoryEntryKwargs",
    "create_memory",
]
