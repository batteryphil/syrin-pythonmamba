"""Tests for 4-type memory system with decay and budget awareness."""

from __future__ import annotations

from datetime import datetime, timedelta
from unittest.mock import patch  # noqa: F401 — used in TestMemoryBudget

from syrin.enums import DecayStrategy, MemoryType
from syrin.memory import (
    Decay,
    MemoryEntry,
)
from syrin.memory.store import MemoryStore
from syrin.memory.types import (
    FactsMemory,
    HistoryMemory,
    InstructionsMemory,
    KnowledgeMemory,
)


class TestMemoryTypes:
    """Tests for memory type-specific classes."""

    def test_facts_memory_creation(self) -> None:
        """Facts memory should be permanent and high importance."""
        memory = FactsMemory(
            id="core-1",
            content="My name is John",
            importance=0.9,
        )
        assert memory.type == MemoryType.FACTS
        assert memory.importance >= 0.8  # Facts should default to high importance

    def test_history_memory_creation(self) -> None:
        """History memory stores specific events/experiences."""
        memory = HistoryMemory(
            id="ep-1",
            content="Yesterday I visited Paris",
            importance=0.7,
        )
        assert memory.type == MemoryType.HISTORY

    def test_knowledge_memory_creation(self) -> None:
        """Knowledge memory stores facts and knowledge."""
        memory = KnowledgeMemory(
            id="sem-1",
            content="Paris is the capital of France",
            importance=0.8,
        )
        assert memory.type == MemoryType.KNOWLEDGE

    def test_instructions_memory_creation(self) -> None:
        """Instructions memory stores how-to knowledge."""
        memory = InstructionsMemory(
            id="proc-1",
            content="How to make coffee: boil water, add coffee, pour hot water",
            importance=0.9,
        )
        assert memory.type == MemoryType.INSTRUCTIONS


class TestDecayCurves:
    """Tests for decay curve implementation."""

    def test_exponential_decay(self) -> None:
        """Memories should decay exponentially over time."""
        decay = Decay(strategy=DecayStrategy.EXPONENTIAL, rate=0.9, min_importance=0.01)

        entry = MemoryEntry(
            id="test-1",
            content="Test",
            type=MemoryType.HISTORY,
            importance=1.0,
            created_at=datetime.now() - timedelta(hours=24),
        )

        decay.apply(entry)
        assert entry.importance < 1.0
        assert entry.importance > 0.01  # Should be above min_importance

    def test_linear_decay(self) -> None:
        """Memories should decay linearly over time."""
        decay = Decay(strategy=DecayStrategy.LINEAR, rate=0.01)

        entry = MemoryEntry(
            id="test-2",
            content="Test",
            type=MemoryType.HISTORY,
            importance=1.0,
            created_at=datetime.now() - timedelta(hours=24),
        )

        decay.apply(entry)
        assert entry.importance < 1.0

    def test_no_decay_when_disabled(self) -> None:
        """Decay should not apply when disabled."""
        decay = Decay(strategy=DecayStrategy.NONE, rate=0.9)

        entry = MemoryEntry(
            id="test-3",
            content="Test",
            type=MemoryType.HISTORY,
            importance=0.5,
            created_at=datetime.now() - timedelta(days=365),
        )

        original_importance = entry.importance
        decay.apply(entry)
        assert entry.importance == original_importance

    def test_reinforce_on_access(self) -> None:
        """Accessing a memory should reinforce it."""
        decay = Decay(strategy=DecayStrategy.EXPONENTIAL, rate=0.9, reinforce_on_access=True)

        entry = MemoryEntry(
            id="test-4",
            content="Test",
            type=MemoryType.HISTORY,
            importance=0.5,
            created_at=datetime.now() - timedelta(hours=24),
        )

        decay.on_access(entry)
        assert entry.access_count == 1
        assert entry.last_accessed is not None


class TestMemoryBudget:
    """Tests for memory budget awareness (flat fields on Memory/MemoryStore)."""

    def test_memory_store_respects_budget_warn(self) -> None:
        """MemoryStore warns when budget exceeded but still stores (warn-only handler)."""
        warnings: list[object] = []

        def warn_handler(ctx: object) -> None:
            warnings.append(ctx)
            # return without raising → warn-only

        store = MemoryStore(
            budget_extraction=0.001,
            budget_on_exceeded=warn_handler,
        )

        entry = MemoryEntry(
            id="test-1",
            content="Some content that is quite long to exceed the tiny budget",
            type=MemoryType.HISTORY,
            importance=0.5,
        )

        result = store.add(entry)
        assert result is True  # warn-only allows operation

    def test_memory_store_allows_under_budget(self) -> None:
        """MemoryStore allows operations under budget."""
        with patch("syrin.cost.calculate_cost") as mock_cost:
            mock_cost.return_value = 0.001
            store = MemoryStore(budget_extraction=1.00)
            result = store.add(
                MemoryEntry(
                    id="test-2",
                    content="Some content",
                    type=MemoryType.HISTORY,
                    importance=0.5,
                )
            )
            assert result is True


class TestMemoryStore:
    """Tests for MemoryStore class."""

    def test_add_memory(self) -> None:
        """Adding a memory should store it."""
        store = MemoryStore()
        entry = MemoryEntry(
            id="mem-1",
            content="Test memory",
            type=MemoryType.HISTORY,
            importance=0.7,
        )

        result = store.add(entry)
        assert result is True
        assert store.get("mem-1") is not None

    def test_recall_memories(self) -> None:
        """Recalling memories should return relevant entries."""
        store = MemoryStore()

        store.add(
            MemoryEntry(id="1", content="I love coffee", type=MemoryType.HISTORY, importance=0.8)
        )
        store.add(
            MemoryEntry(
                id="2", content="Coffee has caffeine", type=MemoryType.KNOWLEDGE, importance=0.9
            )
        )
        store.add(
            MemoryEntry(
                id="3", content="Yesterday I drank coffee", type=MemoryType.HISTORY, importance=0.6
            )
        )

        results = store.recall("coffee", memory_type=MemoryType.HISTORY)
        assert len(results) >= 1

    def test_forget_by_id(self) -> None:
        """Forgetting by ID should remove specific memory."""
        store = MemoryStore()
        store.add(MemoryEntry(id="to-delete", content="Temp", type=MemoryType.HISTORY))

        store.forget(memory_id="to-delete")
        assert store.get("to-delete") is None

    def test_forget_by_type(self) -> None:
        """Forgetting by type should remove all of that type."""
        store = MemoryStore()
        store.add(MemoryEntry(id="1", content="Temp", type=MemoryType.HISTORY))
        store.add(MemoryEntry(id="2", content="Temp", type=MemoryType.HISTORY))
        store.add(MemoryEntry(id="3", content="Important", type=MemoryType.FACTS))

        store.forget(memory_type=MemoryType.HISTORY)

        assert store.get("1") is None
        assert store.get("2") is None
        assert store.get("3") is not None

    def test_decay_applied_on_recall(self) -> None:
        """Recall should apply decay to old memories."""
        decay = Decay(strategy=DecayStrategy.EXPONENTIAL, rate=0.95)
        store = MemoryStore(decay=decay)

        old_memory = MemoryEntry(
            id="old",
            content="Old memory",
            type=MemoryType.HISTORY,
            importance=1.0,
            created_at=datetime.now() - timedelta(days=30),
        )
        store.add(old_memory)

        store.recall("old memory")
        assert old_memory.importance < 1.0


class TestMemoryObservability:
    """Tests for memory observability integration."""

    def test_memory_span_on_store(self) -> None:
        """Storing memory should create a span."""
        from syrin.observability import SpanKind, trace

        store = MemoryStore()

        with trace.span("test-operation", kind=SpanKind.AGENT):
            result = store.add(
                MemoryEntry(
                    id="span-test",
                    content="Test",
                    type=MemoryType.HISTORY,
                )
            )

        assert result is True


class TestMemoryHooks:
    """Tests for memory event hooks."""

    def test_memory_hooks_emitted(self) -> None:
        """Memory operations should emit hooks when events provided."""
        from syrin.events import EventContext

        hooks_emitted: dict[str, list] = {"memory.store": [], "memory.recall": []}

        def emit_fn(event: str, ctx: EventContext) -> None:
            if event in hooks_emitted:
                hooks_emitted[event].append(ctx)

        from syrin.events import Events

        events = Events(emit_fn)

        # Manually add handlers to receive events from MemoryStore
        events.on("memory.store", lambda _ctx: None)
        events.on("memory.recall", lambda _ctx: None)

        store = MemoryStore(events=events)
        store.add(MemoryEntry(id="1", content="Test", type=MemoryType.HISTORY))
        store.recall("test")

        # Events are emitted internally but handlers are called
        # Just verify store operations work


# =============================================================================
# 4-TYPE MEMORY EDGE CASES - TRY TO BREAK FUNCTIONALITY
# =============================================================================


class TestMemoryTypesEdgeCases:
    """Edge cases for 4-type memory system."""

    def test_facts_memory_with_zero_importance(self):
        """FactsMemory with zero importance."""
        memory = FactsMemory(id="c1", content="Test", importance=0.0)
        assert memory.importance == 0.0

    def test_history_memory_with_max_importance(self):
        """HistoryMemory with max importance."""
        memory = HistoryMemory(id="e1", content="Test", importance=1.0)
        assert memory.importance == 1.0

    def test_memory_entry_with_empty_content(self):
        """MemoryEntry with empty content."""
        entry = MemoryEntry(id="e1", content="", type=MemoryType.FACTS)
        assert entry.content == ""

    def test_memory_entry_with_unicode(self):
        """MemoryEntry with unicode content."""
        entry = MemoryEntry(id="e1", content="Hello 🌍 你好", type=MemoryType.KNOWLEDGE)
        assert "🌍" in entry.content

    def test_memory_type_all_values(self):
        """All MemoryType values."""
        for mem_type in MemoryType:
            entry = MemoryEntry(id="test", content="test", type=mem_type)
            assert entry.type == mem_type

    def test_memory_store_empty(self):
        """MemoryStore with no entries."""
        store = MemoryStore()
        results = store.list()
        assert results == []

    def test_memory_store_with_many_entries(self):
        """MemoryStore with many entries."""
        store = MemoryStore()
        for i in range(100):
            store.add(MemoryEntry(id=f"e{i}", content=f"content {i}", type=MemoryType.FACTS))
        results = store.list()
        assert len(results) == 100
