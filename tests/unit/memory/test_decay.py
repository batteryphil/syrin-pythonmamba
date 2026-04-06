"""Tests for memory decay and reinforce_on_access behavior."""

from __future__ import annotations

from datetime import datetime, timedelta

from syrin.enums import DecayStrategy, MemoryType
from syrin.memory import Decay, MemoryStore
from syrin.memory.config import MemoryEntry


class TestDecayApply:
    """Decay.apply() reduces importance by age."""

    def test_decay_none_strategy_unchanged(self) -> None:
        """DecayStrategy.NONE leaves importance unchanged."""
        decay = Decay(strategy=DecayStrategy.NONE, rate=0.995)
        entry = MemoryEntry(
            id="e1",
            content="test",
            type=MemoryType.HISTORY,
            importance=1.0,
            created_at=datetime.now() - timedelta(days=30),
        )
        decay.apply(entry)
        assert entry.importance == 1.0

    def test_decay_exponential_reduces_importance(self) -> None:
        """Exponential decay reduces importance for older entries."""
        decay = Decay(strategy=DecayStrategy.EXPONENTIAL, rate=0.995, min_importance=0.1)
        entry = MemoryEntry(
            id="e1",
            content="old",
            type=MemoryType.HISTORY,
            importance=1.0,
            created_at=datetime.now() - timedelta(hours=24 * 7),
        )
        decay.apply(entry)
        assert entry.importance < 1.0
        assert entry.importance >= 0.1

    def test_decay_min_importance_floor(self) -> None:
        """Importance never goes below min_importance."""
        decay = Decay(
            strategy=DecayStrategy.EXPONENTIAL,
            rate=0.9,
            min_importance=0.2,
        )
        entry = MemoryEntry(
            id="e1",
            content="very old",
            type=MemoryType.HISTORY,
            importance=0.5,
            created_at=datetime.now() - timedelta(days=365),
        )
        decay.apply(entry)
        assert entry.importance >= 0.2


class TestReinforceOnAccess:
    """reinforce_on_access boosts importance on recall."""

    def test_on_access_increments_access_count(self) -> None:
        """on_access increments access_count when reinforce_on_access=True."""
        decay = Decay(reinforce_on_access=True, rate=0.995)
        entry = MemoryEntry(
            id="e1",
            content="test",
            type=MemoryType.HISTORY,
            importance=0.5,
            access_count=0,
        )
        decay.on_access(entry)
        assert entry.access_count == 1
        decay.on_access(entry)
        assert entry.access_count == 2

    def test_on_access_boosts_importance(self) -> None:
        """on_access adds a small boost to importance (capped at 1.0)."""
        decay = Decay(reinforce_on_access=True, rate=0.995)
        entry = MemoryEntry(
            id="e1",
            content="test",
            type=MemoryType.HISTORY,
            importance=0.5,
            access_count=0,
        )
        decay.on_access(entry)
        assert entry.importance > 0.5
        assert entry.importance <= 1.0
        assert entry.last_accessed is not None

    def test_reinforce_on_access_false_no_boost(self) -> None:
        """When reinforce_on_access=False, on_access does not change entry."""
        decay = Decay(reinforce_on_access=False, rate=0.995)
        entry = MemoryEntry(
            id="e1",
            content="test",
            type=MemoryType.HISTORY,
            importance=0.5,
            access_count=0,
        )
        decay.on_access(entry)
        assert entry.access_count == 0
        assert entry.importance == 0.5


class TestStoreRecallAppliesDecay:
    """MemoryStore.recall() applies decay and on_access when apply_decay=True."""

    def test_recall_with_decay_applies_and_reinforces(self) -> None:
        """recall(apply_decay=True) runs decay and on_access on results."""
        decay = Decay(
            strategy=DecayStrategy.EXPONENTIAL,
            rate=0.995,
            reinforce_on_access=True,
        )
        store = MemoryStore(decay=decay)
        entry = MemoryEntry(
            id="e1",
            content="memory",
            type=MemoryType.HISTORY,
            importance=1.0,
            access_count=0,
        )
        store.add(entry=entry)

        results = store.recall(query="memory", apply_decay=True)
        assert len(results) == 1
        # Decay may have reduced; on_access may have boosted
        assert results[0].access_count == 1
        assert results[0].last_accessed is not None

    def test_recall_apply_decay_false_skips_decay(self) -> None:
        """recall(apply_decay=False) does not modify entries."""
        decay = Decay(strategy=DecayStrategy.EXPONENTIAL, rate=0.9, reinforce_on_access=True)
        store = MemoryStore(decay=decay)
        entry = MemoryEntry(
            id="e1",
            content="memory",
            type=MemoryType.HISTORY,
            importance=0.8,
            access_count=0,
        )
        store.add(entry=entry)

        results = store.recall(query="memory", apply_decay=False)
        assert len(results) == 1
        assert results[0].importance == 0.8
        assert results[0].access_count == 0
