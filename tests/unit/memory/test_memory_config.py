"""Memory configuration: budget and consolidation flat fields, Memory.types API.

Verifies:
- MemoryBudget and Consolidation classes removed; their fields absorbed as flat
  prefixed params on Memory
- Memory.types field (None = all types, list = restrict to subset)
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from syrin.enums import MemoryType
from syrin.memory import Memory, MemoryEntry, MemoryStore

# ---------------------------------------------------------------------------
# Budget flat fields
# ---------------------------------------------------------------------------


class TestMemoryBudgetFlatFields:
    """Memory exposes budget_extraction, budget_consolidation, budget_on_exceeded."""

    def test_defaults_have_no_budget(self) -> None:
        """Memory() with no args has no budget constraints."""
        mem = Memory()
        assert mem.budget_extraction is None
        assert mem.budget_consolidation is None
        assert mem.budget_on_exceeded is None

    def test_budget_extraction_accepted(self) -> None:
        """budget_extraction is stored correctly."""
        mem = Memory(budget_extraction=0.05)
        assert mem.budget_extraction == 0.05

    def test_budget_consolidation_accepted(self) -> None:
        """budget_consolidation is stored correctly."""
        mem = Memory(budget_consolidation=0.10)
        assert mem.budget_consolidation == 0.10

    def test_budget_on_exceeded_callable_accepted(self) -> None:
        """budget_on_exceeded stores callback."""
        handler_calls: list[object] = []

        def handler(ctx: object) -> None:
            handler_calls.append(ctx)

        mem = Memory(budget_on_exceeded=handler)
        assert mem.budget_on_exceeded is handler

    def test_budget_extraction_zero_is_invalid(self) -> None:
        """budget_extraction must be > 0."""
        with pytest.raises(ValidationError):
            Memory(budget_extraction=0.0)

    def test_budget_extraction_negative_is_invalid(self) -> None:
        """budget_extraction must be > 0 — negative rejected."""
        with pytest.raises(ValidationError):
            Memory(budget_extraction=-1.0)

    def test_budget_consolidation_zero_is_invalid(self) -> None:
        """budget_consolidation must be > 0."""
        with pytest.raises(ValidationError):
            Memory(budget_consolidation=0.0)

    def test_budget_consolidation_negative_is_invalid(self) -> None:
        """budget_consolidation must be > 0 — negative rejected."""
        with pytest.raises(ValidationError):
            Memory(budget_consolidation=-0.001)

    def test_all_budget_fields_together(self) -> None:
        """All budget flat fields accepted simultaneously."""
        blocked: list[object] = []

        def on_exceeded(ctx: object) -> None:
            blocked.append(ctx)
            raise RuntimeError("blocked")

        mem = Memory(
            budget_extraction=0.01,
            budget_consolidation=0.05,
            budget_on_exceeded=on_exceeded,
        )
        assert mem.budget_extraction == 0.01
        assert mem.budget_consolidation == 0.05
        assert mem.budget_on_exceeded is on_exceeded

    def test_memory_budget_class_not_exported_from_syrin(self) -> None:
        """MemoryBudget is no longer exported from syrin top-level."""
        import syrin

        assert not hasattr(syrin, "MemoryBudget"), "MemoryBudget must not be a public export"

    def test_memory_budget_class_not_exported_from_syrin_memory(self) -> None:
        """MemoryBudget is no longer exported from syrin.memory."""
        import syrin.memory as mem_module

        assert not hasattr(mem_module, "MemoryBudget"), (
            "MemoryBudget must not be exported from syrin.memory"
        )

    def test_budget_blocks_store_when_on_exceeded_raises(self) -> None:
        """When budget_on_exceeded raises, remember() returns False (SYNC mode)."""
        from syrin.enums import WriteMode

        def raiser(ctx: object) -> None:
            raise RuntimeError("budget exceeded")

        mem = Memory(
            budget_extraction=0.000001,  # tiny — exceeded by "x" * 5000
            budget_on_exceeded=raiser,
            write_mode=WriteMode.SYNC,  # ASYNC would return True immediately
        )
        result = mem.remember("x" * 5000, memory_type=MemoryType.HISTORY)
        assert result is False

    def test_budget_allows_store_when_on_exceeded_returns(self) -> None:
        """When budget_on_exceeded returns (doesn't raise), store still succeeds (SYNC mode)."""
        from syrin.enums import WriteMode

        warnings: list[object] = []

        def warn_handler(ctx: object) -> None:
            warnings.append(ctx)
            # return without raising → warn-only, store proceeds

        mem = Memory(
            budget_extraction=0.000001,  # tiny
            budget_on_exceeded=warn_handler,
            write_mode=WriteMode.SYNC,
        )
        result = mem.remember("x" * 5000, memory_type=MemoryType.HISTORY)
        assert result is True  # warn-only, store allowed
        assert len(warnings) > 0


# ---------------------------------------------------------------------------
# Consolidation flat fields
# ---------------------------------------------------------------------------


class TestMemoryConsolidationFlatFields:
    """Memory exposes consolidation_interval, _deduplicate, _compress_after, etc."""

    def test_defaults_consolidation_disabled(self) -> None:
        """Memory() default: consolidation_interval is None (disabled)."""
        mem = Memory()
        assert mem.consolidation_interval is None

    def test_consolidation_deduplicate_default_true(self) -> None:
        """consolidation_deduplicate defaults to True."""
        mem = Memory()
        assert mem.consolidation_deduplicate is True

    def test_consolidation_compress_after_default(self) -> None:
        """consolidation_compress_after defaults to None (no compression)."""
        mem = Memory()
        assert mem.consolidation_compress_after is None

    def test_consolidation_resolve_contradictions_default_true(self) -> None:
        """consolidation_resolve_contradictions defaults to True."""
        mem = Memory()
        assert mem.consolidation_resolve_contradictions is True

    def test_consolidation_model_default_none(self) -> None:
        """consolidation_model defaults to None (use agent model)."""
        mem = Memory()
        assert mem.consolidation_model is None

    def test_consolidation_interval_accepted(self) -> None:
        """consolidation_interval stored correctly."""
        mem = Memory(consolidation_interval="1h")
        assert mem.consolidation_interval == "1h"

    def test_consolidation_full_config(self) -> None:
        """All consolidation fields accepted simultaneously."""
        mem = Memory(
            consolidation_interval="2h",
            consolidation_deduplicate=False,
            consolidation_compress_after="30d",
            consolidation_resolve_contradictions=False,
            consolidation_model="gpt-4o-mini",
        )
        assert mem.consolidation_interval == "2h"
        assert mem.consolidation_deduplicate is False
        assert mem.consolidation_compress_after == "30d"
        assert mem.consolidation_resolve_contradictions is False
        assert mem.consolidation_model == "gpt-4o-mini"

    def test_consolidation_class_not_exported_from_syrin(self) -> None:
        """Consolidation is no longer exported from syrin top-level."""
        import syrin

        assert not hasattr(syrin, "Consolidation"), "Consolidation must not be a public export"

    def test_consolidation_class_not_exported_from_syrin_memory(self) -> None:
        """Consolidation is no longer exported from syrin.memory."""
        import syrin.memory as mem_module

        assert not hasattr(mem_module, "Consolidation"), (
            "Consolidation must not be exported from syrin.memory"
        )

    def test_consolidate_uses_consolidation_deduplicate(self) -> None:
        """consolidate() method uses consolidation_deduplicate flat field."""
        mem = Memory(consolidation_deduplicate=True)
        mem.remember("same content", memory_type=MemoryType.HISTORY)
        mem.remember("same content", memory_type=MemoryType.HISTORY)
        removed = mem.consolidate()
        assert isinstance(removed, int)
        assert removed >= 0

    def test_consolidate_with_deduplicate_false(self) -> None:
        """consolidate(deduplicate=False) overrides consolidation_deduplicate."""
        mem = Memory(consolidation_deduplicate=True)
        mem.remember("dup", memory_type=MemoryType.HISTORY)
        mem.remember("dup", memory_type=MemoryType.HISTORY)
        removed = mem.consolidate(deduplicate=False)
        assert removed == 0  # override to False → nothing removed


# ---------------------------------------------------------------------------
# Memory.types API
# ---------------------------------------------------------------------------


class TestMemoryRestrictTo:
    """Memory.types field API (was restrict_to)."""

    def test_default_types_is_none(self) -> None:
        """Memory() default: types is None (meaning all types enabled)."""
        mem = Memory()
        assert mem.types is None

    def test_types_accepts_subset(self) -> None:
        """types accepts a subset of memory types."""
        mem = Memory(types=[MemoryType.FACTS, MemoryType.HISTORY])
        assert MemoryType.FACTS in (mem.types or [])
        assert MemoryType.HISTORY in (mem.types or [])
        assert MemoryType.KNOWLEDGE not in (mem.types or [])
        assert MemoryType.INSTRUCTIONS not in (mem.types or [])

    def test_types_accepts_single_type(self) -> None:
        """types accepts a single memory type."""
        mem = Memory(types=[MemoryType.FACTS])
        assert mem.types == [MemoryType.FACTS]

    def test_restrict_to_field_does_not_exist(self) -> None:
        """Memory no longer has a 'restrict_to' field — use types instead."""
        mem = Memory()
        assert not hasattr(mem, "restrict_to"), "Memory.restrict_to must be removed; use types"

    def test_memory_budget_not_accepted_as_kwarg(self) -> None:
        """memory_budget= kwarg no longer accepted by Memory constructor."""
        with pytest.raises(TypeError):
            Memory(memory_budget=object())  # type: ignore[call-arg]

    def test_consolidation_not_accepted_as_kwarg(self) -> None:
        """consolidation= kwarg no longer accepted by Memory constructor."""
        with pytest.raises(TypeError):
            Memory(consolidation=object())  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# MemoryStore direct budget params
# ---------------------------------------------------------------------------


class TestMemoryStoreBudgetParams:
    """MemoryStore now accepts individual budget params (not MemoryBudget object)."""

    def test_store_default_no_budget(self) -> None:
        """MemoryStore() with no args has no budget constraints."""
        store = MemoryStore()
        entry = MemoryEntry(id="1", content="hello", type=MemoryType.HISTORY)
        result = store.add(entry)
        assert result is True

    def test_store_budget_extraction_blocks_large_content(self) -> None:
        """MemoryStore with tiny budget_extraction blocks large content when handler raises."""
        blocked: list[object] = []

        def on_exceeded(ctx: object) -> None:
            blocked.append(ctx)
            raise RuntimeError("blocked")

        store = MemoryStore(
            budget_extraction=0.000001,  # 0.000001 USD; 5000 chars → 5000/10000 = 0.5 USD estimated
            budget_on_exceeded=on_exceeded,
        )
        entry = MemoryEntry(id="2", content="x" * 5000, type=MemoryType.HISTORY)
        result = store.add(entry)
        assert result is False
        assert len(blocked) > 0

    def test_store_budget_extraction_allows_small_content(self) -> None:
        """MemoryStore with reasonable budget allows normal content."""
        store = MemoryStore(budget_extraction=10.0)
        entry = MemoryEntry(id="3", content="small", type=MemoryType.HISTORY)
        result = store.add(entry)
        assert result is True

    def test_store_no_longer_accepts_budget_object(self) -> None:
        """MemoryStore.budget kwarg no longer exists."""
        with pytest.raises(TypeError):
            MemoryStore(budget=object())  # type: ignore[call-arg]
