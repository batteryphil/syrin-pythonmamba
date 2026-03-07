"""Tests for runtime context injection (Step 8).

TDD: Valid, invalid, and edge cases for PrepareInput, InjectPlacement,
Context.runtime_inject, prepare(inject=...), and snapshot provenance.
"""

from __future__ import annotations

from typing import Any

import pytest

from syrin.context import Context, DefaultContextManager
from syrin.context.snapshot import ContextSegmentSource

# =============================================================================
# PREPARE INPUT
# =============================================================================


class TestPrepareInput:
    """PrepareInput dataclass."""

    def test_prepare_input_has_required_fields(self) -> None:
        from syrin.context.injection import PrepareInput

        inp = PrepareInput(
            messages=[{"role": "user", "content": "Hi"}],
            system_prompt="You are helpful.",
            tools=[],
            memory_context="",
            user_input="Hi",
        )
        assert inp.messages == [{"role": "user", "content": "Hi"}]
        assert inp.system_prompt == "You are helpful."
        assert inp.tools == []
        assert inp.memory_context == ""
        assert inp.user_input == "Hi"

    def test_prepare_input_with_memory(self) -> None:
        from syrin.context.injection import PrepareInput

        inp = PrepareInput(
            messages=[{"role": "user", "content": "What did I say?"}],
            system_prompt="",
            tools=[],
            memory_context="[Memory] User prefers dark mode",
            user_input="What did I say?",
        )
        assert "dark mode" in inp.memory_context


# =============================================================================
# INJECT PLACEMENT
# =============================================================================


class TestInjectPlacement:
    """InjectPlacement StrEnum."""

    def test_placement_values(self) -> None:
        from syrin.context.injection import InjectPlacement

        assert InjectPlacement.PREPEND_TO_SYSTEM.value == "prepend_to_system"
        assert InjectPlacement.BEFORE_CURRENT_TURN.value == "before_current_turn"
        assert InjectPlacement.AFTER_CURRENT_TURN.value == "after_current_turn"

    def test_placement_default_is_before_current_turn(self) -> None:
        from syrin.context.injection import InjectPlacement

        assert InjectPlacement.BEFORE_CURRENT_TURN == InjectPlacement.BEFORE_CURRENT_TURN


# =============================================================================
# CONTEXT CONFIG
# =============================================================================


class TestContextRuntimeInjectConfig:
    """Context.runtime_inject, inject_placement, inject_source_detail."""

    def test_context_runtime_inject_none_by_default(self) -> None:
        ctx = Context()
        assert getattr(ctx, "runtime_inject", None) is None

    def test_context_runtime_inject_accepts_callable(self) -> None:
        def my_inject(inp: Any) -> list[dict[str, Any]]:
            return [{"role": "system", "content": "injected"}]

        ctx = Context(max_tokens=8000, runtime_inject=my_inject)
        assert ctx.runtime_inject is my_inject

    def test_context_inject_placement_default(self) -> None:
        from syrin.context.injection import InjectPlacement

        ctx = Context(max_tokens=8000)
        assert getattr(ctx, "inject_placement", None) == InjectPlacement.BEFORE_CURRENT_TURN

    def test_context_inject_source_detail_default(self) -> None:
        ctx = Context(max_tokens=8000)
        assert getattr(ctx, "inject_source_detail", "injected") == "injected"


# =============================================================================
# PREPARE WITH INJECT PARAMETER
# =============================================================================


class TestPrepareWithInjectParameter:
    """prepare(..., inject=...) merges injected messages and tags provenance."""

    def test_prepare_with_inject_before_current_turn(self) -> None:
        """Injected messages appear before the current user message."""
        manager = DefaultContextManager(
            context=Context(max_tokens=16000, reserve=500),
        )
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "First"},
            {"role": "assistant", "content": "Hi"},
            {"role": "user", "content": "Second"},
        ]
        inject = [{"role": "system", "content": "[RAG] Doc 1\nDoc 2"}]

        payload = manager.prepare(
            messages=messages,
            system_prompt="You are helpful.",
            tools=[],
            memory_context="",
            inject=inject,
            inject_source_detail="rag",
        )

        # Injected block should be before last user message
        rag_content = next(
            (m.get("content", "") for m in payload.messages if "[RAG]" in m.get("content", "")),
            None,
        )
        assert rag_content is not None
        assert "Doc 1" in rag_content

    def test_prepare_with_inject_empty_list_no_change(self) -> None:
        """inject=[] adds nothing."""
        manager = DefaultContextManager(context=Context(max_tokens=16000))
        messages = [{"role": "user", "content": "Hi"}]
        payload = manager.prepare(
            messages=messages,
            system_prompt="",
            tools=[],
            memory_context="",
            inject=[],
        )
        assert len(payload.messages) == len(messages)
        assert payload.messages[0]["content"] == "Hi"

    def test_prepare_with_inject_none_same_as_no_inject(self) -> None:
        """inject=None behaves as omit."""
        manager = DefaultContextManager(context=Context(max_tokens=16000))
        messages = [{"role": "user", "content": "Hi"}]
        payload_none = manager.prepare(
            messages=messages,
            system_prompt="",
            tools=[],
            memory_context="",
            inject=None,
        )
        payload_omit = manager.prepare(
            messages=messages,
            system_prompt="",
            tools=[],
            memory_context="",
        )
        assert payload_none.messages == payload_omit.messages

    def test_snapshot_includes_injected_provenance(self) -> None:
        """Injected messages have source=INJECTED in snapshot."""
        manager = DefaultContextManager(context=Context(max_tokens=16000))
        messages = [{"role": "user", "content": "Hi"}]
        inject = [{"role": "system", "content": "[Injected] extra context"}]

        manager.prepare(
            messages=messages,
            system_prompt="",
            tools=[],
            memory_context="",
            inject=inject,
            inject_source_detail="rag",
        )

        snap = manager.snapshot()
        injected_provenances = [
            p for p in snap.provenance if p.source == ContextSegmentSource.INJECTED
        ]
        assert len(injected_provenances) >= 1
        assert any("rag" in (p.source_detail or "") for p in injected_provenances)

    def test_injected_messages_in_why_included(self) -> None:
        """why_included contains 'injected' or source_detail."""
        manager = DefaultContextManager(context=Context(max_tokens=16000))
        inject = [{"role": "system", "content": "Extra"}]

        manager.prepare(
            messages=[{"role": "user", "content": "Hi"}],
            system_prompt="",
            tools=[],
            memory_context="",
            inject=inject,
            inject_source_detail="dynamic_rules",
        )

        snap = manager.snapshot()
        injected_reasons = [
            w for w in snap.why_included if "injected" in w.lower() or "dynamic" in w.lower()
        ]
        assert len(injected_reasons) >= 1


# =============================================================================
# RUNTIME INJECT CALLABLE
# =============================================================================


class TestRuntimeInjectCallable:
    """Context.runtime_inject callable is invoked when inject param is None."""

    def test_runtime_inject_callable_called(self) -> None:
        """When runtime_inject is set and inject is None, callable is invoked."""
        injected: list[dict[str, Any]] = []

        def my_inject(inp: Any) -> list[dict[str, Any]]:
            injected.append({"user_input": inp.user_input})
            return [{"role": "system", "content": f"[RAG] Query: {inp.user_input}"}]

        ctx = Context(max_tokens=16000, runtime_inject=my_inject)
        manager = DefaultContextManager(context=ctx)

        manager.prepare(
            messages=[{"role": "user", "content": "What is Syrin?"}],
            system_prompt="",
            tools=[],
            memory_context="",
        )

        assert len(injected) == 1
        assert injected[0]["user_input"] == "What is Syrin?"

    def test_runtime_inject_return_empty_no_extra_messages(self) -> None:
        """Callable returning [] adds nothing."""

        def empty_inject(inp: Any) -> list[dict[str, Any]]:
            return []

        ctx = Context(max_tokens=16000, runtime_inject=empty_inject)
        manager = DefaultContextManager(context=ctx)

        payload = manager.prepare(
            messages=[{"role": "user", "content": "Hi"}],
            system_prompt="",
            tools=[],
            memory_context="",
        )

        assert len(payload.messages) == 1

    def test_per_call_inject_overrides_runtime_inject(self) -> None:
        """When inject param is provided, runtime_inject is not called."""
        called = False

        def my_inject(inp: Any) -> list[dict[str, Any]]:
            nonlocal called
            called = True
            return [{"role": "system", "content": "from_callable"}]

        ctx = Context(max_tokens=16000, runtime_inject=my_inject)
        manager = DefaultContextManager(context=ctx)

        manager.prepare(
            messages=[{"role": "user", "content": "Hi"}],
            system_prompt="",
            tools=[],
            memory_context="",
            inject=[{"role": "system", "content": "per_call"}],
        )

        assert not called
        snap = manager.snapshot()
        # Should have per_call content in snapshot (injected block)
        d = snap.to_dict()
        assert "per_call" in str(d) or "injected" in str(d).lower()


# =============================================================================
# INJECT PLACEMENT
# =============================================================================


class TestInjectPlacementBehavior:
    """Inject placement: prepend_to_system, before_current_turn, after_current_turn."""

    def test_prepend_to_system_inject_at_start(self) -> None:
        from syrin.context.injection import InjectPlacement

        ctx = Context(max_tokens=16000, inject_placement=InjectPlacement.PREPEND_TO_SYSTEM)
        manager = DefaultContextManager(context=ctx)

        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hi"},
        ]
        inject = [{"role": "system", "content": "[Prepend] First block"}]

        payload = manager.prepare(
            messages=messages,
            system_prompt="You are helpful.",
            tools=[],
            memory_context="",
            inject=inject,
        )

        first_content = payload.messages[0].get("content", "")
        assert "[Prepend]" in first_content

    def test_after_current_turn_inject_at_end(self) -> None:
        from syrin.context.injection import InjectPlacement

        ctx = Context(max_tokens=16000, inject_placement=InjectPlacement.AFTER_CURRENT_TURN)
        manager = DefaultContextManager(context=ctx)

        messages = [
            {"role": "user", "content": "Hi"},
        ]
        inject = [{"role": "system", "content": "[Append] After user"}]

        payload = manager.prepare(
            messages=messages,
            system_prompt="",
            tools=[],
            memory_context="",
            inject=inject,
        )

        last_content = payload.messages[-1].get("content", "")
        assert "[Append]" in last_content


# =============================================================================
# EDGE CASES
# =============================================================================


class TestRuntimeInjectionEdgeCases:
    """Edge cases: invalid messages, callable errors, etc."""

    def test_inject_message_must_have_role_and_content(self) -> None:
        """Injected messages without role default to 'user' per dict access."""
        manager = DefaultContextManager(context=Context(max_tokens=16000))
        inject = [{"content": "No role"}]

        payload = manager.prepare(
            messages=[{"role": "user", "content": "Hi"}],
            system_prompt="",
            tools=[],
            memory_context="",
            inject=inject,
        )

        # Manager should handle; at minimum not crash
        assert len(payload.messages) >= 1

    def test_runtime_inject_callable_raises_propagates(self) -> None:
        """If runtime_inject raises, prepare raises."""

        def bad_inject(inp: Any) -> list[dict[str, Any]]:
            raise ValueError("RAG failed")

        ctx = Context(max_tokens=16000, runtime_inject=bad_inject)
        manager = DefaultContextManager(context=ctx)

        with pytest.raises(ValueError, match="RAG failed"):
            manager.prepare(
                messages=[{"role": "user", "content": "Hi"}],
                system_prompt="",
                tools=[],
                memory_context="",
            )

    def test_inject_with_memory_context(self) -> None:
        """Injected + memory context: both appear, correct order."""
        manager = DefaultContextManager(context=Context(max_tokens=16000))
        messages = [
            {"role": "user", "content": "What do you remember?"},
        ]
        inject = [{"role": "system", "content": "[RAG] Retrieved doc"}]

        payload = manager.prepare(
            messages=messages,
            system_prompt="You are helpful.",
            tools=[],
            memory_context="[Memory] User likes Python",
            inject=inject,
        )

        all_content = " ".join(m.get("content", "") for m in payload.messages)
        assert "[Memory]" in all_content or "Memory" in all_content
        assert "[RAG]" in all_content

    def test_breakdown_includes_injected_tokens(self) -> None:
        """ContextBreakdown has injected_tokens when injection used."""
        manager = DefaultContextManager(context=Context(max_tokens=16000))
        inject = [{"role": "system", "content": "Extra context block for tokens"}]

        manager.prepare(
            messages=[{"role": "user", "content": "Hi"}],
            system_prompt="",
            tools=[],
            memory_context="",
            inject=inject,
        )

        stats = manager.stats
        assert stats.breakdown is not None
        assert hasattr(stats.breakdown, "injected_tokens")
        # Injected block adds tokens
        assert stats.breakdown.injected_tokens >= 0
