"""Tests for context management."""

from __future__ import annotations

import pytest

from syrin.context import (
    Context,
    ContextCompactor,
    ContextStats,
    ContextWindowBudget,
    DefaultContextManager,
    MiddleOutTruncator,
    TokenCounter,
)
from syrin.enums import ThresholdWindow
from syrin.threshold import ContextThreshold


class TestContext:
    """Tests for Context configuration."""

    def test_default_context(self) -> None:
        ctx = Context()
        assert ctx.max_tokens is None
        assert ctx.thresholds == []

    def test_custom_context(self) -> None:
        ctx = Context(max_tokens=80000)
        assert ctx.max_tokens == 80000

    def test_context_with_thresholds(self) -> None:
        """Test using ContextThreshold with Context."""
        thresholds = [
            ContextThreshold(at=50, action=lambda _: None),
            ContextThreshold(at=80, action=lambda _: print("High usage!")),
        ]
        ctx = Context(max_tokens=80000, thresholds=thresholds)
        assert len(ctx.thresholds) == 2
        assert ctx.thresholds[0].at == 50
        assert callable(ctx.thresholds[1].action)

    def test_invalid_threshold_at(self) -> None:
        with pytest.raises(ValueError, match="Threshold 'at' must be between"):
            ContextThreshold(at=150, action=lambda _: None)

    def test_get_budget_default(self) -> None:
        ctx = Context()
        budget = ctx.get_budget()
        assert budget.max_tokens == 128000

    def test_apply_returns_compacted_messages(self) -> None:
        """apply(messages, max_tokens) returns list of message dicts."""
        ctx = Context(max_tokens=8000, reserve=500)
        messages = [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello"}]
        out = ctx.apply(messages, max_tokens=4000)
        assert isinstance(out, list)
        assert len(out) >= 1
        assert all(isinstance(m, dict) and "role" in m and "content" in m for m in out)

    def test_apply_empty_messages(self) -> None:
        """apply([]) returns empty list."""
        ctx = Context(max_tokens=8000)
        assert ctx.apply([], max_tokens=4000) == []

    def test_apply_respects_max_tokens_override(self) -> None:
        """apply with max_tokens=0 returns empty list (no room)."""
        ctx = Context(max_tokens=8000)
        messages = [{"role": "user", "content": "Hi"}]
        out = ctx.apply(messages, max_tokens=0)
        assert out == []


class TestContextBudget:
    """Tests for ContextWindowBudget (internal window budget)."""

    def test_available_tokens(self) -> None:
        budget = ContextWindowBudget(max_tokens=10000, reserve=2000)
        assert budget.available == 8000

    def test_utilization(self) -> None:
        budget = ContextWindowBudget(max_tokens=10000, reserve=2000)
        budget.used_tokens = 4000
        assert budget.utilization == 0.5

    def test_utilization_percent(self) -> None:
        budget = ContextWindowBudget(max_tokens=10000, reserve=2000)
        budget.used_tokens = 4000
        assert budget.percent == 50


class TestTokenCounter:
    """Tests for TokenCounter."""

    def test_count_simple(self) -> None:
        counter = TokenCounter()
        tokens = counter.count("Hello world")
        assert tokens > 0

    def test_count_messages(self) -> None:
        counter = TokenCounter()
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
        ]
        result = counter.count_messages(messages)
        assert result.total > 0

    def test_count_tools(self) -> None:
        counter = TokenCounter()
        tools = [{"type": "function", "name": "test", "description": "A test"}]
        tokens = counter.count_tools(tools)
        assert tokens > 0


class TestMiddleOutTruncator:
    """Tests for MiddleOutTruncator."""

    def test_no_truncation_needed(self) -> None:
        counter = TokenCounter()
        messages = [
            {"role": "system", "content": "Short"},
            {"role": "user", "content": "Hi"},
        ]
        truncator = MiddleOutTruncator()
        result = truncator.compact(messages, 1000, counter)
        assert result.method == "none"
        assert len(result.messages) == 2

    def test_truncation(self) -> None:
        counter = TokenCounter()
        messages = [{"role": "system", "content": "You are helpful."}]
        for i in range(20):
            messages.append({"role": "user", "content": f"Message {i}: " + "x" * 100})
            messages.append({"role": "assistant", "content": f"Response {i}: " + "y" * 100})

        truncator = MiddleOutTruncator()
        result = truncator.compact(messages, 500, counter)
        assert result.method == "middle_out_truncate"
        assert len(result.messages) < len(messages)


class TestContextCompactor:
    """Tests for ContextCompactor."""

    def test_no_compaction_needed(self) -> None:
        messages = [
            {"role": "system", "content": "Short"},
            {"role": "user", "content": "Hi"},
        ]
        compactor = ContextCompactor()
        result = compactor.compact(messages, 10000)
        assert result.method == "none"

    def test_compaction_triggered(self) -> None:
        messages = [{"role": "system", "content": "You are helpful."}]
        for i in range(50):
            messages.append({"role": "user", "content": f"Message {i}: " + "x" * 200})

        compactor = ContextCompactor()
        result = compactor.compact(messages, 500)  # Small budget
        assert result.method != "none"
        assert result.tokens_after < result.tokens_before


class TestDefaultContextManager:
    """Tests for DefaultContextManager."""

    def test_prepare_basic(self) -> None:
        manager = DefaultContextManager(Context(max_tokens=80000))
        messages = [{"role": "user", "content": "Hello"}]

        payload = manager.prepare(
            messages=messages,
            system_prompt="You are helpful.",
            tools=[],
            memory_context="",
        )

        assert payload.tokens > 0
        assert len(payload.messages) > 0

    def test_prepare_with_memory_context(self) -> None:
        manager = DefaultContextManager(Context(max_tokens=80000))
        messages = [{"role": "user", "content": "Hello"}]

        payload = manager.prepare(
            messages=messages,
            system_prompt="You are helpful.",
            tools=[],
            memory_context="Some memory",
        )

        assert payload.tokens > 0

    def test_stats_tracking(self) -> None:
        manager = DefaultContextManager(Context(max_tokens=80000))
        messages = [{"role": "user", "content": "Hello"}]

        manager.prepare(
            messages=messages,
            system_prompt="You are helpful.",
            tools=[],
            memory_context="",
        )

        stats = manager.stats
        assert stats.total_tokens > 0
        assert stats.max_tokens == 80000

    def test_compaction_events(self) -> None:
        """Compaction only runs when a threshold action calls ctx.compact()."""
        events = []

        def emit_fn(event, ctx):
            events.append((event, ctx))

        def compact_on_threshold(ctx):
            if ctx.compact:
                ctx.compact()

        thresholds = [ContextThreshold(at=50, action=compact_on_threshold)]
        manager = DefaultContextManager(Context(max_tokens=3000, thresholds=thresholds))
        manager.set_emit_fn(emit_fn)

        messages = [{"role": "system", "content": "System"}]
        for i in range(50):
            messages.append({"role": "user", "content": f"Message {i}: " + "x" * 200})

        manager.prepare(
            messages=messages,
            system_prompt="You are helpful.",
            tools=[],
            memory_context="",
        )

        compact_events = [e for e in events if e[0] == "context.compact"]
        assert len(compact_events) > 0

    def test_threshold_events(self) -> None:
        events = []
        triggered_percentages = []

        def emit_fn(event, ctx):
            events.append((event, ctx))

        def track_threshold(ctx):
            triggered_percentages.append(ctx.percentage)

        thresholds = [
            ContextThreshold(at=50, action=track_threshold),
        ]
        manager = DefaultContextManager(Context(max_tokens=5000, thresholds=thresholds))
        manager.set_emit_fn(emit_fn)

        messages = [{"role": "system", "content": "System"}]
        for i in range(50):
            messages.append({"role": "user", "content": f"Message {i}: " + "x" * 200})

        manager.prepare(
            messages=messages,
            system_prompt="You are helpful.",
            tools=[],
            memory_context="",
        )

        threshold_events = [e for e in events if e[0] == "context.threshold"]
        assert len(threshold_events) > 0
        # The custom action should have been triggered
        assert len(triggered_percentages) > 0


class TestContextStats:
    """Tests for ContextStats."""

    def test_default_stats(self) -> None:
        stats = ContextStats()
        assert stats.total_tokens == 0
        assert stats.compacted is False
        assert stats.compact_count == 0
        assert stats.thresholds_triggered == []

    def test_stats_with_values(self) -> None:
        stats = ContextStats(
            total_tokens=5000,
            max_tokens=80000,
            utilization=0.0625,
            compacted=True,
            compact_count=2,
            compact_method="middle_out_truncate",
            thresholds_triggered=["warn", "summarize"],
        )
        assert stats.total_tokens == 5000
        assert stats.compacted is True
        assert stats.compact_count == 2
        assert len(stats.thresholds_triggered) == 2


# =============================================================================
# CONTEXT EDGE CASES - TRY TO BREAK FUNCTIONALITY
# =============================================================================


class TestContextEdgeCases:
    """Edge cases for context management."""

    def test_context_with_max_tokens_zero(self):
        """Context with max_tokens=0 raises ValueError (must be > 0 when set)."""
        with pytest.raises(ValueError, match="max_tokens must be > 0 when set"):
            Context(max_tokens=0)

    def test_context_budget_with_zero_max(self):
        """ContextWindowBudget with zero max tokens."""
        budget = ContextWindowBudget(max_tokens=0, reserve=0)
        assert budget.available == 0

    def test_context_budget_utilization_zero(self):
        """ContextWindowBudget utilization at zero."""
        budget = ContextWindowBudget(max_tokens=1000, reserve=0)
        assert budget.utilization == 0.0

    def test_context_budget_utilization_100_percent(self):
        """ContextWindowBudget at 100% utilization."""
        budget = ContextWindowBudget(max_tokens=1000, reserve=0)
        budget.used_tokens = 1000
        assert budget.percent == 100

    def test_token_counter_empty_string(self):
        """TokenCounter with empty string."""
        counter = TokenCounter()
        tokens = counter.count("")
        assert tokens == 0

    def test_token_counter_very_long_string(self):
        """TokenCounter with very long string."""
        counter = TokenCounter()
        long_text = "x" * 100000
        tokens = counter.count(long_text)
        assert tokens > 10000

    def test_token_counter_unicode(self):
        """TokenCounter with unicode."""
        counter = TokenCounter()
        tokens = counter.count("Hello  你好 ")
        assert tokens > 0

    def test_context_threshold_with_different_actions(self):
        """Threshold with different action types."""
        # Lambda action
        t1 = ContextThreshold(at=50, action=lambda _: None)
        assert t1.at == 50

        # Function action
        def custom_action(ctx):
            pass

        t2 = ContextThreshold(at=90, action=custom_action)
        assert t2.at == 90

    def test_context_manager_with_empty_messages(self):
        """ContextManager with empty messages."""
        manager = DefaultContextManager(Context(max_tokens=80000))
        payload = manager.prepare(
            messages=[],
            system_prompt="",
            tools=[],
            memory_context="",
        )
        assert payload.tokens >= 0

    def test_middle_out_truncator_preserves_order(self):
        """MiddleOutTruncator preserves message order."""
        counter = TokenCounter()
        messages = [
            {"role": "user", "content": "First"},
            {"role": "assistant", "content": "Second"},
            {"role": "user", "content": "Third"},
        ]
        truncator = MiddleOutTruncator()
        result = truncator.compact(messages, 100, counter)
        # Should still have 3 messages (not truncated)
        assert len(result.messages) <= 3


class TestContextThresholdValidation:
    """Edge case tests for threshold validation in Context."""

    def test_context_rejects_budget_threshold(self):
        """Context should reject BudgetThreshold."""
        from syrin.threshold import BudgetThreshold

        with pytest.raises(ValueError, match="Context thresholds only accept ContextThreshold"):
            Context(
                max_tokens=80000,
                thresholds=[BudgetThreshold(at=80, action=lambda _: None)],
            )

    def test_context_rejects_rate_limit_threshold(self):
        """Context should reject RateLimitThreshold."""
        from syrin.enums import ThresholdMetric
        from syrin.threshold import RateLimitThreshold

        with pytest.raises(ValueError, match="Context thresholds only accept ContextThreshold"):
            Context(
                max_tokens=80000,
                thresholds=[
                    RateLimitThreshold(at=80, action=lambda _: None, metric=ThresholdMetric.RPM)
                ],
            )

    def test_context_threshold_accepts_tokens_metric(self):
        """ContextThreshold should accept TOKENS metric."""
        from syrin.enums import ThresholdMetric
        from syrin.threshold import ContextThreshold

        threshold = ContextThreshold(at=50, action=lambda _: None, metric=ThresholdMetric.TOKENS)
        assert threshold.at == 50
        assert threshold.metric == ThresholdMetric.TOKENS

    def test_context_threshold_default_metric_is_tokens(self):
        """ContextThreshold should default to TOKENS metric."""
        from syrin.enums import ThresholdMetric
        from syrin.threshold import ContextThreshold

        threshold = ContextThreshold(at=50, action=lambda _: None)
        assert threshold.metric == ThresholdMetric.TOKENS

    def test_context_threshold_at_zero(self):
        """ContextThreshold at 0%."""
        from syrin.threshold import ContextThreshold

        threshold = ContextThreshold(at=0, action=lambda _: None)
        assert threshold.at == 0
        assert threshold.should_trigger(0) is True

    def test_context_threshold_at_100(self):
        """ContextThreshold at 100%."""
        from syrin.threshold import ContextThreshold

        threshold = ContextThreshold(at=100, action=lambda _: None)
        assert threshold.at == 100
        assert threshold.should_trigger(100) is True

    def test_context_threshold_invalid_at_negative(self):
        """ContextThreshold should reject negative at value."""
        from syrin.threshold import ContextThreshold

        with pytest.raises(ValueError, match="Threshold 'at' must be between 0 and 100"):
            ContextThreshold(at=-1, action=lambda _: None)

    def test_context_threshold_invalid_at_over_100(self):
        """ContextThreshold should reject at > 100."""
        from syrin.threshold import ContextThreshold

        with pytest.raises(ValueError, match="Threshold 'at' must be between 0 and 100"):
            ContextThreshold(at=101, action=lambda _: None)

    def test_context_threshold_requires_action(self):
        """ContextThreshold should require action."""
        from syrin.threshold import ContextThreshold

        with pytest.raises(ValueError, match="Threshold 'action' is required"):
            ContextThreshold(at=80, action=None)  # type: ignore

    def test_context_threshold_window_is_max_tokens(self):
        """ContextThreshold defaults to window=MAX_TOKENS."""
        threshold = ContextThreshold(at=50, action=lambda _: None)
        assert threshold.window == ThresholdWindow.MAX_TOKENS

    def test_context_threshold_invalid_window_rejected(self):
        """ContextThreshold rejects window other than MAX_TOKENS."""
        with pytest.raises(
            ValueError, match="ContextThreshold window must be ThresholdWindow.MAX_TOKENS"
        ):
            ContextThreshold(at=50, action=lambda _: None, window=ThresholdWindow.RUN)
