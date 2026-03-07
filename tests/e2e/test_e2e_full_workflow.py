"""
E2E: Full agent workflow tests with every feature combined.

No internal mocks. Uses Almock provider for real call stack execution.
Tests: agent + tools + memory + budget + threshold + ratelimit + hooks +
       guardrails + context + observability + checkpoint + structured output.
"""

from __future__ import annotations

import asyncio

import pytest

from syrin import (
    Agent,
    Budget,
    ContentFilter,
    Context,
    GuardrailChain,
    Hook,
    LengthGuardrail,
    Memory,
    MemoryType,
    Model,
    RateLimit,
    ReactLoop,
    SingleShotLoop,
    TokenLimits,
    TokenRateLimit,
    raise_on_exceeded,
    warn_on_exceeded,
)
from syrin.budget import stop_on_exceeded
from syrin.exceptions import BudgetExceededError, BudgetThresholdError
from syrin.memory import Memory
from syrin.threshold import BudgetThreshold
from syrin.tool import tool


def _almock(**kwargs) -> Model:
    defaults = {"latency_seconds": 0.01, "lorem_length": 50}
    defaults.update(kwargs)
    return Model.Almock(**defaults)


# =============================================================================
# 1. BASIC FULL WORKFLOW
# =============================================================================


class TestBasicFullWorkflow:
    """Agent creation → response → valid output (no mocks)."""

    def test_agent_creates_and_responds(self) -> None:
        agent = Agent(model=_almock(), system_prompt="You are a test bot.")
        r = agent.response("Say hello")
        assert r.content
        assert isinstance(r.content, str)
        assert len(r.content) > 0

    def test_response_has_all_required_fields(self) -> None:
        agent = Agent(model=_almock())
        r = agent.response("Test")
        assert isinstance(r.cost, float)
        assert r.cost >= 0
        assert r.tokens is not None
        assert r.tokens.total_tokens >= 0
        assert r.tokens.input_tokens >= 0
        assert r.tokens.output_tokens >= 0
        assert r.stop_reason is not None
        assert r.content is not None

    def test_async_response_parity(self) -> None:
        """arun() returns the same shape as response()."""
        agent = Agent(model=_almock())
        sync_r = agent.response("Hello")
        async_r = asyncio.get_event_loop().run_until_complete(agent.arun("Hello"))
        assert type(sync_r) is type(async_r)
        assert isinstance(async_r.content, str)
        assert async_r.cost >= 0
        assert async_r.tokens is not None

    def test_response_with_empty_input(self) -> None:
        agent = Agent(model=_almock())
        r = agent.response("")
        assert r.content is not None

    def test_response_with_unicode_input(self) -> None:
        agent = Agent(model=_almock())
        r = agent.response("你好世界 🌍 مرحبا")
        assert r.content is not None
        assert len(r.content) > 0

    def test_response_with_very_long_input(self) -> None:
        agent = Agent(model=_almock())
        r = agent.response("x" * 10000)
        assert r.content is not None

    def test_multiple_sequential_responses(self) -> None:
        """Same agent instance handles multiple sequential calls."""
        agent = Agent(model=_almock())
        responses = [agent.response(f"Message {i}") for i in range(5)]
        assert len(responses) == 5
        assert all(r.content for r in responses)
        assert all(r.cost >= 0 for r in responses)


# =============================================================================
# 2. AGENT WITH TOOLS
# =============================================================================


class TestAgentWithTools:
    """Agent + tools (Almock doesn't call tools, but schema is set up)."""

    def test_agent_with_single_tool(self) -> None:
        @tool
        def greet(name: str) -> str:
            return f"Hello, {name}!"

        agent = Agent(model=_almock(), tools=[greet])
        r = agent.response("Greet Alice")
        assert r.content is not None

    def test_agent_with_multiple_tools(self) -> None:
        @tool
        def add(a: int, b: int) -> int:
            return a + b

        @tool
        def multiply(a: int, b: int) -> int:
            return a * b

        agent = Agent(model=_almock(), tools=[add, multiply])
        r = agent.response("Calculate 2+3")
        assert r.content is not None

    def test_tool_execution_directly(self) -> None:
        @tool
        def calculate(expr: str) -> str:
            return str(eval(expr))

        agent = Agent(model=_almock(), tools=[calculate])
        result = agent._execute_tool("calculate", {"expr": "2 + 3"})
        assert result == "5"

    def test_tool_execution_error_is_caught(self) -> None:
        from syrin.exceptions import ToolExecutionError

        @tool
        def fail_tool() -> str:
            raise ValueError("Intentional failure")

        agent = Agent(model=_almock(), tools=[fail_tool])
        with pytest.raises(ToolExecutionError, match="Intentional failure"):
            agent._execute_tool("fail_tool", {})

    def test_unknown_tool_raises(self) -> None:
        from syrin.exceptions import ToolExecutionError

        agent = Agent(model=_almock(), tools=[])
        with pytest.raises(ToolExecutionError, match="Unknown tool"):
            agent._execute_tool("nonexistent", {})


# =============================================================================
# 3. AGENT WITH BUDGET (ALL LIMITS)
# =============================================================================


class TestAgentWithBudget:
    """Budget enforcement: run limit, rate limits, thresholds, reserve."""

    def test_budget_tracks_cost(self) -> None:
        agent = Agent(model=_almock(), budget=Budget(run=10.0))
        agent.response("Hello")
        state = agent.budget_state
        assert state is not None
        assert state.spent > 0
        tracker = agent.get_budget_tracker()
        assert tracker is not None
        assert len(tracker.get_state().get("cost_history", [])) >= 1

    def test_budget_remaining_decreases(self) -> None:
        budget = Budget(run=10.0)
        agent = Agent(model=_almock(), budget=budget)
        before = budget.remaining
        agent.response("Hello")
        after = budget.remaining
        assert after is not None and before is not None
        assert after < before

    def test_budget_warn_on_exceeded_continues(self) -> None:
        agent = Agent(
            model=_almock(),
            budget=Budget(run=0.0, on_exceeded=warn_on_exceeded),
        )
        r = agent.response("Hello")
        assert r.content is not None

    def test_budget_raise_on_exceeded_stops(self) -> None:
        agent = Agent(
            model=_almock(),
            budget=Budget(run=0.0, on_exceeded=raise_on_exceeded),
        )
        with pytest.raises(BudgetExceededError) as exc:
            agent.response("Hello")
        assert exc.value.budget_type == "run"

    def test_budget_stop_on_exceeded(self) -> None:
        agent = Agent(
            model=_almock(),
            budget=Budget(run=0.0, on_exceeded=stop_on_exceeded),
        )
        with pytest.raises(BudgetThresholdError):
            agent.response("Hello")

    def test_budget_with_reserve(self) -> None:
        budget = Budget(run=1.0, reserve=0.5)
        assert budget.remaining == 0.5  # effective = run - reserve
        agent = Agent(model=_almock(), budget=budget)
        agent.response("Hello")
        assert budget.remaining is not None

    def test_budget_with_rate_limit(self) -> None:
        agent = Agent(
            model=_almock(),
            budget=Budget(run=10.0, per=RateLimit(hour=100.0)),
        )
        r = agent.response("Hello")
        assert r.content is not None
        tracker = agent.get_budget_tracker()
        assert tracker is not None
        assert tracker.get_summary().hourly_cost >= 0

    def test_budget_threshold_fires(self) -> None:
        fired = []

        def on_threshold(ctx):
            fired.append(ctx.percentage)

        budget = Budget(
            run=10.0,
            thresholds=[BudgetThreshold(at=0, action=on_threshold)],
        )
        agent = Agent(model=_almock(), budget=budget)
        agent.response("Hello")
        # Threshold at 0% should fire after any spend
        assert len(fired) >= 1

    def test_budget_consume_callback(self) -> None:
        budget = Budget(run=10.0)
        agent = Agent(model=_almock(), budget=budget)
        agent._budget.consume(1.5)
        assert agent._budget_tracker.current_run_cost == 1.5
        assert budget.remaining == 8.5

    def test_budget_shared_flag(self) -> None:
        budget = Budget(run=10.0, shared=True)
        assert budget.shared is True
        assert "shared=True" in str(budget)


# =============================================================================
# 4. AGENT WITH TOKEN LIMITS
# =============================================================================


class TestAgentWithTokenLimits:
    """Token-based budget limits (separate from USD budget)."""

    def test_token_limits_run_exceeded(self) -> None:
        agent = Agent(
            model=_almock(),
            budget=Budget(run=100.0, on_exceeded=raise_on_exceeded),
            context=Context(token_limits=TokenLimits(run=1, on_exceeded=raise_on_exceeded)),
        )
        with pytest.raises(BudgetExceededError) as exc:
            agent.response("Hello")
        assert exc.value.budget_type == "run_tokens"

    def test_token_limits_per_hour(self) -> None:
        agent = Agent(
            model=_almock(),
            budget=Budget(run=100.0, on_exceeded=raise_on_exceeded),
            context=Context(
                token_limits=TokenLimits(
                    per=TokenRateLimit(hour=1),
                    on_exceeded=raise_on_exceeded,
                )
            ),
        )
        with pytest.raises(BudgetExceededError) as exc:
            agent.response("Hello")
        assert exc.value.budget_type == "hour_tokens"


# =============================================================================
# 5. AGENT WITH MEMORY (ALL TYPES)
# =============================================================================


class TestAgentWithMemory:
    """Memory: buffer, window, persistent (4-type), decay."""

    def test_memory_retains_conversation(self) -> None:
        """Memory retains conversation; build_messages includes history."""
        mem = Memory()
        mem.add_conversation_segment("Message 1", role="user")
        mem.add_conversation_segment("Reply 1", role="assistant")
        agent = Agent(model=_almock(), memory=mem)
        msgs = agent._build_messages("Message 2")
        assert len(msgs) >= 3

    def test_persistent_memory_remember_recall_forget(self) -> None:
        agent = Agent(model=_almock(), memory=Memory())
        mid = agent.remember("User name is Alice", memory_type=MemoryType.CORE)
        assert isinstance(mid, str) and len(mid) > 0

        entries = agent.recall("Alice")
        assert len(entries) >= 1
        assert any("Alice" in e.content for e in entries)

        deleted = agent.forget(memory_id=mid)
        assert deleted == 1

        entries_after = agent.recall("Alice")
        assert len(entries_after) == 0

    def test_persistent_memory_all_four_types(self) -> None:
        agent = Agent(model=_almock(), memory=Memory())
        agent.remember("My name is John", memory_type=MemoryType.CORE)
        agent.remember("Visited Paris yesterday", memory_type=MemoryType.EPISODIC)
        agent.remember("Python uses indentation", memory_type=MemoryType.SEMANTIC)
        agent.remember("To make tea: boil water", memory_type=MemoryType.PROCEDURAL)

        all_mems = agent.recall()
        assert len(all_mems) == 4

        core = agent.recall(memory_type=MemoryType.CORE)
        assert len(core) == 1
        assert "John" in core[0].content

    def test_persistent_memory_forget_by_query(self) -> None:
        agent = Agent(model=_almock(), memory=Memory())
        agent.remember("Old data")
        agent.remember("Keep this")
        deleted = agent.forget(query="Old")
        assert deleted == 1
        remaining = agent.recall()
        assert len(remaining) == 1

    def test_persistent_memory_forget_no_match(self) -> None:
        agent = Agent(model=_almock(), memory=Memory())
        agent.remember("Some data")
        deleted = agent.forget(query="nonexistent")
        assert deleted == 0

    def test_memory_disabled_with_false(self) -> None:
        agent = Agent(model=_almock(), memory=False)
        r = agent.response("Hello")
        assert r.content is not None
        assert agent._persistent_memory is None

    def test_no_memory_raises_on_remember(self) -> None:
        """Agent with memory=False raises on remember()."""
        agent = Agent(model=_almock(), memory=False)
        with pytest.raises(RuntimeError, match="No persistent memory"):
            agent.remember("test")

    def test_no_memory_raises_on_recall(self) -> None:
        """Agent with memory=False raises on recall()."""
        agent = Agent(model=_almock(), memory=False)
        with pytest.raises(RuntimeError, match="No persistent memory"):
            agent.recall()


# =============================================================================
# 6. AGENT WITH HOOKS / EVENTS
# =============================================================================


class TestAgentWithHooks:
    """Lifecycle hooks fire with correct data."""

    def test_on_start_and_complete_hooks(self) -> None:
        events = []
        agent = Agent(model=_almock())
        agent.events.on(Hook.AGENT_RUN_START, lambda ctx: events.append(("start", ctx)))
        agent.events.on(Hook.AGENT_RUN_END, lambda ctx: events.append(("end", ctx)))
        agent.response("Hello")
        assert len(events) == 2
        assert events[0][0] == "start"
        assert events[1][0] == "end"

    def test_hook_receives_context_data(self) -> None:
        contexts = []
        agent = Agent(model=_almock())
        agent.events.on(Hook.AGENT_RUN_END, lambda ctx: contexts.append(ctx))
        agent.response("Hello")
        assert len(contexts) == 1
        ctx = contexts[0]
        assert "content" in ctx
        assert "cost" in ctx
        assert "tokens" in ctx
        assert "duration" in ctx

    def test_before_handler_modifies_context(self) -> None:
        modified = []
        agent = Agent(model=_almock())
        agent.events.before(
            Hook.AGENT_RUN_START,
            lambda ctx: (ctx.update({"custom_field": True}), modified.append(True)),
        )
        agent.response("Hello")
        assert len(modified) >= 1

    def test_on_all_receives_every_event(self) -> None:
        all_events = []
        agent = Agent(model=_almock())
        agent.events.on_all(lambda hook, _: all_events.append(hook))
        agent.response("Hello")
        assert len(all_events) >= 2  # at least start + end

    def test_multiple_handlers_per_hook(self) -> None:
        counts = {"a": 0, "b": 0}
        agent = Agent(model=_almock())
        agent.events.on(Hook.AGENT_RUN_START, lambda _: counts.update(a=counts["a"] + 1))
        agent.events.on(Hook.AGENT_RUN_START, lambda _: counts.update(b=counts["b"] + 1))
        agent.response("Hello")
        assert counts["a"] == 1
        assert counts["b"] == 1


# =============================================================================
# 7. AGENT WITH GUARDRAILS
# =============================================================================


class TestAgentWithGuardrails:
    """Input/output guardrails block or pass content."""

    def test_content_filter_blocks_input(self) -> None:
        chain = GuardrailChain([ContentFilter(blocked_words=["forbidden"])])
        agent = Agent(model=_almock(), guardrails=chain)
        r = agent.response("This is forbidden content")
        assert r.report.guardrail.blocked is True
        assert r.report.guardrail.input_passed is False

    def test_content_filter_passes_clean_input(self) -> None:
        chain = GuardrailChain([ContentFilter(blocked_words=["forbidden"])])
        agent = Agent(model=_almock(), guardrails=chain)
        r = agent.response("This is clean content")
        assert r.content is not None
        assert r.report.guardrail.input_passed is True

    def test_length_guardrail_blocks_long_input(self) -> None:
        chain = GuardrailChain([LengthGuardrail(max_length=10)])
        agent = Agent(model=_almock(), guardrails=chain)
        r = agent.response("This is a very long input that exceeds the limit")
        assert r.report.guardrail.blocked is True

    def test_length_guardrail_passes_short_input(self) -> None:
        chain = GuardrailChain([LengthGuardrail(max_length=1000)])
        agent = Agent(model=_almock(), guardrails=chain)
        r = agent.response("Short")
        assert r.content is not None

    def test_multiple_guardrails_chained(self) -> None:
        chain = GuardrailChain(
            [
                ContentFilter(blocked_words=["bad"]),
                LengthGuardrail(max_length=1000),
            ]
        )
        agent = Agent(model=_almock(), guardrails=chain)
        r = agent.response("Good content")
        assert r.content is not None


# =============================================================================
# 8. AGENT WITH CONTEXT
# =============================================================================


class TestAgentWithContext:
    """Context management and token tracking."""

    def test_agent_with_context_config(self) -> None:
        ctx = Context(max_tokens=4000)
        agent = Agent(model=_almock(), context=ctx)
        r = agent.response("Hello")
        assert r.content is not None

    def test_context_stats_populated(self) -> None:
        agent = Agent(model=_almock(), context=Context(max_tokens=4000))
        agent.response("Hello")
        stats = agent.context_stats
        assert stats is not None


# =============================================================================
# 9. AGENT WITH OBSERVABILITY
# =============================================================================


class TestAgentWithObservability:
    """Debug mode, tracing, spans."""

    def test_debug_mode_runs_without_error(self) -> None:
        agent = Agent(model=_almock(), debug=True)
        r = agent.response("Hello")
        assert r.content is not None

    def test_agent_report_populated(self) -> None:
        agent = Agent(model=_almock())
        agent.response("Hello")
        report = agent.report
        assert report is not None
        assert report.tokens.total_tokens >= 0


# =============================================================================
# 10. AGENT WITH EVERYTHING COMBINED
# =============================================================================


class TestAgentWithEverything:
    """Agent with ALL features enabled simultaneously."""

    def test_full_feature_agent(self) -> None:
        """Agent with memory + budget + tools + guardrails + hooks + context."""
        events_log = []

        @tool
        def lookup(query: str) -> str:
            return f"Result for: {query}"

        chain = GuardrailChain([ContentFilter(blocked_words=["forbidden"])])

        agent = Agent(
            model=_almock(),
            system_prompt="You are a helpful assistant with full capabilities.",
            tools=[lookup],
            memory=Memory(),
            budget=Budget(run=10.0, per=RateLimit(hour=100.0)),
            context=Context(max_tokens=8000),
            guardrails=chain,
        )
        agent.events.on(Hook.AGENT_RUN_START, lambda _: events_log.append("start"))
        agent.events.on(Hook.AGENT_RUN_END, lambda _: events_log.append("end"))

        # Store memories
        agent.remember("User prefers concise answers", memory_type=MemoryType.CORE)
        agent.remember("Previous session discussed Python", memory_type=MemoryType.EPISODIC)

        # Run
        r = agent.response("Hello, tell me about Python")

        # Verify everything worked
        assert r.content is not None
        assert r.cost >= 0
        assert r.tokens.total_tokens >= 0
        assert "start" in events_log
        assert "end" in events_log
        assert agent.budget_state is not None and agent.budget_state.spent > 0

        # Memory recall
        memories = agent.recall("Python")
        assert len(memories) >= 1

        # Report
        report = agent.report
        assert report is not None
        assert report.guardrail.input_passed is True

    def test_full_feature_agent_guardrail_blocks(self) -> None:
        """Full-feature agent where guardrail blocks the input."""
        chain = GuardrailChain([ContentFilter(blocked_words=["forbidden"])])
        agent = Agent(
            model=_almock(),
            tools=[],
            memory=Memory(),
            budget=Budget(run=10.0),
            guardrails=chain,
        )
        agent.remember("Some context", memory_type=MemoryType.CORE)
        r = agent.response("This is forbidden")
        assert r.report.guardrail.blocked is True

    def test_full_feature_agent_budget_exceeded(self) -> None:
        """Full-feature agent where budget is exceeded."""
        agent = Agent(
            model=_almock(),
            memory=Memory(),
            budget=Budget(run=0.0, on_exceeded=raise_on_exceeded),
        )
        agent.remember("Context", memory_type=MemoryType.CORE)
        with pytest.raises(BudgetExceededError):
            agent.response("Hello")


# =============================================================================
# 11. STREAMING
# =============================================================================


class TestStreaming:
    """Sync and async streaming."""

    def test_sync_stream(self) -> None:
        agent = Agent(model=_almock())
        chunks = list(agent.stream("Hello"))
        assert len(chunks) >= 1
        full_text = "".join(c.text for c in chunks if c.text)
        assert len(full_text) > 0

    def test_async_stream(self) -> None:
        agent = Agent(model=_almock())

        async def run():
            chunks = []
            async for chunk in agent.astream("Hello"):
                chunks.append(chunk)
            return chunks

        chunks = asyncio.get_event_loop().run_until_complete(run())
        assert len(chunks) >= 1


# =============================================================================
# 12. LOOP STRATEGIES
# =============================================================================


class TestLoopStrategies:
    """Different loop strategies work correctly."""

    def test_single_shot_loop(self) -> None:
        agent = Agent(model=_almock(), loop=SingleShotLoop())
        r = agent.response("Hello")
        assert r.content is not None

    def test_react_loop_default(self) -> None:
        agent = Agent(model=_almock(), loop=ReactLoop(max_iterations=5))
        r = agent.response("Hello")
        assert r.content is not None

    def test_react_loop_max_iterations_boundary(self) -> None:
        with pytest.raises(ValueError, match="max_iterations must be int >= 1"):
            ReactLoop(max_iterations=0)

    def test_react_loop_min_iterations(self) -> None:
        loop = ReactLoop(max_iterations=1)
        agent = Agent(model=_almock(), loop=loop)
        r = agent.response("Hello")
        assert r.content is not None


# =============================================================================
# 13. SWITCH MODEL AT RUNTIME
# =============================================================================


class TestSwitchModel:
    """Model switching mid-session."""

    def test_switch_model(self) -> None:
        agent = Agent(model=_almock())
        original_id = agent._model_config.model_id
        agent.switch_model(_almock(lorem_length=100))
        new_id = agent._model_config.model_id
        assert original_id is not None
        assert new_id is not None

    def test_switch_model_affects_response(self) -> None:
        agent = Agent(model=_almock(lorem_length=20))
        r1 = agent.response("Short")
        agent.switch_model(_almock(lorem_length=200))
        r2 = agent.response("Long")
        # The longer lorem produces longer output
        assert len(r2.content) > len(r1.content)


# =============================================================================
# 14. EDGE CASES AND ERROR RECOVERY
# =============================================================================


class TestEdgeCasesAndErrors:
    """Edge cases that should not crash."""

    def test_agent_with_no_system_prompt(self) -> None:
        agent = Agent(model=_almock())
        r = agent.response("Hello")
        assert r.content is not None

    def test_agent_with_empty_system_prompt(self) -> None:
        agent = Agent(model=_almock(), system_prompt="")
        r = agent.response("Hello")
        assert r.content is not None

    def test_agent_with_none_budget(self) -> None:
        agent = Agent(model=_almock(), budget=None)
        r = agent.response("Hello")
        assert r.content is not None

    def test_budget_with_zero_run_and_warn(self) -> None:
        """Zero budget + warn = continues execution."""
        agent = Agent(
            model=_almock(),
            budget=Budget(run=0.0, on_exceeded=warn_on_exceeded),
        )
        r = agent.response("Hello")
        assert r.content is not None

    def test_budget_with_no_on_exceeded(self) -> None:
        """Budget without on_exceeded handler."""
        agent = Agent(model=_almock(), budget=Budget(run=10.0))
        r = agent.response("Hello")
        assert r.content is not None

    def test_special_characters_in_input(self) -> None:
        agent = Agent(model=_almock())
        inputs = [
            "'; DROP TABLE users; --",
            "<script>alert('xss')</script>",
            "\x00\x01\x02\x03",
            "a" * 50000,
        ]
        for inp in inputs:
            r = agent.response(inp)
            assert r.content is not None

    def test_newlines_and_tabs_in_input(self) -> None:
        agent = Agent(model=_almock())
        r = agent.response("Line 1\nLine 2\tTabbed\r\nCR-LF")
        assert r.content is not None

    def test_repeated_calls_track_per_run_cost(self) -> None:
        """Each response() tracks its own run cost. Hourly cost accumulates."""
        budget = Budget(run=100.0, per=RateLimit(hour=100.0))
        agent = Agent(model=_almock(), budget=budget)
        costs = []
        for _ in range(5):
            r = agent.response("Hello")
            costs.append(r.cost)
        # Hourly cost accumulates across calls
        tracker = agent.get_budget_tracker()
        assert tracker is not None
        assert tracker.get_summary().hourly_cost > 0
        # Current run cost
        assert agent.budget_state is not None
        assert agent.budget_state.spent >= 0

    def test_budget_state_keys(self) -> None:
        agent = Agent(model=_almock(), budget=Budget(run=10.0))
        agent.response("Hello")
        state = agent.budget_state
        assert state is not None
        for key in ("limit", "remaining", "spent", "percent_used"):
            assert hasattr(state, key), f"Missing key: {key}"
        d = state.to_dict()
        assert d["limit"] >= 0
        assert d["remaining"] >= 0
        assert d["spent"] >= 0
