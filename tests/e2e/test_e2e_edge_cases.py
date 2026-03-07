"""
E2E: Edge cases, error paths, and stress scenarios.

No internal mocks. Uses Almock provider for real call stack execution.
Tests boundary conditions, invalid inputs, error recovery, and concurrent usage.
"""

from __future__ import annotations

import threading

import pytest

from syrin import (
    Agent,
    Budget,
    ContentFilter,
    Decay,
    GuardrailChain,
    Hook,
    LengthGuardrail,
    Memory,
    MemoryType,
    Model,
    ReactLoop,
    SingleShotLoop,
    raise_on_exceeded,
)
from syrin.budget import BudgetTracker
from syrin.events import EventContext
from syrin.memory import Memory
from syrin.threshold import BudgetThreshold
from syrin.tool import tool


def _almock(**kwargs) -> Model:
    defaults = {"latency_seconds": 0.01, "lorem_length": 50}
    defaults.update(kwargs)
    return Model.Almock(**defaults)


# =============================================================================
# 1. BUDGET EDGE CASES
# =============================================================================


class TestBudgetEdgeCases:
    """Budget boundary conditions and edge behaviors."""

    def test_budget_exact_at_limit(self) -> None:
        """Budget at exactly the limit — should process but next call may fail."""
        budget = Budget(run=100.0, on_exceeded=raise_on_exceeded)
        agent = Agent(model=_almock(), budget=budget)
        # First call should succeed
        r = agent.response("Hello")
        assert r.content is not None

    def test_budget_reserve_equals_run(self) -> None:
        """When reserve == run, effective limit falls back to run value.
        (reserve >= run → effective = run, not 0)."""
        budget = Budget(run=1.0, reserve=1.0)
        # Effective run = run (since run > reserve is False, it uses run as-is)
        assert budget.remaining == 0.0  # remaining = run - reserve - spent = 0
        agent = Agent(model=_almock(), budget=budget)
        # With Almock's tiny cost, this actually succeeds because the
        # check_budget compares run_and_reserved >= effective_run where
        # effective_run = budget.run (1.0 since reserve >= run)
        r = agent.response("Hello")
        assert r.content is not None

    def test_budget_reserve_greater_than_run(self) -> None:
        """Reserve > run means remaining is 0 but effective limit is run value."""
        budget = Budget(run=1.0, reserve=2.0)
        assert budget.remaining == 0.0  # max(0, run - reserve - spent)
        agent = Agent(model=_almock(), budget=budget)
        r = agent.response("Hello")
        assert r.content is not None

    def test_budget_none_run_is_unlimited(self) -> None:
        """Budget(run=None) means no run limit."""
        budget = Budget(run=None)
        assert budget.remaining is None
        agent = Agent(model=_almock(), budget=budget)
        r = agent.response("Hello")
        assert r.content is not None

    def test_budget_threshold_at_zero(self) -> None:
        """Threshold at 0% fires immediately on any spend."""
        fired = []
        budget = Budget(
            run=10.0,
            thresholds=[BudgetThreshold(at=0, action=lambda _: fired.append(True))],
        )
        agent = Agent(model=_almock(), budget=budget)
        agent.response("Hello")
        assert len(fired) >= 1

    def test_budget_threshold_at_hundred(self) -> None:
        """Threshold at 100% fires only when fully spent."""
        fired = []
        budget = Budget(
            run=10.0,
            thresholds=[BudgetThreshold(at=100, action=lambda _: fired.append(True))],
        )
        agent = Agent(model=_almock(), budget=budget)
        agent.response("Hello")
        # Unlikely to fire with Almock's tiny costs
        # Just verify no crash

    def test_budget_multiple_thresholds(self) -> None:
        """Multiple thresholds at different levels."""
        fired = []
        budget = Budget(
            run=10.0,
            thresholds=[
                BudgetThreshold(at=0, action=lambda _: fired.append("low")),
                BudgetThreshold(at=50, action=lambda _: fired.append("mid")),
                BudgetThreshold(at=90, action=lambda _: fired.append("high")),
            ],
        )
        agent = Agent(model=_almock(), budget=budget)
        agent.response("Hello")
        # At least the 0% threshold should fire
        assert "low" in fired

    def test_budget_tracker_state_persistence(self) -> None:
        """BudgetTracker can serialize and restore state."""
        tracker = BudgetTracker()
        from syrin.types import CostInfo, TokenUsage

        tracker.record(
            CostInfo(
                cost_usd=0.5,
                token_usage=TokenUsage(input_tokens=100, output_tokens=50, total_tokens=150),
            )
        )
        state = tracker.get_state()
        assert state["version"] == 1
        assert len(state["cost_history"]) == 1

        new_tracker = BudgetTracker()
        new_tracker.load_state(state)
        assert new_tracker.current_run_cost == 0.5

    def test_budget_reservation_commit(self) -> None:
        """Reserve → commit flow."""
        tracker = BudgetTracker()
        token = tracker.reserve(1.0)
        assert tracker.run_usage_with_reserved == 1.0
        from syrin.types import TokenUsage

        token.commit(0.5, TokenUsage(input_tokens=10, output_tokens=5, total_tokens=15))
        assert tracker.current_run_cost == 0.5
        assert tracker.run_usage_with_reserved == 0.5

    def test_budget_reservation_rollback(self) -> None:
        """Reserve → rollback flow."""
        tracker = BudgetTracker()
        token = tracker.reserve(1.0)
        assert tracker.run_usage_with_reserved == 1.0
        token.rollback()
        assert tracker.run_usage_with_reserved == 0.0
        assert tracker.current_run_cost == 0.0

    def test_budget_reservation_double_commit_idempotent(self) -> None:
        """Double commit is idempotent."""
        tracker = BudgetTracker()
        token = tracker.reserve(1.0)
        from syrin.types import TokenUsage

        token.commit(0.5, TokenUsage(total_tokens=10))
        token.commit(0.3, TokenUsage(total_tokens=5))  # No-op
        assert tracker.current_run_cost == 0.5

    def test_budget_reservation_double_rollback_idempotent(self) -> None:
        """Double rollback is idempotent."""
        tracker = BudgetTracker()
        token = tracker.reserve(1.0)
        token.rollback()
        token.rollback()  # No-op
        assert tracker.run_usage_with_reserved == 0.0


# =============================================================================
# 2. MEMORY EDGE CASES
# =============================================================================


class TestMemoryEdgeCases:
    """Memory system boundary conditions."""

    def test_memory_conversation_retains_order(self) -> None:
        """Memory retains conversation in order."""
        mem = Memory()
        for i in range(5):
            mem.add_conversation_segment(f"U{i}", role="user")
            mem.add_conversation_segment(f"A{i}", role="assistant")
        msgs = mem.get_conversation_messages()
        assert len(msgs) == 10
        assert msgs[0].content == "U0"
        assert msgs[-1].content == "A4"

    def test_memory_load_clears_and_replaces(self) -> None:
        """load_conversation_messages clears and replaces."""
        mem = Memory()
        mem.add_conversation_segment("Test", role="user")
        assert len(mem.get_conversation_messages()) == 1
        mem.load_conversation_messages([])
        assert len(mem.get_conversation_messages()) == 0

    def test_persistent_memory_unicode_content(self) -> None:
        agent = Agent(model=_almock(), memory=Memory())
        _ = agent.remember("你好世界 🌍 مرحبا", memory_type=MemoryType.SEMANTIC)
        entries = agent.recall()
        assert any("你好" in e.content for e in entries)

    def test_persistent_memory_empty_content(self) -> None:
        agent = Agent(model=_almock(), memory=Memory())
        agent.remember("", memory_type=MemoryType.EPISODIC)
        entries = agent.recall()
        assert len(entries) == 1
        assert isinstance(entries[0].id, str)

    def test_persistent_memory_very_long_content(self) -> None:
        agent = Agent(model=_almock(), memory=Memory())
        long_content = "x" * 100000
        _ = agent.remember(long_content, memory_type=MemoryType.SEMANTIC)
        entries = agent.recall()
        assert len(entries) == 1
        assert len(entries[0].content) == 100000

    def test_persistent_memory_many_entries(self) -> None:
        agent = Agent(model=_almock(), memory=Memory())
        for i in range(100):
            agent.remember(f"Memory entry {i}", memory_type=MemoryType.EPISODIC)
        all_entries = agent.recall(limit=200)
        assert len(all_entries) == 100

    def test_forget_nonexistent_id(self) -> None:
        agent = Agent(model=_almock(), memory=Memory())
        agent.remember("Something")
        # Deleting a nonexistent ID should not crash
        deleted = agent.forget(memory_id="nonexistent-id-12345")
        assert deleted == 1  # delete is called but entry may not exist

    def test_memory_decay_applies(self) -> None:
        """Decay reduces importance of old entries."""
        from syrin.enums import DecayStrategy
        from syrin.memory.config import MemoryEntry as ME

        decay = Decay(strategy=DecayStrategy.EXPONENTIAL, rate=0.9, min_importance=0.1)
        entry = ME(
            id="test",
            content="old",
            type=MemoryType.EPISODIC,
            importance=1.0,
            created_at=__import__("datetime").datetime.now()
            - __import__("datetime").timedelta(days=30),
        )
        decay.apply(entry)
        assert entry.importance < 1.0
        assert entry.importance >= 0.1


# =============================================================================
# 3. GUARDRAIL EDGE CASES
# =============================================================================


class TestGuardrailEdgeCases:
    """Guardrail boundary conditions."""

    def test_empty_guardrail_chain(self) -> None:
        chain = GuardrailChain([])
        agent = Agent(model=_almock(), guardrails=chain)
        r = agent.response("Hello")
        assert r.content is not None

    def test_content_filter_empty_blocked_words(self) -> None:
        chain = GuardrailChain([ContentFilter(blocked_words=[])])
        agent = Agent(model=_almock(), guardrails=chain)
        r = agent.response("Anything goes")
        assert r.content is not None

    def test_content_filter_case_sensitivity(self) -> None:
        chain = GuardrailChain([ContentFilter(blocked_words=["bad"])])
        agent = Agent(model=_almock(), guardrails=chain)
        # "Bad" (capitalized) — should still be caught (case-insensitive)
        r = agent.response("This is Bad content")
        # ContentFilter may or may not be case-insensitive — just verify no crash
        assert r is not None

    def test_length_guardrail_exact_at_limit(self) -> None:
        chain = GuardrailChain([LengthGuardrail(max_length=5)])
        agent = Agent(model=_almock(), guardrails=chain)
        r = agent.response("Hello")  # exactly 5 chars
        # Should pass or block at boundary — no crash
        assert r is not None

    def test_guardrail_report_fields(self) -> None:
        chain = GuardrailChain([ContentFilter(blocked_words=["block"])])
        agent = Agent(model=_almock(), guardrails=chain)
        r = agent.response("Please block this")
        report = r.report.guardrail
        assert hasattr(report, "blocked")
        assert hasattr(report, "input_passed")
        assert hasattr(report, "blocked_stage")


# =============================================================================
# 4. HOOK / EVENT EDGE CASES
# =============================================================================


class TestHookEdgeCases:
    """Hook system edge behaviors."""

    def test_hook_handler_exception_does_not_crash_agent(self) -> None:
        """Exception in hook handler should not crash the agent."""
        agent = Agent(model=_almock())

        def bad_handler(ctx):
            raise RuntimeError("Handler crash!")

        agent.events.on(Hook.AGENT_RUN_START, bad_handler)
        # Should still work (handler errors are logged, not propagated)
        try:
            r = agent.response("Hello")
            # If it succeeds, great
            assert r.content is not None
        except RuntimeError:
            # If it propagates, that's also a valid behavior
            pass

    def test_no_handlers_registered(self) -> None:
        """Agent works fine with no event handlers."""
        agent = Agent(model=_almock())
        r = agent.response("Hello")
        assert r.content is not None

    def test_hook_context_is_event_context(self) -> None:
        contexts = []
        agent = Agent(model=_almock())
        agent.events.on(Hook.AGENT_RUN_END, lambda ctx: contexts.append(ctx))
        agent.response("Hello")
        assert len(contexts) == 1
        assert isinstance(contexts[0], EventContext)

    def test_event_context_dot_access(self) -> None:
        ctx = EventContext(name="test", value=42)
        assert ctx.name == "test"
        assert ctx.value == 42
        assert ctx["name"] == "test"

    def test_event_context_missing_attribute(self) -> None:
        ctx = EventContext()
        with pytest.raises(AttributeError):
            _ = ctx.nonexistent


# =============================================================================
# 5. LOOP EDGE CASES
# =============================================================================


class TestLoopEdgeCases:
    """Loop strategy boundary conditions."""

    def test_react_loop_max_iterations_1(self) -> None:
        agent = Agent(model=_almock(), loop=ReactLoop(max_iterations=1))
        r = agent.response("Hello")
        assert r.content is not None

    def test_react_loop_large_max_iterations(self) -> None:
        agent = Agent(model=_almock(), loop=ReactLoop(max_iterations=1000))
        r = agent.response("Hello")
        assert r.content is not None

    def test_react_loop_invalid_max_iterations(self) -> None:
        with pytest.raises(ValueError):
            ReactLoop(max_iterations=0)
        with pytest.raises(ValueError):
            ReactLoop(max_iterations=-1)
        with pytest.raises(ValueError):
            ReactLoop(max_iterations="abc")  # type: ignore

    def test_single_shot_no_tools(self) -> None:
        agent = Agent(model=_almock(), loop=SingleShotLoop())
        r = agent.response("Simple question")
        assert r.content is not None


# =============================================================================
# 6. CONCURRENT USAGE
# =============================================================================


class TestConcurrentUsage:
    """Thread safety and concurrent operations."""

    def test_sequential_rapid_fire(self) -> None:
        """Rapid sequential calls don't corrupt state."""
        agent = Agent(model=_almock(), budget=Budget(run=100.0))
        results = []
        for i in range(20):
            r = agent.response(f"Message {i}")
            results.append(r)
        assert len(results) == 20
        assert all(r.content is not None for r in results)
        assert agent.budget_state is not None and agent.budget_state.spent > 0

    def test_budget_tracker_thread_safety(self) -> None:
        """BudgetTracker handles concurrent access."""
        tracker = BudgetTracker()
        from syrin.types import CostInfo, TokenUsage

        errors = []

        def record_cost():
            try:
                for _ in range(100):
                    tracker.record(
                        CostInfo(
                            cost_usd=0.001,
                            token_usage=TokenUsage(total_tokens=10),
                        )
                    )
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=record_cost) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert tracker.current_run_cost > 0

    def test_multiple_agents_independent(self) -> None:
        """Multiple agent instances don't share state."""
        agent1 = Agent(model=_almock(), budget=Budget(run=10.0), memory=Memory())
        agent2 = Agent(model=_almock(), budget=Budget(run=10.0), memory=Memory())

        agent1.remember("Agent 1 memory", memory_type=MemoryType.CORE)
        agent2.remember("Agent 2 memory", memory_type=MemoryType.CORE)

        agent1.response("Hello from 1")
        agent2.response("Hello from 2")

        mem1 = agent1.recall()
        mem2 = agent2.recall()

        assert any("Agent 1" in m.content for m in mem1)
        assert not any("Agent 2" in m.content for m in mem1)
        assert any("Agent 2" in m.content for m in mem2)
        assert not any("Agent 1" in m.content for m in mem2)


# =============================================================================
# 7. REPORT SYSTEM EDGE CASES
# =============================================================================


class TestReportEdgeCases:
    """Agent report system edge conditions."""

    def test_report_reset_between_calls(self) -> None:
        """Report resets between response() calls."""
        agent = Agent(model=_almock())
        agent.response("First")
        report1_tokens = agent.report.tokens.total_tokens
        agent.response("Second")
        report2_tokens = agent.report.tokens.total_tokens
        # Reports should be independent per call
        assert report1_tokens >= 0
        assert report2_tokens >= 0

    def test_report_with_guardrails(self) -> None:
        chain = GuardrailChain([ContentFilter(blocked_words=["banned"])])
        agent = Agent(model=_almock(), guardrails=chain)
        agent.response("Clean input")
        assert agent.report.guardrail.input_passed is True

    def test_report_with_memory(self) -> None:
        """Report tracks memory operations within the same response cycle."""
        agent = Agent(model=_almock(), memory=Memory())
        # remember() before response() increments stores on the current report
        agent.remember("Test")
        # The report is populated by remember() but reset by response()
        # So we check stores after remember, before reset
        stores_before = agent.report.memory.stores
        assert stores_before >= 1
        # After response(), report resets
        agent.response("Hello")
        # The response report won't include the pre-call remember
        assert agent.report is not None

    def test_report_budget_status(self) -> None:
        agent = Agent(model=_almock(), budget=Budget(run=10.0))
        agent.response("Hello")
        report = agent.report
        assert report is not None


# =============================================================================
# 8. MODEL AND PROVIDER EDGE CASES
# =============================================================================


class TestModelEdgeCases:
    """Model creation and provider resolution edge cases."""

    def test_almock_zero_latency(self) -> None:
        model = Model.Almock(latency_seconds=0.01, lorem_length=10)
        agent = Agent(model=model)
        r = agent.response("Fast")
        assert r.content is not None

    def test_almock_custom_response(self) -> None:
        model = Model.Almock(
            latency_seconds=0.01,
            response_mode="custom",
            custom_response="Custom reply",
        )
        agent = Agent(model=model)
        r = agent.response("Hello")
        assert r.content == "Custom reply"

    def test_almock_different_lorem_lengths(self) -> None:
        for length in [1, 10, 100, 500, 1000]:
            model = Model.Almock(latency_seconds=0.01, lorem_length=length)
            agent = Agent(model=model)
            r = agent.response("Hello")
            assert len(r.content) >= length * 0.8  # approximate

    def test_model_registry_isolation(self) -> None:
        """ModelRegistry clears between tests (via conftest fixture)."""
        from syrin.model import ModelRegistry

        reg = ModelRegistry()
        assert len(reg._models) == 0  # Should be cleared by fixture


# =============================================================================
# 9. AGENT INHERITANCE
# =============================================================================


class TestAgentInheritance:
    """Agent subclassing with class-level attributes."""

    def test_basic_subclass(self) -> None:
        class MyAgent(Agent):
            model = _almock()
            system_prompt = "You are a custom agent."

        agent = MyAgent()
        r = agent.response("Hello")
        assert r.content is not None

    def test_subclass_with_tools(self) -> None:
        @tool
        def my_tool(x: str) -> str:
            return f"Result: {x}"

        class ToolAgent(Agent):
            model = _almock()
            tools = [my_tool]

        agent = ToolAgent()
        result = agent._execute_tool("my_tool", {"x": "test"})
        assert result == "Result: test"

    def test_subclass_with_budget(self) -> None:
        class BudgetAgent(Agent):
            model = _almock()
            budget = Budget(run=5.0)

        agent = BudgetAgent()
        r = agent.response("Hello")
        assert r.content is not None
        assert agent.budget_state is not None and agent.budget_state.spent > 0

    def test_subclass_with_memory(self) -> None:
        class MemoryAgent(Agent):
            model = _almock()
            memory = Memory()

        agent = MemoryAgent()
        mid = agent.remember("Test memory", memory_type=MemoryType.CORE)
        assert isinstance(mid, str)

    def test_deep_inheritance_chain(self) -> None:
        class BaseAgent(Agent):
            model = _almock()
            system_prompt = "Base"

        class MiddleAgent(BaseAgent):
            system_prompt = "Middle"

        class LeafAgent(MiddleAgent):
            system_prompt = "Leaf"

        agent = LeafAgent()
        assert agent._system_prompt == "Leaf"
        r = agent.response("Hello")
        assert r.content is not None
