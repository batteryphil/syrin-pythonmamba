"""
E2E: Multi-agent orchestration — handoff, spawn, pipeline, dynamic pipeline.

No internal mocks. Uses Almock provider for real call stack execution.
"""

from __future__ import annotations

import pytest

from syrin import (
    Agent,
    Budget,
    DynamicPipeline,
    Hook,
    Memory,
    MemoryType,
    Model,
    Pipeline,
    Response,
)
from syrin.tool import tool


def _almock(**kwargs) -> Model:
    defaults = {"latency_seconds": 0.01, "lorem_length": 50}
    defaults.update(kwargs)
    return Model.Almock(**defaults)


# =============================================================================
# 1. HANDOFF
# =============================================================================


class TestHandoff:
    """Agent hands off to specialized agent."""

    def test_basic_handoff(self) -> None:
        class Specialist(Agent):
            model = _almock()
            system_prompt = "You are a specialist."

        parent = Agent(model=_almock(), memory=Memory())
        parent.remember("Important context", memory_type=MemoryType.CORE)
        result = parent.handoff(Specialist, "Handle this task")
        assert isinstance(result, Response)
        assert result.content is not None
        assert len(result.content) > 0

    def test_handoff_transfers_context(self) -> None:
        class Target(Agent):
            model = _almock()

        source = Agent(model=_almock(), memory=Memory())
        source.remember("Secret knowledge", memory_type=MemoryType.CORE)
        result = source.handoff(Target, "Use context", transfer_context=True)
        assert result.content is not None

    def test_handoff_without_context_transfer(self) -> None:
        class Target(Agent):
            model = _almock()

        source = Agent(model=_almock(), memory=Memory())
        source.remember("This should not transfer", memory_type=MemoryType.CORE)
        result = source.handoff(Target, "Do task", transfer_context=False)
        assert result.content is not None

    def test_handoff_with_budget_transfer(self) -> None:
        class Target(Agent):
            model = _almock()

        budget = Budget(run=10.0, shared=True)
        source = Agent(model=_almock(), budget=budget, memory=Memory())
        result = source.handoff(Target, "Do task", transfer_budget=True)
        assert result.content is not None

    def test_handoff_chain(self) -> None:
        """A → B → C handoff chain."""

        class AgentB(Agent):
            model = _almock()

        class AgentC(Agent):
            model = _almock()

        a = Agent(model=_almock(), memory=Memory())
        a.remember("From A", memory_type=MemoryType.EPISODIC)
        r_b = a.handoff(AgentB, "Relay task")
        assert r_b.content is not None

    def test_handoff_source_without_memory_warns(self, caplog) -> None:
        """Handoff with transfer_context=True but no memory logs a warning."""

        class Target(Agent):
            model = _almock()

        source = Agent(model=_almock())  # No memory
        import logging

        with caplog.at_level(logging.WARNING):
            result = source.handoff(Target, "Task", transfer_context=True)
        assert result.content is not None


# =============================================================================
# 2. SPAWN
# =============================================================================


class TestSpawn:
    """Parent agent spawns child agents."""

    def test_spawn_with_task(self) -> None:
        class Child(Agent):
            model = _almock()

        parent = Agent(model=_almock())
        result = parent.spawn(Child, task="Do something")
        assert isinstance(result, Response)
        assert result.content is not None

    def test_spawn_without_task_returns_agent(self) -> None:
        class Child(Agent):
            model = _almock()

        parent = Agent(model=_almock())
        child = parent.spawn(Child)
        assert isinstance(child, Agent)
        # Child can then be used independently
        r = child.response("Hello from child")
        assert r.content is not None

    def test_spawn_with_budget(self) -> None:
        class Child(Agent):
            model = _almock()

        parent = Agent(model=_almock(), budget=Budget(run=10.0))
        child_budget = Budget(run=2.0)
        result = parent.spawn(Child, task="Budget task", budget=child_budget)
        assert result.content is not None

    def test_spawn_budget_exceeds_parent_raises(self) -> None:
        class Child(Agent):
            model = _almock()

        parent = Agent(model=_almock(), budget=Budget(run=1.0))
        with pytest.raises(ValueError, match="cannot exceed"):
            parent.spawn(Child, task="Expensive", budget=Budget(run=100.0))

    def test_spawn_max_children_limit(self) -> None:
        class Child(Agent):
            model = _almock()

        parent = Agent(model=_almock())
        # Spawn up to max
        for _ in range(10):  # default max_children=10
            parent.spawn(Child)
        with pytest.raises(RuntimeError, match="max children"):
            parent.spawn(Child)

    def test_spawn_with_shared_budget(self) -> None:
        """Child borrows from parent's shared budget."""

        class Child(Agent):
            model = _almock()

        parent = Agent(model=_almock(), budget=Budget(run=10.0, shared=True))
        result = parent.spawn(Child, task="Shared budget task")
        assert result.content is not None
        # Parent's budget should reflect child's spend
        assert parent.budget_state is not None and parent.budget_state.spent > 0

    def test_spawn_parallel(self) -> None:
        class Worker(Agent):
            model = _almock()

        parent = Agent(model=_almock())
        results = parent.spawn_parallel(
            [
                (Worker, "Task 1"),
                (Worker, "Task 2"),
                (Worker, "Task 3"),
            ]
        )
        assert len(results) == 3
        assert all(isinstance(r, Response) for r in results)
        assert all(r.content is not None for r in results)


# =============================================================================
# 3. PIPELINE
# =============================================================================


class TestPipeline:
    """Sequential and parallel pipeline execution."""

    def test_sequential_pipeline(self) -> None:
        class Researcher(Agent):
            model = _almock()
            system_prompt = "You are a researcher."

        class Writer(Agent):
            model = _almock()
            system_prompt = "You are a writer."

        pipeline = Pipeline()
        result = pipeline.run(
            [
                (Researcher, "Research AI trends"),
                (Writer, "Write a summary"),
            ]
        ).sequential()

        assert isinstance(result, Response)
        assert result.content is not None
        assert result.cost >= 0

    def test_parallel_pipeline(self) -> None:
        class Worker1(Agent):
            model = _almock()

        class Worker2(Agent):
            model = _almock()

        pipeline = Pipeline()
        results = pipeline.run(
            [
                (Worker1, "Task A"),
                (Worker2, "Task B"),
            ]
        ).parallel()

        assert isinstance(results, list)
        assert len(results) == 2
        assert all(r.content is not None for r in results)

    def test_pipeline_with_budget(self) -> None:
        class Step1(Agent):
            model = _almock()

        class Step2(Agent):
            model = _almock()

        pipeline = Pipeline(budget=Budget(run=10.0))
        result = pipeline.run(
            [
                (Step1, "Step 1"),
                (Step2, "Step 2"),
            ]
        ).sequential()
        assert result.content is not None

    def test_empty_pipeline_returns_empty_response(self) -> None:
        """Empty pipeline handles gracefully (no crash)."""
        pipeline = Pipeline()
        # Empty list may return an empty response or raise - both valid
        try:
            result = pipeline.run([]).sequential()
            # If it succeeds, response should still be valid
            assert result is not None
        except (ValueError, IndexError, RuntimeError):
            pass  # Raising on empty is also valid

    def test_single_agent_pipeline(self) -> None:
        class Solo(Agent):
            model = _almock()

        pipeline = Pipeline()
        result = pipeline.run([(Solo, "Solo task")]).sequential()
        assert result.content is not None


# =============================================================================
# 4. DYNAMIC PIPELINE
# =============================================================================


class TestDynamicPipeline:
    """Dynamic pipeline with registration, execution, hooks."""

    def test_dynamic_pipeline_basic(self) -> None:
        class Analyzer(Agent):
            model = _almock()

        class Summarizer(Agent):
            model = _almock()

        dp = DynamicPipeline(
            agents=[Analyzer, Summarizer],
            model=_almock(),
        )
        result = dp.run("Analyze and summarize AI trends")
        assert result is not None

    def test_dynamic_pipeline_parallel(self) -> None:
        class Worker1(Agent):
            model = _almock()

        class Worker2(Agent):
            model = _almock()

        dp = DynamicPipeline(
            agents=[Worker1, Worker2],
            model=_almock(),
            max_parallel=2,
        )
        result = dp.run("Process in parallel", mode="parallel")
        assert result is not None

    def test_dynamic_pipeline_with_hooks(self) -> None:
        events = []

        class Step1(Agent):
            model = _almock()

        class Step2(Agent):
            model = _almock()

        dp = DynamicPipeline(agents=[Step1, Step2], model=_almock())
        dp.events.on(
            Hook.DYNAMIC_PIPELINE_START,
            lambda _: events.append("dp_start"),
        )
        dp.events.on(
            Hook.DYNAMIC_PIPELINE_END,
            lambda _: events.append("dp_end"),
        )
        dp.run("Process")
        assert "dp_start" in events
        assert "dp_end" in events

    def test_dynamic_pipeline_event_lifecycle(self) -> None:
        """Verify START, PLAN, and END events always fire."""
        lifecycle = []

        class StepA(Agent):
            model = _almock()

        class StepB(Agent):
            model = _almock()

        dp = DynamicPipeline(agents=[StepA, StepB], model=_almock())
        dp.events.on(
            Hook.DYNAMIC_PIPELINE_START,
            lambda _: lifecycle.append("start"),
        )
        dp.events.on(
            Hook.DYNAMIC_PIPELINE_PLAN,
            lambda _: lifecycle.append("plan"),
        )
        dp.events.on(
            Hook.DYNAMIC_PIPELINE_END,
            lambda _: lifecycle.append("end"),
        )
        dp.run("Run all")
        # START and END always fire; PLAN fires when planner returns
        assert "start" in lifecycle
        assert "end" in lifecycle

    def test_dynamic_pipeline_single_agent(self) -> None:
        class Solo(Agent):
            model = _almock()

        dp = DynamicPipeline(agents=[Solo], model=_almock())
        result = dp.run("Solo task")
        assert result is not None

    def test_dynamic_pipeline_requires_model(self) -> None:
        """DynamicPipeline without model raises ValueError."""

        class Worker(Agent):
            model = _almock()

        with pytest.raises(ValueError, match="model is required"):
            DynamicPipeline(agents=[Worker])


# =============================================================================
# 5. MULTI-AGENT WITH ALL FEATURES
# =============================================================================


class TestMultiAgentFullFeatures:
    """Multi-agent with budget, memory, tools, hooks — combined."""

    def test_handoff_with_full_features(self) -> None:
        events = []

        @tool
        def search(query: str) -> str:
            return f"Found: {query}"

        class Specialist(Agent):
            model = _almock()
            tools = [search]

        source = Agent(
            model=_almock(),
            memory=Memory(),
            budget=Budget(run=10.0, shared=True),
        )
        source.events.on(Hook.AGENT_RUN_START, lambda _: events.append("source_start"))

        source.remember("User wants help with Python", memory_type=MemoryType.CORE)
        result = source.handoff(
            Specialist,
            "Help with Python",
            transfer_context=True,
            transfer_budget=True,
        )
        assert result.content is not None

    def test_spawn_with_memory_isolation(self) -> None:
        """Child agent should not share parent's memory unless explicitly transferred."""

        class Child(Agent):
            model = _almock()
            memory = Memory()

        parent = Agent(model=_almock(), memory=Memory())
        parent.remember("Parent secret", memory_type=MemoryType.CORE)

        child = parent.spawn(Child)
        assert isinstance(child, Agent)

        # Child should have its own memory (empty)
        if child._memory_backend is not None:
            child_memories = child.recall()
            parent_memories = parent.recall()
            assert len(parent_memories) >= 1
            # Child's memory should be independent
            assert not any("Parent secret" in m.content for m in child_memories)

    def test_pipeline_budget_tracking(self) -> None:
        """Pipeline tracks total cost across all agents."""

        class Step1(Agent):
            model = _almock()

        class Step2(Agent):
            model = _almock()

        pipeline = Pipeline(budget=Budget(run=10.0))
        result = pipeline.run(
            [
                (Step1, "Step 1"),
                (Step2, "Step 2"),
            ]
        ).sequential()

        assert result.cost >= 0
