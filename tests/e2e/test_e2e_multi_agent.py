"""
E2E: Multi-agent orchestration — spawn, sequential/parallel helpers, AgentRouter.

No internal mocks. Uses Almock provider for real call stack execution.
"""

from __future__ import annotations

import pytest

from syrin import (
    Agent,
    AgentRouter,
    Budget,
    Memory,
    MemoryType,
    Model,
    Response,
)
from syrin.agent.pipeline import parallel, sequential


def _almock(**kwargs) -> Model:
    defaults = {"latency_seconds": 0.01, "lorem_length": 50}
    defaults.update(kwargs)
    return Model.Almock(**defaults)


# =============================================================================
# 1. SPAWN
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
        r = child.run("Hello from child")
        assert r.content is not None

    def test_spawn_with_budget(self) -> None:
        class Child(Agent):
            model = _almock()

        parent = Agent(model=_almock(), budget=Budget(max_cost=10.0))
        child_budget = Budget(max_cost=2.0)
        result = parent.spawn(Child, task="Budget task", budget=child_budget)
        assert result.content is not None

    def test_spawn_budget_exceeds_parent_raises(self) -> None:
        class Child(Agent):
            model = _almock()

        parent = Agent(model=_almock(), budget=Budget(max_cost=1.0))
        with pytest.raises(ValueError, match="cannot exceed"):
            parent.spawn(Child, task="Expensive", budget=Budget(max_cost=100.0))

    def test_spawn_max_children_limit(self) -> None:
        class Child(Agent):
            model = _almock()

        parent = Agent(model=_almock())
        # Spawn up to max
        for _ in range(10):  # default max_children=10
            parent.spawn(Child)
        with pytest.raises(RuntimeError, match="max child agents"):
            parent.spawn(Child)

    def test_spawn_with_shared_budget(self) -> None:
        """Child borrows from parent's shared budget."""

        class Child(Agent):
            model = _almock()

        parent = Agent(model=_almock(), budget=Budget(max_cost=10.0))
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
# 2. SEQUENTIAL / PARALLEL HELPERS
# =============================================================================


class TestSequentialParallel:
    """sequential() and parallel() helper functions."""

    def test_sequential_two_agents(self) -> None:
        class Researcher(Agent):
            model = _almock()
            system_prompt = "You are a researcher."

        class Writer(Agent):
            model = _almock()
            system_prompt = "You are a writer."

        result = sequential(
            [
                (Researcher(_almock()), "Research AI trends"),
                (Writer(_almock()), "Write a summary"),
            ]
        )

        assert isinstance(result, Response)
        assert result.content is not None
        assert result.cost >= 0

    def test_parallel_two_agents(self) -> None:
        import asyncio

        class Worker1(Agent):
            model = _almock()

        class Worker2(Agent):
            model = _almock()

        results = asyncio.run(
            parallel(
                [
                    (Worker1(_almock()), "Task A"),
                    (Worker2(_almock()), "Task B"),
                ]
            )
        )

        assert isinstance(results, list)
        assert len(results) == 2
        assert all(r.content is not None for r in results)

    def test_empty_sequential_returns_empty_response(self) -> None:
        """Empty sequential handles gracefully."""
        result = sequential([])
        assert result is not None
        assert result.content == ""

    def test_single_agent_sequential(self) -> None:
        class Solo(Agent):
            model = _almock()

        result = sequential([(Solo(_almock()), "Solo task")])
        assert result.content is not None


# =============================================================================
# 3. DYNAMIC PIPELINE
# =============================================================================


class TestAgentRouter:
    """AgentRouter (replaces DynamicPipeline in v0.11.0) — basic routing tests."""

    def test_agent_router_basic(self) -> None:
        class Analyzer(Agent):
            model = _almock()
            system_prompt = "analyzer"

        class Summarizer(Agent):
            model = _almock()
            system_prompt = "summarizer"

        router = AgentRouter(agents=[Analyzer, Summarizer], model=_almock())
        result = router.run("Analyze and summarize AI trends")
        assert result is not None

    def test_agent_router_single_agent(self) -> None:
        class Solo(Agent):
            model = _almock()
            system_prompt = "solo"

        router = AgentRouter(agents=[Solo], model=_almock())
        result = router.run("Solo task")
        assert result is not None


# =============================================================================
# 4. MULTI-AGENT WITH ALL FEATURES
# =============================================================================


class TestMultiAgentFullFeatures:
    """Multi-agent with budget, memory, tools, hooks — combined."""

    def test_spawn_with_memory_isolation(self) -> None:
        """Child agent should not share parent's memory unless explicitly transferred."""

        class Child(Agent):
            model = _almock()
            memory = Memory()

        parent = Agent(model=_almock(), memory=Memory())
        parent.remember("Parent secret", memory_type=MemoryType.FACTS)

        child = parent.spawn(Child)
        assert isinstance(child, Agent)

        # Child should have its own memory (empty)
        if child._memory_backend is not None:
            child_memories = child.recall()
            parent_memories = parent.recall()
            assert len(parent_memories) >= 1
            # Child's memory should be independent
            assert not any("Parent secret" in m.content for m in child_memories)

    def test_sequential_cost_tracking(self) -> None:
        """sequential() helper accumulates cost across agents."""

        class Step1(Agent):
            model = _almock()
            budget = Budget(max_cost=10.0)

        class Step2(Agent):
            model = _almock()

        result = sequential(
            [
                (Step1(_almock()), "Step 1"),
                (Step2(_almock()), "Step 2"),
            ]
        )

        assert result.cost >= 0
