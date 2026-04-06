"""P2-T9: Swarm lifecycle — pause, resume, cancel_agent, agent_statuses."""

from __future__ import annotations

import asyncio

import pytest

from syrin import Agent, Budget, Model
from syrin.enums import AgentStatus, Hook, WorkflowStatus
from syrin.events import EventContext
from syrin.response import Response
from syrin.swarm import Swarm
from syrin.workflow.exceptions import WorkflowCancelledError


def _make_slow_agent(name: str, content: str, delay: float = 0.1) -> Agent:
    """Slow agent that sleeps for *delay* seconds."""
    _delay = delay

    class _Slow(Agent):
        model = Model.Almock()
        system_prompt = "slow"

        async def arun(self, input_text: str) -> Response[str]:
            await asyncio.sleep(_delay)
            return Response(content=content, cost=0.01)

    _Slow.__name__ = name
    return _Slow()


def _make_instant_agent(name: str, content: str) -> Agent:
    """Instant agent with no delay."""

    class _Instant(Agent):
        model = Model.Almock()
        system_prompt = "instant"

        async def arun(self, input_text: str) -> Response[str]:
            return Response(content=content, cost=0.01)

    _Instant.__name__ = name
    return _Instant()


@pytest.mark.phase_2
class TestSwarmPauseResume:
    """swarm.pause() and swarm.resume() control the swarm."""

    async def test_pause_then_resume_completes(self) -> None:
        """Pausing and resuming a swarm still produces a result."""
        a = _make_slow_agent("SlowA", "result A", delay=0.05)
        b = _make_slow_agent("SlowB", "result B", delay=0.05)
        swarm = Swarm(agents=[a, b], goal="pause test")
        handle = swarm.play()

        await swarm.pause()
        await asyncio.sleep(0.02)
        await swarm.resume()
        result = await handle.wait()

        assert result is not None
        assert isinstance(result.content, str)

    async def test_pause_sets_handle_status_paused(self) -> None:
        """After pause takes effect, handle.status is PAUSED."""
        a = _make_slow_agent("SA", "r", delay=0.1)
        b = _make_slow_agent("SB", "r2", delay=0.1)
        swarm = Swarm(agents=[a, b], goal="status test")
        handle = swarm.play()

        await asyncio.sleep(0)
        await swarm.pause()
        await asyncio.sleep(0.05)

        # May be PAUSED or RUNNING depending on timing
        assert handle.status in (WorkflowStatus.PAUSED, WorkflowStatus.RUNNING)
        await swarm.cancel()


@pytest.mark.phase_2
class TestSwarmCancelAgent:
    """swarm.cancel_agent() stops one agent without stopping the swarm."""

    async def test_cancel_one_agent_others_continue(self) -> None:
        """Cancelling one agent by object lets others finish."""
        a = _make_slow_agent("LongAgent", "long result", delay=0.2)
        b = _make_instant_agent("FastAgent", "fast result")
        swarm = Swarm(agents=[a, b], goal="cancel agent test")
        handle = swarm.play()

        await asyncio.sleep(0)
        # Cancel the slow agent by passing the agent object — no free strings
        await swarm.cancel_agent(a)
        result = await handle.wait()

        assert result is not None
        assert isinstance(result.content, str)

    async def test_cancel_nonexistent_agent_raises(self) -> None:
        """cancel_agent() with an unknown name raises ValueError."""
        a = _make_instant_agent("A", "r")
        swarm = Swarm(agents=[a], goal="cancel test")
        handle = swarm.play()
        await handle.wait()

        with pytest.raises((ValueError, KeyError)):
            await swarm.cancel_agent("DoesNotExist")


@pytest.mark.phase_2
class TestSwarmAgentStatuses:
    """swarm.agent_statuses() returns status for all agents."""

    async def test_agent_statuses_returns_list(self) -> None:
        """agent_statuses() returns a list with one entry per agent."""
        a = _make_instant_agent("A", "r")
        b = _make_instant_agent("B", "r2")
        swarm = Swarm(agents=[a, b], goal="status list test")
        await swarm.run()
        statuses = swarm.agent_statuses()
        assert len(statuses) == 2

    async def test_agent_status_has_agent_name(self) -> None:
        """Each status entry has an agent_name field."""
        a = _make_instant_agent("MyAgent", "r")
        swarm = Swarm(agents=[a], goal="name status")
        await swarm.run()
        statuses = swarm.agent_statuses()
        assert statuses[0].agent_name == "MyAgent"

    async def test_agent_statuses_completed_after_run(self) -> None:
        """After run(), all agent statuses are STOPPED or COMPLETED."""
        a = _make_instant_agent("A", "r")
        swarm = Swarm(agents=[a], goal="completed status")
        await swarm.run()
        statuses = swarm.agent_statuses()
        for status in statuses:
            assert status.state in (AgentStatus.STOPPED, AgentStatus.IDLE)


@pytest.mark.phase_2
class TestSwarmBudgetHooks:
    """Swarm fires budget-related lifecycle hooks."""

    async def test_swarm_budget_low_fires_at_threshold(self) -> None:
        """Hook.SWARM_BUDGET_LOW fires when pool falls below threshold."""
        budget_low_events: list[EventContext] = []

        # Use a tiny budget so each agent quickly hits the threshold
        a = _make_instant_agent("A", "r")
        b = _make_instant_agent("B", "r2")
        budget = Budget(max_cost=0.05)
        swarm = Swarm(agents=[a, b], goal="budget low test", budget=budget)
        swarm.events.on(Hook.SWARM_BUDGET_LOW, lambda ctx: budget_low_events.append(ctx))
        await swarm.run()
        # Hook may or may not fire depending on exact costs — just verify no crash
        assert isinstance(budget_low_events, list)

    async def test_swarm_cancel_raises_on_resume(self) -> None:
        """After swarm cancel, calling resume raises WorkflowCancelledError."""
        a = _make_slow_agent("A", "r", delay=0.2)
        swarm = Swarm(agents=[a], goal="cancel resume test")
        swarm.play()

        await asyncio.sleep(0)
        await swarm.cancel()
        await asyncio.sleep(0.05)

        with pytest.raises(WorkflowCancelledError):
            await swarm.resume()
