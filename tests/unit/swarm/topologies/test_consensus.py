"""P4-T1: CONSENSUS topology tests."""

from __future__ import annotations

import asyncio

import pytest

from syrin import Agent, Budget, Model
from syrin.enums import ConsensusStrategy, FallbackStrategy, Hook, SwarmTopology
from syrin.events import EventContext
from syrin.response import Response
from syrin.swarm import Swarm, SwarmConfig
from syrin.swarm.topologies._consensus import (
    ConsensusConfig,
    ConsensusResult,
    ConsensusVote,
)

# ---------------------------------------------------------------------------
# Stub agents
# ---------------------------------------------------------------------------


def _make_answer_agent(answer: str, cost: float = 0.01, weight: float = 1.0) -> Agent:
    """Return an agent that always answers *answer* with given *weight*."""
    _answer = answer
    _weight = weight

    class _Stub(Agent):
        model = Model.Almock()
        system_prompt = "stub"

        async def arun(self, input_text: str) -> Response[str]:
            return Response(content=_answer, cost=cost)

    _Stub.__name__ = f"Agent_{answer[:8].replace(' ', '_')}"
    # Attach weight attribute (used by WEIGHTED strategy)
    instance = _Stub()
    instance.weight = _weight  # type: ignore[attr-defined]
    return instance


def _make_failing_agent(name: str = "FailAgent") -> Agent:
    """Return an agent that always raises RuntimeError."""

    class _Fail(Agent):
        model = Model.Almock()
        system_prompt = "fail"

        async def arun(self, input_text: str) -> Response[str]:
            raise RuntimeError(f"{name} failed")

    _Fail.__name__ = name
    return _Fail()


# ---------------------------------------------------------------------------
# P4-T1-1: Construction
# ---------------------------------------------------------------------------


@pytest.mark.phase_4
class TestConsensusConstruction:
    """CONSENSUS topology constructs without error."""

    async def test_swarm_with_consensus_topology_constructs(self) -> None:
        """Swarm with CONSENSUS topology and 3 agents constructs without error."""
        a = _make_answer_agent("ans A")
        b = _make_answer_agent("ans A")
        c = _make_answer_agent("ans C")
        swarm = Swarm(
            agents=[a, b, c],
            goal="What is the capital of France?",
            config=SwarmConfig(topology=SwarmTopology.CONSENSUS),
        )
        assert swarm.agent_count == 3
        assert swarm.config.topology == SwarmTopology.CONSENSUS

    async def test_consensus_config_constructs(self) -> None:
        """ConsensusConfig constructs with default values."""
        cfg = ConsensusConfig()
        assert cfg.strategy == ConsensusStrategy.MAJORITY
        assert cfg.min_agreement == pytest.approx(0.5)

    async def test_consensus_vote_constructs(self) -> None:
        """ConsensusVote constructs with expected fields."""
        vote = ConsensusVote(agent_name="AgentA", answer="Paris", weight=1.0)
        assert vote.agent_name == "AgentA"
        assert vote.answer == "Paris"
        assert vote.weight == pytest.approx(1.0)

    async def test_consensus_result_constructs(self) -> None:
        """ConsensusResult constructs with expected fields."""
        result = ConsensusResult(
            consensus_reached=True,
            content="Paris",
            votes=[],
            winning_answer="Paris",
            agreement_fraction=1.0,
        )
        assert result.consensus_reached is True
        assert result.content == "Paris"


# ---------------------------------------------------------------------------
# P4-T1-2: All agents agree → consensus_reached = True
# ---------------------------------------------------------------------------


@pytest.mark.phase_4
class TestConsensusAllAgreeUnanimous:
    """When all 3 agents return the same answer, consensus is reached."""

    async def test_all_same_answer_consensus_reached(self) -> None:
        """3 agents all returning 'Paris' → consensus_reached True."""
        agents = [_make_answer_agent("Paris") for _ in range(3)]
        swarm = Swarm(
            agents=agents,
            goal="Capital of France?",
            config=SwarmConfig(topology=SwarmTopology.CONSENSUS),
        )
        result = await swarm.run()
        assert result.consensus_result is not None
        assert result.consensus_result.consensus_reached is True
        assert result.consensus_result.content == "Paris"
        assert result.content == "Paris"

    async def test_all_agree_winning_answer_set(self) -> None:
        """winning_answer matches content when consensus is reached."""
        agents = [_make_answer_agent("Rome") for _ in range(2)]
        swarm = Swarm(
            agents=agents,
            goal="Capital of Italy?",
            config=SwarmConfig(topology=SwarmTopology.CONSENSUS),
        )
        result = await swarm.run()
        assert result.consensus_result is not None
        assert result.consensus_result.winning_answer == "Rome"


# ---------------------------------------------------------------------------
# P4-T1-3: 2/3 agree → MAJORITY → consensus_reached True
# ---------------------------------------------------------------------------


@pytest.mark.phase_4
class TestConsensusMajority:
    """MAJORITY strategy: 2/3 agents agree → consensus reached."""

    async def test_majority_two_of_three_agree(self) -> None:
        """2 agents say 'Paris', 1 says 'Lyon' → MAJORITY → reached."""
        a = _make_answer_agent("Paris")
        b = _make_answer_agent("Paris")
        c = _make_answer_agent("Lyon")
        swarm = Swarm(
            agents=[a, b, c],
            goal="Capital?",
            config=SwarmConfig(
                topology=SwarmTopology.CONSENSUS,
                consensus=ConsensusConfig(strategy=ConsensusStrategy.MAJORITY),
            ),
        )
        result = await swarm.run()
        assert result.consensus_result is not None
        assert result.consensus_result.consensus_reached is True
        assert result.consensus_result.content == "Paris"

    async def test_majority_content_is_winning_answer(self) -> None:
        """SwarmResult.content equals the majority answer."""
        a = _make_answer_agent("Berlin")
        b = _make_answer_agent("Berlin")
        c = _make_answer_agent("Munich")
        swarm = Swarm(
            agents=[a, b, c],
            goal="Capital?",
            config=SwarmConfig(
                topology=SwarmTopology.CONSENSUS,
                consensus=ConsensusConfig(strategy=ConsensusStrategy.MAJORITY),
            ),
        )
        result = await swarm.run()
        assert result.content == "Berlin"


# ---------------------------------------------------------------------------
# P4-T1-4: UNANIMITY with 2/3 agreeing → consensus NOT reached
# ---------------------------------------------------------------------------


@pytest.mark.phase_4
class TestConsensusUnanimity:
    """UNANIMITY strategy: all agents must agree."""

    async def test_unanimity_two_of_three_not_reached(self) -> None:
        """2/3 agree → UNANIMITY → consensus_reached False."""
        a = _make_answer_agent("Paris")
        b = _make_answer_agent("Paris")
        c = _make_answer_agent("Lyon")
        swarm = Swarm(
            agents=[a, b, c],
            goal="Capital?",
            config=SwarmConfig(
                topology=SwarmTopology.CONSENSUS,
                consensus=ConsensusConfig(strategy=ConsensusStrategy.UNANIMITY),
            ),
        )
        result = await swarm.run()
        assert result.consensus_result is not None
        assert result.consensus_result.consensus_reached is False

    async def test_unanimity_all_agree_reached(self) -> None:
        """All 3 agree → UNANIMITY → consensus_reached True."""
        agents = [_make_answer_agent("Berlin") for _ in range(3)]
        swarm = Swarm(
            agents=agents,
            goal="Capital?",
            config=SwarmConfig(
                topology=SwarmTopology.CONSENSUS,
                consensus=ConsensusConfig(strategy=ConsensusStrategy.UNANIMITY),
            ),
        )
        result = await swarm.run()
        assert result.consensus_result is not None
        assert result.consensus_result.consensus_reached is True

    async def test_unanimity_content_empty_when_not_reached(self) -> None:
        """Content is empty string when consensus not reached."""
        a = _make_answer_agent("Paris")
        b = _make_answer_agent("Lyon")
        swarm = Swarm(
            agents=[a, b],
            goal="Capital?",
            config=SwarmConfig(
                topology=SwarmTopology.CONSENSUS,
                consensus=ConsensusConfig(strategy=ConsensusStrategy.UNANIMITY),
            ),
        )
        result = await swarm.run()
        assert result.consensus_result is not None
        assert result.consensus_result.consensus_reached is False
        assert result.consensus_result.content == ""


# ---------------------------------------------------------------------------
# P4-T1-5: min_agreement — 2/3 = 0.667 < 0.67 → NOT reached
# ---------------------------------------------------------------------------


@pytest.mark.phase_4
class TestConsensusMinAgreement:
    """min_agreement threshold controls whether majority wins."""

    async def test_min_agreement_067_not_reached(self) -> None:
        """2/3 = 0.667 < 0.67 → consensus NOT reached."""
        a = _make_answer_agent("Paris")
        b = _make_answer_agent("Paris")
        c = _make_answer_agent("Lyon")
        swarm = Swarm(
            agents=[a, b, c],
            goal="Capital?",
            config=SwarmConfig(
                topology=SwarmTopology.CONSENSUS,
                consensus=ConsensusConfig(
                    strategy=ConsensusStrategy.MAJORITY,
                    min_agreement=0.67,
                ),
            ),
        )
        result = await swarm.run()
        assert result.consensus_result is not None
        assert result.consensus_result.consensus_reached is False

    async def test_min_agreement_060_reached(self) -> None:
        """2/3 = 0.667 > 0.60 → consensus IS reached."""
        a = _make_answer_agent("Paris")
        b = _make_answer_agent("Paris")
        c = _make_answer_agent("Lyon")
        swarm = Swarm(
            agents=[a, b, c],
            goal="Capital?",
            config=SwarmConfig(
                topology=SwarmTopology.CONSENSUS,
                consensus=ConsensusConfig(
                    strategy=ConsensusStrategy.MAJORITY,
                    min_agreement=0.60,
                ),
            ),
        )
        result = await swarm.run()
        assert result.consensus_result is not None
        assert result.consensus_result.consensus_reached is True


# ---------------------------------------------------------------------------
# P4-T1-7: Votes
# ---------------------------------------------------------------------------


@pytest.mark.phase_4
class TestConsensusVotes:
    """ConsensusResult.votes has one entry per agent."""

    async def test_votes_one_per_agent(self) -> None:
        """3 agents → 3 votes in ConsensusResult.votes."""
        agents = [_make_answer_agent(f"ans{i}") for i in range(3)]
        # Give them distinct names
        for i, ag in enumerate(agents):
            type(ag).__name__ = f"AgentVote{i}"
        swarm = Swarm(
            agents=agents,
            goal="goal",
            config=SwarmConfig(topology=SwarmTopology.CONSENSUS),
        )
        result = await swarm.run()
        assert result.consensus_result is not None
        assert len(result.consensus_result.votes) == 3

    async def test_votes_have_agent_name_and_answer(self) -> None:
        """Each vote has agent_name and answer set."""
        a = _make_answer_agent("Paris")
        type(a).__name__ = "MyAgent"
        swarm = Swarm(
            agents=[a],
            goal="Capital?",
            config=SwarmConfig(topology=SwarmTopology.CONSENSUS),
        )
        result = await swarm.run()
        assert result.consensus_result is not None
        vote = result.consensus_result.votes[0]
        assert vote.agent_name == "MyAgent"
        assert vote.answer == "Paris"

    async def test_agreement_fraction_correct(self) -> None:
        """agreement_fraction = 2/3 when 2 of 3 agree."""
        a = _make_answer_agent("Paris")
        b = _make_answer_agent("Paris")
        c = _make_answer_agent("Lyon")
        swarm = Swarm(
            agents=[a, b, c],
            goal="Capital?",
            config=SwarmConfig(topology=SwarmTopology.CONSENSUS),
        )
        result = await swarm.run()
        assert result.consensus_result is not None
        assert result.consensus_result.agreement_fraction == pytest.approx(2 / 3, abs=0.001)


# ---------------------------------------------------------------------------
# P4-T1-8: Budget split equally
# ---------------------------------------------------------------------------


@pytest.mark.phase_4
class TestConsensusBudget:
    """Budget is split equally among all consensus agents."""

    async def test_budget_report_has_per_agent_entries(self) -> None:
        """SwarmResult.budget_report.per_agent has an entry for each agent."""
        a = _make_answer_agent("Paris", cost=0.02)
        b = _make_answer_agent("Paris", cost=0.02)
        budget = Budget(max_cost=5.00)
        swarm = Swarm(
            agents=[a, b],
            goal="Capital?",
            budget=budget,
            config=SwarmConfig(topology=SwarmTopology.CONSENSUS),
        )
        result = await swarm.run()
        assert result.budget_report is not None
        assert len(result.budget_report.per_agent) == 2

    async def test_budget_total_spent_reflects_all_agents(self) -> None:
        """total_spent = sum of all agent costs."""
        a = _make_answer_agent("Paris", cost=0.02)
        b = _make_answer_agent("Paris", cost=0.03)
        budget = Budget(max_cost=5.00)
        swarm = Swarm(
            agents=[a, b],
            goal="Capital?",
            budget=budget,
            config=SwarmConfig(topology=SwarmTopology.CONSENSUS),
        )
        result = await swarm.run()
        assert result.budget_report is not None
        assert result.budget_report.total_spent == pytest.approx(0.05)


# ---------------------------------------------------------------------------
# P4-T1-9: Independence — each agent receives same goal text
# ---------------------------------------------------------------------------


@pytest.mark.phase_4
class TestConsensusIndependence:
    """Each agent receives the swarm goal, not other agents' outputs."""

    async def test_each_agent_receives_goal_text(self) -> None:
        """All agents are called with swarm.goal as input_text."""
        received: list[str] = []

        class _TrackAgent(Agent):
            model = Model.Almock()
            system_prompt = "track"

            async def arun(self, input_text: str) -> Response[str]:
                received.append(input_text)
                return Response(content="tracked", cost=0.01)

        agents = [_TrackAgent() for _ in range(2)]
        for i, ag in enumerate(agents):
            type(ag).__name__ = f"Tracker{i}"

        goal = "What is 2+2?"
        swarm = Swarm(
            agents=agents,
            goal=goal,
            config=SwarmConfig(topology=SwarmTopology.CONSENSUS),
        )
        await swarm.run()
        assert all(t == goal for t in received)


# ---------------------------------------------------------------------------
# P4-T1-10: Concurrency — all agents start concurrently
# ---------------------------------------------------------------------------


@pytest.mark.phase_4
class TestConsensusConcurrency:
    """All agents run concurrently via asyncio.gather."""

    async def test_agents_run_concurrently(self) -> None:
        """Three 50ms agents complete in <120ms total."""
        import time

        def _make_slow(answer: str) -> Agent:
            class _Slow(Agent):
                model = Model.Almock()
                system_prompt = "slow"

                async def arun(self, input_text: str) -> Response[str]:
                    await asyncio.sleep(0.05)
                    return Response(content=answer, cost=0.01)

            _Slow.__name__ = f"Slow_{answer}"
            return _Slow()

        agents = [_make_slow("A"), _make_slow("A"), _make_slow("B")]
        swarm = Swarm(
            agents=agents,
            goal="concurrent test",
            config=SwarmConfig(topology=SwarmTopology.CONSENSUS),
        )
        start = time.monotonic()
        await swarm.run()
        elapsed = time.monotonic() - start
        assert elapsed < 0.12, f"Expected parallel (<0.12s), got {elapsed:.3f}s"


# ---------------------------------------------------------------------------
# P4-T1-11/12/13/14: Hooks
# ---------------------------------------------------------------------------


@pytest.mark.phase_4
class TestConsensusHooks:
    """CONSENSUS topology fires lifecycle hooks correctly."""

    async def test_swarm_started_fires(self) -> None:
        """Hook.SWARM_STARTED fires before any agent starts."""
        fired: list[EventContext] = []
        a = _make_answer_agent("Paris")
        swarm = Swarm(
            agents=[a],
            goal="hook test",
            config=SwarmConfig(topology=SwarmTopology.CONSENSUS),
        )
        swarm.events.on(Hook.SWARM_STARTED, lambda ctx: fired.append(ctx))
        await swarm.run()
        assert len(fired) == 1

    async def test_agent_joined_fires_per_agent(self) -> None:
        """Hook.AGENT_JOINED_SWARM fires once per agent."""
        joined: list[EventContext] = []
        agents = [_make_answer_agent(f"ans{i}") for i in range(3)]
        swarm = Swarm(
            agents=agents,
            goal="join hooks",
            config=SwarmConfig(topology=SwarmTopology.CONSENSUS),
        )
        swarm.events.on(Hook.AGENT_JOINED_SWARM, lambda ctx: joined.append(ctx))
        await swarm.run()
        assert len(joined) == 3

    async def test_agent_left_fires_per_agent(self) -> None:
        """Hook.AGENT_LEFT_SWARM fires once per agent when it completes."""
        left: list[EventContext] = []
        agents = [_make_answer_agent(f"ans{i}") for i in range(2)]
        swarm = Swarm(
            agents=agents,
            goal="leave hooks",
            config=SwarmConfig(topology=SwarmTopology.CONSENSUS),
        )
        swarm.events.on(Hook.AGENT_LEFT_SWARM, lambda ctx: left.append(ctx))
        await swarm.run()
        assert len(left) == 2

    async def test_swarm_ended_fires_with_consensus_reached(self) -> None:
        """Hook.SWARM_ENDED fires with consensus_reached in context."""
        ended: list[EventContext] = []
        agents = [_make_answer_agent("Paris") for _ in range(2)]
        swarm = Swarm(
            agents=agents,
            goal="end hook",
            config=SwarmConfig(topology=SwarmTopology.CONSENSUS),
        )
        swarm.events.on(Hook.SWARM_ENDED, lambda ctx: ended.append(ctx))
        await swarm.run()
        assert len(ended) == 1
        assert "consensus_reached" in ended[0]

    async def test_hook_order_started_before_joined(self) -> None:
        """SWARM_STARTED fires before AGENT_JOINED_SWARM, SWARM_ENDED fires last."""
        order: list[str] = []
        a = _make_answer_agent("Paris")
        swarm = Swarm(
            agents=[a],
            goal="order test",
            config=SwarmConfig(topology=SwarmTopology.CONSENSUS),
        )
        swarm.events.on(Hook.SWARM_STARTED, lambda _ctx: order.append("STARTED"))
        swarm.events.on(Hook.AGENT_JOINED_SWARM, lambda _ctx: order.append("JOINED"))
        swarm.events.on(Hook.SWARM_ENDED, lambda _ctx: order.append("ENDED"))
        await swarm.run()
        assert order[0] == "STARTED"
        assert order[-1] == "ENDED"


# ---------------------------------------------------------------------------
# P4-T1-15: One agent fails → SKIP_AND_CONTINUE → vote from 2 remaining
# ---------------------------------------------------------------------------


@pytest.mark.phase_4
class TestConsensusFailure:
    """Agent failure with SKIP_AND_CONTINUE excludes failed agent from vote."""

    async def test_one_fail_skip_and_continue(self) -> None:
        """1 failing agent + 2 agreeing → no crash, consensus from 2 votes."""
        a = _make_answer_agent("Paris")
        b = _make_answer_agent("Paris")
        bad = _make_failing_agent("BadConsensusAgent")
        swarm = Swarm(
            agents=[a, b, bad],
            goal="Capital?",
            config=SwarmConfig(
                topology=SwarmTopology.CONSENSUS,
                on_agent_failure=FallbackStrategy.SKIP_AND_CONTINUE,
            ),
        )
        result = await swarm.run()
        assert result.consensus_result is not None
        # Only 2 votes (failed agent skipped)
        assert len(result.consensus_result.votes) == 2
        # Both remaining agreed
        assert result.consensus_result.consensus_reached is True
        assert result.consensus_result.content == "Paris"

    async def test_agent_failed_hook_fires_on_failure(self) -> None:
        """Hook.AGENT_FAILED fires when a consensus agent raises."""
        failed: list[EventContext] = []
        bad = _make_failing_agent("FailConsensus")
        a = _make_answer_agent("Paris")
        swarm = Swarm(
            agents=[a, bad],
            goal="fail hook",
            config=SwarmConfig(
                topology=SwarmTopology.CONSENSUS,
                on_agent_failure=FallbackStrategy.SKIP_AND_CONTINUE,
            ),
        )
        swarm.events.on(Hook.AGENT_FAILED, lambda ctx: failed.append(ctx))
        await swarm.run()
        assert len(failed) >= 1


# ---------------------------------------------------------------------------
# P4-T1-16: WEIGHTED strategy
# ---------------------------------------------------------------------------


@pytest.mark.phase_4
class TestConsensusWeighted:
    """WEIGHTED strategy uses agent.weight attribute for vote counting."""

    async def test_weighted_heavy_agent_wins(self) -> None:
        """Agent with weight=3.0 on 'Lyon' beats 2 agents with weight=1.0 on 'Paris'."""
        a = _make_answer_agent("Paris", weight=1.0)
        b = _make_answer_agent("Paris", weight=1.0)
        c = _make_answer_agent("Lyon", weight=3.0)
        swarm = Swarm(
            agents=[a, b, c],
            goal="Capital?",
            config=SwarmConfig(
                topology=SwarmTopology.CONSENSUS,
                consensus=ConsensusConfig(
                    strategy=ConsensusStrategy.WEIGHTED,
                    min_agreement=0.5,
                ),
            ),
        )
        result = await swarm.run()
        assert result.consensus_result is not None
        assert result.consensus_result.consensus_reached is True
        assert result.consensus_result.content == "Lyon"

    async def test_weighted_default_weight_one(self) -> None:
        """Agents without weight attribute default to weight=1.0."""

        # Use regular agents (no explicit weight)
        class _NoWeight(Agent):
            model = Model.Almock()
            system_prompt = "no-weight"

            async def arun(self, input_text: str) -> Response[str]:
                return Response(content="Paris", cost=0.01)

        agents = [_NoWeight() for _ in range(2)]
        for i, ag in enumerate(agents):
            type(ag).__name__ = f"NoWeight{i}"

        swarm = Swarm(
            agents=agents,
            goal="Capital?",
            config=SwarmConfig(
                topology=SwarmTopology.CONSENSUS,
                consensus=ConsensusConfig(strategy=ConsensusStrategy.WEIGHTED),
            ),
        )
        result = await swarm.run()
        assert result.consensus_result is not None
        assert result.consensus_result.consensus_reached is True
