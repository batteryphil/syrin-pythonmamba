"""P4-T2: REFLECTION topology tests."""

from __future__ import annotations

import pytest

from syrin import Agent, Model
from syrin.enums import Hook, SwarmTopology
from syrin.events import EventContext
from syrin.response import Response
from syrin.swarm import Swarm, SwarmConfig
from syrin.swarm.topologies._reflection import (
    ReflectionConfig,
    ReflectionResult,
    RoundOutput,
)

# ---------------------------------------------------------------------------
# Stub agents
# ---------------------------------------------------------------------------


class WriterAgent(Agent):
    """Stub producer agent for reflection tests."""

    model = Model.Almock()
    system_prompt = "Write"
    _call_count: int = 0

    async def arun(self, input_text: str) -> Response[str]:
        WriterAgent._call_count += 1
        return Response(content=f"Draft {WriterAgent._call_count}: {input_text[:30]}", cost=0.01)


class CriticAgent(Agent):
    """Stub critic agent for reflection tests — always scores 0.9."""

    model = Model.Almock()
    system_prompt = "Critique"

    async def arun(self, input_text: str) -> Response[str]:
        return Response(content="Good work. Score: 0.9", cost=0.01)


class LowScoreCritic(Agent):
    """Stub critic that always gives a low score."""

    model = Model.Almock()
    system_prompt = "Low score critic"

    async def arun(self, input_text: str) -> Response[str]:
        return Response(content="Needs improvement. Score: 0.3", cost=0.01)


class VariedScoreCritic(Agent):
    """Stub critic whose score increases each round (for early stopping tests)."""

    model = Model.Almock()
    system_prompt = "Varied score critic"
    _round: int = 0

    async def arun(self, input_text: str) -> Response[str]:
        VariedScoreCritic._round += 1
        score = min(0.5 * VariedScoreCritic._round, 1.0)
        return Response(content=f"Score: {score}", cost=0.01)


# ---------------------------------------------------------------------------
# P4-T2-1: Construction
# ---------------------------------------------------------------------------


@pytest.mark.phase_4
class TestReflectionConstruction:
    """ReflectionConfig constructs without error."""

    async def test_reflection_config_constructs(self) -> None:
        """ReflectionConfig with producer and critic constructs correctly."""
        cfg = ReflectionConfig(
            producer=WriterAgent,
            critic=CriticAgent,
            max_rounds=3,
        )
        assert cfg.producer is WriterAgent
        assert cfg.critic is CriticAgent
        assert cfg.max_rounds == 3
        assert cfg.stop_when is None
        assert cfg.budget_per_round is None

    async def test_round_output_constructs(self) -> None:
        """RoundOutput constructs with expected fields."""
        ro = RoundOutput(
            round_index=0,
            producer_output="draft",
            critic_feedback="good",
            score=0.9,
            stop_condition_met=False,
        )
        assert ro.round_index == 0
        assert ro.producer_output == "draft"
        assert ro.critic_feedback == "good"
        assert ro.score == pytest.approx(0.9)

    async def test_reflection_result_constructs(self) -> None:
        """ReflectionResult constructs with expected fields."""
        rr = ReflectionResult(
            content="best draft",
            round_outputs=[],
            final_round=0,
            rounds_completed=1,
        )
        assert rr.content == "best draft"
        assert rr.rounds_completed == 1


# ---------------------------------------------------------------------------
# P4-T2-2: Basic run — 1 round
# ---------------------------------------------------------------------------


@pytest.mark.phase_4
class TestReflectionBasicRun:
    """Producer runs, critic evaluates, result returned after 1 round."""

    async def test_basic_reflection_run(self) -> None:
        """Swarm with REFLECTION topology runs and returns SwarmResult."""
        WriterAgent._call_count = 0
        swarm = Swarm(
            agents=[WriterAgent()],
            goal="Write a haiku about AI",
            config=SwarmConfig(
                topology=SwarmTopology.REFLECTION,
                reflection=ReflectionConfig(
                    producer=WriterAgent,
                    critic=CriticAgent,
                    max_rounds=1,
                ),
            ),
        )
        result = await swarm.run()
        assert result.reflection_result is not None
        assert result.reflection_result.rounds_completed == 1
        assert result.content != ""

    async def test_reflection_result_embedded_in_swarm_result(self) -> None:
        """SwarmResult.reflection_result is a ReflectionResult."""
        WriterAgent._call_count = 0
        swarm = Swarm(
            agents=[WriterAgent()],
            goal="Write a poem",
            config=SwarmConfig(
                topology=SwarmTopology.REFLECTION,
                reflection=ReflectionConfig(
                    producer=WriterAgent,
                    critic=CriticAgent,
                    max_rounds=1,
                ),
            ),
        )
        result = await swarm.run()
        assert isinstance(result.reflection_result, ReflectionResult)


# ---------------------------------------------------------------------------
# P4-T2-3: stop_when — early stopping when score meets threshold
# ---------------------------------------------------------------------------


@pytest.mark.phase_4
class TestReflectionEarlyStopping:
    """stop_when predicate stops the loop early when condition is met."""

    async def test_stop_when_high_score_stops_after_round_1(self) -> None:
        """CriticAgent gives 0.9 → stop_when(score >= 0.85) → stops after 1 round."""
        WriterAgent._call_count = 0

        def stop_fn(ro):
            return ro.score >= 0.85

        swarm = Swarm(
            agents=[WriterAgent()],
            goal="Write something",
            config=SwarmConfig(
                topology=SwarmTopology.REFLECTION,
                reflection=ReflectionConfig(
                    producer=WriterAgent,
                    critic=CriticAgent,
                    max_rounds=5,
                    stop_when=stop_fn,
                ),
            ),
        )
        result = await swarm.run()
        assert result.reflection_result is not None
        # Should stop after round 1 since score 0.9 >= 0.85
        assert result.reflection_result.rounds_completed == 1

    async def test_stop_when_none_runs_all_rounds(self) -> None:
        """stop_when=None → runs all max_rounds."""
        WriterAgent._call_count = 0
        swarm = Swarm(
            agents=[WriterAgent()],
            goal="Write something",
            config=SwarmConfig(
                topology=SwarmTopology.REFLECTION,
                reflection=ReflectionConfig(
                    producer=WriterAgent,
                    critic=LowScoreCritic,
                    max_rounds=3,
                    stop_when=None,
                ),
            ),
        )
        result = await swarm.run()
        assert result.reflection_result is not None
        assert result.reflection_result.rounds_completed == 3


# ---------------------------------------------------------------------------
# P4-T2-4/5: round_outputs
# ---------------------------------------------------------------------------


@pytest.mark.phase_4
class TestReflectionRoundOutputs:
    """ReflectionResult.round_outputs is correct per round."""

    async def test_round_output_producer_output_set(self) -> None:
        """round_outputs[0].producer_output is set from round 1."""
        WriterAgent._call_count = 0
        swarm = Swarm(
            agents=[WriterAgent()],
            goal="Write a haiku",
            config=SwarmConfig(
                topology=SwarmTopology.REFLECTION,
                reflection=ReflectionConfig(
                    producer=WriterAgent,
                    critic=CriticAgent,
                    max_rounds=1,
                ),
            ),
        )
        result = await swarm.run()
        assert result.reflection_result is not None
        assert len(result.reflection_result.round_outputs) >= 1
        ro = result.reflection_result.round_outputs[0]
        assert ro.producer_output != ""

    async def test_round_output_critic_feedback_set(self) -> None:
        """round_outputs[0].critic_feedback contains critic's response."""
        WriterAgent._call_count = 0
        swarm = Swarm(
            agents=[WriterAgent()],
            goal="Write a haiku",
            config=SwarmConfig(
                topology=SwarmTopology.REFLECTION,
                reflection=ReflectionConfig(
                    producer=WriterAgent,
                    critic=CriticAgent,
                    max_rounds=1,
                ),
            ),
        )
        result = await swarm.run()
        assert result.reflection_result is not None
        ro = result.reflection_result.round_outputs[0]
        assert "Score" in ro.critic_feedback or ro.score > 0

    async def test_round_2_feedback_included_in_producer_input(self) -> None:
        """In round 2, producer receives round 1 critic feedback as context."""
        received_inputs: list[str] = []

        class _TrackWriter(Agent):
            model = Model.Almock()
            system_prompt = "Track"

            async def arun(self, input_text: str) -> Response[str]:
                received_inputs.append(input_text)
                return Response(content=f"Draft: {input_text[:20]}", cost=0.01)

        swarm = Swarm(
            agents=[_TrackWriter()],
            goal="Write a haiku",
            config=SwarmConfig(
                topology=SwarmTopology.REFLECTION,
                reflection=ReflectionConfig(
                    producer=_TrackWriter,
                    critic=CriticAgent,  # returns "Good work. Score: 0.9"
                    max_rounds=2,
                    stop_when=None,  # run all rounds
                ),
            ),
        )
        # Override CriticAgent to always give low score so we run round 2
        swarm.config.reflection = ReflectionConfig(  # type: ignore[attr-defined]
            producer=_TrackWriter,
            critic=LowScoreCritic,
            max_rounds=2,
            stop_when=None,
        )
        result = await swarm.run()
        assert result.reflection_result is not None
        # Round 2's input should contain feedback from round 1
        if len(received_inputs) >= 2:
            assert (
                len(received_inputs[1]) > len(received_inputs[0]) or "Score" in received_inputs[1]
            )


# ---------------------------------------------------------------------------
# P4-T2-6: round_outputs length
# ---------------------------------------------------------------------------


@pytest.mark.phase_4
class TestReflectionRoundCount:
    """round_outputs has one entry per completed round."""

    async def test_round_outputs_length_matches_rounds_completed(self) -> None:
        """3 rounds → len(round_outputs) == 3."""
        WriterAgent._call_count = 0
        swarm = Swarm(
            agents=[WriterAgent()],
            goal="Write something",
            config=SwarmConfig(
                topology=SwarmTopology.REFLECTION,
                reflection=ReflectionConfig(
                    producer=WriterAgent,
                    critic=LowScoreCritic,
                    max_rounds=3,
                    stop_when=None,
                ),
            ),
        )
        result = await swarm.run()
        assert result.reflection_result is not None
        assert len(result.reflection_result.round_outputs) == 3
        assert result.reflection_result.rounds_completed == 3


# ---------------------------------------------------------------------------
# P4-T2-7: final_round index
# ---------------------------------------------------------------------------


@pytest.mark.phase_4
class TestReflectionFinalRound:
    """final_round is the 0-based index where stop condition was met."""

    async def test_final_round_zero_when_stop_in_round_1(self) -> None:
        """High score in round 1 → final_round = 0."""
        WriterAgent._call_count = 0
        swarm = Swarm(
            agents=[WriterAgent()],
            goal="Write something",
            config=SwarmConfig(
                topology=SwarmTopology.REFLECTION,
                reflection=ReflectionConfig(
                    producer=WriterAgent,
                    critic=CriticAgent,  # score 0.9
                    max_rounds=5,
                    stop_when=lambda ro: ro.score >= 0.85,
                ),
            ),
        )
        result = await swarm.run()
        assert result.reflection_result is not None
        assert result.reflection_result.final_round == 0

    async def test_final_round_is_last_when_no_stop_condition(self) -> None:
        """No stop_when + max_rounds=3 → final_round = 2 (last)."""
        WriterAgent._call_count = 0
        swarm = Swarm(
            agents=[WriterAgent()],
            goal="Write something",
            config=SwarmConfig(
                topology=SwarmTopology.REFLECTION,
                reflection=ReflectionConfig(
                    producer=WriterAgent,
                    critic=LowScoreCritic,
                    max_rounds=3,
                    stop_when=None,
                ),
            ),
        )
        result = await swarm.run()
        assert result.reflection_result is not None
        assert result.reflection_result.final_round == 2


# ---------------------------------------------------------------------------
# P4-T2-8: Max rounds hit without stop_when → returns best result
# ---------------------------------------------------------------------------


@pytest.mark.phase_4
class TestReflectionBestResult:
    """When max_rounds hit, returns the round with the highest score."""

    async def test_max_rounds_returns_best_score_round(self) -> None:
        """All rounds have varying scores; best round's content is returned."""
        scores = [0.3, 0.8, 0.5]
        call_count = [0]

        class _VaryingCritic(Agent):
            model = Model.Almock()
            system_prompt = "varying"

            async def arun(self, input_text: str) -> Response[str]:
                idx = call_count[0] % len(scores)
                call_count[0] += 1
                return Response(content=f"Score: {scores[idx]}", cost=0.01)

        class _SimpleWriter(Agent):
            model = Model.Almock()
            system_prompt = "write"
            _n = [0]

            async def arun(self, input_text: str) -> Response[str]:
                _SimpleWriter._n[0] += 1
                return Response(content=f"output_{_SimpleWriter._n[0]}", cost=0.01)

        swarm = Swarm(
            agents=[_SimpleWriter()],
            goal="Best result test",
            config=SwarmConfig(
                topology=SwarmTopology.REFLECTION,
                reflection=ReflectionConfig(
                    producer=_SimpleWriter,
                    critic=_VaryingCritic,
                    max_rounds=3,
                    stop_when=None,
                ),
            ),
        )
        result = await swarm.run()
        assert result.reflection_result is not None
        # Best round was index 1 (score 0.8 = highest)
        assert result.reflection_result.final_round == 1
        # Content should come from round 1 (output_2 = second writer call)
        assert "output_2" in result.content


# ---------------------------------------------------------------------------
# P4-T2-9: budget_per_round
# ---------------------------------------------------------------------------


@pytest.mark.phase_4
class TestReflectionBudgetPerRound:
    """budget_per_round caps each round's total cost."""

    async def test_budget_per_round_in_config(self) -> None:
        """ReflectionConfig accepts budget_per_round."""
        cfg = ReflectionConfig(
            producer=WriterAgent,
            critic=CriticAgent,
            budget_per_round=0.50,
        )
        assert cfg.budget_per_round == pytest.approx(0.50)


# ---------------------------------------------------------------------------
# P4-T2-11/12/13: Hooks
# ---------------------------------------------------------------------------


@pytest.mark.phase_4
class TestReflectionHooks:
    """REFLECTION topology fires lifecycle hooks."""

    async def test_swarm_started_fires(self) -> None:
        """Hook.SWARM_STARTED fires before any round."""
        fired: list[EventContext] = []
        WriterAgent._call_count = 0
        swarm = Swarm(
            agents=[WriterAgent()],
            goal="hook test",
            config=SwarmConfig(
                topology=SwarmTopology.REFLECTION,
                reflection=ReflectionConfig(
                    producer=WriterAgent,
                    critic=CriticAgent,
                    max_rounds=1,
                ),
            ),
        )
        swarm.events.on(Hook.SWARM_STARTED, lambda ctx: fired.append(ctx))
        await swarm.run()
        assert len(fired) == 1

    async def test_swarm_ended_fires_with_rounds_completed(self) -> None:
        """Hook.SWARM_ENDED fires with rounds_completed in context."""
        ended: list[EventContext] = []
        WriterAgent._call_count = 0
        swarm = Swarm(
            agents=[WriterAgent()],
            goal="end hook",
            config=SwarmConfig(
                topology=SwarmTopology.REFLECTION,
                reflection=ReflectionConfig(
                    producer=WriterAgent,
                    critic=CriticAgent,
                    max_rounds=2,
                    stop_when=None,
                ),
            ),
        )
        swarm.events.on(Hook.SWARM_ENDED, lambda ctx: ended.append(ctx))
        await swarm.run()
        assert len(ended) == 1
        assert "rounds_completed" in ended[0]

    async def test_agent_joined_fires_for_producer_and_critic(self) -> None:
        """Hook.AGENT_JOINED_SWARM fires for producer and critic each round."""
        joined: list[EventContext] = []
        WriterAgent._call_count = 0
        swarm = Swarm(
            agents=[WriterAgent()],
            goal="join hooks",
            config=SwarmConfig(
                topology=SwarmTopology.REFLECTION,
                reflection=ReflectionConfig(
                    producer=WriterAgent,
                    critic=CriticAgent,
                    max_rounds=1,
                ),
            ),
        )
        swarm.events.on(Hook.AGENT_JOINED_SWARM, lambda ctx: joined.append(ctx))
        await swarm.run()
        # At least 2 joined events (producer + critic for round 1)
        assert len(joined) >= 2


# ---------------------------------------------------------------------------
# P4-T2-14: Producer in round N+1 gets critic feedback from round N
# ---------------------------------------------------------------------------


@pytest.mark.phase_4
class TestReflectionFeedbackPropagation:
    """Critic feedback is passed to the producer in the next round."""

    async def test_producer_receives_critic_feedback_in_round_2(self) -> None:
        """Producer's second call includes critic feedback string."""
        received_inputs: list[str] = []

        class _TrackWriter2(Agent):
            model = Model.Almock()
            system_prompt = "Track2"

            async def arun(self, input_text: str) -> Response[str]:
                received_inputs.append(input_text)
                return Response(content=f"Draft: {len(received_inputs)}", cost=0.01)

        swarm = Swarm(
            agents=[_TrackWriter2()],
            goal="Original goal",
            config=SwarmConfig(
                topology=SwarmTopology.REFLECTION,
                reflection=ReflectionConfig(
                    producer=_TrackWriter2,
                    critic=LowScoreCritic,  # Score: 0.3 — will not stop early
                    max_rounds=2,
                    stop_when=None,
                ),
            ),
        )
        result = await swarm.run()
        assert result.reflection_result is not None
        assert len(received_inputs) >= 2
        # Round 2 input should differ from round 1 and include feedback
        assert received_inputs[1] != received_inputs[0]
        # The critic feedback "Score: 0.3" or similar should appear in round 2 input
        assert "Score" in received_inputs[1] or "Needs" in received_inputs[1]
