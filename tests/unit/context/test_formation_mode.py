"""Tests for formation_mode PUSH vs PULL (Step 10).

PULL uses agent's Memory for segment storage and retrieval; no separate context_store.
"""

from __future__ import annotations

from syrin import Agent, Model
from syrin.context import Context
from syrin.enums import FormationMode
from syrin.memory import Memory


def _almock() -> Model:
    return Model.Almock(latency_seconds=0.01)


class TestFormationModePullConfig:
    """Context config validation."""

    def test_formation_mode_pull_valid_without_context_store(self) -> None:
        """formation_mode=PULL is valid; uses agent's Memory, not context_store."""
        ctx = Context(
            formation_mode=FormationMode.PULL,
            pull_top_k=5,
            pull_threshold=0.5,
        )
        assert ctx.formation_mode == FormationMode.PULL
        assert ctx.pull_top_k == 5
        assert ctx.pull_threshold == 0.5

    def test_formation_mode_push_default(self) -> None:
        """formation_mode defaults to PUSH."""
        ctx = Context()
        assert ctx.formation_mode == FormationMode.PUSH


class TestFormationModePullAgentFlow:
    """Agent with formation_mode=PULL uses Memory for segment storage and retrieval."""

    def test_pull_mode_empty_memory_only_current_user(self) -> None:
        """Pull mode with empty Memory: only system + current user in context."""
        agent = Agent(
            model=_almock(),
            system_prompt="Help.",
            memory=Memory(),
            context=Context(
                formation_mode=FormationMode.PULL,
                pull_top_k=5,
            ),
        )
        msgs = agent._build_messages("Hello")
        roles = [m.role.value for m in msgs]
        assert "system" in roles
        assert "user" in roles
        assert len(msgs) <= 3  # system, maybe memory placeholder, user

    def test_pull_mode_segments_added_and_retrieved(self) -> None:
        """After turns, segments in Memory; next turn pulls relevant ones."""
        agent = Agent(
            model=_almock(),
            system_prompt="Help.",
            memory=Memory(),
            context=Context(
                formation_mode=FormationMode.PULL,
                pull_top_k=5,
                pull_threshold=0.0,
            ),
        )
        agent.response("Tell me about Python")
        agent.response("What about JavaScript?")
        r = agent.response("Remember Python?")
        assert r.content is not None
        snap = agent.context.snapshot()
        assert hasattr(snap, "pulled_segments")
        assert hasattr(snap, "pull_scores")

    def test_pull_mode_snapshot_has_pulled_data(self) -> None:
        """Snapshot includes pulled_segments and pull_scores when pull mode."""
        memory = Memory()
        memory.add_conversation_segment("Python is great", role="user")
        memory.add_conversation_segment("Yes", role="assistant")
        agent = Agent(
            model=_almock(),
            system_prompt="Help.",
            memory=memory,
            context=Context(
                formation_mode=FormationMode.PULL,
                pull_top_k=5,
                pull_threshold=0.0,
            ),
        )
        agent.response("Tell me about Python")
        snap = agent.context.snapshot()
        assert isinstance(snap.pulled_segments, list)
        assert isinstance(snap.pull_scores, list)
