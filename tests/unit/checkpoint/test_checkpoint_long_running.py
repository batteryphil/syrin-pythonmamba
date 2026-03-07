"""Tests for checkpoint long-running session support: messages and context_snapshot restore."""

from __future__ import annotations

from syrin import Agent, CheckpointConfig, Model
from syrin.memory import Memory


class TestCheckpointStoresMessages:
    """Save checkpoint stores conversation messages."""

    def test_save_checkpoint_includes_messages(self) -> None:
        """Checkpoint state includes messages from memory."""
        mem = Memory()
        mem.add_conversation_segment("Hello", role="user")
        mem.add_conversation_segment("Hi there", role="assistant")

        agent = Agent(
            model=Model.Almock(),
            system_prompt="You are helpful.",
            memory=mem,
            checkpoint=CheckpointConfig(storage="memory"),
        )
        checkpoint_id = agent.save_checkpoint()

        assert checkpoint_id is not None
        state = agent._checkpointer.load(checkpoint_id)
        assert state is not None
        assert len(state.messages) == 2
        assert state.messages[0].get("content") == "Hello"
        assert state.messages[1].get("content") == "Hi there"

    def test_save_checkpoint_empty_conversation(self) -> None:
        """Checkpoint with no conversation memory stores empty messages."""
        agent = Agent(
            model=Model.Almock(),
            system_prompt="Hi",
            checkpoint=CheckpointConfig(storage="memory"),
        )
        checkpoint_id = agent.save_checkpoint()
        assert checkpoint_id is not None
        state = agent._checkpointer.load(checkpoint_id)
        assert state is not None
        assert state.messages == []

    def test_save_checkpoint_includes_context_snapshot_in_metadata(self) -> None:
        """Checkpoint metadata includes context_snapshot when available."""
        mem = Memory()
        agent = Agent(
            model=Model.Almock(),
            system_prompt="Hi",
            memory=mem,
            checkpoint=CheckpointConfig(storage="memory"),
        )
        # Run a response to trigger prepare and populate snapshot
        agent.response("Hello")
        checkpoint_id = agent.save_checkpoint()
        assert checkpoint_id is not None
        state = agent._checkpointer.load(checkpoint_id)
        assert state is not None
        ctx_snap = state.metadata.get("context_snapshot")
        assert ctx_snap is not None
        assert isinstance(ctx_snap, dict)
        assert "utilization_pct" in ctx_snap or "total_tokens" in ctx_snap


class TestLoadCheckpointRestoresMessages:
    """Load checkpoint restores conversation messages and iteration."""

    def test_load_checkpoint_restores_messages(self) -> None:
        """Loading checkpoint restores messages to memory."""
        mem = Memory()
        mem.add_conversation_segment("A", role="user")
        mem.add_conversation_segment("B", role="assistant")

        agent = Agent(
            model=Model.Almock(),
            system_prompt="Hi",
            memory=mem,
            checkpoint=CheckpointConfig(storage="memory"),
        )
        cid = agent.save_checkpoint()

        # Clear and add different content
        mem.load_conversation_messages([])
        mem.add_conversation_segment("X", role="user")

        ok = agent.load_checkpoint(cid)
        assert ok is True
        restored = agent.messages
        assert len(restored) == 2
        assert restored[0].content == "A"
        assert restored[1].content == "B"

    def test_load_checkpoint_restores_iteration(self) -> None:
        """Loading checkpoint restores iteration count."""
        agent = Agent(
            model=Model.Almock(),
            system_prompt="Hi",
            checkpoint=CheckpointConfig(storage="memory"),
        )
        agent._last_iteration = 5
        cid = agent.save_checkpoint()

        agent._last_iteration = 0
        ok = agent.load_checkpoint(cid)
        assert ok is True
        assert agent.iteration == 5

    def test_load_checkpoint_no_memory_skips_message_restore(self) -> None:
        """Load with memory=None does not fail; iteration still restored."""
        agent = Agent(
            model=Model.Almock(),
            system_prompt="Hi",
            memory=None,
            checkpoint=CheckpointConfig(storage="memory"),
        )
        agent._last_iteration = 3
        cid = agent.save_checkpoint()
        assert agent._persistent_memory is None

        agent._last_iteration = 0
        ok = agent.load_checkpoint(cid)
        assert ok is True
        assert agent.iteration == 3

    def test_load_checkpoint_nonexistent_returns_false(self) -> None:
        """Loading nonexistent checkpoint returns False."""
        agent = Agent(
            model=Model.Almock(),
            checkpoint=CheckpointConfig(storage="memory"),
        )
        ok = agent.load_checkpoint("nonexistent_xyz")
        assert ok is False


class TestCheckpointRoundtrip:
    """Full roundtrip: multi-turn conversation, save, load, continue."""

    def test_roundtrip_messages_preserved(self) -> None:
        """Save after multi-turn, load, verify messages then continue."""
        mem = Memory()
        agent = Agent(
            model=Model.Almock(),
            system_prompt="You are helpful.",
            memory=mem,
            checkpoint=CheckpointConfig(storage="memory"),
        )
        agent.response("What is 2+2?")
        agent.response("And 3+3?")
        assert len(agent.messages) >= 4  # user, assistant, user, assistant
        cid = agent.save_checkpoint()

        # New agent instance (simulate restart), load checkpoint
        mem2 = Memory()
        agent2 = Agent(
            model=Model.Almock(),
            system_prompt="You are helpful.",
            memory=mem2,
            checkpoint=agent._checkpointer,
        )
        ok = agent2.load_checkpoint(cid)
        assert ok is True
        assert len(agent2.messages) >= 4
        # Content should match (order preserved)
        for i, m in enumerate(agent.messages[:4]):
            if i < len(agent2.messages):
                assert agent2.messages[i].content == m.content
                assert agent2.messages[i].role == m.role
