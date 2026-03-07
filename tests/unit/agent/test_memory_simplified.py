"""Tests for simplified memory: single Memory type, default Memory()."""

from __future__ import annotations

import pytest

from syrin import Agent, Memory, Model


def _almock() -> Model:
    return Model.Almock(latency_seconds=0.01)


class TestMemoryDefault:
    """Default memory = Memory() for multi-turn."""

    def test_agent_without_memory_param_has_memory(self) -> None:
        """Agent(model=...) with no memory param gets Memory() by default."""
        agent = Agent(model=_almock())
        assert agent._persistent_memory is not None
        assert isinstance(agent._persistent_memory, Memory)

    def test_agent_default_memory_multi_turn_works(self) -> None:
        """Default Memory retains conversation across turns."""
        agent = Agent(model=_almock(), system_prompt="Be brief.")
        agent.response("Hello")
        agent.response("What did I say first?")
        msgs = agent.messages
        assert len(msgs) >= 2
        contents = [m.content for m in msgs]
        assert "Hello" in contents

    def test_agent_default_memory_build_messages_includes_history(self) -> None:
        """build_messages includes conversation history when default memory."""
        agent = Agent(model=_almock(), system_prompt="Bot.")
        agent.record_conversation_turn("First", "Hi")
        messages = agent._build_messages("Second")
        contents = [m.content for m in messages]
        assert "First" in contents
        assert "Hi" in contents
        assert "Second" in contents


class TestMemoryNone:
    """memory=None explicitly disables memory."""

    def test_agent_memory_none_no_conversation(self) -> None:
        """Agent(memory=None) has no memory."""
        agent = Agent(model=_almock(), memory=None)
        assert agent._persistent_memory is None
        assert agent._memory_backend is None

    def test_agent_memory_none_build_messages_no_history(self) -> None:
        """build_messages with memory=None has no history."""
        agent = Agent(model=_almock(), memory=None)
        messages = agent._build_messages("Hello")
        assert len(messages) >= 1
        assert messages[-1].content == "Hello"

    def test_agent_memory_false_same_as_none(self) -> None:
        """memory=False same as memory=None."""
        agent = Agent(model=_almock(), memory=False)
        assert agent._persistent_memory is None


class TestMemoryExplicit:
    """Explicit memory=Memory()."""

    def test_agent_memory_explicit(self) -> None:
        """memory=Memory() works."""
        agent = Agent(model=_almock(), memory=Memory())
        assert agent._persistent_memory is not None
        assert isinstance(agent._persistent_memory, Memory)

    def test_agent_memory_persistent_multi_turn_works(self) -> None:
        """memory=Memory() stores conversation; build_messages includes history."""
        agent = Agent(model=_almock(), memory=Memory(), system_prompt="Be brief.")
        agent.record_conversation_turn("First", "Hi")
        messages = agent._build_messages("Second")
        contents = [m.content for m in messages]
        assert "First" in contents
        assert "Hi" in contents
        assert "Second" in contents

    def test_agent_memory_true_gives_persistent(self) -> None:
        """memory=True gives persistent Memory with CORE+EPISODIC."""
        agent = Agent(model=_almock(), memory=True)
        assert agent._persistent_memory is not None


class TestConversationMemoryParamRemoved:
    """conversation_memory param is removed; use memory= instead."""

    def test_conversation_memory_param_raises_type_error(self) -> None:
        """conversation_memory is no longer a valid Agent param."""
        with pytest.raises(TypeError, match="conversation_memory"):
            Agent(model=_almock(), conversation_memory=Memory())


class TestContextNeedsMemoryWarning:
    """Warn when context needs memory but memory=None."""

    def test_context_mode_intelligent_with_memory_none_warns(self) -> None:
        """context_mode=intelligent with memory=None emits warning."""
        from syrin.context import Context
        from syrin.enums import ContextMode

        with pytest.warns(UserWarning, match="memory=None|provide memory"):
            Agent(
                model=_almock(),
                memory=None,
                context=Context(context_mode=ContextMode.INTELLIGENT),
            )


class TestMemoryProperty:
    """agent.memory property returns active memory."""

    def test_memory_property_returns_memory_when_set(self) -> None:
        """memory property returns memory when set."""
        agent = Agent(model=_almock())
        assert agent.memory is agent._persistent_memory

    def test_memory_property_none_when_disabled(self) -> None:
        """memory property is None when memory=None."""
        agent = Agent(model=_almock(), memory=None)
        assert agent.memory is None


class TestRecordConversationTurn:
    """record_conversation_turn works with default memory."""

    def test_record_turn_default_memory(self) -> None:
        """record_conversation_turn adds to default Memory."""
        agent = Agent(model=_almock())
        agent.record_conversation_turn("Hello", "Hi there!")
        msgs = agent.messages
        assert len(msgs) == 2
        assert msgs[0].content == "Hello"
        assert msgs[1].content == "Hi there!"

    def test_record_turn_memory_none_no_op(self) -> None:
        """record_conversation_turn with memory=None is no-op."""
        agent = Agent(model=_almock(), memory=None)
        agent.record_conversation_turn("Hello", "Hi")
        assert agent._persistent_memory is None
