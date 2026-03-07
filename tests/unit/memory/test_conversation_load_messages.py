"""Tests for Memory.load_conversation_messages - checkpoint restore."""

from __future__ import annotations

from syrin.enums import MessageRole
from syrin.memory import Memory
from syrin.types import Message


class TestMemoryLoadConversationMessages:
    """Tests for Memory.load_conversation_messages()."""

    def test_load_empty_replaces_all(self) -> None:
        """Load empty list clears segments."""
        mem = Memory()
        mem.add_conversation_segment("Hello", role="user")
        mem.add_conversation_segment("Hi", role="assistant")
        mem.load_conversation_messages([])
        assert mem.get_conversation_messages() == []

    def test_load_replaces_existing(self) -> None:
        """Load replaces all existing segments."""
        mem = Memory()
        mem.add_conversation_segment("A", role="user")
        mem.add_conversation_segment("B", role="assistant")
        new_msgs = [
            Message(role=MessageRole.USER, content="X"),
            Message(role=MessageRole.ASSISTANT, content="Y"),
        ]
        mem.load_conversation_messages(new_msgs)
        got = mem.get_conversation_messages()
        assert len(got) == 2
        assert got[0].content == "X"
        assert got[1].content == "Y"

    def test_load_preserves_order(self) -> None:
        """Load preserves message order."""
        mem = Memory()
        msgs = [
            Message(role=MessageRole.USER, content="1"),
            Message(role=MessageRole.ASSISTANT, content="2"),
            Message(role=MessageRole.USER, content="3"),
        ]
        mem.load_conversation_messages(msgs)
        got = mem.get_conversation_messages()
        assert [m.content for m in got] == ["1", "2", "3"]

    def test_load_from_dict_checkpoint_serialization(self) -> None:
        """Load accepts dicts (from checkpoint serialization)."""
        mem = Memory()
        msgs_dict = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
        ]
        mem.load_conversation_messages(msgs_dict)
        got = mem.get_conversation_messages()
        assert len(got) == 2
        assert got[0].content == "Hello"
        assert got[1].content == "Hi"
