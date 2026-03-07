"""Tests for Memory conversation segment storage (add/get/load)."""

from __future__ import annotations

from syrin.memory import Memory
from syrin.types import Message


def test_memory_add_and_get_conversation() -> None:
    """Memory stores and returns conversation segments."""
    mem = Memory()
    mem.add_conversation_segment("Hi", role="user")
    mem.add_conversation_segment("Hello", role="assistant")
    msgs = mem.get_conversation_messages()
    assert len(msgs) == 2
    assert msgs[0].content == "Hi"
    assert msgs[1].content == "Hello"


def test_memory_load_replaces_all() -> None:
    """load_conversation_messages replaces existing segments."""
    mem = Memory()
    mem.add_conversation_segment("A", role="user")
    mem.add_conversation_segment("B", role="assistant")
    mem.load_conversation_messages(
        [
            Message(role="user", content="X"),
            Message(role="assistant", content="Y"),
        ]
    )
    got = mem.get_conversation_messages()
    assert len(got) == 2
    assert got[0].content == "X"
    assert got[1].content == "Y"


def test_memory_with_long_content() -> None:
    """Memory with very long content."""
    mem = Memory()
    long_content = "x" * 10000
    mem.add_conversation_segment(long_content, role="user")
    msgs = mem.get_conversation_messages()
    assert len(msgs) == 1
    assert len(msgs[0].content) == 10000


def test_memory_with_unicode() -> None:
    """Memory with unicode content."""
    mem = Memory()
    mem.add_conversation_segment("Hello 🌍 你好 🔥", role="user")
    msgs = mem.get_conversation_messages()
    assert "🌍" in msgs[0].content


def test_memory_preserves_order() -> None:
    """Memory preserves message order."""
    mem = Memory()
    for i in range(10):
        mem.add_conversation_segment(f"user {i}", role="user")
        mem.add_conversation_segment(f"assistant {i}", role="assistant")
    msgs = mem.get_conversation_messages()
    for i in range(10):
        assert msgs[i * 2].content == f"user {i}"
        assert msgs[i * 2 + 1].content == f"assistant {i}"


def test_message_with_all_roles() -> None:
    """Message with all possible roles."""
    roles = ["system", "user", "assistant", "tool"]
    for role in roles:
        msg = Message(role=role, content="test")
        assert msg.role == role


def test_message_with_tool_call_id() -> None:
    """Message with tool_call_id."""
    msg = Message(role="tool", content="result", tool_call_id="call_123")
    assert msg.tool_call_id == "call_123"
