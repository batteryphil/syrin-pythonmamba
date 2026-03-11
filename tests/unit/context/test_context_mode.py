"""Tests for Context.context_mode: full, focused, intelligent."""

from __future__ import annotations

import pytest

from syrin.context import Context, DefaultContextManager
from syrin.context.config import ContextWindowCapacity
from syrin.enums import ContextMode


def _msg(role: str, content: str) -> dict:
    return {"role": role, "content": content}


class TestContextModeConfig:
    """Context mode config: full, focused, intelligent."""

    def test_default_context_mode_is_full(self) -> None:
        ctx = Context()
        assert ctx.context_mode == ContextMode.FULL

    def test_context_mode_focused(self) -> None:
        ctx = Context(context_mode=ContextMode.FOCUSED, focused_keep=10)
        assert ctx.context_mode == ContextMode.FOCUSED
        assert ctx.focused_keep == 10

    def test_focused_keep_default(self) -> None:
        ctx = Context(context_mode=ContextMode.FOCUSED)
        assert ctx.focused_keep == 10

    def test_focused_keep_invalid_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="focused_keep must be >= 1"):
            Context(context_mode=ContextMode.FOCUSED, focused_keep=0)

    def test_focused_keep_invalid_negative_raises(self) -> None:
        with pytest.raises(ValueError, match="focused_keep must be >= 1"):
            Context(context_mode=ContextMode.FOCUSED, focused_keep=-1)


class TestContextModeFull:
    """context_mode=full: no filtering, current behavior."""

    def test_full_mode_keeps_all_messages(self) -> None:
        manager = DefaultContextManager(
            context=Context(max_tokens=16000, context_mode=ContextMode.FULL)
        )
        messages = [
            _msg("user", "u1"),
            _msg("assistant", "a1"),
            _msg("user", "u2"),
            _msg("assistant", "a2"),
            _msg("user", "u3 current"),
        ]
        capacity = ContextWindowCapacity(max_tokens=16000)
        payload = manager.prepare(
            messages=messages,
            system_prompt="Sys",
            tools=[],
            capacity=capacity,
        )
        assert len(payload.messages) >= 5
        contents = [m.get("content", "") for m in payload.messages]
        assert "u1" in contents
        assert "a1" in contents
        assert "u3 current" in contents

    def test_full_mode_snapshot_no_dropped(self) -> None:
        manager = DefaultContextManager(
            context=Context(max_tokens=16000, context_mode=ContextMode.FULL)
        )
        messages = [_msg("user", "u1"), _msg("assistant", "a1"), _msg("user", "u2")]
        capacity = ContextWindowCapacity(max_tokens=16000)
        manager.prepare(messages=messages, system_prompt="", tools=[], capacity=capacity)
        snap = manager.snapshot()
        assert snap.context_mode == "full"
        assert snap.context_mode_dropped_count == 0


class TestContextModeFocused:
    """context_mode=focused: keep last N turns (user+assistant pairs)."""

    def test_focused_keeps_last_2_turns(self) -> None:
        manager = DefaultContextManager(
            context=Context(
                max_tokens=16000,
                context_mode=ContextMode.FOCUSED,
                focused_keep=2,
            )
        )
        messages = [
            _msg("user", "u1"),
            _msg("assistant", "a1"),
            _msg("user", "u2"),
            _msg("assistant", "a2"),
            _msg("user", "u3"),
            _msg("assistant", "a3"),
            _msg("user", "current question"),
        ]
        capacity = ContextWindowCapacity(max_tokens=16000)
        payload = manager.prepare(
            messages=messages,
            system_prompt="Sys",
            tools=[],
            capacity=capacity,
        )
        contents = [m.get("content", "") for m in payload.messages]
        assert "current question" in contents
        assert "u3" in contents
        assert "a3" in contents
        assert "u2" in contents
        assert "a2" in contents
        assert "u1" not in contents
        assert "a1" not in contents

    def test_focused_keeps_all_when_fewer_turns_than_keep(self) -> None:
        manager = DefaultContextManager(
            context=Context(
                max_tokens=16000,
                context_mode=ContextMode.FOCUSED,
                focused_keep=10,
            )
        )
        messages = [
            _msg("user", "u1"),
            _msg("assistant", "a1"),
            _msg("user", "u2 current"),
        ]
        capacity = ContextWindowCapacity(max_tokens=16000)
        payload = manager.prepare(
            messages=messages,
            system_prompt="",
            tools=[],
            capacity=capacity,
        )
        contents = [m.get("content", "") for m in payload.messages]
        assert "u1" in contents
        assert "a1" in contents
        assert "u2 current" in contents

    def test_focused_with_system_and_memory(self) -> None:
        manager = DefaultContextManager(
            context=Context(
                max_tokens=16000,
                context_mode=ContextMode.FOCUSED,
                focused_keep=1,
            )
        )
        messages = [
            _msg("user", "u1"),
            _msg("assistant", "a1"),
            _msg("user", "u2"),
            _msg("assistant", "a2"),
            _msg("user", "current"),
        ]
        capacity = ContextWindowCapacity(max_tokens=16000)
        payload = manager.prepare(
            messages=messages,
            system_prompt="System prompt here",
            tools=[],
            memory_context="Recalled: foo",
            capacity=capacity,
        )
        contents = [m.get("content", "") for m in payload.messages]
        assert "System prompt here" in contents or any("System" in c for c in contents)
        assert "Recalled: foo" in contents or any("[Memory]" in c for c in contents)
        assert "current" in contents
        assert "u2" in contents
        assert "a2" in contents
        assert "u1" not in contents
        assert "a1" not in contents

    def test_focused_single_current_user_no_history(self) -> None:
        manager = DefaultContextManager(
            context=Context(
                max_tokens=16000,
                context_mode=ContextMode.FOCUSED,
                focused_keep=5,
            )
        )
        messages = [_msg("user", "first message")]
        capacity = ContextWindowCapacity(max_tokens=16000)
        payload = manager.prepare(
            messages=messages,
            system_prompt="",
            tools=[],
            capacity=capacity,
        )
        assert len(payload.messages) >= 1
        assert payload.messages[-1].get("content") == "first message"

    def test_focused_snapshot_shows_dropped_count(self) -> None:
        manager = DefaultContextManager(
            context=Context(
                max_tokens=16000,
                context_mode=ContextMode.FOCUSED,
                focused_keep=1,
            )
        )
        messages = [
            _msg("user", "u1"),
            _msg("assistant", "a1"),
            _msg("user", "u2"),
            _msg("assistant", "a2"),
            _msg("user", "current"),
        ]
        capacity = ContextWindowCapacity(max_tokens=16000)
        manager.prepare(
            messages=messages,
            system_prompt="",
            tools=[],
            capacity=capacity,
        )
        snap = manager.snapshot()
        assert snap.context_mode == "focused"
        assert snap.context_mode_dropped_count == 2  # u1, a1 dropped

    def test_focused_turn_boundaries_no_orphan_assistant(self) -> None:
        """Focused keeps whole turns; no assistant without its user question."""
        manager = DefaultContextManager(
            context=Context(
                max_tokens=16000,
                context_mode=ContextMode.FOCUSED,
                focused_keep=1,
            )
        )
        messages = [
            _msg("user", "q1"),
            _msg("assistant", "ans1"),
            _msg("user", "q2"),
            _msg("assistant", "ans2"),
            _msg("user", "q3"),
        ]
        capacity = ContextWindowCapacity(max_tokens=16000)
        payload = manager.prepare(
            messages=messages,
            system_prompt="",
            tools=[],
            capacity=capacity,
        )
        contents = [m.get("content", "") for m in payload.messages]
        assert "q3" in contents
        assert "ans2" in contents
        assert "q2" in contents
        assert "ans1" not in contents
        assert "q1" not in contents
