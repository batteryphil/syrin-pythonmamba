"""Tests for Response object (response.py)."""

from __future__ import annotations

from syrin.enums import StopReason
from syrin.response import Response, TraceStep
from syrin.types import TokenUsage


def test_response_str_is_content() -> None:
    r = Response(content="Hello world")
    assert str(r) == "Hello world"
    assert r.content == "Hello world"


def test_response_has_cost_and_trace() -> None:
    r = Response(
        content="Hi",
        raw="Hi",
        cost=0.01,
        tokens=TokenUsage(input_tokens=10, output_tokens=5),
        model="gpt-4",
        duration=0.1,
        trace=[TraceStep(step_type="llm", timestamp=0, tokens=15, cost_usd=0.01)],
        stop_reason=StopReason.END_TURN,
    )
    assert r.cost == 0.01
    assert r.model == "gpt-4"
    assert len(r.trace) == 1
    assert r.trace[0].step_type == "llm"
    assert r.duration == 0.1


# =============================================================================
# RESPONSE EDGE CASES - TRY TO BREAK FUNCTIONALITY
# =============================================================================


def test_response_with_empty_content() -> None:
    """Response with empty content."""
    r = Response(content="")
    assert r.content == ""


def test_response_with_none_content() -> None:
    """Response with None content."""
    r = Response(content=None)
    assert r.content is None


def test_response_with_very_long_content() -> None:
    """Response with very long content."""
    long_content = "x" * 100000
    r = Response(content=long_content)
    assert len(r.content) == 100000


def test_response_with_unicode_content() -> None:
    """Response with unicode content."""
    r = Response(content="Hello 🌍 你好 🔥")
    assert "🌍" in r.content


def test_response_with_zero_cost() -> None:
    """Response with zero cost."""
    r = Response(content="test", cost=0.0)
    assert r.cost == 0.0


def test_response_with_zero_duration() -> None:
    """Response with zero duration."""
    r = Response(content="test", duration=0.0)
    assert r.duration == 0.0


def test_response_with_large_cost() -> None:
    """Response with large cost."""
    r = Response(content="test", cost=1_000_000.0)
    assert r.cost == 1_000_000.0


def test_response_with_empty_trace() -> None:
    """Response with empty trace."""
    r = Response(content="test", trace=[])
    assert r.trace == []


def test_response_with_multiple_trace_steps() -> None:
    """Response with multiple trace steps."""
    trace = [
        TraceStep(step_type="llm", timestamp=0, tokens=10, cost_usd=0.01),
        TraceStep(step_type="tool", timestamp=1, tokens=0, cost_usd=0.0),
        TraceStep(step_type="llm", timestamp=2, tokens=20, cost_usd=0.02),
    ]
    r = Response(content="done", trace=trace)
    assert len(r.trace) == 3
    assert r.trace[0].step_type == "llm"
    assert r.trace[1].step_type == "tool"
    assert r.trace[2].step_type == "llm"


def test_response_stop_reason_values() -> None:
    """Response with various stop reasons."""
    for reason in StopReason:
        r = Response(content="test", stop_reason=reason)
        assert r.stop_reason == reason


def test_response_with_tool_calls() -> None:
    """Response with tool calls."""
    from syrin.types import ToolCall

    tool_calls = [ToolCall(id="call_1", name="search", arguments={"q": "test"})]
    r = Response(content="done", tool_calls=tool_calls)
    assert len(r.tool_calls) == 1
    assert r.tool_calls[0].name == "search"


def test_response_repr() -> None:
    """Response repr should contain content."""
    r = Response(content="Hello world")
    assert "Hello world" in repr(r)


def test_response_output_returns_content_when_no_structured() -> None:
    """response.output returns content string for plain text agents (no structured output)."""
    r = Response(content="plain text")
    assert r.output == "plain text"


def test_response_output_returns_structured_parsed() -> None:
    """response.output returns structured.parsed when structured is set."""
    from syrin.response import StructuredOutput

    mock_parsed = type("MockParsed", (), {"name": "Alice"})()
    r = Response(
        content="{}",
        structured=StructuredOutput(
            raw='{"name": "Alice"}',
            parsed=mock_parsed,
            _data={"name": "Alice"},
        ),
    )
    assert r.output is mock_parsed


def test_response_total_tokens_convenience() -> None:
    """response.total_tokens is shortcut for response.tokens.total_tokens."""
    from syrin.types import TokenUsage

    r = Response(
        content="hi", tokens=TokenUsage(input_tokens=10, output_tokens=20, total_tokens=30)
    )
    assert r.total_tokens == 30
