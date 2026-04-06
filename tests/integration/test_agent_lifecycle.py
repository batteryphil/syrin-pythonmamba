"""Agent lifecycle: response()/arun() follow same sequence; no silent path differences."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch

from syrin import Agent
from syrin.loop import SingleShotLoop
from syrin.model import Model
from syrin.types import ProviderResponse, TokenUsage


def _mock_provider_response(
    content: str = "ok",
    input_tokens: int = 10,
    output_tokens: int = 20,
) -> ProviderResponse:
    return ProviderResponse(
        content=content,
        tool_calls=[],
        token_usage=TokenUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
        ),
    )


class TestAgentLifecycleSyncAsyncParity:
    """response() and arun() must produce consistent Response shape and semantics."""

    def test_response_and_arun_return_same_shape(self) -> None:
        """response() and arun() return Response with same fields populated."""
        model = Model("anthropic/claude-3-5-sonnet")
        agent = Agent(model=model, system_prompt="Be brief.")
        mock_resp = _mock_provider_response(content="Hi", input_tokens=5, output_tokens=15)
        with patch.object(
            agent._provider, "complete", new_callable=AsyncMock, return_value=mock_resp
        ):
            r_sync = agent.run("Hello")
            r_async = asyncio.run(agent.arun("Hello"))
        for r in (r_sync, r_async):
            assert hasattr(r, "content") and r.content
            assert hasattr(r, "cost") and isinstance(r.cost, float)
            assert hasattr(r, "tokens") and r.tokens.total_tokens >= 0
            assert hasattr(r, "stop_reason")
            assert hasattr(r, "tool_calls") and isinstance(r.tool_calls, list)
            assert hasattr(r, "model")
            assert hasattr(r, "report")

    def test_response_and_arun_same_content_for_same_input(self) -> None:
        """For same input and mocked provider, response() and arun() return same content."""
        model = Model("anthropic/claude-3-5-sonnet")
        agent = Agent(model=model, system_prompt="Be brief.")
        mock_resp = _mock_provider_response(content="Same reply")
        with patch.object(
            agent._provider, "complete", new_callable=AsyncMock, return_value=mock_resp
        ):
            r_sync = agent.run("Same input")
            r_async = asyncio.run(agent.arun("Same input"))
        assert r_sync.content == r_async.content == "Same reply"

    def test_build_messages_called_before_llm(self) -> None:
        """build_messages is invoked (via run_context) before LLM complete."""
        model = Model("anthropic/claude-3-5-sonnet")
        agent = Agent(model=model, system_prompt="Test.")
        build_messages_calls: list[str] = []
        original_build = agent._build_messages

        def tracked_build(user_input: str):
            build_messages_calls.append(user_input)
            return original_build(user_input)

        agent._build_messages = tracked_build  # type: ignore[method-assign]
        mock_resp = _mock_provider_response(content="Hi")
        with patch.object(
            agent._provider, "complete", new_callable=AsyncMock, return_value=mock_resp
        ):
            agent.run("Hello")
        assert build_messages_calls == ["Hello"]

    def test_record_cost_called_after_llm(self) -> None:
        """record_cost is invoked after LLM complete so Response has cost/tokens."""
        model = Model("anthropic/claude-3-5-sonnet")
        agent = Agent(model=model, system_prompt="Test.")
        mock_resp = _mock_provider_response(content="Hi", input_tokens=100, output_tokens=50)
        with patch.object(
            agent._provider, "complete", new_callable=AsyncMock, return_value=mock_resp
        ):
            r = agent.run("Hello")
        assert r.tokens.input_tokens == 100
        assert r.tokens.output_tokens == 50
        assert r.tokens.total_tokens == 150
        assert r.cost >= 0

    def test_lifecycle_with_single_shot_loop(self) -> None:
        """Single-shot path still populates cost, tokens, stop_reason."""
        model = Model("anthropic/claude-3-5-sonnet")
        agent = Agent(
            model=model,
            system_prompt="Test.",
            loop=SingleShotLoop(),
        )
        mock_resp = _mock_provider_response(content="Done", input_tokens=1, output_tokens=2)
        with patch.object(
            agent._provider, "complete", new_callable=AsyncMock, return_value=mock_resp
        ):
            r = agent.run("Go")
        assert r.content == "Done"
        assert r.tokens.total_tokens == 3
        assert r.stop_reason.value == "end_turn"

    def test_lifecycle_with_react_loop_no_tools(self) -> None:
        """REACT loop with no tool calls returns one iteration and end_turn."""
        model = Model("anthropic/claude-3-5-sonnet")
        agent = Agent(
            model=model,
            system_prompt="Test.",
            tools=[],
        )
        mock_resp = _mock_provider_response(content="Reply", input_tokens=5, output_tokens=10)
        with patch.object(
            agent._provider, "complete", new_callable=AsyncMock, return_value=mock_resp
        ):
            r = agent.run("Hi")
        assert r.content == "Reply"
        assert r.iterations == 1
        assert r.tokens.total_tokens == 15


class TestAgentLifecycleEdgeCases:
    """Edge cases: empty input, no budget, no memory."""

    def test_empty_user_input_still_builds_messages(self) -> None:
        """Empty string input still goes through build_messages and returns Response."""
        model = Model("anthropic/claude-3-5-sonnet")
        agent = Agent(model=model, system_prompt="Test.")
        mock_resp = _mock_provider_response(content="OK")
        with patch.object(
            agent._provider, "complete", new_callable=AsyncMock, return_value=mock_resp
        ):
            r = agent.run("")
        assert r.content == "OK"

    def test_no_budget_no_memory_returns_valid_response(self) -> None:
        """Agent without budget or memory still returns valid Response."""
        model = Model("anthropic/claude-3-5-sonnet")
        agent = Agent(model=model, system_prompt="Test.", budget=None)
        mock_resp = _mock_provider_response(content="OK", input_tokens=0, output_tokens=1)
        with patch.object(
            agent._provider, "complete", new_callable=AsyncMock, return_value=mock_resp
        ):
            r = agent.run("x")
        assert r.content == "OK"
        assert r.cost >= 0
        assert r.report is not None
