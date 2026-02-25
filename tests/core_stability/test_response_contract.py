"""Response: always has correct cost, tokens, tool_calls, stop_reason; no data loss."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

from syrin import Agent
from syrin.enums import StopReason
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


class TestResponseFieldsAlwaysPresent:
    """Every Response returned by agent.response()/arun() has required fields set."""

    def test_normal_response_has_cost_tokens_stop_reason(self) -> None:
        """Normal completion path: cost, tokens, stop_reason, tool_calls populated."""
        model = Model("anthropic/claude-3-5-sonnet")
        agent = Agent(model=model, system_prompt="Test.")
        mock_resp = _mock_provider_response(content="Hi", input_tokens=5, output_tokens=15)
        with patch.object(
            agent._provider,
            "complete",
            new_callable=AsyncMock,
            return_value=mock_resp,
        ):
            r = agent.response("Hello")
        assert isinstance(r.cost, (int, float))
        assert r.tokens is not None
        assert r.tokens.input_tokens == 5
        assert r.tokens.output_tokens == 15
        assert r.tokens.total_tokens == 20
        assert r.stop_reason is not None
        assert isinstance(r.stop_reason, StopReason)
        assert r.tool_calls is not None
        assert isinstance(r.tool_calls, list)

    def test_response_tool_calls_list_when_no_tools_used(self) -> None:
        """When no tool calls, response.tool_calls is empty list."""
        model = Model("anthropic/claude-3-5-sonnet")
        agent = Agent(model=model, system_prompt="Test.")
        mock_resp = _mock_provider_response(content="Hi")
        with patch.object(
            agent._provider,
            "complete",
            new_callable=AsyncMock,
            return_value=mock_resp,
        ):
            r = agent.response("Hello")
        assert r.tool_calls == []

    def test_response_stop_reason_enum(self) -> None:
        """stop_reason is always a StopReason enum value."""
        model = Model("anthropic/claude-3-5-sonnet")
        agent = Agent(model=model, system_prompt="Test.")
        mock_resp = _mock_provider_response(content="Hi")
        with patch.object(
            agent._provider,
            "complete",
            new_callable=AsyncMock,
            return_value=mock_resp,
        ):
            r = agent.response("Hello")
        assert isinstance(r.stop_reason, StopReason)
        assert r.stop_reason in (
            StopReason.END_TURN,
            StopReason.MAX_ITERATIONS,
            StopReason.BUDGET,
            StopReason.GUARDRAIL,
        )

    def test_response_has_report(self) -> None:
        """Response.report is always present and has expected shape."""
        model = Model("anthropic/claude-3-5-sonnet")
        agent = Agent(model=model, system_prompt="Test.")
        with patch.object(
            agent._provider,
            "complete",
            new_callable=AsyncMock,
            return_value=_mock_provider_response(content="Hi"),
        ):
            r = agent.response("Hello")
        assert r.report is not None
        assert hasattr(r.report, "tokens")
        assert hasattr(r.report, "guardrail")


class TestResponseGuardrailPath:
    """When guardrail blocks, Response still has cost, tokens, stop_reason. Flow: input before LLM, output after (no tool calls)."""

    def test_guardrail_block_response_has_stop_reason_guardrail(self) -> None:
        """When input guardrail blocks, returned Response has stop_reason=GUARDRAIL."""
        from syrin.guardrails import GuardrailResult

        model = Model("anthropic/claude-3-5-sonnet")
        agent = Agent(model=model, system_prompt="Test.")
        agent._run_guardrails = lambda _text, _stage: GuardrailResult(passed=False)  # type: ignore[method-assign]
        r = agent.response("Hello")
        assert r.stop_reason == StopReason.GUARDRAIL
        assert r.cost >= 0
        assert r.tokens is not None
        assert r.tool_calls is not None

    def test_input_guardrail_block_does_not_call_llm(self) -> None:
        """When input guardrail blocks, provider.complete is not called."""
        from syrin.guardrails import GuardrailResult

        model = Model("anthropic/claude-3-5-sonnet")
        agent = Agent(model=model, system_prompt="Test.")
        agent._run_guardrails = lambda _text, _stage: GuardrailResult(passed=False)  # type: ignore[method-assign]
        with patch.object(
            agent._provider,
            "complete",
            new_callable=AsyncMock,
        ) as mock_complete:
            agent.response("Blocked input")
        mock_complete.assert_not_called()

    def test_output_guardrail_block_returns_guardrail_response(self) -> None:
        """When output guardrail blocks (no tool calls), response has stop_reason=GUARDRAIL and content empty."""
        from syrin.guardrails import GuardrailResult

        model = Model("anthropic/claude-3-5-sonnet")
        agent = Agent(model=model, system_prompt="Test.")

        def run_guardrails(text: str, stage: object) -> object:
            from syrin.enums import GuardrailStage

            if stage == GuardrailStage.INPUT:
                return GuardrailResult(passed=True)
            return GuardrailResult(passed=False)

        agent._run_guardrails = run_guardrails  # type: ignore[method-assign]
        mock_resp = _mock_provider_response(content="Bad output", input_tokens=5, output_tokens=10)
        with patch.object(
            agent._provider,
            "complete",
            new_callable=AsyncMock,
            return_value=mock_resp,
        ):
            r = agent.response("Hello")
        assert r.stop_reason == StopReason.GUARDRAIL
        assert r.cost >= 0
        assert r.tool_calls is not None


class TestResponseDataLoss:
    """No data loss: tokens and cost from LLM are preserved on Response."""

    def test_token_counts_preserved_from_provider(self) -> None:
        """Token usage from provider response appears on Response.tokens."""
        model = Model("anthropic/claude-3-5-sonnet")
        agent = Agent(model=model, system_prompt="Test.")
        mock_resp = _mock_provider_response(
            content="Hi",
            input_tokens=100,
            output_tokens=200,
        )
        with patch.object(
            agent._provider,
            "complete",
            new_callable=AsyncMock,
            return_value=mock_resp,
        ):
            r = agent.response("Hello")
        assert r.tokens.input_tokens == 100
        assert r.tokens.output_tokens == 200
        assert r.tokens.total_tokens == 300

    def test_content_preserved(self) -> None:
        """Response.content matches LLM response content."""
        model = Model("anthropic/claude-3-5-sonnet")
        agent = Agent(model=model, system_prompt="Test.")
        mock_resp = _mock_provider_response(content="Exact content here")
        with patch.object(
            agent._provider,
            "complete",
            new_callable=AsyncMock,
            return_value=mock_resp,
        ):
            r = agent.response("Hello")
        assert r.content == "Exact content here"


class TestResponseDataSyncAsyncParity:
    """Regression: sync response() and async arun() populate cost, tokens, tool_calls identically."""

    def test_async_arun_has_cost_tokens_tool_calls(self) -> None:
        """arun() returns Response with cost, tokens, tool_calls populated (same as sync)."""
        import asyncio

        model = Model("anthropic/claude-3-5-sonnet")
        agent = Agent(model=model, system_prompt="Test.")
        mock_resp = _mock_provider_response(
            content="Async reply",
            input_tokens=7,
            output_tokens=14,
        )
        with patch.object(
            agent._provider,
            "complete",
            new_callable=AsyncMock,
            return_value=mock_resp,
        ):

            async def run() -> None:
                r = await agent.arun("Hello")
                assert isinstance(r.cost, (int, float))
                assert r.cost >= 0
                assert r.tokens is not None
                assert r.tokens.input_tokens == 7
                assert r.tokens.output_tokens == 14
                assert r.tokens.total_tokens == 21
                assert r.tool_calls is not None
                assert isinstance(r.tool_calls, list)
                assert r.stop_reason is not None

            asyncio.run(run())

    def test_sync_and_async_same_shape_for_single_turn(self) -> None:
        """response() and arun() produce same field types and presence for same mock."""
        import asyncio

        model = Model("anthropic/claude-3-5-sonnet")
        agent = Agent(model=model, system_prompt="Test.")
        mock_resp = _mock_provider_response(
            content="Same",
            input_tokens=3,
            output_tokens=6,
        )
        with patch.object(
            agent._provider,
            "complete",
            new_callable=AsyncMock,
            return_value=mock_resp,
        ):
            r_sync = agent.response("Hi")
            async_run = agent.arun("Hi")
            r_async = asyncio.run(async_run)
        for r in (r_sync, r_async):
            assert hasattr(r, "cost") and isinstance(r.cost, (int, float))
            assert hasattr(r, "tokens") and r.tokens is not None
            assert hasattr(r, "tool_calls") and isinstance(r.tool_calls, list)
            assert r.tokens.input_tokens == 3
            assert r.tokens.output_tokens == 6
            assert r.tokens.total_tokens == 9


class TestResponseToolCallsPopulated:
    """When loop returns tool_calls, Response.tool_calls is populated."""

    def test_response_tool_calls_from_loop_result(self) -> None:
        """Multi-turn loop that returns tool_calls: response.tool_calls has correct shape."""
        from syrin.tool import tool
        from syrin.types import ToolCall

        @tool
        def search(q: str) -> str:
            """Search."""
            return "results"

        model = Model("anthropic/claude-3-5-sonnet")
        agent = Agent(model=model, system_prompt="Test.", tools=[search])
        # First call returns tool_calls, second returns content (loop ends)
        first_resp = ProviderResponse(
            content="",
            tool_calls=[
                ToolCall(id="tc-1", name="search", arguments={"q": "x"}),
            ],
            token_usage=TokenUsage(
                input_tokens=10,
                output_tokens=20,
                total_tokens=30,
            ),
        )
        second_resp = _mock_provider_response(
            content="Done",
            input_tokens=10,
            output_tokens=5,
        )
        call_count = 0

        async def mock_complete(*args: object, **kwargs: object) -> ProviderResponse:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return first_resp
            return second_resp

        with patch.object(
            agent._provider,
            "complete",
            new_callable=AsyncMock,
            side_effect=mock_complete,
        ):
            r = agent.response("Run search")
        assert r.tool_calls is not None
        assert isinstance(r.tool_calls, list)
        assert len(r.tool_calls) >= 1
        assert r.cost >= 0
        assert r.tokens.total_tokens >= 0
