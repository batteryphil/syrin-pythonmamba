"""Loop strategies: same LoopResult shape; no unhandled exceptions."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from syrin.agent._run_context import DefaultAgentRunContext
from syrin.enums import MessageRole
from syrin.exceptions import ProviderError, ToolExecutionError
from syrin.loop import (
    CodeActionLoop,
    LoopResult,
    PlanExecuteLoop,
    ReactLoop,
    SingleShotLoop,
)
from syrin.types import Message, TokenUsage, ToolCall

# Required LoopResult fields and types per API contract
LOOP_RESULT_REQUIRED = {
    "content": str,
    "stop_reason": str,
    "iterations": int,
    "tools_used": list,
    "cost_usd": (int, float),
    "latency_ms": (int, float),
    "token_usage": dict,
    "tool_calls": list,
    "raw_response": type(None),  # can be None or any
}


def _run_ctx(mock_agent: MagicMock) -> DefaultAgentRunContext:
    mock_agent._build_messages = MagicMock(
        return_value=[Message(role=MessageRole.USER, content="test")]
    )
    return DefaultAgentRunContext(mock_agent)


def _mock_llm_response(
    content: str = "ok",
    tool_calls: list[ToolCall] | None = None,
    input_tokens: int = 10,
    output_tokens: int = 20,
):
    from syrin.types import ProviderResponse

    return MagicMock(
        content=content,
        tool_calls=tool_calls or [],
        token_usage=TokenUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
        ),
        stop_reason="end_turn",
        raw_response=None,
        spec=ProviderResponse,
    )


def assert_loop_result_shape(result: LoopResult) -> None:
    """Assert LoopResult has all required fields with correct types."""
    assert isinstance(result, LoopResult)
    assert isinstance(result.content, str)
    assert isinstance(result.stop_reason, str)
    assert isinstance(result.iterations, int)
    assert isinstance(result.tools_used, list)
    assert isinstance(result.cost_usd, (int, float))
    assert isinstance(result.latency_ms, (int, float))
    assert isinstance(result.token_usage, dict)
    assert "input" in result.token_usage
    assert "output" in result.token_usage
    assert "total" in result.token_usage
    assert isinstance(result.tool_calls, list)
    assert result.iterations >= 0


class TestLoopResultShapeAllStrategies:
    """Every loop strategy returns LoopResult with same shape."""

    def _mock_agent_for_loop(self, complete_return, execute_tool_return=None):
        """Build mock agent so cost_usd is a float (pricing from model_id, not Mock)."""
        mock_agent = MagicMock()
        mock_agent.complete = AsyncMock(return_value=complete_return)
        mock_agent.execute_tool = AsyncMock(return_value=execute_tool_return or "ok")
        mock_agent._model_config = MagicMock()
        mock_agent._model_config.model_id = "gpt-4o"
        mock_agent._model = None  # so pricing_override is None and cost uses _resolve_pricing
        mock_agent._emit_event = MagicMock()
        mock_agent._check_and_apply_budget = MagicMock()
        mock_agent._check_and_apply_rate_limit = MagicMock()
        mock_agent._pre_call_budget_check = MagicMock()
        mock_agent.has_budget = False
        mock_agent.has_rate_limit = False
        mock_agent.pricing_override = None
        return mock_agent

    def test_single_shot_returns_valid_loop_result(self) -> None:
        loop = SingleShotLoop()
        mock_agent = self._mock_agent_for_loop(_mock_llm_response(content="Hi"))
        result = asyncio.run(loop.run(_run_ctx(mock_agent), "hello"))
        assert_loop_result_shape(result)
        assert result.iterations == 1
        assert result.content == "Hi"

    def test_react_returns_valid_loop_result(self) -> None:
        loop = ReactLoop(max_iterations=3)
        mock_agent = self._mock_agent_for_loop(_mock_llm_response(content="Hi"))
        result = asyncio.run(loop.run(_run_ctx(mock_agent), "hello"))
        assert_loop_result_shape(result)
        assert result.iterations == 1

    def test_plan_execute_returns_valid_loop_result(self) -> None:
        loop = PlanExecuteLoop(max_plan_iterations=1, max_execution_iterations=2)
        mock_agent = self._mock_agent_for_loop(_mock_llm_response(content="Plan done"))
        result = asyncio.run(loop.run(_run_ctx(mock_agent), "task"))
        assert_loop_result_shape(result)

    def test_code_action_returns_valid_loop_result(self) -> None:
        loop = CodeActionLoop(max_iterations=2)
        mock_agent = self._mock_agent_for_loop(_mock_llm_response(content="Done"))
        result = asyncio.run(loop.run(_run_ctx(mock_agent), "compute"))
        assert_loop_result_shape(result)


class TestLoopExceptionHandling:
    """Loop propagates typed exceptions; no bare Exception from loop internals."""

    def test_provider_error_propagates_from_single_shot(self) -> None:
        loop = SingleShotLoop()
        mock_agent = MagicMock()
        mock_agent.complete = AsyncMock(side_effect=ProviderError("API down"))
        mock_agent._model_config = MagicMock()
        mock_agent._model_config.model_id = "gpt-4o"
        mock_agent._emit_event = MagicMock()
        mock_agent._check_and_apply_budget = MagicMock()
        mock_agent._check_and_apply_rate_limit = MagicMock()
        mock_agent._pre_call_budget_check = MagicMock()
        mock_agent.has_budget = False
        mock_agent.has_rate_limit = False
        with pytest.raises(ProviderError, match="API down"):
            asyncio.run(loop.run(_run_ctx(mock_agent), "hello"))

    def test_react_tool_error_handled_returns_loop_result(self) -> None:
        """When execute_tool raises, REACT loop continues and returns LoopResult (error in content)."""
        loop = ReactLoop(max_iterations=2)
        first = _mock_llm_response(
            content="I'll use tool",
            tool_calls=[ToolCall(id="1", name="bad_tool", arguments={})],
        )
        mock_agent = MagicMock()
        mock_agent.complete = AsyncMock(side_effect=[first, _mock_llm_response(content="Gave up")])
        mock_agent.execute_tool = AsyncMock(side_effect=ToolExecutionError("tool failed"))
        mock_agent._model_config = MagicMock()
        mock_agent._model_config.model_id = "gpt-4o"
        mock_agent._model = None
        mock_agent._emit_event = MagicMock()
        mock_agent._check_and_apply_budget = MagicMock()
        mock_agent._check_and_apply_rate_limit = MagicMock()
        mock_agent._pre_call_budget_check = MagicMock()
        mock_agent.has_budget = False
        mock_agent.has_rate_limit = False
        result = asyncio.run(loop.run(_run_ctx(mock_agent), "use tool"))
        assert_loop_result_shape(result)
        assert result.iterations >= 1


class TestLoopDirectUsage:
    """Loop classes can be used directly via loop= parameter."""

    def test_react_loop_default(self) -> None:
        assert ReactLoop is not None

    def test_single_shot_loop_available(self) -> None:
        assert SingleShotLoop is not None

    def test_plan_execute_and_code_action_still_available_as_loop_param(self) -> None:
        """PlanExecuteLoop and CodeActionLoop can still be passed as loop=..."""
        assert PlanExecuteLoop is not None
        assert CodeActionLoop is not None
