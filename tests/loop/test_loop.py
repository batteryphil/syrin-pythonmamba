"""Tests for the simplified Loop system."""

import asyncio
from unittest.mock import AsyncMock, MagicMock

from syrin import Agent
from syrin.agent._run_context import DefaultAgentRunContext
from syrin.enums import MessageRole
from syrin.loop import (
    HITL,
    REACT,
    SINGLE_SHOT,
    HumanInTheLoop,
    LoopResult,
    ReactLoop,
    SingleShotLoop,
)
from syrin.types import Message, ModelConfig, TokenUsage, ToolCall


def _run_ctx(mock_agent: MagicMock) -> DefaultAgentRunContext:
    """Wrap a mock agent so loop receives AgentRunContext. Ensures _build_messages returns a list."""
    mock_agent._build_messages = MagicMock(
        return_value=[Message(role=MessageRole.USER, content="test")]
    )
    return DefaultAgentRunContext(mock_agent)


class TestLoopResult:
    """Test LoopResult dataclass."""

    def test_creation(self):
        result = LoopResult(content="test", stop_reason="end_turn", iterations=1)
        assert result.content == "test"
        assert result.stop_reason == "end_turn"
        assert result.iterations == 1

    def test_with_tools(self):
        result = LoopResult(
            content="test",
            stop_reason="end_turn",
            iterations=2,
            tools_used=["search", "calculate"],
        )
        assert result.tools_used == ["search", "calculate"]

    def test_default_fields(self):
        """Test that default fields are properly initialized."""
        result = LoopResult(content="test", stop_reason="end_turn", iterations=1)
        assert result.cost_usd == 0.0
        assert result.latency_ms == 0.0
        assert result.token_usage == {"input": 0, "output": 0, "total": 0}
        assert result.tool_calls == []
        assert result.raw_response is None

    def test_all_fields(self):
        """Test all fields can be set."""
        result = LoopResult(
            content="test",
            stop_reason="end_turn",
            iterations=3,
            tools_used=["tool1", "tool2"],
            cost_usd=0.005,
            latency_ms=1500.0,
            token_usage={"input": 100, "output": 50, "total": 150},
            tool_calls=[{"id": "call_1", "name": "test", "arguments": {}}],
            raw_response={"raw": "data"},
        )
        assert result.cost_usd == 0.005
        assert result.latency_ms == 1500.0
        assert result.token_usage == {"input": 100, "output": 50, "total": 150}
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0]["name"] == "test"
        assert result.raw_response == {"raw": "data"}


class TestSingleShotLoop:
    """Test SingleShotLoop."""

    def test_creation(self):
        loop = SingleShotLoop()
        assert loop.name == "single_shot"

    def test_single_iteration(self):
        """Verifies single shot runs once."""
        loop = SingleShotLoop()

        mock_agent = MagicMock()
        mock_agent.complete = AsyncMock(
            return_value=MagicMock(
                content="response",
                tool_calls=[],
            )
        )
        mock_agent._model_config = MagicMock()
        mock_agent._model_config.model_id = "gpt-4o"
        mock_agent._emit_event = MagicMock()

        result = asyncio.run(loop.run(_run_ctx(mock_agent), "test"))
        assert result.iterations == 1

    def test_populates_cost(self):
        """Verifies SingleShotLoop calculates and populates cost_usd."""
        from syrin.types import TokenUsage

        loop = SingleShotLoop()

        mock_response = MagicMock()
        mock_response.content = "test response"
        mock_response.tool_calls = []
        mock_response.token_usage = TokenUsage(input_tokens=100, output_tokens=50, total_tokens=150)
        mock_response.raw_response = {"raw": "data"}

        mock_agent = MagicMock()
        mock_agent.complete = AsyncMock(return_value=mock_response)
        mock_agent._model_config = MagicMock()
        mock_agent._model_config.model_id = "gpt-4o-mini"
        mock_agent._model = None
        mock_agent._emit_event = MagicMock()

        result = asyncio.run(loop.run(_run_ctx(mock_agent), "test"))

        # Cost should be calculated (gpt-4o-mini pricing: $0.00015/1k input, $0.0006/1k output)
        assert result.cost_usd > 0
        assert isinstance(result.cost_usd, float)

    def test_populates_token_usage(self):
        """Verifies SingleShotLoop populates token_usage field."""
        from syrin.types import TokenUsage

        loop = SingleShotLoop()

        mock_response = MagicMock()
        mock_response.content = "test response"
        mock_response.tool_calls = []
        mock_response.token_usage = TokenUsage(input_tokens=100, output_tokens=50, total_tokens=150)
        mock_response.raw_response = None

        mock_agent = MagicMock()
        mock_agent.complete = AsyncMock(return_value=mock_response)
        mock_agent._model_config = MagicMock()
        mock_agent._model_config.model_id = "gpt-4o-mini"
        mock_agent._model = None
        mock_agent._emit_event = MagicMock()

        result = asyncio.run(loop.run(_run_ctx(mock_agent), "test"))

        assert result.token_usage == {"input": 100, "output": 50, "total": 150}

    def test_populates_tool_calls(self):
        """Verifies SingleShotLoop populates tool_calls when present."""
        from syrin.types import TokenUsage, ToolCall

        loop = SingleShotLoop()

        mock_response = MagicMock()
        mock_response.content = "I'll search for that"
        mock_response.tool_calls = [
            ToolCall(id="call_123", name="web_search", arguments={"query": "AI news"})
        ]
        mock_response.token_usage = TokenUsage(input_tokens=50, output_tokens=30, total_tokens=80)
        mock_response.raw_response = None

        mock_agent = MagicMock()
        mock_agent.complete = AsyncMock(return_value=mock_response)
        mock_agent._model_config = MagicMock()
        mock_agent._model_config.model_id = "gpt-4o-mini"
        mock_agent._model = None
        mock_agent._emit_event = MagicMock()

        result = asyncio.run(loop.run(_run_ctx(mock_agent), "test"))

        assert len(result.tool_calls) == 1
        assert result.tool_calls[0]["id"] == "call_123"
        assert result.tool_calls[0]["name"] == "web_search"
        assert result.tool_calls[0]["arguments"] == {"query": "AI news"}
        assert "web_search" in result.tools_used

    def test_populates_raw_response(self):
        """Verifies SingleShotLoop populates raw_response."""
        from syrin.types import TokenUsage

        loop = SingleShotLoop()

        raw_response_data = {"model": "gpt-4o-mini", "finish_reason": "stop"}
        mock_response = MagicMock()
        mock_response.content = "test"
        mock_response.tool_calls = []
        mock_response.token_usage = TokenUsage()
        mock_response.raw_response = raw_response_data

        mock_agent = MagicMock()
        mock_agent.complete = AsyncMock(return_value=mock_response)
        mock_agent._model_config = MagicMock()
        mock_agent._model_config.model_id = "gpt-4o-mini"
        mock_agent._model = None
        mock_agent._emit_event = MagicMock()

        result = asyncio.run(loop.run(_run_ctx(mock_agent), "test"))

        assert result.raw_response == raw_response_data

    def test_empty_response_handling(self):
        """Edge case: empty response content."""
        from syrin.types import TokenUsage

        loop = SingleShotLoop()

        mock_response = MagicMock()
        mock_response.content = ""
        mock_response.tool_calls = []
        mock_response.token_usage = TokenUsage(input_tokens=0, output_tokens=0, total_tokens=0)
        mock_response.raw_response = None

        mock_agent = MagicMock()
        mock_agent.complete = AsyncMock(return_value=mock_response)
        mock_agent._model_config = MagicMock()
        mock_agent._model_config.model_id = "gpt-4o-mini"
        mock_agent._model = None
        mock_agent._emit_event = MagicMock()

        result = asyncio.run(loop.run(_run_ctx(mock_agent), "test"))

        assert result.content == ""
        assert result.cost_usd == 0.0
        assert result.token_usage == {"input": 0, "output": 0, "total": 0}


class TestReactLoop:
    """Test ReactLoop."""

    def test_creation(self):
        loop = ReactLoop(max_iterations=5)
        assert loop.name == "react"
        assert loop.max_iterations == 5

    def test_default_max_iterations(self):
        loop = ReactLoop()
        assert loop.max_iterations == 10

    def test_populates_cost(self):
        """Verifies ReactLoop calculates and populates cost_usd."""
        from syrin.types import TokenUsage

        loop = ReactLoop(max_iterations=2)

        mock_response = MagicMock()
        mock_response.content = "Final response"
        mock_response.tool_calls = []
        mock_response.token_usage = TokenUsage(input_tokens=100, output_tokens=50, total_tokens=150)
        mock_response.raw_response = {"raw": "data"}

        mock_agent = MagicMock()
        mock_agent.complete = AsyncMock(return_value=mock_response)
        mock_agent._model_config = MagicMock()
        mock_agent._model_config.model_id = "gpt-4o-mini"
        mock_agent._model = None
        mock_agent._emit_event = MagicMock()
        mock_agent._check_and_apply_budget = MagicMock()

        result = asyncio.run(loop.run(_run_ctx(mock_agent), "test"))

        assert result.cost_usd > 0
        assert result.iterations == 1

    def test_populates_token_usage(self):
        """Verifies ReactLoop populates token_usage field."""
        from syrin.types import TokenUsage

        loop = ReactLoop()

        mock_response = MagicMock()
        mock_response.content = "Final response"
        mock_response.tool_calls = []
        mock_response.token_usage = TokenUsage(
            input_tokens=200, output_tokens=100, total_tokens=300
        )
        mock_response.raw_response = None

        mock_agent = MagicMock()
        mock_agent.complete = AsyncMock(return_value=mock_response)
        mock_agent._model_config = MagicMock()
        mock_agent._model_config.model_id = "gpt-4o-mini"
        mock_agent._model = None
        mock_agent._emit_event = MagicMock()
        mock_agent._check_and_apply_budget = MagicMock()

        result = asyncio.run(loop.run(_run_ctx(mock_agent), "test"))

        assert result.token_usage == {"input": 200, "output": 100, "total": 300}

    def test_populates_tool_calls_at_end(self):
        """Verifies ReactLoop populates tool_calls from final response."""
        from syrin.types import TokenUsage, ToolCall

        loop = ReactLoop()

        # First call returns tool calls, second returns final response (no more tool calls)
        call_count = [0]

        def mock_complete(_messages, _tools=None):
            call_count[0] += 1
            if call_count[0] == 1:
                mock_resp = MagicMock()
                mock_resp.content = "I'll search for that"
                mock_resp.tool_calls = [
                    ToolCall(id="call_1", name="search", arguments={"q": "test"})
                ]
                mock_resp.token_usage = TokenUsage(
                    input_tokens=50, output_tokens=20, total_tokens=70
                )
                mock_resp.raw_response = None
                return mock_resp
            else:
                mock_resp = MagicMock()
                mock_resp.content = "Found it"
                mock_resp.tool_calls = []  # Final response has no tool calls
                mock_resp.token_usage = TokenUsage(
                    input_tokens=30, output_tokens=10, total_tokens=40
                )
                mock_resp.raw_response = None
                return mock_resp

        mock_agent = MagicMock()
        mock_agent.complete = AsyncMock(side_effect=mock_complete)
        mock_agent.execute_tool = AsyncMock(return_value="Search results")
        mock_agent._model_config = MagicMock()
        mock_agent._model_config.model_id = "gpt-4o-mini"
        mock_agent._model = None
        mock_agent._emit_event = MagicMock()
        mock_agent._check_and_apply_budget = MagicMock()

        result = asyncio.run(loop.run(_run_ctx(mock_agent), "test"))

        # tools_used should track all tools across iterations
        assert "search" in result.tools_used
        # tool_calls should now be populated (our fix)
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0]["name"] == "search"

    def test_zero_iterations_no_tools(self):
        """Edge case: no tool calls, single iteration."""
        from syrin.types import TokenUsage

        loop = ReactLoop()

        mock_response = MagicMock()
        mock_response.content = "Direct answer"
        mock_response.tool_calls = []
        mock_response.token_usage = TokenUsage(input_tokens=10, output_tokens=5, total_tokens=15)
        mock_response.raw_response = None

        mock_agent = MagicMock()
        mock_agent.complete = AsyncMock(return_value=mock_response)
        mock_agent._model_config = MagicMock()
        mock_agent._model_config.model_id = "gpt-4o-mini"
        mock_agent._model = None
        mock_agent._emit_event = MagicMock()
        mock_agent._check_and_apply_budget = MagicMock()

        result = asyncio.run(loop.run(_run_ctx(mock_agent), "test"))

        assert result.iterations == 1
        assert result.tools_used == []
        assert result.tool_calls == []


class TestHumanInTheLoop:
    """Test HumanInTheLoop."""

    def test_creation_with_callback(self):
        async def approve(_name, _args):
            return True

        loop = HumanInTheLoop(approve=approve, max_iterations=5)
        assert loop.name == "human_in_the_loop"
        assert loop.approve is approve
        assert loop.max_iterations == 5

    def test_creation_without_callback(self):
        loop = HumanInTheLoop()
        assert loop.name == "human_in_the_loop"
        assert loop.approve is None

    def test_approval_called(self):
        """Verifies approval callback is called with tool info."""
        approval_calls = []

        async def approve(name, args):
            approval_calls.append((name, args))
            return True

        loop = HumanInTheLoop(approve=approve)

        async def mock_complete(_messages, _tools=None):
            response = MagicMock()
            if not hasattr(mock_complete, "_called"):
                mock_complete._called = True
                response.content = ""
                response.tool_calls = [
                    ToolCall(id="call_1", name="search", arguments={"query": "test"})
                ]
            else:
                response.tool_calls = []
            return response

        mock_agent = MagicMock()
        mock_agent.complete = mock_complete
        mock_agent.execute_tool = AsyncMock(return_value="results")
        mock_agent._model_config = MagicMock()
        mock_agent._model_config.model_id = "gpt-4o"
        mock_agent._emit_event = MagicMock()
        mock_agent._check_and_apply_budget = MagicMock()

        asyncio.run(loop.run(_run_ctx(mock_agent), "test"))

        assert len(approval_calls) == 1
        assert approval_calls[0][0] == "search"

    def test_rejected_tool_blocks_execution(self):
        """Verifies rejected tools don't execute; approval callback is still invoked."""

        approval_calls = []

        async def approve(name, args):
            approval_calls.append((name, args))
            return False  # Reject all

        loop = HumanInTheLoop(approve=approve)

        mock_agent = MagicMock()
        mock_agent.complete = AsyncMock(
            return_value=MagicMock(
                content="",
                tool_calls=[ToolCall(id="call_1", name="delete", arguments={})],
            )
        )
        mock_agent.execute_tool = AsyncMock()
        mock_agent._model_config = MagicMock()
        mock_agent._model_config.model_id = "gpt-4o"
        mock_agent._emit_event = MagicMock()
        mock_agent._check_and_apply_budget = MagicMock()

        asyncio.run(loop.run(_run_ctx(mock_agent), "test"))

        assert len(approval_calls) >= 1
        assert approval_calls[0][0] == "delete"
        # Tool should NOT have been executed when approval returns False
        mock_agent.execute_tool.assert_not_called()

    def test_populates_cost(self):
        """Verifies HumanInTheLoop calculates and populates cost_usd."""
        from syrin.types import TokenUsage

        async def approve(_name, _args):
            return True

        loop = HumanInTheLoop(approve=approve)

        mock_response = MagicMock()
        mock_response.content = "Final response"
        mock_response.tool_calls = []
        mock_response.token_usage = TokenUsage(input_tokens=100, output_tokens=50, total_tokens=150)
        mock_response.raw_response = {"raw": "data"}

        mock_agent = MagicMock()
        mock_agent.complete = AsyncMock(return_value=mock_response)
        mock_agent._model_config = MagicMock()
        mock_agent._model_config.model_id = "gpt-4o-mini"
        mock_agent._model = None
        mock_agent._emit_event = MagicMock()

        result = asyncio.run(loop.run(_run_ctx(mock_agent), "test"))

        assert result.cost_usd > 0

    def test_populates_token_usage(self):
        """Verifies HumanInTheLoop populates token_usage field."""
        from syrin.types import TokenUsage

        async def approve(_name, _args):
            return True

        loop = HumanInTheLoop(approve=approve)

        mock_response = MagicMock()
        mock_response.content = "Final response"
        mock_response.tool_calls = []
        mock_response.token_usage = TokenUsage(input_tokens=250, output_tokens=75, total_tokens=325)
        mock_response.raw_response = None

        mock_agent = MagicMock()
        mock_agent.complete = AsyncMock(return_value=mock_response)
        mock_agent._model_config = MagicMock()
        mock_agent._model_config.model_id = "gpt-4o-mini"
        mock_agent._model = None
        mock_agent._emit_event = MagicMock()

        result = asyncio.run(loop.run(_run_ctx(mock_agent), "test"))

        assert result.token_usage == {"input": 250, "output": 75, "total": 325}


class TestConstants:
    """Test module constants."""

    def test_react_constant(self):
        assert REACT is ReactLoop

    def test_single_shot_constant(self):
        assert SINGLE_SHOT is SingleShotLoop

    def test_hitl_constant(self):
        assert HITL is HumanInTheLoop


class TestCustomLoop:
    """Test custom loop implementation."""

    def test_simple_class(self):
        """Any class with run() works."""

        class MyLoop:
            name = "mine"

            async def run(self, _agent, _user_input):
                return LoopResult(content="custom", stop_reason="done", iterations=1)

        loop = MyLoop()
        assert loop.name == "mine"

        # Works with Agent - pass the instance
        agent = Agent(
            model=ModelConfig(name="test", provider="openai", model_id="gpt-4o"),
            loop=loop,  # Pass instance
        )
        assert agent._loop.name == "mine"


class TestAgentIntegration:
    """Test Agent with loops."""

    def test_agent_accepts_loop_instance(self):
        agent = Agent(
            model=ModelConfig(name="test", provider="openai", model_id="gpt-4o"),
            loop=ReactLoop(),
        )
        assert agent._loop is not None

    def test_agent_accepts_loop_class(self):
        agent = Agent(
            model=ModelConfig(name="test", provider="openai", model_id="gpt-4o"),
            loop=SingleShotLoop,
        )
        assert agent._loop is not None

    def test_agent_default_loop(self):
        agent = Agent(
            model=ModelConfig(name="test", provider="openai", model_id="gpt-4o"),
        )
        assert agent._loop is not None


class TestAgentRunLoopResponse:
    """Test Agent._run_loop_response properly converts LoopResult to Response."""

    def test_response_cost_populated(self):
        """Verifies Response gets cost from LoopResult."""

        # Create a custom loop that returns known values
        class TestLoop:
            name = "test"

            async def run(self, _agent, _user_input):
                return LoopResult(
                    content="test response",
                    stop_reason="end_turn",
                    iterations=1,
                    cost_usd=0.0015,
                    latency_ms=1000.0,
                    token_usage={"input": 100, "output": 50, "total": 150},
                    tool_calls=[],
                    raw_response=None,
                )

        agent = Agent(
            model=ModelConfig(name="test", provider="openai", model_id="gpt-4o-mini"),
            loop=TestLoop(),
        )

        result = agent.response("test")

        assert result.cost == 0.0015

    def test_response_tokens_populated(self):
        """Verifies Response gets tokens from LoopResult."""

        class TestLoop:
            name = "test"

            async def run(self, _agent, _user_input):
                return LoopResult(
                    content="test response",
                    stop_reason="end_turn",
                    iterations=1,
                    cost_usd=0.001,
                    latency_ms=500.0,
                    token_usage={"input": 200, "output": 100, "total": 300},
                    tool_calls=[],
                    raw_response=None,
                )

        agent = Agent(
            model=ModelConfig(name="test", provider="openai", model_id="gpt-4o-mini"),
            loop=TestLoop(),
        )

        result = agent.response("test")

        assert result.tokens.input_tokens == 200
        assert result.tokens.output_tokens == 100
        assert result.tokens.total_tokens == 300

    def test_response_tool_calls_populated(self):
        """Verifies Response gets tool_calls from LoopResult."""

        class TestLoop:
            name = "test"

            async def run(self, _agent, _user_input):
                return LoopResult(
                    content="test response",
                    stop_reason="end_turn",
                    iterations=1,
                    cost_usd=0.001,
                    latency_ms=500.0,
                    token_usage={"input": 50, "output": 25, "total": 75},
                    tool_calls=[{"id": "call_1", "name": "search", "arguments": {"query": "test"}}],
                    raw_response=None,
                )

        agent = Agent(
            model=ModelConfig(name="test", provider="openai", model_id="gpt-4o-mini"),
            loop=TestLoop(),
        )

        result = agent.response("test")

        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].name == "search"
        assert result.tool_calls[0].arguments == {"query": "test"}

    def test_response_stop_reason_populated(self):
        """Verifies Response gets stop_reason from LoopResult."""

        class TestLoop:
            name = "test"

            async def run(self, _agent, _user_input):
                return LoopResult(
                    content="test response",
                    stop_reason="max_iterations",
                    iterations=1,
                    cost_usd=0.001,
                    latency_ms=500.0,
                    token_usage={"input": 50, "output": 25, "total": 75},
                    tool_calls=[],
                    raw_response=None,
                )

        agent = Agent(
            model=ModelConfig(name="test", provider="openai", model_id="gpt-4o-mini"),
            loop=TestLoop(),
        )

        result = agent.response("test")

        assert result.stop_reason.value == "max_iterations"

    def test_response_duration_populated(self):
        """Verifies Response gets duration from LoopResult."""

        class TestLoop:
            name = "test"

            async def run(self, _agent, _user_input):
                return LoopResult(
                    content="test response",
                    stop_reason="end_turn",
                    iterations=1,
                    cost_usd=0.001,
                    latency_ms=2500.0,
                    token_usage={"input": 50, "output": 25, "total": 75},
                    tool_calls=[],
                    raw_response=None,
                )

        agent = Agent(
            model=ModelConfig(name="test", provider="openai", model_id="gpt-4o-mini"),
            loop=TestLoop(),
        )

        result = agent.response("test")

        assert result.duration == 2.5  # 2500ms = 2.5 seconds


# =============================================================================
# LOOP EDGE CASES - TRY TO BREAK FUNCTIONALITY
# =============================================================================


class TestLoopEdgeCases:
    """Edge cases for loop implementations."""

    def test_react_loop_max_iterations_zero(self):
        """ReactLoop with max_iterations=0 should handle gracefully."""
        loop = ReactLoop(max_iterations=0)
        assert loop.max_iterations == 0

    def test_react_loop_max_iterations_very_high(self):
        """ReactLoop with very high max_iterations."""
        loop = ReactLoop(max_iterations=10000)
        assert loop.max_iterations == 10000

    def test_loop_result_with_null_fields(self):
        """LoopResult with None fields should work."""
        result = LoopResult(
            content=None,
            stop_reason=None,
            iterations=0,
            tool_calls=None,
            latency_ms=None,
            cost_usd=None,
            token_usage=None,
        )
        assert result.content is None
        assert result.stop_reason is None

    def test_loop_result_fields_accessible(self):
        """LoopResult fields should be accessible."""
        result = LoopResult(
            content="test",
            stop_reason="end_turn",
            iterations=1,
            cost_usd=0.001,
            latency_ms=100.0,
            token_usage={"input": 10, "output": 5, "total": 15},
            tool_calls=[{"id": "1", "name": "tool", "arguments": {}}],
        )
        # All fields should be accessible
        assert result.content == "test"
        assert result.iterations == 1
        assert result.cost_usd == 0.001
        assert result.latency_ms == 100.0
        assert len(result.tool_calls) == 1

    def test_human_in_the_loop_no_approval_callback(self):
        """HITL without approval callback should still work."""
        loop = HumanInTheLoop()
        assert loop.approve is None

    def test_single_shot_loop_with_very_long_content(self):
        """SingleShotLoop handles very long content."""
        loop = SingleShotLoop()

        mock_response = MagicMock()
        mock_response.content = "x" * 100000  # 100KB response
        mock_response.tool_calls = []
        mock_response.token_usage = TokenUsage(
            input_tokens=1000, output_tokens=50000, total_tokens=51000
        )
        mock_response.raw_response = None

        mock_agent = MagicMock()
        mock_agent.complete = AsyncMock(return_value=mock_response)
        mock_agent._model_config = MagicMock()
        mock_agent._model_config.model_id = "gpt-4o-mini"
        mock_agent._model = None
        mock_agent._emit_event = MagicMock()

        result = asyncio.run(loop.run(_run_ctx(mock_agent), "test"))
        assert len(result.content) == 100000
        assert result.cost_usd > 0

    def test_react_loop_many_iterations(self):
        """ReactLoop with many iterations."""
        loop = ReactLoop(max_iterations=50)

        mock_response = MagicMock()
        mock_response.content = "Final response"
        mock_response.tool_calls = []
        mock_response.token_usage = TokenUsage(input_tokens=10, output_tokens=5, total_tokens=15)
        mock_response.raw_response = None

        mock_agent = MagicMock()
        mock_agent.complete = AsyncMock(return_value=mock_response)
        mock_agent._model_config = MagicMock()
        mock_agent._model_config.model_id = "gpt-4o-mini"
        mock_agent._model = None
        mock_agent._emit_event = MagicMock()
        mock_agent._check_and_apply_budget = MagicMock()

        result = asyncio.run(loop.run(_run_ctx(mock_agent), "test"))
        assert result.iterations == 1  # Only 1 since no tool calls

    def test_loop_with_custom_stop_reason(self):
        """Custom stop reason values should be accepted."""
        result = LoopResult(
            content="test",
            stop_reason="max_tokens",  # Custom reason
            iterations=1,
        )
        assert result.stop_reason == "max_tokens"

    def test_loop_result_tools_used_tracking(self):
        """tools_used should track unique tools across iterations."""
        result = LoopResult(
            content="done",
            stop_reason="end_turn",
            iterations=5,
            tools_used=["search", "fetch", "search", "calculate"],
        )
        assert len(result.tools_used) == 4
        # Should track all, including duplicates (that's current behavior)
        assert result.tools_used == ["search", "fetch", "search", "calculate"]
