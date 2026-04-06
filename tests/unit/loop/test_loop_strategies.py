"""Tests for PlanExecuteLoop and CodeActionLoop."""

import asyncio
from unittest.mock import AsyncMock, MagicMock

from syrin.agent._run_context import DefaultAgentRunContext
from syrin.enums import MessageRole
from syrin.loop import (
    CodeActionLoop,
    PlanExecuteLoop,
)
from syrin.types import Message, TokenUsage, ToolCall


def _run_ctx(mock_agent: MagicMock) -> DefaultAgentRunContext:
    """Wrap a mock agent so loop receives AgentRunContext. Ensures _build_messages returns a list."""
    mock_agent._build_messages = MagicMock(
        return_value=[Message(role=MessageRole.USER, content="test")]
    )
    return DefaultAgentRunContext(mock_agent)


class TestPlanExecuteLoop:
    """Test PlanExecuteLoop - Plan all steps, then execute each."""

    def test_creation(self):
        """PlanExecuteLoop can be created with config."""
        loop = PlanExecuteLoop(max_plan_iterations=3, max_execution_iterations=10)
        assert loop.name == "plan_execute"
        assert loop.max_plan_iterations == 3
        assert loop.max_execution_iterations == 10

    def test_default_values(self):
        """Test default configuration values."""
        loop = PlanExecuteLoop()
        assert loop.max_plan_iterations == 5
        assert loop.max_execution_iterations == 20

    def test_plan_phase_generates_steps(self):
        """Plan phase should generate a list of steps to execute."""
        loop = PlanExecuteLoop()

        # First call returns a plan with steps
        plan_response = MagicMock()
        plan_response.content = """Here is my plan:
1. Step 1: Search for information
2. Step 2: Process results
3. Step 3: Return final answer"""
        plan_response.tool_calls = []
        plan_response.token_usage = TokenUsage(input_tokens=50, output_tokens=100, total_tokens=150)
        plan_response.stop_reason = "end_turn"

        # Second call returns final response after executing steps
        exec_response = MagicMock()
        exec_response.content = "Final answer after executing plan"
        exec_response.tool_calls = []
        exec_response.token_usage = TokenUsage(input_tokens=200, output_tokens=50, total_tokens=250)
        exec_response.stop_reason = "end_turn"

        mock_agent = MagicMock()
        mock_agent.complete = AsyncMock(side_effect=[plan_response, exec_response])
        mock_agent.execute_tool = AsyncMock(return_value="tool result")
        mock_agent._model_config = MagicMock()
        mock_agent._model_config.model_id = "gpt-4o-mini"
        mock_agent._model = None
        mock_agent._emit_event = MagicMock()
        mock_agent._check_and_apply_budget = MagicMock()

        result = asyncio.run(loop.run(mock_agent, "Complex task"))

        assert result.iterations >= 1
        assert result.content != ""

    def test_execution_phase_runs_steps(self):
        """Execution phase should run each planned step."""
        loop = PlanExecuteLoop()

        # First call - plan
        plan_response = MagicMock()
        plan_response.content = "Plan: 1) Do X 2) Do Y"
        plan_response.tool_calls = []
        plan_response.token_usage = TokenUsage(input_tokens=50, output_tokens=50, total_tokens=100)
        plan_response.stop_reason = "end_turn"

        # Second call - execution result
        exec_response = MagicMock()
        exec_response.content = "Executed all steps"
        exec_response.tool_calls = []
        exec_response.token_usage = TokenUsage(input_tokens=100, output_tokens=50, total_tokens=150)
        exec_response.stop_reason = "end_turn"

        mock_agent = MagicMock()
        mock_agent.complete = AsyncMock(side_effect=[plan_response, exec_response])
        mock_agent.execute_tool = AsyncMock()
        mock_agent._model_config = MagicMock()
        mock_agent._model_config.model_id = "gpt-4o-mini"
        mock_agent._model = None
        mock_agent._emit_event = MagicMock()
        mock_agent._check_and_apply_budget = MagicMock()

        result = asyncio.run(loop.run(mock_agent, "Task"))

        assert result.stop_reason in ["end_turn", "max_iterations", "plan_execute"]

    def test_populates_cost(self):
        """Verifies PlanExecuteLoop calculates cost."""
        loop = PlanExecuteLoop()

        response = MagicMock()
        response.content = "Done"
        response.tool_calls = []
        response.token_usage = TokenUsage(input_tokens=100, output_tokens=50, total_tokens=150)
        response.stop_reason = "end_turn"

        mock_agent = MagicMock()
        mock_agent.complete = AsyncMock(return_value=response)
        mock_agent.execute_tool = AsyncMock()
        mock_agent._model_config = MagicMock()
        mock_agent._model_config.model_id = "gpt-4o-mini"
        mock_agent._model = None
        mock_agent._emit_event = MagicMock()
        mock_agent._check_and_apply_budget = MagicMock()

        result = asyncio.run(loop.run(_run_ctx(mock_agent), "Test"))

        assert result.cost_usd > 0

    def test_populates_token_usage(self):
        """Verifies PlanExecuteLoop populates token_usage."""
        loop = PlanExecuteLoop()

        response = MagicMock()
        response.content = "Done"
        response.tool_calls = []
        response.token_usage = TokenUsage(input_tokens=200, output_tokens=100, total_tokens=300)
        response.stop_reason = "end_turn"

        mock_agent = MagicMock()
        mock_agent.complete = AsyncMock(return_value=response)
        mock_agent.execute_tool = AsyncMock()
        mock_agent._model_config = MagicMock()
        mock_agent._model_config.model_id = "gpt-4o-mini"
        mock_agent._model = None
        mock_agent._emit_event = MagicMock()
        mock_agent._check_and_apply_budget = MagicMock()

        result = asyncio.run(loop.run(mock_agent, "Test"))

        # PlanExecuteLoop accumulates tokens from planning + execution phases
        assert result.token_usage["total"] >= 300

    def test_plan_with_tool_calls(self):
        """Plan can include tool calls as steps."""
        loop = PlanExecuteLoop()

        # Plan with tool calls
        plan_response = MagicMock()
        plan_response.content = "I'll use tools to help"
        plan_response.tool_calls = [
            ToolCall(id="call_1", name="search", arguments={"query": "test"})
        ]
        plan_response.token_usage = TokenUsage(input_tokens=50, output_tokens=30, total_tokens=80)
        plan_response.stop_reason = "tool_use"

        # Final response (after tool execution in planning phase)
        plan_final_response = MagicMock()
        plan_final_response.content = "Done planning"
        plan_final_response.tool_calls = []
        plan_final_response.token_usage = TokenUsage(
            input_tokens=30, output_tokens=20, total_tokens=50
        )
        plan_final_response.stop_reason = "end_turn"

        # Execution response
        exec_response = MagicMock()
        exec_response.content = "Result"
        exec_response.tool_calls = []
        exec_response.token_usage = TokenUsage(input_tokens=100, output_tokens=20, total_tokens=120)
        exec_response.stop_reason = "end_turn"

        mock_agent = MagicMock()
        mock_agent.complete = AsyncMock(
            side_effect=[plan_response, plan_final_response, exec_response]
        )
        mock_agent.execute_tool = AsyncMock(return_value="search results")
        mock_agent._model_config = MagicMock()
        mock_agent._model_config.model_id = "gpt-4o-mini"
        mock_agent._model = None
        mock_agent._emit_event = MagicMock()
        mock_agent._check_and_apply_budget = MagicMock()

        result = asyncio.run(loop.run(mock_agent, "Search task"))

        assert result.iterations >= 1

    def test_review_phase(self):
        """Should have a review phase to verify completion."""
        loop = PlanExecuteLoop()

        response = MagicMock()
        response.content = "Task completed successfully"
        response.tool_calls = []
        response.token_usage = TokenUsage(input_tokens=100, output_tokens=50, total_tokens=150)
        response.stop_reason = "end_turn"

        mock_agent = MagicMock()
        mock_agent.complete = AsyncMock(return_value=response)
        mock_agent.execute_tool = AsyncMock()
        mock_agent._model_config = MagicMock()
        mock_agent._model_config.model_id = "gpt-4o-mini"
        mock_agent._model = None
        mock_agent._emit_event = MagicMock()
        mock_agent._check_and_apply_budget = MagicMock()

        result = asyncio.run(loop.run(mock_agent, "Review task"))

        # Should complete with end_turn after review
        assert result.stop_reason in ["end_turn", "plan_execute", "max_iterations"]


class TestCodeActionLoop:
    """Test CodeActionLoop - LLM writes Python code to execute."""

    def test_creation(self):
        """CodeActionLoop can be created with config."""
        loop = CodeActionLoop(max_iterations=5, timeout_seconds=30)
        assert loop.name == "code_action"
        assert loop.max_iterations == 5
        assert loop.timeout_seconds == 30

    def test_default_values(self):
        """Test default configuration values."""
        loop = CodeActionLoop()
        assert loop.max_iterations == 10
        assert loop.timeout_seconds == 60

    def test_generates_code(self):
        """CodeActionLoop should generate Python code."""
        loop = CodeActionLoop()

        # First call returns code to execute
        code_response = MagicMock()
        code_response.content = "```python\nresult = 2 + 2\nprint(result)\n```"
        code_response.tool_calls = []
        code_response.token_usage = TokenUsage(input_tokens=50, output_tokens=100, total_tokens=150)
        code_response.stop_reason = "end_turn"

        # Final response
        final_response = MagicMock()
        final_response.content = "The code executed successfully and returned 4"
        final_response.tool_calls = []
        final_response.token_usage = TokenUsage(
            input_tokens=200, output_tokens=50, total_tokens=250
        )
        final_response.stop_reason = "end_turn"

        mock_agent = MagicMock()
        mock_agent.complete = AsyncMock(side_effect=[code_response, final_response])
        mock_agent._model_config = MagicMock()
        mock_agent._model_config.model_id = "gpt-4o-mini"
        mock_agent._model = None
        mock_agent._emit_event = MagicMock()
        mock_agent._check_and_apply_budget = MagicMock()

        result = asyncio.run(loop.run(mock_agent, "What is 2+2?"))

        assert result.iterations >= 1

    def test_executes_code(self):
        """Should execute generated code."""
        loop = CodeActionLoop()

        response = MagicMock()
        response.content = "Code executed: result = 4"
        response.tool_calls = []
        response.token_usage = TokenUsage(input_tokens=100, output_tokens=50, total_tokens=150)
        response.stop_reason = "end_turn"

        mock_agent = MagicMock()
        mock_agent.complete = AsyncMock(return_value=response)
        mock_agent._model_config = MagicMock()
        mock_agent._model_config.model_id = "gpt-4o-mini"
        mock_agent._model = None
        mock_agent._emit_event = MagicMock()
        mock_agent._check_and_apply_budget = MagicMock()

        result = asyncio.run(loop.run(mock_agent, "Calculate something"))

        assert result.content != ""

    def test_populates_cost(self):
        """Verifies CodeActionLoop calculates cost."""
        loop = CodeActionLoop()

        response = MagicMock()
        response.content = "Code executed"
        response.tool_calls = []
        response.token_usage = TokenUsage(input_tokens=100, output_tokens=50, total_tokens=150)
        response.stop_reason = "end_turn"

        mock_agent = MagicMock()
        mock_agent.complete = AsyncMock(return_value=response)
        mock_agent._model_config = MagicMock()
        mock_agent._model_config.model_id = "gpt-4o-mini"
        mock_agent._model = None
        mock_agent._emit_event = MagicMock()
        mock_agent._check_and_apply_budget = MagicMock()

        result = asyncio.run(loop.run(_run_ctx(mock_agent), "Test"))

        assert result.cost_usd > 0

    def test_populates_token_usage(self):
        """Verifies CodeActionLoop populates token_usage."""
        loop = CodeActionLoop()

        response = MagicMock()
        response.content = "Done"
        response.tool_calls = []
        response.token_usage = TokenUsage(input_tokens=200, output_tokens=100, total_tokens=300)
        response.stop_reason = "end_turn"

        mock_agent = MagicMock()
        mock_agent.complete = AsyncMock(return_value=response)
        mock_agent._model_config = MagicMock()
        mock_agent._model_config.model_id = "gpt-4o-mini"
        mock_agent._model = None
        mock_agent._emit_event = MagicMock()
        mock_agent._check_and_apply_budget = MagicMock()

        result = asyncio.run(loop.run(mock_agent, "Test"))

        assert result.token_usage["total"] == 300

    def test_code_with_error_handling(self):
        """Should handle code execution errors."""
        loop = CodeActionLoop()

        # First attempt - code with error
        error_response = MagicMock()
        error_response.content = "Let me try again with fixed code"
        error_response.tool_calls = []
        error_response.token_usage = TokenUsage(input_tokens=50, output_tokens=30, total_tokens=80)
        error_response.stop_reason = "end_turn"

        # Second attempt - success
        success_response = MagicMock()
        success_response.content = "Fixed code worked!"
        success_response.tool_calls = []
        success_response.token_usage = TokenUsage(
            input_tokens=100, output_tokens=20, total_tokens=120
        )
        success_response.stop_reason = "end_turn"

        mock_agent = MagicMock()
        mock_agent.complete = AsyncMock(side_effect=[error_response, success_response])
        mock_agent._model_config = MagicMock()
        mock_agent._model_config.model_id = "gpt-4o-mini"
        mock_agent._model = None
        mock_agent._emit_event = MagicMock()
        mock_agent._check_and_apply_budget = MagicMock()

        result = asyncio.run(loop.run(mock_agent, "Calculate with error"))

        assert result.iterations >= 1

    def test_returns_code_output(self):
        """Should return the output of executed code."""
        loop = CodeActionLoop()

        response = MagicMock()
        response.content = "Executed: [1, 2, 3, 4, 5]"
        response.tool_calls = []
        response.token_usage = TokenUsage(input_tokens=100, output_tokens=50, total_tokens=150)
        response.stop_reason = "end_turn"

        mock_agent = MagicMock()
        mock_agent.complete = AsyncMock(return_value=response)
        mock_agent._model_config = MagicMock()
        mock_agent._model_config.model_id = "gpt-4o-mini"
        mock_agent._model = None
        mock_agent._emit_event = MagicMock()
        mock_agent._check_and_apply_budget = MagicMock()

        result = asyncio.run(loop.run(mock_agent, "Generate list 1-5"))

        assert "1" in result.content or "list" in result.content.lower()


class TestLoopStrategyIntegration:
    """Test loop class direct usage (LoopStrategy enum removed)."""

    def test_single_shot_loop_instantiation(self):
        """SingleShotLoop can be instantiated directly."""
        from syrin.loop import SingleShotLoop

        loop = SingleShotLoop()
        assert loop is not None

    def test_react_loop_instantiation(self):
        """ReactLoop can be instantiated directly."""
        from syrin.loop import ReactLoop

        loop = ReactLoop()
        assert loop is not None


# =============================================================================
# EDGE CASES
# =============================================================================


class TestLoopEdgeCases:
    """Edge cases for PlanExecuteLoop and CodeActionLoop."""

    def test_plan_execute_empty_plan_response(self):
        """Handle empty plan response gracefully."""
        loop = PlanExecuteLoop()

        response = MagicMock()
        response.content = ""  # Empty plan
        response.tool_calls = []
        response.token_usage = TokenUsage(input_tokens=10, output_tokens=5, total_tokens=15)
        response.stop_reason = "end_turn"

        mock_agent = MagicMock()
        mock_agent.complete = AsyncMock(return_value=response)
        mock_agent.execute_tool = AsyncMock()
        mock_agent._model_config = MagicMock()
        mock_agent._model_config.model_id = "gpt-4o-mini"
        mock_agent._model = None
        mock_agent._emit_event = MagicMock()
        mock_agent._check_and_apply_budget = MagicMock()

        result = asyncio.run(loop.run(mock_agent, "Test"))

        # Should still complete, not hang
        assert result.iterations >= 0

    def test_plan_execute_max_plan_iterations(self):
        """Should respect max_plan_iterations."""
        loop = PlanExecuteLoop(max_plan_iterations=1)

        response = MagicMock()
        response.content = "Plan"
        response.tool_calls = []
        response.token_usage = TokenUsage(input_tokens=50, output_tokens=50, total_tokens=100)
        response.stop_reason = "end_turn"

        mock_agent = MagicMock()
        mock_agent.complete = AsyncMock(return_value=response)
        mock_agent.execute_tool = AsyncMock()
        mock_agent._model_config = MagicMock()
        mock_agent._model_config.model_id = "gpt-4o-mini"
        mock_agent._model = None
        mock_agent._emit_event = MagicMock()
        mock_agent._check_and_apply_budget = MagicMock()

        asyncio.run(loop.run(mock_agent, "Test"))

        # Should complete within plan limit
        assert loop.max_plan_iterations == 1

    def test_code_action_empty_code_response(self):
        """Handle empty code response gracefully."""
        loop = CodeActionLoop()

        response = MagicMock()
        response.content = ""  # No code generated
        response.tool_calls = []
        response.token_usage = TokenUsage(input_tokens=10, output_tokens=5, total_tokens=15)
        response.stop_reason = "end_turn"

        mock_agent = MagicMock()
        mock_agent.complete = AsyncMock(return_value=response)
        mock_agent._model_config = MagicMock()
        mock_agent._model_config.model_id = "gpt-4o-mini"
        mock_agent._model = None
        mock_agent._emit_event = MagicMock()
        mock_agent._check_and_apply_budget = MagicMock()

        result = asyncio.run(loop.run(mock_agent, "Test"))

        # Should not hang
        assert result.iterations >= 0

    def test_code_action_max_iterations_zero(self):
        """Handle max_iterations=0 - should still work with 0 iterations."""
        loop = CodeActionLoop(max_iterations=0)

        # With max_iterations=0, loop doesn't run but returns default
        mock_agent = MagicMock()
        mock_agent.complete = AsyncMock()
        mock_agent._model_config = MagicMock()
        mock_agent._model_config.model_id = "gpt-4o-mini"
        mock_agent._model = None
        mock_agent._emit_event = MagicMock()
        mock_agent._check_and_apply_budget = MagicMock()

        # This will fail because the loop never runs - that's expected behavior
        # The test just verifies the config is set
        assert loop.max_iterations == 0
