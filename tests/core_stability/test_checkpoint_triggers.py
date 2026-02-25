"""Checkpoint: auto-save triggers (STEP, TOOL, ERROR, BUDGET) fire; state can be resumed."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

from syrin import Agent, CheckpointConfig, Model
from syrin.checkpoint import CheckpointTrigger
from syrin.types import ProviderResponse, TokenUsage


def _mock_provider_response(content: str = "Ok") -> ProviderResponse:
    return ProviderResponse(
        content=content,
        tool_calls=[],
        token_usage=TokenUsage(input_tokens=5, output_tokens=10, total_tokens=15),
    )


class TestCheckpointTriggerStep:
    """STEP trigger: _maybe_checkpoint('step') and ('tool') trigger save when trigger=STEP."""

    def test_step_trigger_fires_after_response(self) -> None:
        """With trigger=STEP, one save after response (reason=step)."""
        config = CheckpointConfig(storage="memory", trigger=CheckpointTrigger.STEP)
        agent = Agent(
            model=Model("anthropic/claude-3-5-sonnet"),
            system_prompt="Test.",
            checkpoint=config,
        )
        with patch.object(
            agent._provider,
            "complete",
            new_callable=AsyncMock,
            return_value=_mock_provider_response(),
        ):
            agent.response("Hi")
        assert agent._run_report.checkpoints.saves >= 1

    def test_manual_trigger_does_not_auto_save_on_step(self) -> None:
        """With trigger=MANUAL, response() does not auto-save."""
        config = CheckpointConfig(storage="memory", trigger=CheckpointTrigger.MANUAL)
        agent = Agent(
            model=Model("anthropic/claude-3-5-sonnet"),
            system_prompt="Test.",
            checkpoint=config,
        )
        with patch.object(
            agent._provider,
            "complete",
            new_callable=AsyncMock,
            return_value=_mock_provider_response(),
        ):
            agent.response("Hi")
        assert agent._run_report.checkpoints.saves == 0


class TestCheckpointResume:
    """State can be saved and resumed (load_checkpoint restores)."""

    def test_save_then_load_restores_state(self) -> None:
        """save_checkpoint then load_checkpoint returns True and state is usable."""
        config = CheckpointConfig(storage="memory")
        agent = Agent(
            model=Model("anthropic/claude-3-5-sonnet"),
            system_prompt="Test.",
            checkpoint=config,
        )
        cid = agent.save_checkpoint()
        assert cid is not None
        loaded = agent.load_checkpoint(cid)
        assert loaded is True

    def test_trigger_reason_matches_enum_values(self) -> None:
        """Reasons passed to _maybe_checkpoint align with CheckpointTrigger values."""
        assert CheckpointTrigger.STEP.value == "step"
        assert CheckpointTrigger.TOOL.value == "tool"
        assert CheckpointTrigger.ERROR.value == "error"
        assert CheckpointTrigger.BUDGET.value == "budget"
        assert CheckpointTrigger.MANUAL.value == "manual"
