"""Checkpoint: auto-save triggers (STEP, TOOL, ERROR, BUDGET) fire; state can be resumed."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from syrin import Agent, Budget, CheckpointConfig, Model
from syrin.checkpoint import CheckpointTrigger


def _almock_model() -> Model:
    """Almock model with no delay for fast tests."""
    return Model.Almock(latency_min=0, latency_max=0)


class TestCheckpointTriggerStep:
    """STEP trigger: _maybe_checkpoint('step') and ('tool') trigger save when trigger=STEP."""

    def test_step_trigger_fires_after_response(self) -> None:
        """With trigger=STEP, one save after response (reason=step)."""
        config = CheckpointConfig(storage="memory", trigger=CheckpointTrigger.STEP)
        agent = Agent(
            model=_almock_model(),
            system_prompt="Test.",
            checkpoint=config,
        )
        agent.run("Hi")
        assert agent._run_report.checkpoints.saves >= 1

    def test_manual_trigger_does_not_auto_save_on_step(self) -> None:
        """With trigger=MANUAL, response() does not auto-save."""
        config = CheckpointConfig(storage="memory", trigger=CheckpointTrigger.MANUAL)
        agent = Agent(
            model=_almock_model(),
            system_prompt="Test.",
            checkpoint=config,
        )
        agent.run("Hi")
        assert agent._run_report.checkpoints.saves == 0


class TestCheckpointResume:
    """State can be saved and resumed (load_checkpoint restores)."""

    def test_save_then_load_restores_state(self) -> None:
        """save_checkpoint then load_checkpoint returns True and state is usable."""
        config = CheckpointConfig(storage="memory")
        agent = Agent(
            model=_almock_model(),
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


class TestCheckpointTriggerTool:
    """TOOL trigger: checkpoint saved when tool path is taken (with Almock we only get step)."""

    def test_tool_trigger_config(self) -> None:
        """Config with trigger=TOOL is accepted."""
        config = CheckpointConfig(storage="memory", trigger=CheckpointTrigger.TOOL)
        agent = Agent(
            model=_almock_model(),
            system_prompt="Test.",
            checkpoint=config,
        )
        assert agent._checkpoint_config is not None
        assert agent._checkpoint_config.trigger == CheckpointTrigger.TOOL


class TestCheckpointTriggerError:
    """ERROR trigger: checkpoint saved when an exception is raised."""

    def test_error_trigger_fires_on_exception(self) -> None:
        """With trigger=ERROR, run that raises saves checkpoint then re-raises."""
        config = CheckpointConfig(storage="memory", trigger=CheckpointTrigger.ERROR)
        agent = Agent(
            model=_almock_model(),
            system_prompt="Test.",
            checkpoint=config,
        )
        with (
            patch.object(
                agent._provider,
                "complete",
                new_callable=AsyncMock,
                side_effect=RuntimeError("simulated"),
            ),
            pytest.raises(RuntimeError, match="simulated"),
        ):
            agent.run("Hi")
        assert agent._run_report.checkpoints.saves >= 1

    def test_error_trigger_config(self) -> None:
        """Config with trigger=ERROR is accepted."""
        config = CheckpointConfig(storage="memory", trigger=CheckpointTrigger.ERROR)
        agent = Agent(
            model=_almock_model(),
            system_prompt="Test.",
            checkpoint=config,
        )
        assert agent._checkpoint_config.trigger == CheckpointTrigger.ERROR


class TestCheckpointTriggerBudget:
    """BUDGET trigger: checkpoint saved when budget exceeded."""

    def test_budget_trigger_fires_on_budget_exceeded(self) -> None:
        """With trigger=BUDGET, run that exceeds budget saves checkpoint then raises."""
        from syrin.exceptions import BudgetExceededError

        config = CheckpointConfig(storage="memory", trigger=CheckpointTrigger.BUDGET)
        agent = Agent(
            model=_almock_model(),
            system_prompt="Test.",
            budget=Budget(max_cost=0.0001),  # very low so one call exceeds
            checkpoint=config,
        )
        with pytest.raises(BudgetExceededError):
            agent.run("Hi")
        assert agent._run_report.checkpoints.saves >= 1

    def test_budget_trigger_config(self) -> None:
        """Config with trigger=BUDGET is accepted."""
        config = CheckpointConfig(storage="memory", trigger=CheckpointTrigger.BUDGET)
        agent = Agent(
            model=_almock_model(),
            system_prompt="Test.",
            checkpoint=config,
        )
        assert agent._checkpoint_config.trigger == CheckpointTrigger.BUDGET
