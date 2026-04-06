"""Tests for Response contract, Error contract, and config validation.

Ensures Response shape, all public exceptions extend SyrinError, and
Budget/CheckpointConfig/Context accept valid inputs and reject invalid ones.
"""

from __future__ import annotations

import pytest

from syrin.budget import Budget
from syrin.checkpoint import CheckpointConfig, CheckpointTrigger
from syrin.context import Context
from syrin.enums import ExceedPolicy, StopReason
from syrin.exceptions import (
    BudgetExceededError,
    BudgetThresholdError,
    CodegenError,
    ModelNotFoundError,
    ProviderError,
    ProviderNotFoundError,
    SyrinError,
    TaskError,
    ToolExecutionError,
    ValidationError,
)
from syrin.response import AgentReport, BudgetStatus, Response
from syrin.threshold import BudgetThreshold

# -----------------------------------------------------------------------------
# Response contract
# -----------------------------------------------------------------------------


RESPONSE_CONTRACT_FIELDS = [
    "content",
    "raw",
    "cost",
    "tokens",
    "model",
    "duration",
    "budget_remaining",
    "budget_used",
    "trace",
    "tool_calls",
    "stop_reason",
    "structured",
    "iterations",
    "report",
    "context_stats",
    "context",
    "blocked",
    "block_reason",
    "output",
]


def test_response_has_all_contract_fields() -> None:
    """Response must have all required contract fields."""
    r = Response(content="test")
    for field in RESPONSE_CONTRACT_FIELDS:
        assert hasattr(r, field), f"Response missing contract field: {field}"


def test_response_budget_property_returns_budget_status() -> None:
    """response.budget must return BudgetStatus."""
    r = Response(content="x", budget_remaining=10.0, budget_used=0.5)
    b = r.budget
    assert isinstance(b, BudgetStatus)
    assert b.remaining == 10.0
    assert b.used == 0.5
    assert b.cost == 0.5 or b.cost == r.cost


def test_response_output_property_when_structured_none() -> None:
    """response.output returns content string when structured is None (plain text agent)."""
    r = Response(content="x")
    assert r.output == "x"


def test_response_stop_reason_is_enum() -> None:
    """stop_reason must be StopReason enum."""
    r = Response(content="x", stop_reason=StopReason.END_TURN)
    assert r.stop_reason == StopReason.END_TURN
    r2 = Response(content="x", stop_reason=StopReason.BUDGET)
    assert r2.stop_reason == StopReason.BUDGET


def test_response_report_is_agent_report() -> None:
    """report must be AgentReport."""
    r = Response(content="x")
    assert isinstance(r.report, AgentReport)


# -----------------------------------------------------------------------------
# Error contract: all public exceptions extend SyrinError
# -----------------------------------------------------------------------------


PUBLIC_EXCEPTIONS = [
    SyrinError,
    BudgetExceededError,
    BudgetThresholdError,
    ModelNotFoundError,
    ToolExecutionError,
    TaskError,
    ProviderError,
    ProviderNotFoundError,
    CodegenError,
    ValidationError,
]


@pytest.mark.parametrize("exc_cls", PUBLIC_EXCEPTIONS)
def test_all_public_exceptions_extend_syrin_error(exc_cls: type[Exception]) -> None:
    """Every public exception must extend SyrinError."""
    assert issubclass(exc_cls, SyrinError), f"{exc_cls.__name__} must extend SyrinError"


def test_validation_error_has_attempts_and_last_error() -> None:
    """ValidationError has attempts and last_error."""
    e = ValidationError("invalid", attempts=["err1"], last_error=ValueError("parse"))
    assert e.attempts == ["err1"]
    assert e.last_error is not None
    assert isinstance(e.last_error, ValueError)


# -----------------------------------------------------------------------------
# Budget config: valid and invalid
# -----------------------------------------------------------------------------


def test_budget_valid_max_cost_and_policy() -> None:
    """Budget accepts max_cost=float and exceed_policy=ExceedPolicy."""
    b = Budget(max_cost=0.5, exceed_policy=ExceedPolicy.STOP)
    assert b.max_cost == 0.5
    assert callable(b._handler)


def test_budget_valid_with_thresholds() -> None:
    """Budget accepts list of BudgetThreshold."""
    b = Budget(
        run=1.0,
        exceed_policy=ExceedPolicy.STOP,
        thresholds=[BudgetThreshold(at=80, action=lambda _ctx: None)],
    )
    assert len(b.thresholds) == 1
    assert b.thresholds[0].at == 80


def test_budget_invalid_threshold_not_budget_threshold_raises() -> None:
    """Budget must only accept BudgetThreshold in thresholds."""
    with pytest.raises(TypeError, match="Budget only accepts BudgetThreshold"):
        Budget(max_cost=1.0, thresholds=[{"at": 80}])  # type: ignore[arg-type]


def test_budget_valid_safety_margin() -> None:
    """Budget accepts safety_margin."""
    b = Budget(max_cost=10.0, safety_margin=1.0)
    assert b.safety_margin == 1.0


# -----------------------------------------------------------------------------
# CheckpointConfig: valid and invalid
# -----------------------------------------------------------------------------


def test_checkpoint_config_valid_storage_values() -> None:
    """CheckpointConfig accepts storage memory, sqlite, postgres, filesystem."""
    for storage in ("memory", "sqlite", "postgres", "filesystem"):
        c = CheckpointConfig(storage=storage)
        assert c.storage == storage


def test_checkpoint_config_trigger_enum() -> None:
    """CheckpointConfig accepts CheckpointTrigger enum."""
    c = CheckpointConfig(storage="memory", trigger=CheckpointTrigger.STEP)
    assert c.trigger == CheckpointTrigger.STEP
    assert c.trigger.value == "step"


def test_checkpoint_config_invalid_storage_raises() -> None:
    """CheckpointConfig rejects invalid storage."""
    with pytest.raises(ValueError):
        CheckpointConfig(storage="invalid_backend")


# -----------------------------------------------------------------------------
# Context config: valid and invalid
# -----------------------------------------------------------------------------


def test_context_valid_max_tokens_and_thresholds() -> None:
    """Context accepts max_tokens and thresholds list."""
    from syrin.threshold import ContextThreshold, compact_if_available

    c = Context(max_tokens=8000, thresholds=[ContextThreshold(at=75, action=compact_if_available)])
    assert c.max_tokens == 8000
    assert len(c.thresholds) == 1


def test_context_invalid_reserve_negative_raises() -> None:
    """Context rejects negative reserve."""
    with pytest.raises(ValueError, match="reserve"):
        Context(max_tokens=8000, reserve=-1)


def test_context_invalid_max_tokens_zero_raises() -> None:
    """Context rejects max_tokens <= 0 when set."""
    with pytest.raises(ValueError, match="max_tokens"):
        Context(max_tokens=0)


def test_context_invalid_thresholds_not_context_threshold_raises() -> None:
    """Context thresholds must be ContextThreshold only."""
    with pytest.raises(ValueError, match="ContextThreshold"):
        Context(max_tokens=8000, thresholds=[BudgetThreshold(at=75, action=lambda _ctx: None)])  # type: ignore[arg-type]
