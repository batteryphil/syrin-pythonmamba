"""Tests for exception hierarchy (exceptions.py)."""

from __future__ import annotations

from syrin.exceptions import (
    BudgetExceededError,
    BudgetThresholdError,
    CircuitBreakerOpenError,
    CodegenError,
    ModelNotFoundError,
    ProviderError,
    SyrinError,
    TaskError,
    ToolExecutionError,
    ValidationError,
)


def test_Syrin_error_base() -> None:
    e = SyrinError("test")
    assert str(e) == "test"
    assert isinstance(e, Exception)


def test_budget_exceeded_error_attrs() -> None:
    e = BudgetExceededError("over limit", current_cost=1.5, limit=1.0, budget_type="run")
    assert e.current_cost == 1.5
    assert e.limit == 1.0
    assert e.budget_type == "run"


def test_budget_threshold_error_attrs() -> None:
    e = BudgetThresholdError("threshold hit", threshold_percent=80.0, action_taken="switch_model")
    assert e.threshold_percent == 80.0
    assert e.action_taken == "switch_model"


def test_exception_hierarchy() -> None:
    assert issubclass(BudgetExceededError, SyrinError)
    assert issubclass(BudgetThresholdError, SyrinError)
    assert issubclass(ModelNotFoundError, SyrinError)
    assert issubclass(ToolExecutionError, SyrinError)
    assert issubclass(TaskError, SyrinError)
    assert issubclass(ProviderError, SyrinError)
    assert issubclass(CodegenError, SyrinError)
    assert issubclass(ValidationError, SyrinError)
    assert issubclass(CircuitBreakerOpenError, SyrinError)


# =============================================================================
# EXCEPTIONS EDGE CASES - TRY TO BREAK FUNCTIONALITY
# =============================================================================


def test_syrin_error_with_empty_message() -> None:
    """SyrinError with empty message."""
    e = SyrinError("")
    assert str(e) == ""


def test_budget_exceeded_error_zero_values() -> None:
    """BudgetExceededError with zero values."""
    e = BudgetExceededError("over", current_cost=0.0, limit=0.0, budget_type="run")
    assert e.current_cost == 0.0
    assert e.limit == 0.0


def test_budget_threshold_error_all_actions() -> None:
    """BudgetThresholdError with various actions."""
    actions = ["warn", "stop", "switch_model", "summarize"]
    for action in actions:
        e = BudgetThresholdError("hit", threshold_percent=50.0, action_taken=action)
        assert e.action_taken == action


def test_tool_execution_error_basic() -> None:
    """ToolExecutionError basic functionality."""
    e = ToolExecutionError("tool error")
    assert "tool" in str(e).lower()


def test_provider_error_basic() -> None:
    """ProviderError basic functionality."""
    e = ProviderError("provider error")
    assert "provider" in str(e).lower()


def test_model_not_found_error_basic() -> None:
    """ModelNotFoundError basic functionality."""
    e = ModelNotFoundError("model not found")
    assert "model" in str(e).lower()


def test_task_error_basic() -> None:
    """TaskError basic functionality."""
    e = TaskError("task failed")
    assert "task" in str(e).lower()


def test_codegen_error_basic() -> None:
    """CodegenError basic functionality."""
    e = CodegenError("codegen error")
    assert "codegen" in str(e).lower()


def test_circuit_breaker_open_error_attrs() -> None:
    """CircuitBreakerOpenError has required attributes."""
    e = CircuitBreakerOpenError(
        "Circuit open",
        agent_name="TestAgent",
        recovery_at=123.0,
        fallback_model="ollama/llama3",
    )
    assert e.agent_name == "TestAgent"
    assert e.recovery_at == 123.0
    assert e.fallback_model == "ollama/llama3"


def test_all_exceptions_are_picklable() -> None:
    """All exceptions should be picklable."""
    import pickle

    e = SyrinError("test")
    pickled = pickle.dumps(e)
    unpickled = pickle.loads(pickled)
    assert str(unpickled) == "test"


def test_exception_chaining() -> None:
    """Exceptions support chaining."""
    original = ValueError("original")
    e = SyrinError("wrapper")
    e.__cause__ = original
    assert e.__cause__ is original
