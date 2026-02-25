"""spawn(): return type (Response vs Agent) and budget inheritance correct."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from syrin import Agent, Budget, Model, Response
from syrin.types import TokenUsage


def _mock_provider():
    mock = MagicMock()
    mock.complete = AsyncMock(
        return_value=MagicMock(
            content="Ok",
            tool_calls=[],
            token_usage=TokenUsage(input_tokens=5, output_tokens=10, total_tokens=15),
        )
    )
    return mock


@patch("syrin.agent._resolve_provider")
def test_spawn_with_task_returns_response(mock_resolve: MagicMock) -> None:
    """spawn(Child, task='...') returns Response, not Agent."""
    mock_resolve.return_value = _mock_provider()

    class Parent(Agent):
        model = Model("test/model")

    class Child(Agent):
        model = Model("test/model")

    parent = Parent()
    result = parent.spawn(Child, task="Do it")
    assert isinstance(result, Response)
    assert not isinstance(result, Agent)
    assert hasattr(result, "content")
    assert hasattr(result, "cost")


@patch("syrin.agent._resolve_provider")
def test_spawn_without_task_returns_agent(mock_resolve: MagicMock) -> None:
    """spawn(Child) with no task returns the spawned Agent instance."""
    mock_resolve.return_value = _mock_provider()

    class Parent(Agent):
        model = Model("test/model")

    class Child(Agent):
        model = Model("test/model")

    parent = Parent()
    child = parent.spawn(Child)
    assert isinstance(child, Agent)
    assert isinstance(child, Child)
    assert not isinstance(child, Response)


@patch("syrin.agent._resolve_provider")
def test_spawn_budget_inheritance_shared_parent_remaining_decreases(
    mock_resolve: MagicMock,
) -> None:
    """With shared budget, after child runs, parent's remaining reflects child spend."""
    mock_resolve.return_value = _mock_provider()

    class Parent(Agent):
        model = Model("test/model")

    class Child(Agent):
        model = Model("test/model")

    parent = Parent(budget=Budget(run=10.0, shared=True))
    initial_remaining = parent._budget.remaining
    assert initial_remaining is not None

    result = parent.spawn(Child, task="Run")
    assert result.cost >= 0
    # Child shared parent tracker; parent's spent should include child's cost
    assert parent._budget._spent == parent._budget_tracker.current_run_cost
    assert parent._budget.remaining is not None
    assert parent._budget.remaining <= initial_remaining


@patch("syrin.agent._resolve_provider")
def test_spawn_pocket_money_child_has_budget_cap(mock_resolve: MagicMock) -> None:
    """spawn(Child, budget=Budget(run=0.50)) gives child that run cap."""
    mock_resolve.return_value = _mock_provider()

    class Parent(Agent):
        model = Model("test/model")

    class Child(Agent):
        model = Model("test/model")

    parent = Parent(budget=Budget(run=5.0))
    child = parent.spawn(Child, budget=Budget(run=0.50))
    assert child._budget is not None
    assert child._budget.run == 0.50


@patch("syrin.agent._resolve_provider")
def test_spawn_pocket_money_exceeds_parent_raises(mock_resolve: MagicMock) -> None:
    """spawn(Child, budget=Budget(run=...)) with run > parent remaining raises ValueError."""
    mock_resolve.return_value = _mock_provider()

    class Parent(Agent):
        model = Model("test/model")

    class Child(Agent):
        model = Model("test/model")

    parent = Parent(budget=Budget(run=1.0))
    with pytest.raises(ValueError, match="cannot exceed parent"):
        parent.spawn(Child, budget=Budget(run=2.0))
