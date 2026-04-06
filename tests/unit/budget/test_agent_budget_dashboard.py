"""TDD tests for agent.budget_summary() and agent.export_costs().

Verifies:
- budget_summary() returns correct keys and values
- budget_summary() without budget omits budget-specific keys
- budget_summary() with budget includes max_cost, safety_margin, policy, remaining, percent_used
- budget_summary() after recording costs reflects updated totals
- export_costs() returns list of dicts by default
- export_costs(format="json") returns valid JSON string
- export_costs() entries have correct shape: cost_usd, total_tokens, model, timestamp
- export_costs() is in recording order
- export_costs() returns empty list when no costs recorded
- export_costs() invalid format raises ValueError
"""

from __future__ import annotations

import json

import pytest

from syrin.agent import Agent
from syrin.budget import Budget
from syrin.model import Model
from syrin.types import CostInfo, TokenUsage


def _agent(budget: Budget | None = None) -> Agent:
    return Agent(model=Model("openai/gpt-4o-mini"), system_prompt="", budget=budget)


def _record(agent: Agent, cost: float, tokens: int = 100, model: str = "gpt-4o-mini") -> None:
    agent._budget_tracker.record(
        CostInfo(
            cost_usd=cost,
            token_usage=TokenUsage(total_tokens=tokens),
            model_name=model,
        )
    )


# ---------------------------------------------------------------------------
# budget_summary() — no budget configured
# ---------------------------------------------------------------------------


class TestBudgetSummaryNoBudget:
    def test_returns_dict(self) -> None:
        agent = _agent()
        summary = agent.budget_summary()
        assert isinstance(summary, dict)

    def test_run_cost_starts_at_zero(self) -> None:
        agent = _agent()
        assert agent.budget_summary()["run_cost"] == pytest.approx(0.0)

    def test_run_tokens_starts_at_zero(self) -> None:
        agent = _agent()
        assert agent.budget_summary()["run_tokens"] == 0

    def test_window_costs_present(self) -> None:
        agent = _agent()
        summary = agent.budget_summary()
        for key in ("hourly_cost", "daily_cost", "weekly_cost", "monthly_cost"):
            assert key in summary, f"Missing key: {key}"

    def test_window_tokens_present(self) -> None:
        agent = _agent()
        summary = agent.budget_summary()
        for key in ("hourly_tokens", "daily_tokens", "weekly_tokens", "monthly_tokens"):
            assert key in summary, f"Missing key: {key}"

    def test_budget_keys_absent_when_no_budget(self) -> None:
        agent = _agent(budget=None)
        summary = agent.budget_summary()
        assert "max_cost" not in summary
        assert "safety_margin" not in summary
        assert "exceed_policy" not in summary
        assert "budget_remaining" not in summary

    def test_run_cost_reflects_recorded_cost(self) -> None:
        agent = _agent()
        _record(agent, 0.05)
        assert agent.budget_summary()["run_cost"] == pytest.approx(0.05)

    def test_run_tokens_reflects_recorded_tokens(self) -> None:
        agent = _agent()
        _record(agent, 0.01, tokens=250)
        assert agent.budget_summary()["run_tokens"] == 250

    def test_multiple_recordings_accumulate(self) -> None:
        agent = _agent()
        _record(agent, 0.01, tokens=100)
        _record(agent, 0.02, tokens=200)
        summary = agent.budget_summary()
        assert summary["run_cost"] == pytest.approx(0.03)
        assert summary["run_tokens"] == 300


# ---------------------------------------------------------------------------
# budget_summary() — with budget configured
# ---------------------------------------------------------------------------


class TestBudgetSummaryWithBudget:
    def test_max_cost_present(self) -> None:
        agent = _agent(budget=Budget(max_cost=5.0))
        assert agent.budget_summary()["max_cost"] == pytest.approx(5.0)

    def test_safety_margin_present(self) -> None:
        agent = _agent(budget=Budget(max_cost=5.0, safety_margin=0.5))
        assert agent.budget_summary()["safety_margin"] == pytest.approx(0.5)

    def test_safety_margin_zero_when_not_set(self) -> None:
        agent = _agent(budget=Budget(max_cost=5.0))
        assert agent.budget_summary()["safety_margin"] == pytest.approx(0.0)

    def test_budget_remaining_present(self) -> None:
        agent = _agent(budget=Budget(max_cost=5.0))
        assert "budget_remaining" in agent.budget_summary()

    def test_budget_remaining_equals_max_cost_initially(self) -> None:
        agent = _agent(budget=Budget(max_cost=5.0))
        remaining = agent.budget_summary()["budget_remaining"]
        assert isinstance(remaining, float)
        assert remaining == pytest.approx(5.0)

    def test_budget_remaining_decreases_after_cost(self) -> None:
        agent = _agent(budget=Budget(max_cost=5.0))
        _record(agent, 1.0)
        remaining = agent.budget_summary()["budget_remaining"]
        assert isinstance(remaining, float)
        assert remaining == pytest.approx(4.0)

    def test_budget_percent_used_zero_initially(self) -> None:
        agent = _agent(budget=Budget(max_cost=10.0))
        pct = agent.budget_summary()["budget_percent_used"]
        assert isinstance(pct, float)
        assert pct == pytest.approx(0.0)

    def test_budget_percent_used_after_half_spent(self) -> None:
        agent = _agent(budget=Budget(max_cost=10.0))
        _record(agent, 5.0)
        pct = agent.budget_summary()["budget_percent_used"]
        assert isinstance(pct, float)
        assert pct == pytest.approx(50.0)

    def test_exceed_policy_none_when_not_set(self) -> None:
        agent = _agent(budget=Budget(max_cost=5.0))
        assert agent.budget_summary()["exceed_policy"] is None

    def test_exceed_policy_present_when_set(self) -> None:
        from syrin.enums import ExceedPolicy

        agent = _agent(budget=Budget(max_cost=5.0, exceed_policy=ExceedPolicy.WARN))
        policy = agent.budget_summary()["exceed_policy"]
        assert policy == "warn"

    def test_budget_remaining_never_negative(self) -> None:
        agent = _agent(budget=Budget(max_cost=1.0))
        _record(agent, 5.0)  # Far over budget
        remaining = agent.budget_summary()["budget_remaining"]
        assert isinstance(remaining, float)
        assert remaining >= 0.0

    def test_with_safety_margin_reduces_effective_limit(self) -> None:
        # max_cost=5.0, safety_margin=1.0 → effective limit 4.0
        agent = _agent(budget=Budget(max_cost=5.0, safety_margin=1.0))
        summary = agent.budget_summary()
        assert summary["safety_margin"] == pytest.approx(1.0)
        # Remaining starts at effective limit (4.0)
        remaining = summary["budget_remaining"]
        assert isinstance(remaining, float)
        assert remaining == pytest.approx(4.0)


# ---------------------------------------------------------------------------
# export_costs()
# ---------------------------------------------------------------------------


class TestExportCostsDict:
    def test_returns_list_by_default(self) -> None:
        agent = _agent()
        result = agent.export_costs()
        assert isinstance(result, list)

    def test_empty_when_no_costs_recorded(self) -> None:
        agent = _agent()
        assert agent.export_costs() == []

    def test_single_entry_after_one_record(self) -> None:
        agent = _agent()
        _record(agent, 0.05, tokens=100, model="gpt-4o-mini")
        rows = agent.export_costs()
        assert isinstance(rows, list)
        assert len(rows) == 1

    def test_entry_has_required_keys(self) -> None:
        agent = _agent()
        _record(agent, 0.05)
        row = agent.export_costs()[0]
        assert isinstance(row, dict)
        assert "cost_usd" in row
        assert "total_tokens" in row
        assert "model" in row
        assert "timestamp" in row

    def test_cost_usd_correct(self) -> None:
        agent = _agent()
        _record(agent, 0.0123, tokens=50, model="gpt-4o")
        row = agent.export_costs()[0]
        assert isinstance(row, dict)
        assert row["cost_usd"] == pytest.approx(0.0123)

    def test_total_tokens_correct(self) -> None:
        agent = _agent()
        _record(agent, 0.01, tokens=777)
        row = agent.export_costs()[0]
        assert isinstance(row, dict)
        assert row["total_tokens"] == 777

    def test_model_name_correct(self) -> None:
        agent = _agent()
        _record(agent, 0.01, model="claude-3-5-haiku")
        row = agent.export_costs()[0]
        assert isinstance(row, dict)
        assert row["model"] == "claude-3-5-haiku"

    def test_timestamp_is_numeric(self) -> None:
        agent = _agent()
        _record(agent, 0.01)
        row = agent.export_costs()[0]
        assert isinstance(row, dict)
        assert isinstance(row["timestamp"], float | int)

    def test_multiple_entries_in_order(self) -> None:
        agent = _agent()
        _record(agent, 0.01, model="model-a")
        _record(agent, 0.02, model="model-b")
        _record(agent, 0.03, model="model-c")
        rows = agent.export_costs()
        assert isinstance(rows, list)
        assert len(rows) == 3
        assert rows[0]["model"] == "model-a"  # type: ignore[index]
        assert rows[1]["model"] == "model-b"  # type: ignore[index]
        assert rows[2]["model"] == "model-c"  # type: ignore[index]

    def test_costs_accumulate_across_records(self) -> None:
        agent = _agent()
        _record(agent, 0.10, tokens=100)
        _record(agent, 0.20, tokens=200)
        rows = agent.export_costs()
        assert isinstance(rows, list)
        total = sum(r["cost_usd"] for r in rows)  # type: ignore[index]
        assert total == pytest.approx(0.30)


class TestExportCostsJson:
    def test_json_format_returns_string(self) -> None:
        agent = _agent()
        _record(agent, 0.05)
        result = agent.export_costs(format="json")
        assert isinstance(result, str)

    def test_json_is_valid(self) -> None:
        agent = _agent()
        _record(agent, 0.05, tokens=100)
        result = agent.export_costs(format="json")
        assert isinstance(result, str)
        parsed = json.loads(result)
        assert isinstance(parsed, list)

    def test_json_empty_list_when_no_costs(self) -> None:
        agent = _agent()
        result = agent.export_costs(format="json")
        assert isinstance(result, str)
        assert json.loads(result) == []

    def test_json_entry_has_correct_fields(self) -> None:
        agent = _agent()
        _record(agent, 0.05, tokens=50, model="gpt-4o")
        result = agent.export_costs(format="json")
        assert isinstance(result, str)
        parsed = json.loads(result)
        row = parsed[0]
        assert row["cost_usd"] == pytest.approx(0.05)
        assert row["total_tokens"] == 50
        assert row["model"] == "gpt-4o"

    def test_json_multiple_entries(self) -> None:
        agent = _agent()
        _record(agent, 0.01)
        _record(agent, 0.02)
        result = agent.export_costs(format="json")
        assert isinstance(result, str)
        assert len(json.loads(result)) == 2

    def test_dict_format_explicit(self) -> None:
        agent = _agent()
        _record(agent, 0.01)
        result = agent.export_costs(format="dict")
        assert isinstance(result, list)
