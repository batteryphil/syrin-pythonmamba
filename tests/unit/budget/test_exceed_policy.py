"""Tests for ExceedPolicy enum integration with Budget and TokenLimits.

Verifies:
- ExceedPolicy enum values and string representation
- Budget.exceed_policy maps to a callable _handler
- TokenLimits.exceed_policy maps to a callable _handler
- STOP policy raises BudgetExceededError
- WARN policy logs and continues
- IGNORE policy silently continues
- exceed_policy=None leaves _handler as None
"""

from __future__ import annotations

import logging

import pytest

from syrin.budget import (
    Budget,
    BudgetExceededContext,
    BudgetLimitType,
    TokenLimits,
)
from syrin.enums import ExceedPolicy
from syrin.exceptions import BudgetExceededError


def _make_ctx(
    budget_type: BudgetLimitType = BudgetLimitType.RUN,
    current: float = 10.0,
    limit: float = 5.0,
) -> BudgetExceededContext:
    return BudgetExceededContext(
        current_cost=current,
        limit=limit,
        budget_type=budget_type,
        message=f"Budget exceeded: ${current:.4f} >= ${limit:.4f}",
    )


# ---------------------------------------------------------------------------
# ExceedPolicy enum
# ---------------------------------------------------------------------------


class TestExceedPolicyEnum:
    def test_stop_value(self) -> None:
        assert ExceedPolicy.STOP == "stop"

    def test_warn_value(self) -> None:
        assert ExceedPolicy.WARN == "warn"

    def test_ignore_value(self) -> None:
        assert ExceedPolicy.IGNORE == "ignore"

    def test_is_str(self) -> None:
        """ExceedPolicy values are plain strings (StrEnum)."""
        assert isinstance(ExceedPolicy.STOP, str)
        assert isinstance(ExceedPolicy.WARN, str)

    def test_all_values_unique(self) -> None:
        values = [p.value for p in ExceedPolicy]
        assert len(values) == len(set(values))

    def test_from_string(self) -> None:
        assert ExceedPolicy("stop") is ExceedPolicy.STOP
        assert ExceedPolicy("warn") is ExceedPolicy.WARN
        assert ExceedPolicy("ignore") is ExceedPolicy.IGNORE

    def test_invalid_value_raises(self) -> None:
        with pytest.raises(ValueError):
            ExceedPolicy("invalid")


# ---------------------------------------------------------------------------
# Budget.exceed_policy integration
# ---------------------------------------------------------------------------


class TestBudgetExceedPolicy:
    def test_no_policy_leaves_handler_none(self) -> None:
        b = Budget(max_cost=1.0)
        assert b._handler is None

    def test_stop_policy_raises(self) -> None:
        b = Budget(max_cost=1.0, exceed_policy=ExceedPolicy.STOP)
        assert b._handler is not None
        with pytest.raises(BudgetExceededError):
            b._handler(_make_ctx())

    def test_warn_policy_logs_and_continues(self, caplog: pytest.LogCaptureFixture) -> None:
        b = Budget(max_cost=1.0, exceed_policy=ExceedPolicy.WARN)
        assert b._handler is not None
        with caplog.at_level(logging.WARNING):
            b._handler(_make_ctx())  # Must NOT raise
        assert len(caplog.records) > 0

    def test_ignore_policy_continues_silently(self) -> None:
        b = Budget(max_cost=1.0, exceed_policy=ExceedPolicy.IGNORE)
        assert b._handler is not None
        result = b._handler(_make_ctx())
        assert result is None

    def test_switch_policy_no_longer_exists(self) -> None:
        """ExceedPolicy.SWITCH has been removed."""
        assert not hasattr(ExceedPolicy, "SWITCH")

    def test_exceed_policy_none_leaves_handler_none(self) -> None:
        b = Budget(max_cost=1.0, exceed_policy=None)
        assert b._handler is None

    def test_stop_policy_str_roundtrip(self) -> None:
        b = Budget(max_cost=1.0, exceed_policy=ExceedPolicy.STOP)
        assert b.exceed_policy == "stop"
        assert b.exceed_policy is ExceedPolicy.STOP


# ---------------------------------------------------------------------------
# TokenLimits.exceed_policy integration
# ---------------------------------------------------------------------------


class TestTokenLimitsExceedPolicy:
    def test_no_policy_leaves_handler_none(self) -> None:
        tl = TokenLimits(max_tokens=1000)
        assert tl._handler is None

    def test_stop_policy_raises(self) -> None:
        tl = TokenLimits(max_tokens=1000, exceed_policy=ExceedPolicy.STOP)
        assert tl._handler is not None
        with pytest.raises(BudgetExceededError):
            tl._handler(_make_ctx(budget_type=BudgetLimitType.RUN_TOKENS))

    def test_warn_policy_continues(self, caplog: pytest.LogCaptureFixture) -> None:
        tl = TokenLimits(max_tokens=1000, exceed_policy=ExceedPolicy.WARN)
        assert tl._handler is not None
        with caplog.at_level(logging.WARNING):
            tl._handler(_make_ctx(budget_type=BudgetLimitType.RUN_TOKENS))
        assert len(caplog.records) > 0

    def test_ignore_policy_continues_silently(self) -> None:
        tl = TokenLimits(max_tokens=1000, exceed_policy=ExceedPolicy.IGNORE)
        assert tl._handler is not None
        result = tl._handler(_make_ctx(budget_type=BudgetLimitType.RUN_TOKENS))
        assert result is None


# ---------------------------------------------------------------------------
# STOP vs WARN behaviour contracts via ExceedPolicy
# ---------------------------------------------------------------------------


class TestPolicyHandlerContracts:
    def test_stop_policy_raises_budget_error(self) -> None:
        b = Budget(max_cost=1.0, exceed_policy=ExceedPolicy.STOP)
        ctx = _make_ctx(current=5.0, limit=1.0)
        with pytest.raises(BudgetExceededError):
            b._handler(ctx)  # type: ignore[misc]

    def test_stop_policy_includes_current_and_limit(self) -> None:
        b = Budget(max_cost=1.0, exceed_policy=ExceedPolicy.STOP)
        ctx = _make_ctx(current=7.5, limit=2.0)
        with pytest.raises(BudgetExceededError) as exc_info:
            b._handler(ctx)  # type: ignore[misc]
        err = exc_info.value
        assert err.current_cost == pytest.approx(7.5)
        assert err.limit == pytest.approx(2.0)

    def test_warn_policy_does_not_raise(self, caplog: pytest.LogCaptureFixture) -> None:
        b = Budget(max_cost=1.0, exceed_policy=ExceedPolicy.WARN)
        ctx = _make_ctx()
        with caplog.at_level(logging.WARNING):
            b._handler(ctx)  # type: ignore[misc]  # Must NOT raise

    def test_warn_policy_logs_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        b = Budget(max_cost=1.0, exceed_policy=ExceedPolicy.WARN)
        ctx = _make_ctx(current=9.99, limit=5.0)
        with caplog.at_level(logging.WARNING):
            b._handler(ctx)  # type: ignore[misc]
        assert any(
            "budget" in r.message.lower() or "exceed" in r.message.lower() for r in caplog.records
        )

    def test_stop_policy_budget_type_preserved(self) -> None:
        b = Budget(max_cost=1.0, exceed_policy=ExceedPolicy.STOP)
        ctx = _make_ctx(budget_type=BudgetLimitType.HOUR)
        with pytest.raises(BudgetExceededError) as exc_info:
            b._handler(ctx)  # type: ignore[misc]
        assert exc_info.value.budget_type == BudgetLimitType.HOUR.value
