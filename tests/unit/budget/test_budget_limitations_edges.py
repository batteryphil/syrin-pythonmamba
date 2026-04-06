"""Tests documenting Budget limitations and edge cases.

These tests encode documented behavior: rate limits in-memory unless store is used,
effective run limit (run - reserve), token_limits boundary, check-after-call overshoot,
and fixed rate windows (e.g. 30 days for month).
"""

from __future__ import annotations

import time

from syrin.budget import (
    Budget,
    BudgetLimitType,
    BudgetStatus,
    BudgetTracker,
    RateLimit,
    TokenLimits,
    TokenRateLimit,
)
from syrin.enums import ThresholdMetric
from syrin.threshold import BudgetThreshold
from syrin.types import CostInfo, TokenUsage


def test_rate_limits_not_shared_between_trackers() -> None:
    """Rate limits are per BudgetTracker; without budget_store they do not persist.

    Two separate tracker instances have independent state. So rate limits (hour/day/week/month)
    only apply within the lifetime of one process and one tracker unless you use a
    BudgetStore to load/save state across runs.
    """
    t1 = BudgetTracker()
    t2 = BudgetTracker()
    t1.record(CostInfo(cost_usd=5.0, token_usage=TokenUsage()))
    t2.record(CostInfo(cost_usd=5.0, token_usage=TokenUsage()))
    budget = Budget(rate_limits=RateLimit(hour=10.0))
    assert t1.check_budget(budget).status == BudgetStatus.OK
    assert t2.check_budget(budget).status == BudgetStatus.OK
    t1.record(CostInfo(cost_usd=6.0, token_usage=TokenUsage()))
    assert t1.check_budget(budget).status == BudgetStatus.EXCEEDED
    assert t2.check_budget(budget).status == BudgetStatus.OK


def test_run_reserve_effective_limit_in_check_budget() -> None:
    """Effective run limit is run - safety_margin when run > reserve; exceed when cost >= effective."""
    tracker = BudgetTracker()
    tracker.record(CostInfo(cost_usd=7.0, token_usage=TokenUsage()))
    budget = Budget(max_cost=10.0, safety_margin=2.0)
    result = tracker.check_budget(budget)
    assert result.status == BudgetStatus.OK
    assert result.exceeded_limit is None
    tracker.record(CostInfo(cost_usd=1.5, token_usage=TokenUsage()))
    result = tracker.check_budget(budget)
    assert result.status == BudgetStatus.EXCEEDED
    assert result.exceeded_limit == BudgetLimitType.RUN


def test_run_reserve_greater_than_run_uses_run_as_limit() -> None:
    """When safety_margin >= run, effective limit is still run (no negative)."""
    tracker = BudgetTracker()
    tracker.record(CostInfo(cost_usd=5.0, token_usage=TokenUsage()))
    budget = Budget(max_cost=5.0, safety_margin=10.0)
    result = tracker.check_budget(budget)
    assert result.status == BudgetStatus.EXCEEDED
    assert result.exceeded_limit == BudgetLimitType.RUN


def test_run_tokens_exactly_at_limit_exceeded() -> None:
    """token_limits.run_tokens check uses >= ; exactly at limit is EXCEEDED."""
    tracker = BudgetTracker()
    tracker.record(CostInfo(cost_usd=0.0, token_usage=TokenUsage(total_tokens=100)))
    result = tracker.check_budget(
        Budget(max_cost=10.0),
        token_limits=TokenLimits(max_tokens=100),
    )
    assert result.status == BudgetStatus.EXCEEDED
    assert result.exceeded_limit == BudgetLimitType.RUN_TOKENS


def test_run_tokens_just_under_limit_ok() -> None:
    """Just under token_limits.run_tokens limit is OK."""
    tracker = BudgetTracker()
    tracker.record(CostInfo(cost_usd=0.0, token_usage=TokenUsage(total_tokens=99)))
    result = tracker.check_budget(
        Budget(max_cost=10.0),
        token_limits=TokenLimits(max_tokens=100),
    )
    assert result.status == BudgetStatus.OK
    assert result.exceeded_limit is None


def test_check_after_record_exceeded_one_call_can_overshoot() -> None:
    """Budget is checked after each record; one call can push over the limit.

    So you can overshoot by at most one LLM call's cost (or one chunk when streaming).
    """
    tracker = BudgetTracker()
    budget = Budget(max_cost=5.0)
    tracker.record(CostInfo(cost_usd=4.0, token_usage=TokenUsage()))
    assert tracker.check_budget(budget).status == BudgetStatus.OK
    tracker.record(CostInfo(cost_usd=2.0, token_usage=TokenUsage()))
    result = tracker.check_budget(budget)
    assert result.status == BudgetStatus.EXCEEDED
    assert result.exceeded_limit == BudgetLimitType.RUN
    assert tracker.current_run_cost == 6.0


def test_month_window_uses_fixed_30_days() -> None:
    """Rate-limit month window defaults to 30 days (wall-clock); configurable via month_days."""
    tracker = BudgetTracker()
    tracker.record(CostInfo(cost_usd=100.0, token_usage=TokenUsage()))
    budget = Budget(rate_limits=RateLimit(month=50.0))
    result = tracker.check_budget(budget)
    assert result.status == BudgetStatus.EXCEEDED
    assert result.exceeded_limit == BudgetLimitType.MONTH


def test_run_none_no_run_limit_check() -> None:
    """When run is None, no per-run cost limit is enforced."""
    tracker = BudgetTracker()
    tracker.record(CostInfo(cost_usd=1e6, token_usage=TokenUsage()))
    budget = Budget(max_cost=None, rate_limits=RateLimit(month=1e9))
    result = tracker.check_budget(budget)
    assert result.status == BudgetStatus.OK


# =============================================================================
# RATE LIMIT: month_days (USD only; token caps via TokenLimits)
# =============================================================================


def test_rate_limit_month_days_default() -> None:
    """RateLimit.month_days defaults to 30."""
    r = RateLimit(month=100.0)
    assert r.month_days == 30


def test_rate_limit_month_days_valid_range() -> None:
    """RateLimit.month_days accepts 1 to 31."""
    r = RateLimit(month=100.0, month_days=1)
    assert r.month_days == 1
    r = RateLimit(month=100.0, month_days=31)
    assert r.month_days == 31


def test_rate_limit_month_days_invalid_low() -> None:
    """RateLimit.month_days must be >= 1."""
    import pytest
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        RateLimit(month=100.0, month_days=0)


def test_rate_limit_month_days_invalid_high() -> None:
    """RateLimit.month_days must be <= 31."""
    import pytest
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        RateLimit(month=100.0, month_days=32)


# =============================================================================
# BUDGET TRACKER: month_days and token windows (TokenLimits)
# =============================================================================


def test_tracker_month_days_from_budget() -> None:
    """BudgetTracker uses budget.per.month_days for month window when checking."""
    tracker = BudgetTracker()
    tracker.record(CostInfo(cost_usd=100.0, token_usage=TokenUsage()))
    budget_30 = Budget(rate_limits=RateLimit(month=50.0, month_days=30))
    budget_7 = Budget(rate_limits=RateLimit(month=50.0, month_days=7))
    assert tracker.check_budget(budget_30).exceeded_limit == BudgetLimitType.MONTH
    assert tracker._month_days == 30
    assert tracker.check_budget(budget_7).exceeded_limit == BudgetLimitType.MONTH
    assert tracker._month_days == 7


def test_tracker_hourly_daily_weekly_monthly_tokens() -> None:
    """BudgetTracker exposes hourly_tokens, daily_tokens, weekly_tokens, monthly_tokens."""
    tracker = BudgetTracker()
    tracker.record(CostInfo(cost_usd=0.1, token_usage=TokenUsage(total_tokens=100)))
    tracker.record(CostInfo(cost_usd=0.1, token_usage=TokenUsage(total_tokens=50)))
    assert tracker.hourly_tokens == 150
    assert tracker.daily_tokens == 150
    assert tracker.weekly_tokens == 150
    assert tracker.monthly_tokens == 150


def test_tracker_check_budget_exceeded_hour_tokens() -> None:
    """When token_limits.per.hour exceeded, status EXCEEDED and exceeded_limit is hour_tokens."""
    tracker = BudgetTracker()
    tracker.record(CostInfo(cost_usd=0.0, token_usage=TokenUsage(total_tokens=150_000)))
    result = tracker.check_budget(
        Budget(max_cost=10.0, rate_limits=RateLimit(hour=100.0)),
        token_limits=TokenLimits(rate_limits=TokenRateLimit(hour=100_000)),
    )
    assert result.status == BudgetStatus.EXCEEDED
    assert result.exceeded_limit == BudgetLimitType.HOUR_TOKENS


def test_tracker_check_budget_exceeded_day_tokens() -> None:
    """When token_limits.per.day exceeded, exceeded_limit is day_tokens."""
    tracker = BudgetTracker()
    tracker.record(CostInfo(cost_usd=0.0, token_usage=TokenUsage(total_tokens=200_000)))
    result = tracker.check_budget(
        Budget(max_cost=10.0, rate_limits=RateLimit(day=100.0)),
        token_limits=TokenLimits(rate_limits=TokenRateLimit(day=100_000)),
    )
    assert result.status == BudgetStatus.EXCEEDED
    assert result.exceeded_limit == BudgetLimitType.DAY_TOKENS


def test_tracker_check_budget_exceeded_month_tokens() -> None:
    """When token_limits.per.month exceeded, exceeded_limit is month_tokens."""
    tracker = BudgetTracker()
    tracker.record(CostInfo(cost_usd=0.0, token_usage=TokenUsage(total_tokens=500_000)))
    result = tracker.check_budget(
        Budget(max_cost=10.0, rate_limits=RateLimit(month=100.0)),
        token_limits=TokenLimits(rate_limits=TokenRateLimit(month=400_000)),
    )
    assert result.status == BudgetStatus.EXCEEDED
    assert result.exceeded_limit == BudgetLimitType.MONTH_TOKENS


def test_tracker_check_budget_token_limits_under_ok() -> None:
    """When token usage under token_limits, status OK."""
    tracker = BudgetTracker()
    tracker.record(CostInfo(cost_usd=0.0, token_usage=TokenUsage(total_tokens=50_000)))
    result = tracker.check_budget(
        Budget(max_cost=10.0, rate_limits=RateLimit(hour=100.0)),
        token_limits=TokenLimits(rate_limits=TokenRateLimit(hour=100_000)),
    )
    assert result.status == BudgetStatus.OK
    assert result.exceeded_limit is None


def test_tracker_get_state_load_state_preserves_month_days() -> None:
    """get_state/load_state round-trip preserves month_days."""
    tracker = BudgetTracker()
    tracker._month_days = 14
    tracker.record(CostInfo(cost_usd=0.1, token_usage=TokenUsage()))
    state = tracker.get_state()
    assert state.get("month_days") == 14
    tracker2 = BudgetTracker()
    tracker2.load_state(state)
    assert tracker2._month_days == 14


def test_tracker_load_state_missing_month_days_defaults_to_30() -> None:
    """load_state with no month_days uses default 30 (backward compat)."""
    tracker = BudgetTracker()
    state = {"cost_history": [], "run_start": time.time()}
    tracker.load_state(state)
    assert tracker._month_days == 30


def test_budget_summary_includes_token_windows() -> None:
    """BudgetSummary includes hourly_tokens, daily_tokens, weekly_tokens, monthly_tokens."""
    tracker = BudgetTracker()
    tracker.record(CostInfo(cost_usd=0.1, token_usage=TokenUsage(total_tokens=100)))
    s = tracker.get_summary()
    assert s.hourly_tokens == 100
    assert s.daily_tokens == 100
    assert s.weekly_tokens == 100
    assert s.monthly_tokens == 100
    d = s.to_dict()
    assert d["hourly_tokens"] == 100
    assert d["monthly_tokens"] == 100


# =============================================================================
# CALENDAR MONTH
# =============================================================================


def test_rate_limit_calendar_month_default_false() -> None:
    """RateLimit.calendar_month defaults to False."""
    r = RateLimit(month=100.0)
    assert r.calendar_month is False


def test_tracker_calendar_month_sums_current_month_only() -> None:
    """When calendar_month=True, monthly_cost and monthly_tokens sum only current calendar month."""
    tracker = BudgetTracker()
    tracker._use_calendar_month = True
    tracker.record(CostInfo(cost_usd=10.0, token_usage=TokenUsage(total_tokens=1000)))
    cost = tracker.monthly_cost
    tokens = tracker.monthly_tokens
    assert cost == 10.0
    assert tokens == 1000


# =============================================================================
# TOKEN THRESHOLDS AND at_range
# =============================================================================


def test_budget_threshold_at_range_triggers_in_band() -> None:
    """Threshold with at_range=(70, 75) triggers when 70 <= pct <= 75."""
    from syrin.threshold import BudgetThreshold

    triggered_pct: list[int] = []

    def capture(ctx):
        triggered_pct.append(ctx.percentage)

    tracker = BudgetTracker()
    tracker.record(CostInfo(cost_usd=72.0, token_usage=TokenUsage()))
    budget = Budget(
        max_cost=100.0,
        thresholds=[
            BudgetThreshold(at_range=(70, 75), action=capture, window="run"),
        ],
    )
    tracker.check_thresholds(budget)
    assert triggered_pct == [72]


def test_budget_threshold_at_range_does_not_trigger_below_band() -> None:
    """Threshold with at_range=(70, 75) does not trigger when pct < 70."""
    from syrin.threshold import BudgetThreshold

    triggered: list[int] = []

    def capture(ctx):
        triggered.append(ctx.percentage)

    tracker = BudgetTracker()
    tracker.record(CostInfo(cost_usd=50.0, token_usage=TokenUsage()))
    budget = Budget(
        max_cost=100.0,
        thresholds=[BudgetThreshold(at_range=(70, 75), action=capture, window="run")],
    )
    tracker.check_thresholds(budget)
    assert triggered == []


def test_budget_threshold_at_range_does_not_trigger_above_band() -> None:
    """Threshold with at_range=(70, 75) does not trigger when pct > 75."""
    from syrin.threshold import BudgetThreshold

    triggered: list[int] = []

    def capture(ctx):
        triggered.append(ctx.percentage)

    tracker = BudgetTracker()
    tracker.record(CostInfo(cost_usd=90.0, token_usage=TokenUsage()))
    budget = Budget(
        max_cost=100.0,
        thresholds=[BudgetThreshold(at_range=(70, 75), action=capture, window="run")],
    )
    tracker.check_thresholds(budget)
    assert triggered == []


def test_budget_threshold_tokens_hour_window() -> None:
    """Token threshold with window=hour triggers at 80% of token_limits.per.hour."""
    triggered: list[int] = []

    def capture(ctx):
        triggered.append(ctx.percentage)

    tracker = BudgetTracker()
    tracker.record(CostInfo(cost_usd=0.0, token_usage=TokenUsage(total_tokens=85_000)))
    budget = Budget(
        max_cost=10.0,
        rate_limits=RateLimit(hour=100.0),
        thresholds=[
            BudgetThreshold(
                at=80,
                action=capture,
                metric=ThresholdMetric.TOKENS,
                window="hour",
            ),
        ],
    )
    token_limits = TokenLimits(rate_limits=TokenRateLimit(hour=100_000))
    tracker.check_thresholds(budget, token_limits=token_limits)
    assert triggered == [85]


def test_reserve_commit_records_actual_and_releases() -> None:
    """reserve(amount) returns a token; commit(actual_cost) records cost and releases reservation."""
    tracker = BudgetTracker()
    tracker.record(CostInfo(cost_usd=1.0, token_usage=TokenUsage()))
    token = tracker.reserve(2.0)
    assert tracker.current_run_cost == 1.0
    token.commit(0.5, TokenUsage(input_tokens=10, output_tokens=5, total_tokens=15))
    assert tracker.current_run_cost == 1.0 + 0.5
    assert tracker.current_run_tokens == 15


def test_reserve_rollback_releases_without_recording() -> None:
    """rollback() releases the reservation without recording any cost."""
    tracker = BudgetTracker()
    tracker.record(CostInfo(cost_usd=1.0, token_usage=TokenUsage()))
    token = tracker.reserve(3.0)
    before = tracker.current_run_cost
    token.rollback()
    assert tracker.current_run_cost == before


def test_check_budget_exceeded_when_run_plus_reserved_hits_limit() -> None:
    """check_budget treats (current_run_cost + _reserved) as effective run usage."""
    tracker = BudgetTracker()
    tracker.record(CostInfo(cost_usd=4.0, token_usage=TokenUsage()))
    budget = Budget(max_cost=10.0)
    assert tracker.check_budget(budget).status == BudgetStatus.OK
    token = tracker.reserve(6.0)
    result = tracker.check_budget(budget)
    assert result.status == BudgetStatus.EXCEEDED
    assert result.exceeded_limit == BudgetLimitType.RUN
    token.rollback()
    assert tracker.check_budget(budget).status == BudgetStatus.OK


def test_reservation_double_commit_or_rollback_idempotent() -> None:
    """Calling commit or rollback more than once is a no-op after the first."""
    tracker = BudgetTracker()
    token = tracker.reserve(1.0)
    token.commit(0.1, TokenUsage())
    token.commit(0.2, TokenUsage())
    assert tracker.current_run_cost == 0.1
    token2 = tracker.reserve(1.0)
    token2.rollback()
    token2.rollback()
    assert tracker.check_budget(Budget(max_cost=10.0)).status == BudgetStatus.OK
