"""Unified threshold system for syrin.

Provides separate threshold classes for different use cases.

Example:
    >>> from syrin import Budget
    >>> from syrin.threshold import BudgetThreshold
    >>>
    >>> budget = Budget(
    ...     run=10.0,
    ...     thresholds=[
    ...         BudgetThreshold(at=80, action=lambda ctx: print(f"At {ctx.percentage}%"))
    ...     ]
    ... )
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, TypeVar

from syrin.enums import ThresholdMetric, ThresholdWindow


def _noop_compact() -> None:
    """No-op compactor when not inside context prepare."""
    pass


def compact_if_available(ctx: ThresholdContext) -> None:
    """Threshold action: run context compaction when triggered (e.g. during prepare). No-op otherwise.

    Use as the action for ContextThreshold when you want to compact at a utilization percentage
    without writing a lambda. Only has effect when the context manager is inside prepare (when
    the threshold is evaluated); otherwise ctx.compact() is a no-op.

    Example:
        >>> from syrin.threshold import ContextThreshold, compact_if_available
        >>> ContextThreshold(at=75, action=compact_if_available)
    """
    ctx.compact()


@dataclass
class ThresholdContext:
    """Event object passed to threshold actions (Budget, Context, RateLimit).

    Use when you need to react at a utilization percentage (e.g. compact at 75%).
    For context thresholds, call **compact()** to run compaction.
    """

    percentage: int
    """Utilization percentage (0-100) that triggered this threshold."""
    metric: Any  # ThresholdMetric
    """Metric being tracked (e.g. ThresholdMetric.TOKENS, ThresholdMetric.COST)."""
    current_value: float
    """Current value (e.g. tokens used, cost so far)."""
    limit_value: float
    """Limit or cap (e.g. max_tokens, run budget)."""
    budget_run: float = 0.0
    """Alias for limit_value (for COST metric)."""
    parent: Any = None
    """Parent object (e.g. Agent, Budget) when available."""
    metadata: dict[str, Any] = field(default_factory=dict)
    """Extra key-value data."""
    compact: Callable[[], None] = field(default=_noop_compact)
    """(Context only.) Call to compact; no-op when not inside prepare."""


# Alias: event passed to threshold action (avoids confusion with Context config).


# Type alias for threshold action handlers
def _threshold_action_handler() -> None:
    pass


ThresholdAction = type(_threshold_action_handler)


T = TypeVar("T", bound="BaseThreshold")


@dataclass
class BaseThreshold:
    """Base class for thresholds."""

    at: int = 0
    action: Any = None  # Callable[[ThresholdContext], None]
    metric: Any = None  # ThresholdMetric
    at_range: tuple[int, int] | None = None  # (min, max): trigger when min <= pct <= max

    def __post_init__(self) -> None:
        if self.at_range is not None:
            lo, hi = self.at_range
            if not 0 <= lo <= hi <= 100:
                raise ValueError(
                    f"Threshold 'at_range' must be (0-100, 0-100) with lo<=hi, got {self.at_range}"
                )
        else:
            if not 0 <= self.at <= 100:
                raise ValueError(f"Threshold 'at' must be between 0 and 100, got {self.at}")
        if self.action is None:
            raise ValueError("Threshold 'action' is required")

    def should_trigger(self, percentage: int, metric: Any = None) -> bool:
        """Check if this threshold should trigger."""
        if self.at_range is not None:
            lo, hi = self.at_range
            in_range = lo <= percentage <= hi
        else:
            in_range = percentage >= self.at
        if metric is not None and hasattr(self.metric, "value") and hasattr(metric, "value"):
            return in_range and str(self.metric.value) == str(metric.value)
        return in_range

    def execute(self, ctx: ThresholdContext) -> None:
        self.action(ctx)


@dataclass
class BudgetThreshold(BaseThreshold):
    """Threshold for Budget (cost in USD or tokens).

    Args:
        at: Percentage (0-100) at which to trigger (use when pct >= at).
        at_range: Optional (min, max): trigger only when min <= pct <= max.
        action: Function to call when threshold is crossed. Receives ThresholdContext.
        metric: ThresholdMetric.COST (default) or ThresholdMetric.TOKENS.
        window: ThresholdWindow.RUN (default), HOUR, DAY, WEEK, or MONTH.
            For TOKENS, limits come from TokenLimits (run and per=TokenRateLimit).

    Example:
        >>> BudgetThreshold(at=80, action=lambda ctx: print(f"At {ctx.percentage}%"))
        >>> BudgetThreshold(at=80, action=warn, window=ThresholdWindow.DAY)
        >>> BudgetThreshold(at=80, metric=ThresholdMetric.TOKENS, window=ThresholdWindow.HOUR, action=...)
        >>> BudgetThreshold(at_range=(70, 75), action=alert)
    """

    metric: Any = ThresholdMetric.COST
    window: ThresholdWindow = ThresholdWindow.RUN

    def __post_init__(self) -> None:
        super().__post_init__()
        if isinstance(self.window, str):
            self.window = ThresholdWindow(self.window)
        if self.window not in (
            ThresholdWindow.RUN,
            ThresholdWindow.HOUR,
            ThresholdWindow.DAY,
            ThresholdWindow.WEEK,
            ThresholdWindow.MONTH,
        ):
            raise ValueError(
                f"BudgetThreshold window must be ThresholdWindow.RUN/HOUR/DAY/WEEK/MONTH, got {self.window!r}"
            )


@dataclass
class ContextThreshold(BaseThreshold):
    """Threshold for Context (tracks token usage vs max_tokens).

    Same shape as BudgetThreshold for consistency: at, at_range, action, metric, window.
    Context only supports window=MAX_TOKENS (current context window).

    Args:
        at: Percentage (0-100) at which to trigger (use when pct >= at).
        at_range: Optional (min, max): trigger only when min <= pct <= max.
        action: Function to call when threshold is crossed. Receives ThresholdContext.
            Use ctx.compact() inside the action to compact context when triggered.
        metric: ThresholdMetric.TOKENS (fixed for context).
        window: ThresholdWindow.MAX_TOKENS only (current context window).

    Example:
        >>> from syrin.threshold import ContextThreshold, compact_if_available
        >>> ContextThreshold(at=80, action=lambda ctx: print(f"At {ctx.percentage}%"))
        >>> ContextThreshold(at=75, action=compact_if_available)
        >>> ContextThreshold(at_range=(70, 75), action=warn)
    """

    metric: Any = ThresholdMetric.TOKENS
    window: Any = ThresholdWindow.MAX_TOKENS

    def __post_init__(self) -> None:
        super().__post_init__()
        w = self.window if hasattr(self.window, "value") else ThresholdWindow(str(self.window))
        if w != ThresholdWindow.MAX_TOKENS:
            raise ValueError(
                f"ContextThreshold window must be ThresholdWindow.MAX_TOKENS, got {self.window!r}"
            )


@dataclass
class RateLimitThreshold(BaseThreshold):
    """Threshold for APIRateLimit (tracks RPM, TPM, or RPD).

    Requires metric to be specified. Use the action callback to do anything when
    the threshold is crossed (e.g. raise, wait, switch model, log).

    Args:
        at: Percentage (0-100) at which to trigger
        action: Function to call when threshold is crossed. Receives ThresholdContext.
        metric: The rate limit metric (ThresholdMetric.RPM, TPM, or RPD)

    Example:
        >>> from syrin.threshold import RateLimitThreshold
        >>> from syrin.enums import ThresholdMetric
        >>>
        >>> def on_threshold(ctx):
        ...     if ctx.percentage >= 80:
        ...         raise RuntimeError("Rate limit threshold reached")
        >>> RateLimitThreshold(at=80, action=on_threshold, metric=ThresholdMetric.RPM)
    """

    metric: Any = None

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.metric is None:
            raise ValueError(
                "RateLimitThreshold 'metric' is required - use ThresholdMetric.RPM, TPM, or RPD"
            )
        valid_metrics = {ThresholdMetric.RPM, ThresholdMetric.TPM, ThresholdMetric.RPD}
        metric_val = self.metric.value if hasattr(self.metric, "value") else self.metric
        if metric_val not in valid_metrics:
            raise ValueError(f"RateLimitThreshold only supports {valid_metrics}, got {metric_val}")


__all__ = [
    "ThresholdContext",
    "ThresholdAction",
    "BaseThreshold",
    "BudgetThreshold",
    "ThresholdWindow",
    "ContextThreshold",
    "RateLimitThreshold",
    "compact_if_available",
]
