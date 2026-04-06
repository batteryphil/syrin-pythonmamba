"""Metrics aggregation system for observability.

This module provides metrics collection, aggregation, and reporting
for agent operations including costs, latency, tokens, and errors.

Example:
    >>> from syrin.observability.metrics import MetricsCollector, get_metrics
    >>>
    >>> collector = get_metrics()
    >>> collector.increment("requests.total")
    >>>
    >>> # Get aggregated metrics
    >>> cost = collector.get("cost.total")
"""

from __future__ import annotations

import threading
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

from syrin.observability import Span


@dataclass
class MetricPoint:
    """A single metric data point."""

    timestamp: datetime = field(default_factory=datetime.now)
    value: float = 0.0
    tags: dict[str, str] = field(default_factory=dict)


class MetricAggregator:
    """Aggregates metrics by operation type."""

    def __init__(self) -> None:
        self._values: dict[str, list[MetricPoint]] = defaultdict(list)
        self._lock = threading.Lock()

    def record(self, name: str, value: float, tags: dict[str, str] | None = None) -> None:
        """Record a metric value."""
        with self._lock:
            self._values[name].append(
                MetricPoint(
                    value=value,
                    tags=tags or {},
                )
            )

    def get(
        self,
        name: str,
        window: timedelta | None = None,
        tags: dict[str, str] | None = None,
    ) -> list[MetricPoint]:
        """Get metric values, optionally filtered by window and tags."""
        with self._lock:
            points = list(self._values.get(name, []))

        if window:
            cutoff = datetime.now(UTC) - window
            points = [p for p in points if p.timestamp >= cutoff]

        if tags:
            points = [p for p in points if all(p.tags.get(k) == v for k, v in tags.items())]

        return points

    def sum(self, name: str, window: timedelta | None = None) -> float:
        """Get sum of metric values."""
        points = self.get(name, window)
        return sum(p.value for p in points)

    def avg(self, name: str, window: timedelta | None = None) -> float:
        """Get average of metric values."""
        points = self.get(name, window)
        if not points:
            return 0.0
        return sum(p.value for p in points) / len(points)

    def count(self, name: str, window: timedelta | None = None) -> int:
        """Get count of metric values."""
        return len(self.get(name, window))

    def p95(self, name: str, window: timedelta | None = None) -> float:
        """Get 95th percentile of metric values."""
        points = self.get(name, window)
        if not points:
            return 0.0
        sorted_values = sorted(p.value for p in points)
        idx = int(len(sorted_values) * 0.95)
        return sorted_values[min(idx, len(sorted_values) - 1)]

    def clear(self, name: str | None = None) -> None:
        """Clear metrics."""
        with self._lock:
            if name:
                self._values.pop(name, None)
            else:
                self._values.clear()


class MetricsCollector:
    """Collects and aggregates metrics from agent operations.

    This collector automatically records metrics from spans and provides
    aggregation functions for analysis.
    """

    def __init__(self) -> None:
        self._counters: dict[str, float] = defaultdict(float)
        self._gauges: dict[str, float] = {}
        self._aggregator = MetricAggregator()
        self._lock = threading.Lock()

    def increment(self, name: str, value: float = 1.0, tags: dict[str, str] | None = None) -> None:
        """Increment a counter metric."""
        with self._lock:
            self._counters[name] += value
            self._aggregator.record(name, value, tags)

    def decrement(self, name: str, value: float = 1.0) -> None:
        """Decrement a counter metric."""
        with self._lock:
            self._counters[name] -= value

    def gauge(self, name: str, value: float, tags: dict[str, str] | None = None) -> None:
        """Set a gauge metric."""
        with self._lock:
            self._gauges[name] = value
            self._aggregator.record(name, value, tags)

    def record(self, name: str, value: float, tags: dict[str, str] | None = None) -> None:
        """Record a metric value."""
        self._aggregator.record(name, value, tags)

    def timing(self, name: str, duration_ms: float, tags: dict[str, str] | None = None) -> None:
        """Record a timing metric."""
        self._aggregator.record(f"{name}.count", 1.0, tags)
        self._aggregator.record(f"{name}.sum", duration_ms, tags)
        self._aggregator.record(f"{name}.max", duration_ms, tags)

    def get_counter(self, name: str) -> float:
        """Get current counter value."""
        return self._counters.get(name, 0.0)

    def get_gauge(self, name: str) -> float | None:
        """Get current gauge value."""
        return self._gauges.get(name)

    def get(
        self,
        name: str,
        window: timedelta | None = None,
        aggregate: str = "sum",
    ) -> float:
        """Get aggregated metric value."""
        if aggregate == "sum":
            return self._aggregator.sum(name, window)
        elif aggregate == "avg":
            return self._aggregator.avg(name, window)
        elif aggregate == "count":
            return self._aggregator.count(name, window)
        elif aggregate == "p95":
            return self._aggregator.p95(name, window)
        return 0.0

    def from_span(self, span: Span) -> None:
        """Extract metrics from a span."""
        attrs = span.attributes

        # LLM metrics
        if span.kind.value == "llm":
            tokens_input = attrs.get("llm.tokens.input", 0) or 0
            tokens_output = attrs.get("llm.tokens.output", 0) or 0
            tokens_total_attr = attrs.get("llm.tokens.total", 0) or 0
            tokens_total = (tokens_input + tokens_output) or tokens_total_attr  # type: ignore[operator]
            cost = attrs.get("llm.cost", 0)

            self.increment("llm.requests.total", tags={"model": attrs.get("llm.model", "")})  # type: ignore[dict-item]
            self.record("llm.tokens.input", tokens_input)  # type: ignore[arg-type]
            self.record("llm.tokens.output", tokens_output)  # type: ignore[arg-type]
            self.record("llm.tokens.total", tokens_total)  # type: ignore[arg-type]
            self.record("llm.cost", cost)  # type: ignore[arg-type]
            self.timing("llm.latency", span.duration_ms)

        # Tool metrics
        elif span.kind.value == "tool":
            tool_name = attrs.get("tool.name", "unknown")
            self.increment("tool.calls.total", tags={"tool": tool_name})  # type: ignore[dict-item]
            self.timing(f"tool.{tool_name}.latency", span.duration_ms)

        # Agent metrics
        elif span.kind.value == "agent":
            iterations = attrs.get("iterations", 1)
            cost = (
                attrs.get("llm.cost", 0) or attrs.get("cost.usd", 0) or attrs.get("budget.used", 0)
            )

            self.increment("agent.runs.total")
            self.record("agent.cost", cost)  # type: ignore[arg-type]
            self.record("agent.iterations", iterations)  # type: ignore[arg-type]
            self.timing("agent.latency", span.duration_ms)

            if span.status.value == "error":
                self.increment("agent.errors.total")

        # Guardrail metrics
        elif span.kind.value == "guardrail":
            passed = attrs.get("guardrail.passed", True)
            stage = attrs.get("guardrail.stage", "unknown")

            if passed:
                self.increment("guardrail.passed.total", tags={"stage": stage})  # type: ignore[dict-item]
            else:
                self.increment("guardrail.blocked.total", tags={"stage": stage})  # type: ignore[dict-item]

        # Memory metrics
        elif span.kind.value == "memory":
            op = attrs.get("memory.operation", "unknown")
            results = attrs.get("memory.results.count", 0)

            self.increment(f"memory.{op}.total")
            self.record(f"memory.{op}.results", results)  # type: ignore[arg-type]

        # Budget metrics
        elif span.kind.value == "budget":
            remaining = attrs.get("budget.remaining")
            if remaining is not None:
                self.gauge("budget.remaining", remaining)  # type: ignore[arg-type]

    def get_summary(self, window: timedelta | None = None) -> dict[str, object]:
        """Get a summary of all metrics."""
        return {
            "counters": dict(self._counters),
            "gauges": dict(self._gauges),
            "llm": {
                "requests": self.get("llm.requests.total", window),
                "tokens_input": self.get("llm.tokens.input", window),
                "tokens_output": self.get("llm.tokens.output", window),
                "tokens_total": self.get("llm.tokens.total", window),
                "cost": self.get("llm.cost", window),
                "latency_avg": self.get("llm.latency", window, "avg"),
                "latency_p95": self.get("llm.latency", window, "p95"),
            },
            "agent": {
                "runs": self.get("agent.runs.total", window),
                "errors": self.get("agent.errors.total", window),
                "cost": self.get("agent.cost", window),
                "latency_avg": self.get("agent.latency", window, "avg"),
                "latency_p95": self.get("agent.latency", window, "p95"),
            },
            "tool": {
                "calls": self.get("tool.calls.total", window),
            },
            "guardrail": {
                "passed": self.get("guardrail.passed.total", window),
                "blocked": self.get("guardrail.blocked.total", window),
            },
            "memory": {
                "recalls": self.get("memory.recall.total", window),
                "stores": self.get("memory.store.total", window),
                "forgets": self.get("memory.forget.total", window),
            },
        }

    def clear(self) -> None:
        """Clear all metrics."""
        with self._lock:
            self._counters.clear()
            self._gauges.clear()
            self._aggregator.clear()


# Singleton metrics collector
_metrics: MetricsCollector | None = None
_metrics_lock = threading.Lock()


def get_metrics() -> MetricsCollector:
    """Get the global metrics collector."""
    global _metrics
    if _metrics is None:
        with _metrics_lock:
            if _metrics is None:
                _metrics = MetricsCollector()
    return _metrics


def record_span_metrics(span: Span) -> None:
    """Record metrics from a span to the global collector."""
    collector = get_metrics()
    collector.from_span(span)


__all__ = [
    "MetricsCollector",
    "MetricAggregator",
    "MetricPoint",
    "get_metrics",
    "record_span_metrics",
]
