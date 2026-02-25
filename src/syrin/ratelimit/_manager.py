"""Rate limit manager implementation."""

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from syrin.ratelimit.backends import (
    RateLimitBackend,
    RateLimitState,
)
from syrin.ratelimit.config import APIRateLimit, RateLimitStats
from syrin.threshold import RateLimitThreshold, ThresholdContext

_SEC_PER_MINUTE = 60.0
_SEC_PER_DAY = 86400.0


@dataclass
class RateLimitEntry:
    """Single usage entry for rate tracking."""

    requests: int = 0
    tokens: int = 0
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {"requests": self.requests, "tokens": self.tokens, "timestamp": self.timestamp}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RateLimitEntry":
        return cls(
            requests=data.get("requests", 0),
            tokens=data.get("tokens", 0),
            timestamp=data.get("timestamp", time.time()),
        )


@runtime_checkable
class RateLimitManager(Protocol):
    """Protocol for custom rate limit management strategies."""

    def check(self, tokens_used: int = 0) -> tuple[bool, str]:
        """Check if request is allowed.

        Returns:
            (allowed, reason)
        """
        ...

    def record(self, tokens_used: int = 0) -> None:
        """Record a request for rate tracking."""
        ...

    def get_stats(self) -> RateLimitStats:
        """Get current rate limit statistics."""
        ...

    def get_triggered_action(self) -> Any:
        """Get the currently triggered threshold action if any."""
        ...

    @property
    def config(self) -> Any:
        """Get the rate limit configuration."""
        ...


class _NullSpan:
    """Null context manager for when no tracer is set."""

    def __enter__(self) -> "_NullSpan":
        return self

    def __exit__(self, *args: Any) -> None:
        pass

    def set_attribute(self, key: str, value: Any) -> None:
        pass


class _RateLimitSpan:
    """Wrapper for tracer spans."""

    def __init__(self, tracer: Any, name: str, **attrs: Any):
        self._tracer = tracer
        self._name = name
        self._attrs = attrs
        self._span: Any | None = None

    def __enter__(self) -> Any:
        if hasattr(self._tracer, "span"):
            self._span = self._tracer.span(self._name)
            for k, v in self._attrs.items():
                if self._span is not None:
                    self._span.set_attribute(k, v)
            if self._span is not None:
                return self._span.__enter__()
        return self

    def __exit__(self, *args: Any) -> None:
        if self._span is not None:
            self._span.__exit__(*args)

    def set_attribute(self, key: str, value: Any) -> None:
        if self._span is not None:
            self._span.set_attribute(key, value)


@dataclass
class DefaultRateLimitManager:
    """Default rate limit manager with threshold actions.

    Features:
    - Tracks RPM, TPM, RPD usage with rolling windows
    - Threshold-based actions via custom functions
    - Full observability via events and spans
    - Persistence backends for multi-instance deployments
    """

    config: APIRateLimit = field(default_factory=APIRateLimit)
    _entries: list[RateLimitEntry] = field(default_factory=list)
    _emit_fn: Callable[[str, dict[str, Any]], None] | None = field(default=None, repr=False)
    _tracer: Any = field(default=None, repr=False)
    _stats: RateLimitStats = field(default_factory=RateLimitStats)
    _backend: RateLimitBackend | None = field(default=None, repr=False)
    _key: str = "default"

    def _emit(self, event: str, ctx: dict[str, Any]) -> None:
        """Emit an event if emit_fn is configured."""
        if self._emit_fn:
            self._emit_fn(event, ctx)

    def _span(self, name: str, **attrs: Any) -> _RateLimitSpan | _NullSpan:
        """Create a span if tracer is configured."""
        if self._tracer is None:
            return _NullSpan()
        return _RateLimitSpan(self._tracer, name, **attrs)

    def set_emit_fn(self, emit_fn: Callable[[str, dict[str, Any]], None]) -> None:
        """Set the event emit function for lifecycle hooks."""
        self._emit_fn = emit_fn

    def set_tracer(self, tracer: Any) -> None:
        """Set the tracer for observability."""
        self._tracer = tracer

    def set_backend(self, backend: RateLimitBackend, key: str = "default") -> None:
        """Set a persistence backend for multi-instance support.

        Args:
            backend: RateLimitBackend instance (memory, sqlite, redis)
            key: Key to use for storage (useful for multiple agents)
        """
        self._backend = backend
        self._key = key

    def save(self) -> None:
        """Save current state to backend (if configured)."""
        if self._backend is None:
            return
        self._prune_old_entries()
        state = RateLimitState(
            entries=[e.to_dict() for e in self._entries],
            last_updated=time.time(),
        )
        self._backend.save(self._key, state)

    def load(self) -> bool:
        """Load state from backend (if configured).

        Returns:
            True if state was loaded, False if not found or not configured
        """
        if self._backend is None:
            return False
        state = self._backend.load(self._key)
        if state:
            self._entries = [RateLimitEntry.from_dict(e) for e in state.entries]
            return True
        return False

    def _prune_old_entries(self) -> None:
        """Remove entries older than 1 day."""
        now = time.time()
        cutoff = now - _SEC_PER_DAY
        self._entries = [e for e in self._entries if e.timestamp >= cutoff]

    def _current_rpm(self) -> int:
        """Get current requests per minute."""
        self._prune_old_entries()
        now = time.time()
        cutoff = now - _SEC_PER_MINUTE
        return sum(e.requests for e in self._entries if e.timestamp >= cutoff)

    def _current_tpm(self) -> int:
        """Get current tokens per minute."""
        self._prune_old_entries()
        now = time.time()
        cutoff = now - _SEC_PER_MINUTE
        return sum(e.tokens for e in self._entries if e.timestamp >= cutoff)

    def _current_rpd(self) -> int:
        """Get current requests per day."""
        self._prune_old_entries()
        return sum(e.requests for e in self._entries)

    def record(self, tokens_used: int = 0) -> None:
        """Record a request for rate tracking."""
        self._entries.append(RateLimitEntry(requests=1, tokens=tokens_used, timestamp=time.time()))
        self._update_stats()
        if self._backend is not None:
            self.save()

    def _update_stats(self) -> None:
        """Update rate limit statistics."""
        rpm_used = self._current_rpm()
        tpm_used = self._current_tpm()
        rpd_used = self._current_rpd()

        self._stats = RateLimitStats(
            rpm_used=rpm_used,
            rpm_limit=self.config.rpm or 0,
            tpm_used=tpm_used,
            tpm_limit=self.config.tpm or 0,
            rpd_used=rpd_used,
            rpd_limit=self.config.rpd or 0,
        )

    def check(self, _tokens_used: int = 0) -> tuple[bool, str]:
        """Check if request is allowed.

        Returns:
            (allowed, reason)
        """
        with self._span("ratelimit.check") as span:
            if span:
                span.set_attribute("ratelimit.rpm", self._current_rpm())
                span.set_attribute("ratelimit.tpm", self._current_tpm())

            # Check thresholds - they emit events but don't block
            self._check_thresholds()

            # Check actual limits - these always block
            rpm = self._current_rpm()
            tpm = self._current_tpm()
            rpd = self._current_rpd()

            if self.config.rpm and rpm >= self.config.rpm:
                self._emit(
                    "ratelimit.exceeded", {"metric": "rpm", "used": rpm, "limit": self.config.rpm}
                )
                return False, f"RPM exceeded: {rpm}/{self.config.rpm}"

            if self.config.tpm and tpm >= self.config.tpm:
                self._emit(
                    "ratelimit.exceeded", {"metric": "tpm", "used": tpm, "limit": self.config.tpm}
                )
                return False, f"TPM exceeded: {tpm}/{self.config.tpm}"

            if self.config.rpd and rpd >= self.config.rpd:
                self._emit(
                    "ratelimit.exceeded", {"metric": "rpd", "used": rpd, "limit": self.config.rpd}
                )
                return False, f"RPD exceeded: {rpd}/{self.config.rpd}"

            return True, "OK"

    def _check_thresholds(self) -> list[str]:
        """Check and trigger thresholds.

        Returns list of triggered threshold actions (as strings for backwards compat).
        """
        triggered = []
        rpm = self._current_rpm()
        tpm = self._current_tpm()
        rpd = self._current_rpd()

        for threshold in self.config.thresholds:
            # Get metric values
            metric_values = {
                "rpm": (rpm, self.config.rpm or 0),
                "tpm": (tpm, self.config.tpm or 0),
                "rpd": (rpd, self.config.rpd or 0),
            }

            used, limit = metric_values.get(threshold.metric, (0, 0))

            if limit <= 0:
                continue

            percentage = int((used / limit) * 100)

            if percentage >= threshold.at:
                triggered.append(threshold.metric)

                threshold_event = {
                    "at": threshold.at,
                    "percent": percentage,
                    "metric": threshold.metric,
                    "used": used,
                    "limit": limit,
                }
                self._emit("ratelimit.threshold", threshold_event)

                # Execute the threshold action
                ctx = ThresholdContext(
                    percentage=percentage,
                    metric=threshold.metric,
                    current_value=float(used),
                    limit_value=float(limit),
                )
                threshold.execute(ctx)

        self._stats.thresholds_triggered = triggered
        return triggered

    def get_triggered_threshold(self) -> RateLimitThreshold | None:
        """Get the highest priority triggered threshold.

        Returns the first threshold that should be executed.
        """
        rpm = self._current_rpm()
        tpm = self._current_tpm()
        rpd = self._current_rpd()

        for threshold in self.config.thresholds:
            metric_values = {
                "rpm": (rpm, self.config.rpm or 0),
                "tpm": (tpm, self.config.tpm or 0),
                "rpd": (rpd, self.config.rpd or 0),
            }

            used, limit = metric_values.get(threshold.metric, (0, 0))

            if limit <= 0:
                continue

            percentage = int((used / limit) * 100)

            if percentage >= threshold.at:
                return threshold

        return None

    def get_triggered_action(self) -> Any:
        """Get the currently triggered threshold action if any.

        Threshold behavior is fully controlled by the user's action callback
        (executed in _check_thresholds). This method returns None; the Protocol
        retains it for compatibility.
        """
        return None

    @property
    def stats(self) -> RateLimitStats:
        """Get current rate limit statistics."""
        self._update_stats()
        return self._stats

    @property
    def current_rpm(self) -> int:
        """Get current RPM usage."""
        return self._current_rpm()

    @property
    def current_tpm(self) -> int:
        """Get current TPM usage."""
        return self._current_tpm()

    @property
    def current_rpd(self) -> int:
        """Get current RPD usage."""
        return self._current_rpd()


def create_rate_limit_manager(
    config: APIRateLimit,
    emit_fn: Callable[[str, dict[str, Any]], None] | None = None,
    tracer: Any = None,
) -> DefaultRateLimitManager:
    """Create a default rate limit manager from config.

    Args:
        config: APIRateLimit configuration (required)
        emit_fn: Optional event emit function for lifecycle hooks
        tracer: Optional tracer for observability

    Returns:
        Configured DefaultRateLimitManager
    """
    manager = DefaultRateLimitManager(config=config)
    if emit_fn:
        manager.set_emit_fn(emit_fn)
    if tracer:
        manager.set_tracer(tracer)
    return manager


__all__ = [
    "RateLimitEntry",
    "RateLimitManager",
    "DefaultRateLimitManager",
    "create_rate_limit_manager",
]
