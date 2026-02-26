"""Circuit breaker state machine for cascading failure prevention."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from syrin.enums import CircuitState


@dataclass
class CircuitBreakerState:
    """Current state of the circuit breaker."""

    state: CircuitState
    failures: int
    last_failure_time: float | None
    last_success_time: float | None
    half_open_attempts: int


class CircuitBreaker:
    """Circuit breaker for LLM provider failures.

    States: CLOSED → OPEN (on threshold) → HALF_OPEN (after timeout) → CLOSED.

    Example:
        >>> cb = CircuitBreaker(failure_threshold=3, recovery_timeout=60)
        >>> if cb.is_open():
        ...     raise CircuitBreakerOpenError(...)
        >>> try:
        ...     result = await provider.complete(...)
        ...     cb.record_success()
        ... except Exception as e:
        ...     cb.record_failure(e)
        ...     raise
    """

    def __init__(
        self,
        *,
        failure_threshold: int = 5,
        recovery_timeout: int = 60,
        half_open_max: int = 1,
        fallback: str | Any = None,
        on_trip: Callable[[CircuitBreakerState], None] | None = None,
    ) -> None:
        if failure_threshold < 1:
            raise ValueError(f"failure_threshold must be >= 1, got {failure_threshold}")
        if recovery_timeout < 1:
            raise ValueError(f"recovery_timeout must be >= 1, got {recovery_timeout}")
        if half_open_max < 1:
            raise ValueError(f"half_open_max must be >= 1, got {half_open_max}")
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_max = half_open_max
        self._fallback = fallback
        self._on_trip = on_trip

        self._state = CircuitState.CLOSED
        self._failures = 0
        self._last_failure_time: float | None = None
        self._last_success_time: float | None = None
        self._half_open_attempts = 0

    @property
    def fallback(self) -> str | Any | None:
        return self._fallback

    def get_state(self) -> CircuitBreakerState:
        """Return current circuit breaker state."""
        return CircuitBreakerState(
            state=self._state,
            failures=self._failures,
            last_failure_time=self._last_failure_time,
            last_success_time=self._last_success_time,
            half_open_attempts=self._half_open_attempts,
        )

    def is_open(self) -> bool:
        """Return True if circuit is open (blocking requests)."""
        now = time.monotonic()

        if self._state == CircuitState.CLOSED:
            return False

        if self._state == CircuitState.OPEN:
            elapsed = now - (self._last_failure_time or 0)
            if elapsed >= self.recovery_timeout:
                self._state = CircuitState.HALF_OPEN
                self._half_open_attempts = 0
                return False
            return True

        if self._state == CircuitState.HALF_OPEN:
            return False

        return False

    def record_success(self) -> None:
        """Record a successful LLM call. Resets failures when closed."""
        now = time.monotonic()
        self._last_success_time = now

        if self._state == CircuitState.HALF_OPEN:
            self._state = CircuitState.CLOSED
            self._failures = 0
            self._half_open_attempts = 0
        elif self._state == CircuitState.CLOSED:
            self._failures = 0

    def record_failure(self, error: Exception | None = None) -> None:
        """Record a failed LLM call. Trips when threshold reached."""
        now = time.monotonic()
        self._last_failure_time = now

        if self._state == CircuitState.HALF_OPEN:
            self._state = CircuitState.OPEN
            self._half_open_attempts = 0
            self._emit_trip()
        elif self._state == CircuitState.CLOSED:
            self._failures += 1
            if self._failures >= self.failure_threshold:
                self._state = CircuitState.OPEN
                self._emit_trip()

    def allow_request(self) -> bool:
        """Return True if a request is allowed (circuit closed or half-open with capacity)."""
        now = time.monotonic()

        if self._state == CircuitState.CLOSED:
            return True

        if self._state == CircuitState.OPEN:
            elapsed = now - (self._last_failure_time or 0)
            if elapsed >= self.recovery_timeout:
                self._state = CircuitState.HALF_OPEN
                self._half_open_attempts = 1
                return True
            return False

        if self._state == CircuitState.HALF_OPEN:
            if self._half_open_attempts < self.half_open_max:
                self._half_open_attempts += 1
                return True
            return False

        return False

    def _emit_trip(self) -> None:
        if self._on_trip is not None:
            self._on_trip(self.get_state())
