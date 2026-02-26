"""Tests for circuit breaker state machine (TDD)."""

from __future__ import annotations

import time
from unittest.mock import MagicMock

import pytest

from syrin.circuit import CircuitBreaker
from syrin.enums import CircuitState
from syrin.exceptions import CircuitBreakerOpenError


class TestCircuitBreakerClosed:
    """Circuit starts CLOSED; allows requests."""

    def test_closed_allows_requests(self) -> None:
        cb = CircuitBreaker(failure_threshold=3, recovery_timeout=60)
        assert cb.allow_request() is True
        assert cb.get_state().state == CircuitState.CLOSED

    def test_success_resets_failures(self) -> None:
        cb = CircuitBreaker(failure_threshold=3, recovery_timeout=60)
        cb.record_failure(Exception("err"))
        cb.record_failure(Exception("err"))
        cb.record_success()
        assert cb.get_state().failures == 0
        assert cb.get_state().state == CircuitState.CLOSED


class TestCircuitBreakerTrip:
    """Failures reach threshold → circuit trips to OPEN."""

    def test_trip_after_threshold(self) -> None:
        cb = CircuitBreaker(failure_threshold=3, recovery_timeout=60)
        cb.record_failure(Exception("1"))
        cb.record_failure(Exception("2"))
        assert cb.get_state().state == CircuitState.CLOSED
        cb.record_failure(Exception("3"))
        assert cb.get_state().state == CircuitState.OPEN

    def test_open_blocks_requests(self) -> None:
        cb = CircuitBreaker(failure_threshold=2, recovery_timeout=10)
        cb.record_failure(Exception("1"))
        cb.record_failure(Exception("2"))
        assert cb.allow_request() is False
        assert cb.is_open() is True

    def test_on_trip_callback_fired(self) -> None:
        on_trip = MagicMock()
        cb = CircuitBreaker(failure_threshold=2, recovery_timeout=10, on_trip=on_trip)
        cb.record_failure(Exception("1"))
        on_trip.assert_not_called()
        cb.record_failure(Exception("2"))
        on_trip.assert_called_once()
        state = on_trip.call_args[0][0]
        assert state.state == CircuitState.OPEN
        assert state.failures == 2


class TestCircuitBreakerRecovery:
    """After recovery_timeout, circuit goes HALF_OPEN."""

    def test_half_open_after_timeout(self) -> None:
        cb = CircuitBreaker(failure_threshold=1, recovery_timeout=1)
        cb.record_failure(Exception("1"))
        assert cb.allow_request() is False

        time.sleep(1.1)
        assert cb.allow_request() is True
        assert cb.get_state().state == CircuitState.HALF_OPEN

    def test_half_open_success_closes_circuit(self) -> None:
        cb = CircuitBreaker(failure_threshold=1, recovery_timeout=1)
        cb.record_failure(Exception("1"))
        time.sleep(1.1)
        assert cb.allow_request() is True
        cb.record_success()
        assert cb.get_state().state == CircuitState.CLOSED
        assert cb.get_state().failures == 0

    def test_half_open_failure_reopens(self) -> None:
        cb = CircuitBreaker(failure_threshold=1, recovery_timeout=1)
        cb.record_failure(Exception("1"))
        time.sleep(1.1)
        assert cb.allow_request() is True
        cb.record_failure(Exception("2"))
        assert cb.get_state().state == CircuitState.OPEN
        assert cb.allow_request() is False


class TestCircuitBreakerHalfOpenLimit:
    """HALF_OPEN allows at most half_open_max probes."""

    def test_half_open_max_one_by_default(self) -> None:
        cb = CircuitBreaker(failure_threshold=1, recovery_timeout=1, half_open_max=1)
        cb.record_failure(Exception("1"))
        time.sleep(1.1)
        assert cb.allow_request() is True
        assert cb.allow_request() is False
        cb.record_success()
        assert cb.allow_request() is True

    def test_half_open_max_two(self) -> None:
        cb = CircuitBreaker(failure_threshold=1, recovery_timeout=1, half_open_max=2)
        cb.record_failure(Exception("1"))
        time.sleep(1.1)
        assert cb.allow_request() is True
        assert cb.allow_request() is True
        assert cb.allow_request() is False


class TestCircuitBreakerInvalidConfig:
    """Invalid config raises."""

    def test_failure_threshold_zero_invalid(self) -> None:
        with pytest.raises(ValueError, match="failure_threshold"):
            CircuitBreaker(failure_threshold=0)

    def test_recovery_timeout_zero_invalid(self) -> None:
        with pytest.raises(ValueError, match="recovery_timeout"):
            CircuitBreaker(recovery_timeout=0)


class TestCircuitBreakerOpenError:
    """CircuitBreakerOpenError has required attributes."""

    def test_error_attributes(self) -> None:
        err = CircuitBreakerOpenError(
            "Circuit open",
            agent_name="TestAgent",
            recovery_at=123.0,
            fallback_model="ollama/llama3",
        )
        assert err.agent_name == "TestAgent"
        assert err.recovery_at == 123.0
        assert err.fallback_model == "ollama/llama3"
