"""Integration tests: Agent with CircuitBreaker trips and uses fallback."""

from unittest.mock import AsyncMock, patch

import pytest

from syrin import Agent, CircuitBreaker, Model
from syrin.exceptions import CircuitBreakerOpenError, ProviderError


def _almock() -> Model:
    return Model.Almock(latency_seconds=0.01, lorem_length=50)


class TestCircuitBreakerAgent:
    """Agent with circuit breaker trips after failures, uses fallback when open."""

    def test_agent_without_circuit_breaker_runs_normally(self) -> None:
        """Agent without circuit breaker runs as usual."""
        agent = Agent(model=_almock(), system_prompt="Hi")
        r = agent.response("Hello")
        assert r.content is not None

    def test_agent_with_circuit_breaker_success_resets_failures(self) -> None:
        """Successful calls keep circuit closed."""
        cb = CircuitBreaker(failure_threshold=3, recovery_timeout=60)
        agent = Agent(model=_almock(), system_prompt="Hi", circuit_breaker=cb)
        agent.response("Hello")
        agent.response("Hi again")
        assert cb.get_state().state.value == "closed"
        assert cb.get_state().failures == 0

    def test_agent_circuit_trips_after_failures(self) -> None:
        """Circuit trips after failure_threshold provider errors."""
        cb = CircuitBreaker(failure_threshold=2, recovery_timeout=10)
        agent = Agent(model=_almock(), system_prompt="Hi", circuit_breaker=cb)
        with patch.object(
            agent._provider,
            "complete",
            new_callable=AsyncMock,
            side_effect=ProviderError("API down"),
        ):
            with pytest.raises(ProviderError):
                agent.response("Hello")
            assert cb.get_state().failures == 1
            with pytest.raises(ProviderError):
                agent.response("Hi")
            assert cb.get_state().state.value == "open"

    def test_agent_circuit_open_no_fallback_raises(self) -> None:
        """When circuit open and no fallback, raises CircuitBreakerOpenError."""
        cb = CircuitBreaker(failure_threshold=1, recovery_timeout=60)
        agent = Agent(model=_almock(), system_prompt="Hi", circuit_breaker=cb)
        with (
            patch.object(
                agent._provider,
                "complete",
                new_callable=AsyncMock,
                side_effect=ProviderError("API down"),
            ),
            pytest.raises(ProviderError),
        ):
            agent.response("Hello")
        with pytest.raises(CircuitBreakerOpenError):
            agent.response("Second call")

    def test_agent_circuit_open_with_fallback_uses_fallback(self) -> None:
        """When circuit open and fallback set, uses fallback model."""
        fallback_model = Model.Almock(latency_seconds=0.01, lorem_length=30)
        cb = CircuitBreaker(
            failure_threshold=1,
            recovery_timeout=60,
            fallback=fallback_model,
        )
        agent = Agent(
            model=_almock(),
            system_prompt="Hi",
            circuit_breaker=cb,
        )
        with (
            patch.object(
                agent._provider,
                "complete",
                new_callable=AsyncMock,
                side_effect=ProviderError("API down"),
            ),
            pytest.raises(ProviderError),
        ):
            agent.response("Hello")
        r = agent.response("Second call - should use fallback")
        assert r.content is not None

    def test_agent_circuit_breaker_invalid_type_rejected(self) -> None:
        """Agent rejects non-CircuitBreaker for circuit_breaker."""
        with pytest.raises(TypeError, match="circuit_breaker must be CircuitBreaker"):
            Agent(model=_almock(), circuit_breaker="invalid")
