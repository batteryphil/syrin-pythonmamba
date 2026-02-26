"""Circuit breaker for LLM provider failures."""

from syrin.circuit._breaker import CircuitBreaker, CircuitBreakerState

__all__ = ["CircuitBreaker", "CircuitBreakerState"]
