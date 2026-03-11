"""Circuit Breaker Example.

Demonstrates:
- Circuit breaker for LLM provider failures
- Trip after N consecutive failures, then use a fallback model
- Hooks: CIRCUIT_TRIP, CIRCUIT_RESET

Run: python examples/15_advanced/circuit_breaker.py
"""

from syrin import Agent, CircuitBreaker, Hook, Model

# Primary model (Almock for demo; in production use Model.Anthropic(...) etc.)
primary = Model.Almock(latency_seconds=0.01, lorem_length=50)

# Fallback model when circuit trips (e.g., a cheaper or local model)
fallback = Model.Almock(latency_seconds=0.01, lorem_length=30)

cb = CircuitBreaker(
    failure_threshold=3,
    recovery_timeout=30,
    fallback=fallback,
)


class CircuitBreakerAgent(Agent):
    _agent_name = "circuit-breaker"
    _agent_description = "Agent with circuit breaker for LLM failures"
    model = primary
    system_prompt = "You are helpful."
    circuit_breaker = cb


if __name__ == "__main__":
    agent = CircuitBreakerAgent()

    # Listen for circuit breaker lifecycle hooks
    agent.events.on(Hook.CIRCUIT_TRIP, lambda c: print(f"  [CIRCUIT TRIP]  {c}"))
    agent.events.on(Hook.CIRCUIT_RESET, lambda _: print("  [CIRCUIT RESET]"))

    print("--- Circuit Breaker Example ---")
    r = agent.response("What is 2+2?")
    print(f"Response: {r.content[:120]}")
    print(f"Cost: ${r.cost:.6f}")
    print("Done.")

    # Optional: serve the agent with playground UI
    # agent.serve(port=8000, enable_playground=True, debug=True)
