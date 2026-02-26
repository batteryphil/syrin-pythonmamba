"""Circuit Breaker Example (v0.3.0+).

Demonstrates:
- Circuit breaker for LLM provider failures
- Trip after N failures, use fallback model when open
- Hooks: CIRCUIT_TRIP, CIRCUIT_RESET

Run: python -m examples.15_advanced.circuit_breaker
"""

from pathlib import Path

from dotenv import load_dotenv

from examples.models.models import almock
from syrin import Agent, CircuitBreaker, Hook, Model

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

# Primary model (Almock for demo; in production: Model.Anthropic(...))
# Fallback when circuit trips: local Ollama or cheaper model
fallback = Model.Almock(latency_seconds=0.01, lorem_length=30)

cb = CircuitBreaker(
    failure_threshold=3,
    recovery_timeout=30,
    fallback=fallback,
)

agent = Agent(
    model=almock,
    system_prompt="You are helpful.",
    circuit_breaker=cb,
)

# Hook into circuit events
agent.events.on(Hook.CIRCUIT_TRIP, lambda c: print(f"[CIRCUIT TRIP] {c}"))
agent.events.on(Hook.CIRCUIT_RESET, lambda _: print("[CIRCUIT RESET]"))

# Normal run
r = agent.response("What is 2+2?")
print(f"Response: {r.content[:80]}...")
