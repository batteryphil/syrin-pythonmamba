"""Comprehensive Tracing Example.

Demonstrates:
- Response trace steps (step_type, model, tokens, cost, latency)
- Session-level tracing
- Debug mode for verbose output
- Span inspection from agent.report

Run: python -m examples.10_observability.comprehensive_tracing
"""

from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv

import syrin
from examples.models.models import almock
from syrin import Agent, Budget

load_dotenv(Path(__file__).resolve().parent.parent / ".env")


# 1. Trace steps from response
agent = Agent(model=almock, system_prompt="You are a helpful assistant.")
result = agent.response("What is Python?")
print(f"Total trace steps: {len(result.trace)}")
for i, step in enumerate(result.trace):
    print(f"  Step {i + 1}: {step.step_type}, cost=${step.cost_usd:.6f}")

# 2. Multi-call trace aggregation
agent = Agent(model=almock, budget=Budget(run=1.0))
total_latency = 0.0
total_steps = 0
for q in ["What is AI?", "What is ML?", "What is DL?"]:
    result = agent.response(q)
    for step in result.trace:
        total_latency += step.latency_ms
        total_steps += 1
print(f"Total steps: {total_steps}, latency: {total_latency:.1f}ms")
print(f"Budget state: {agent.budget_state}")


# 3. Debug mode
class DebugAgent(Agent):
    model = almock
    debug = True


agent = DebugAgent()
result = agent.response("Hello debug!")
print(f"Result: {result.content[:60]}...")

# 4. Global trace config
syrin.configure(trace=True)
result = syrin.run("Traced call", model=almock)
print(f"Trace steps: {len(result.trace)}")
syrin.configure(trace=False)
