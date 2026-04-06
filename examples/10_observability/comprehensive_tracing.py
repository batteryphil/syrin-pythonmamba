"""Comprehensive Tracing Example.

Demonstrates:
- Response trace steps (step_type, model, tokens, cost, latency)
- Session-level tracing
- Debug mode for verbose output
- Span inspection from agent.report

Run: python -m examples.10_observability.comprehensive_tracing
Visit: http://localhost:8000/playground
Requires: uv pip install syrin[serve]
"""

from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv

import syrin
from syrin import Agent, Budget, Model

load_dotenv(Path(__file__).resolve().parent.parent / ".env")


# 1. Trace steps from response
agent = Agent(
    model=Model.mock(latency_min=1, latency_max=3, lorem_length=800, pricing_tier="high"),
    system_prompt="You are a helpful assistant.",
)
result = agent.run("What is Python?")
print(f"Total trace steps: {len(result.trace)}")
for i, step in enumerate(result.trace):
    print(f"  Step {i + 1}: {step.step_type}, cost=${step.cost_usd:.6f}")

# 2. Multi-call trace aggregation
agent = Agent(
    model=Model.mock(latency_min=1, latency_max=3, lorem_length=800, pricing_tier="high"),
    budget=Budget(max_cost=1.0),
)
total_latency = 0.0
total_steps = 0
for q in ["What is AI?", "What is ML?", "What is DL?"]:
    result = agent.run(q)
    for step in result.trace:
        total_latency += step.latency_ms
        total_steps += 1
print(f"Total steps: {total_steps}, latency: {total_latency:.1f}ms")
print(f"Budget state: {agent.budget_state}")


# 3. Debug mode
class DebugAgent(Agent):
    model = Model.mock(latency_min=1, latency_max=3, lorem_length=800, pricing_tier="high")
    debug = True


agent = DebugAgent()
result = agent.run("Hello debug!")
print(f"Result: {result.content[:60]}...")

# 4. Global trace config
syrin.configure(trace=True)
result = syrin.run(
    "Traced call",
    model=Model.mock(latency_min=1, latency_max=3, lorem_length=800, pricing_tier="high"),
)
print(f"Trace steps: {len(result.trace)}")
syrin.configure(trace=False)


class TracingDemoAgent(Agent):
    name = "tracing-demo"
    description = "Agent with comprehensive tracing"
    model = Model.mock(latency_min=1, latency_max=3, lorem_length=800, pricing_tier="high")
    system_prompt = "You are a helpful assistant."
    debug = True


if __name__ == "__main__":
    agent = TracingDemoAgent()
    print("Serving at http://localhost:8000/playground")
    agent.serve(port=8000, enable_playground=True, debug=True)
