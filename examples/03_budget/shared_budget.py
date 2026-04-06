"""Shared Budget — Multiple agents drawing from one budget pool.

When you pass a Budget to a Swarm or to a parent agent that spawns children,
budget sharing is automatic. No extra flags needed.

Demonstrates:
- Parent agent spawning children — budget borrowed automatically
- Swarm with a shared Budget pool
- Budget tracking across parent and child agents

No API key needed (uses Almock).

Run:
    python examples/03_budget/shared_budget.py
"""

from __future__ import annotations

import asyncio

from syrin import Agent, Budget, ExceedPolicy, Model
from syrin.enums import SwarmTopology
from syrin.swarm import Swarm, SwarmConfig

model = Model.mock()

# ---------------------------------------------------------------------------
# 1. Parent agent spawning a child — child borrows parent's remaining budget
# ---------------------------------------------------------------------------
print("=" * 60)
print("1. Parent spawns a child — budget borrowed automatically")
print("=" * 60)

parent = Agent(model=model, budget=Budget(max_cost=10.0, exceed_policy=ExceedPolicy.WARN))
result = parent.run("Hello from parent")
print(f"   Parent cost:   ${result.cost:.6f}")
print(f"   Budget state:  {parent.budget_state}")


class Child(Agent):
    """Child agent that inherits the parent's remaining budget."""

    model = model


spawned = parent.spawn(Child, task="Do work")
print(f"   Child result:  {spawned.content[:60]}...")
print(f"   Budget after:  {parent.budget_state}")

# ---------------------------------------------------------------------------
# 2. Multiple children — each borrows from parent sequentially
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("2. Three children sharing one budget pool (sequential)")
print("=" * 60)

parent2 = Agent(model=model, budget=Budget(max_cost=10.0))
for i in range(3):
    parent2.spawn(Child, task=f"Task {i + 1}")
    print(f"   After child {i + 1}: {parent2.budget_state}")

# ---------------------------------------------------------------------------
# 3. Swarm — budget is shared across all agents automatically
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("3. Swarm — budget shared across agents automatically")
print("=" * 60)


async def run_swarm() -> None:
    class ResearchAgent(Agent):
        model = Model.mock()
        system_prompt = "Research the topic concisely."

    class WriterAgent(Agent):
        model = Model.mock()
        system_prompt = "Write a one-paragraph summary."

    swarm = Swarm(
        agents=[ResearchAgent(), WriterAgent()],
        goal="Research AI trends and write a summary.",
        budget=Budget(max_cost=10.00, exceed_policy=ExceedPolicy.WARN),
        config=SwarmConfig(topology=SwarmTopology.PARALLEL),
    )
    result = await swarm.run()
    print(f"   Swarm output:  {result.content[:80]}...")
    if result.budget_report:
        print(f"   Total spent:   ${result.budget_report.total_spent:.6f}")
        print(f"   Within p95:    {result.budget_report.was_within_p95}")


asyncio.run(run_swarm())
