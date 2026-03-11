"""Shared Budget — Multiple agents sharing a single budget pool.

Demonstrates:
- Budget(shared=True) for multi-agent shared budgets
- Spawn with shared budget (child borrows from parent)
- Budget tracking across parent and child agents

No API key needed (uses Almock).

Run:
    python examples/03_budget/shared_budget.py
"""

from __future__ import annotations

from syrin import Agent, Budget, Model, warn_on_exceeded

# Create a mock model — no API key needed
model = Model.Almock()

# ---------------------------------------------------------------------------
# 1. Shared budget across agents
# ---------------------------------------------------------------------------
print("=" * 60)
print("1. Parent agent with shared budget ($10.00)")
print("=" * 60)

shared = Budget(run=10.0, shared=True, on_exceeded=warn_on_exceeded)
parent = Agent(model=model, budget=shared)
result = parent.response("Hello from parent")
print(f"   Parent cost:   ${result.cost:.6f}")
print(f"   Budget state:  {parent.budget_state}")

# ---------------------------------------------------------------------------
# 2. Spawn child that borrows from shared budget
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("2. Spawn child — borrows from parent's shared budget")
print("=" * 60)


class Child(Agent):
    """Child agent that inherits the parent's shared budget."""

    model = model


result = parent.spawn(Child, task="Do work")
print(f"   Child result:  {result.content[:60]}...")
print(f"   Budget after:  {parent.budget_state}")

# ---------------------------------------------------------------------------
# 3. Multiple children sharing budget
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("3. Three children sharing one budget pool")
print("=" * 60)

parent2 = Agent(model=model, budget=Budget(run=10.0, shared=True))
for i in range(3):
    parent2.spawn(Child, task=f"Task {i + 1}")
    print(f"   After child {i + 1}: {parent2.budget_state}")

# ---------------------------------------------------------------------------
# 4. Class-level shared budget (reusable agent definition)
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("4. Class-level: shared budget parent")
print("=" * 60)


class SharedBudgetParent(Agent):
    """Parent agent with shared budget for spawning children."""

    _agent_name = "shared-budget"
    _agent_description = "Agent with shared budget (spawn children that borrow)"
    model = model
    budget = Budget(run=10.0, shared=True, on_exceeded=warn_on_exceeded)


agent = SharedBudgetParent()
result = agent.response("Plan a project in three steps.")
print(f"   Cost:          ${result.cost:.6f}")
print(f"   Budget state:  {agent.budget_state}")

# --- Serve with web playground (uncomment to try) ---
# agent.serve(port=8000, enable_playground=True, debug=True)
# Visit http://localhost:8000/playground
