"""Budget Persistence Example.

Demonstrates:
- BudgetStore with backend="file" for persisting budget state across restarts
- key= parameter for per-user/per-org isolation
- Rate limits that survive process restarts

Run: python -m examples.03_budget.budget_persistence
Visit: http://localhost:8000/playground

Requires: uv pip install syrin[serve]
"""

from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv

from syrin import Agent, Budget, ExceedPolicy, Model, RateLimit
from syrin.budget_store import BudgetStore

load_dotenv(Path(__file__).resolve().parent.parent / ".env")


# 1. BudgetStore with backend="file" — persist to disk
store_path = Path(__file__).resolve().parent.parent / "data" / "budget_example.json"
store_path.parent.mkdir(parents=True, exist_ok=True)


class PersistentAgent(Agent):
    name = "persistent-budget"
    description = "Agent with BudgetStore file backend (persists across restarts)"
    model = Model.mock()
    system_prompt = "You are concise."
    budget = Budget(
        max_cost=0.10,
        rate_limits=RateLimit(day=5.00, month=50.00, month_days=30),
        exceed_policy=ExceedPolicy.STOP,
    )


agent = PersistentAgent(
    budget_store=BudgetStore(key="example_user", backend="file", path=store_path),
)
result = agent.run("Summarize Python in two sentences.")
print(f"Cost: ${result.cost:.6f}")
print(f"Budget state: {agent.budget_state}")

# 2. Per-user isolation via key=
agent_alice = PersistentAgent(
    budget_store=BudgetStore(key="alice", backend="file", path=store_path),
)
agent_bob = PersistentAgent(
    budget_store=BudgetStore(key="bob", backend="file", path=store_path),
)
agent_alice.run("Hello from Alice")
agent_bob.run("Hello from Bob")
print(f"Alice budget: {agent_alice.budget_state}")
print(f"Bob budget: {agent_bob.budget_state}")


if __name__ == "__main__":
    agent = PersistentAgent(
        budget_store=BudgetStore(key="example_user", backend="file", path=store_path),
    )
    print("Serving at http://localhost:8000/playground")
    agent.serve(port=8000, enable_playground=True, debug=True)
