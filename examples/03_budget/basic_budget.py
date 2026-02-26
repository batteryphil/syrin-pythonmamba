"""Basic Budget Example.

Demonstrates:
- Budget(run=) to set per-run cost limit
- on_exceeded callbacks: raise_on_exceeded, warn_on_exceeded
- budget_state for tracking
- Budget with Agent class-level definition

Run: python -m examples.03_budget.basic_budget
"""

from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv

from examples.models.models import almock
from syrin import Agent, Budget, raise_on_exceeded, warn_on_exceeded
from syrin.exceptions import BudgetExceededError

load_dotenv(Path(__file__).resolve().parent.parent / ".env")


# 1. Basic budget — run limit
agent = Agent(model=almock, budget=Budget(run=0.50))
result = agent.response("What is machine learning?")
print(f"Cost: ${result.cost:.6f}")
print(f"Budget: {agent.budget_state}")

# 2. on_exceeded=warn_on_exceeded
agent = Agent(model=almock, budget=Budget(run=0.05, on_exceeded=warn_on_exceeded))
result = agent.response("Summarize Python in two sentences.")
print(f"Cost: ${result.cost:.6f}")
print(f"Budget: {agent.budget_state}")

# 3. on_exceeded=raise_on_exceeded
agent = Agent(model=almock, budget=Budget(run=0.0001, on_exceeded=raise_on_exceeded))
try:
    agent.response("This might exceed the budget")
except BudgetExceededError as e:
    print(f"BudgetExceededError: {e}")


# 4. Class-level budget definition
class CostAwareAgent(Agent):
    model = almock
    system_prompt = "You are a concise assistant."
    budget = Budget(run=1.00, on_exceeded=warn_on_exceeded)


agent = CostAwareAgent()
result = agent.response("Hello!")
print(f"Cost: ${result.cost:.6f}")
print(f"Budget: {agent.budget_state}")
