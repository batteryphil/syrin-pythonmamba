"""Basic Budget — Per-run cost limits, exceed policies, and tracking.

Shows how to set a dollar budget, handle exceeded limits, and track spending.

Run:
    python examples/03_budget/basic_budget.py
"""

from syrin import Agent, Budget, Model
from syrin.enums import ExceedPolicy
from syrin.exceptions import BudgetExceededError

model = Model.mock()

# ============================================================
# 1. Simple run budget — spend up to $0.50 per run
# ============================================================
agent = Agent(model=model, budget=Budget(max_cost=0.50))
result = agent.run("What is machine learning?")

print("=== Simple Budget ===")
print(f"Answer: {result.content[:60]}...")
print(f"Cost:   ${result.cost:.6f}")
print(f"State:  {agent.budget_state}")
print()

# ============================================================
# 2. WARN policy — log a warning but keep running
# ============================================================
agent2 = Agent(model=model, budget=Budget(max_cost=0.05, exceed_policy=ExceedPolicy.WARN))
result2 = agent2.run("Summarize Python in two sentences.")

print("=== WARN Policy ===")
print(f"Cost:  ${result2.cost:.6f}")
print(f"State: {agent2.budget_state}")
print()

# ============================================================
# 3. STOP policy — hard stop when budget is hit
# ============================================================
agent3 = Agent(model=model, budget=Budget(max_cost=0.0001, exceed_policy=ExceedPolicy.STOP))

print("=== STOP Policy ===")
try:
    agent3.run("This will likely exceed the tiny budget")
except BudgetExceededError as e:
    print(f"Caught: {e}")
print()


# ============================================================
# 4. Class-level budget — reusable agent definition
# ============================================================
class CostAwareAgent(Agent):
    model = model
    system_prompt = "You are a concise assistant."
    budget = Budget(max_cost=1.00, exceed_policy=ExceedPolicy.WARN)


agent5 = CostAwareAgent()
result5 = agent5.run("Hello!")
print("=== Class-level Budget ===")
print(f"Cost: ${result5.cost:.6f}")
print(f"State: {agent5.budget_state}")

# --- Serve (uncomment to try playground) ---
# agent5.serve(port=8000, enable_playground=True)
