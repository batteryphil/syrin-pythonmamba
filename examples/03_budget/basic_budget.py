"""Basic Budget — Per-run cost limits, callbacks, and tracking.

Shows how to set a dollar budget, handle exceeded limits, and track spending.

Run:
    python examples/03_budget/basic_budget.py
"""

from syrin import Agent, Budget, Model, raise_on_exceeded, warn_on_exceeded
from syrin.exceptions import BudgetExceededError

model = Model.Almock()

# ============================================================
# 1. Simple run budget — spend up to $0.50 per run
# ============================================================
agent = Agent(model=model, budget=Budget(run=0.50))
result = agent.response("What is machine learning?")

print("=== Simple Budget ===")
print(f"Answer: {result.content[:60]}...")
print(f"Cost:   ${result.cost:.6f}")
print(f"State:  {agent.budget_state}")
print()

# ============================================================
# 2. warn_on_exceeded — log a warning but keep running
# ============================================================
agent2 = Agent(model=model, budget=Budget(run=0.05, on_exceeded=warn_on_exceeded))
result2 = agent2.response("Summarize Python in two sentences.")

print("=== Warn on Exceeded ===")
print(f"Cost:  ${result2.cost:.6f}")
print(f"State: {agent2.budget_state}")
print()

# ============================================================
# 3. raise_on_exceeded — hard stop when budget is hit
# ============================================================
agent3 = Agent(model=model, budget=Budget(run=0.0001, on_exceeded=raise_on_exceeded))

print("=== Raise on Exceeded ===")
try:
    agent3.response("This will likely exceed the tiny budget")
except BudgetExceededError as e:
    print(f"Caught: {e}")
print()

# ============================================================
# 4. Custom callback — do whatever you want
# ============================================================
exceeded_events = []


def my_callback(ctx):
    """Custom handler: log the event and continue."""
    exceeded_events.append(ctx.message)
    print(f"  [custom callback] {ctx.message}")


agent4 = Agent(model=model, budget=Budget(run=0.0001, on_exceeded=my_callback))
agent4.response("Hello!")
print(f"Events captured: {len(exceeded_events)}")
print()


# ============================================================
# 5. Class-level budget — reusable agent definition
# ============================================================
class CostAwareAgent(Agent):
    model = model
    system_prompt = "You are a concise assistant."
    budget = Budget(run=1.00, on_exceeded=warn_on_exceeded)


agent5 = CostAwareAgent()
result5 = agent5.response("Hello!")
print("=== Class-level Budget ===")
print(f"Cost: ${result5.cost:.6f}")
print(f"State: {agent5.budget_state}")

# --- Serve (uncomment to try playground) ---
# agent5.serve(port=8000, enable_playground=True)
