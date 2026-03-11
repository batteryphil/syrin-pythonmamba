"""Agent with Budget — Control AI spending.

Set a dollar limit so your agent never overspends.

Run:
    python examples/01_minimal/agent_with_budget.py
"""

from syrin import Agent, Budget, Model, raise_on_exceeded, warn_on_exceeded
from syrin.exceptions import BudgetExceededError

model = Model.Almock()

# --- Example 1: Budget with warning ---
# warn_on_exceeded: logs a warning but keeps running
agent = Agent(
    model=model,
    system_prompt="Be concise.",
    budget=Budget(run=0.50, on_exceeded=warn_on_exceeded),
)

response = agent.response("What is machine learning?")
print(f"Answer: {response.content[:80]}...")
print(f"Cost:   ${response.cost:.6f}")
print(f"Budget: {agent.budget_state}")
print()

# --- Example 2: Budget with hard stop ---
# raise_on_exceeded: raises BudgetExceededError when limit hit
agent2 = Agent(
    model=model,
    system_prompt="Be concise.",
    budget=Budget(run=0.0001, on_exceeded=raise_on_exceeded),
)

try:
    agent2.response("This might exceed the budget")
except BudgetExceededError as e:
    print(f"Budget exceeded (expected): {e}")
print()

# --- Example 3: Class-level budget ---
class CostAwareAgent(Agent):
    model = model
    system_prompt = "You are a concise assistant."
    budget = Budget(run=1.00, on_exceeded=warn_on_exceeded)


agent3 = CostAwareAgent()
response3 = agent3.response("Hello!")
print(f"Class agent cost: ${response3.cost:.6f}")
print(f"Budget remaining: {agent3.budget_state}")
