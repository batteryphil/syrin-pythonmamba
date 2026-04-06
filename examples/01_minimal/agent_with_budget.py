"""Agent with Budget — Control AI spending.

Set a dollar limit so your agent never overspends.

Run:
    python examples/01_minimal/agent_with_budget.py
"""

from syrin import Agent, Budget, Model
from syrin.enums import ExceedPolicy
from syrin.exceptions import BudgetExceededError

model = Model.mock()

# --- Example 1: Budget with WARN policy ---
agent = Agent(
    model=model,
    system_prompt="Be concise.",
    budget=Budget(max_cost=0.50, exceed_policy=ExceedPolicy.WARN),
)

response = agent.run("What is machine learning?")
print(f"Answer: {response.content[:80]}...")
print(f"Cost:   ${response.cost:.6f}")
print(f"Budget: {agent.budget_state}")
print()

# --- Example 2: Budget with STOP policy ---
agent2 = Agent(
    model=model,
    system_prompt="Be concise.",
    budget=Budget(max_cost=0.0001, exceed_policy=ExceedPolicy.STOP),
)

try:
    agent2.run("This might exceed the budget")
except BudgetExceededError as e:
    print(f"Budget exceeded (expected): {e}")
print()


# --- Example 3: Class-level budget ---
class CostAwareAgent(Agent):
    model = model
    system_prompt = "You are a concise assistant."
    budget = Budget(max_cost=1.00, exceed_policy=ExceedPolicy.WARN)


agent3 = CostAwareAgent()
response3 = agent3.run("Hello!")
print(f"Class agent cost: ${response3.cost:.6f}")
print(f"Budget remaining: {agent3.budget_state}")
