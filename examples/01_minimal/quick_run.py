"""Quick Run — Fastest way to use Syrin.

Two ways to get started, from simplest to most flexible.

Run:
    python examples/01_minimal/quick_run.py
"""

from syrin import Agent, Budget, ExceedPolicy, Model

model = Model.mock()  # No API key needed

# --- Way 1: Inline (one-off query) ---
agent = Agent(model=model, system_prompt="Explain like I'm five years old.")
response = agent.run("What is gravity?")
print(f"Inline:  {response.content[:80]}...")
print()


# --- Way 2: Class-based (best for reuse) ---
class MyAgent(Agent):
    model = model
    system_prompt = "You are helpful and concise."
    budget = Budget(max_cost=1.00, exceed_policy=ExceedPolicy.WARN)


agent2 = MyAgent()
response2 = agent2.run("Hello!")
print(f"Class:   {response2.content}")
