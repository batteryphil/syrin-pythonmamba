"""Response Object — Everything you get back from an agent.

Shows all the fields on the Response object.

Run:
    python examples/01_minimal/response_object.py
"""

from syrin import Agent, Budget, Model

model = Model.Almock()

agent = Agent(
    model=model,
    system_prompt="You are a helpful assistant. Be concise.",
    budget=Budget(run=0.50),
)

response = agent.response("What are the three primary colors?")

# --- Core fields ---
print("=== Response Object ===")
print(f"Content:    {response.content}")         # The answer text
print(f"Cost:       ${response.cost:.6f}")        # USD spent
print(f"Tokens:     {response.tokens}")           # Total tokens used
print(f"Model:      {response.model}")            # Model that responded
print(f"Duration:   {response.duration:.2f}s")    # Wall-clock time
print(f"Success:    {bool(response)}")            # True if response has content
print()

# --- Budget info (only if budget is set) ---
state = agent.budget_state
if state:
    print("=== Budget State ===")
    print(f"Limit:      ${state.limit:.4f}")
    print(f"Spent:      ${state.spent:.6f}")
    print(f"Remaining:  ${state.remaining:.6f}")
    print(f"Used:       {state.percent_used:.1f}%")
