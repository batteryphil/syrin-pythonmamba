"""Basic Agent — Your first Syrin agent.

Creates an agent and asks it a question. No API key needed (uses Almock).

Run:
    python examples/01_minimal/basic_agent.py
"""

from syrin import Agent, Model

# Create a model — Almock is a built-in mock (no API key needed)
# Replace with Model.OpenAI("gpt-4o-mini") for real usage
model = Model.Almock()


# Define your agent as a class
class Assistant(Agent):
    model = model
    system_prompt = "You are a helpful assistant. Be concise."


# Use the agent
agent = Assistant()
response = agent.response("What is Python?")

print(f"Answer:  {response.content}")
print(f"Cost:    ${response.cost:.6f}")
print(f"Tokens:  {response.tokens}")
print(f"Model:   {response.model}")

# --- Or use the builder pattern (no class needed) ---
agent2 = Agent.builder(model).with_system_prompt("You are helpful.").build()
response2 = agent2.response("What is 2 + 2?")
print(f"\nBuilder agent says: {response2.content}")

# --- Serve with web playground (uncomment to try) ---
# agent.serve(port=8000, enable_playground=True)
# Visit http://localhost:8000/playground
