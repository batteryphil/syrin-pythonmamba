"""Agent with Tools — Give your agent abilities.

Tools are Python functions the agent can call to do real work.

Run:
    python examples/01_minimal/agent_with_tools.py
"""

from syrin import Agent, Model, tool

model = Model.Almock()


# Define tools as regular Python functions with @tool
@tool
def calculate(a: float, b: float, operation: str = "add") -> str:
    """Perform basic arithmetic. operation: add, subtract, multiply, divide."""
    ops = {
        "add": a + b,
        "subtract": a - b,
        "multiply": a * b,
        "divide": a / b if b != 0 else "Error: division by zero",
    }
    return str(ops.get(operation, "Unknown operation"))


@tool
def get_weather(city: str) -> str:
    """Get current weather for a city."""
    # In real usage, call a weather API here
    return f"The weather in {city} is 22°C and sunny."


# Attach tools to the agent
class ToolAgent(Agent):
    model = model
    system_prompt = "You are a helpful assistant. Use tools when needed."
    tools = [calculate, get_weather]


agent = ToolAgent()
response = agent.response("What is 15 * 7?")
print(f"Answer: {response.content}")
print(f"Cost:   ${response.cost:.6f}")

# --- Serve with playground (uncomment to try) ---
# agent.serve(port=8000, enable_playground=True)
