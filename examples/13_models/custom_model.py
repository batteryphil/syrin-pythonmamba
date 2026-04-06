"""Custom Model Example.

Demonstrates:
- Creating a custom Model subclass
- Overriding complete() for any LLM API
- ModelPricing for cost calculation
- with_fallback() for reliability chains
- Using Almock for testing

Run: python -m examples.13_models.custom_model
Visit: http://localhost:8000/playground
Requires: uv pip install syrin[serve]
"""

from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv

from syrin import Agent, Model

load_dotenv(Path(__file__).resolve().parent.parent / ".env")


# 1. Custom Model subclass
class MyCustomModel(Model):
    """Custom model for any LLM API."""

    def complete(self, messages: list, **kwargs: object) -> object:
        print(f"  Custom complete() called with {len(messages)} messages")
        return None


model = MyCustomModel("my-model")
print(f"Custom model: {model}, provider: {model.provider}")

# 2. Almock for testing
mock_model = Model.mock(latency_seconds=0.01, lorem_length=50)
agent = Agent(model=mock_model, system_prompt="You are helpful.")
result = agent.run("Hello!")
print(f"Response: {result.content[:60]}..., cost: ${result.cost:.6f}")

# 3. Almock with latency range
mock_model = Model.mock(latency_min=0, latency_max=0, lorem_length=80)
agent = Agent(model=mock_model)
result = agent.run("Fast response")

# 4. Fallback chains
primary = Model.mock(latency_seconds=0.01)
model = primary.with_fallback(Model.mock(latency_seconds=0.01), Model.mock(latency_seconds=0.01))
agent = Agent(model=model)
result = agent.run("Hello with fallback!")


# 5. Class-level model
class MyAgent(Agent):
    name = "custom-model"
    description = "Agent with custom Model subclass"
    model = Model.mock()
    system_prompt = "You are a specialized assistant."


if __name__ == "__main__":
    agent = MyAgent()
    result = agent.run("Hello specialized!")
    print("Serving at http://localhost:8000/playground")
    agent.serve(port=8000, enable_playground=True, debug=True)
