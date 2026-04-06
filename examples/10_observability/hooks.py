"""Hooks Example — Agent lifecycle events.

Demonstrates:
- agent.events.on_complete(), on_response(), on_tool()
- Tracking cost across requests
- Event-driven observability

Run: python -m examples.10_observability.hooks
Visit: http://localhost:8000/playground
Requires: uv pip install syrin[serve]
"""

from pathlib import Path

from dotenv import load_dotenv

from syrin import Agent, Model

load_dotenv(Path(__file__).resolve().parent.parent / ".env")


total_cost = {"value": 0.0}


def track_cost(ctx):
    total_cost["value"] += ctx.get("cost", 0)


class HooksDemoAgent(Agent):
    name = "hooks-demo"
    description = "Agent with lifecycle event hooks"
    model = Model.mock(latency_min=1, latency_max=3, lorem_length=800, pricing_tier="high")
    system_prompt = "You are helpful."


if __name__ == "__main__":
    agent = HooksDemoAgent()
    agent.events.on_response(track_cost)
    agent.events.on_complete(lambda ctx: print(f"  Done. Cost: ${ctx.get('cost', 0):.6f}"))
    agent.run("Hello")
    agent.run("How are you?")
    print(f"Total cost: ${total_cost['value']:.6f}")
    print("Serving at http://localhost:8000/playground")
    agent.serve(port=8000, enable_playground=True, debug=True)
