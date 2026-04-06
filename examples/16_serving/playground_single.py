"""Playground: Single Agent Example.

Demonstrates:
- Single agent with budget
- Visit http://localhost:8000/playground
- Chat, budget gauge, observability (debug=True)

Run: python -m examples.serving.playground_single
"""

from __future__ import annotations

import sys
from pathlib import Path

_root = Path(__file__).resolve().parents[2]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from dotenv import load_dotenv

from syrin import Agent, Budget, Model, RateLimit

load_dotenv(Path(__file__).resolve().parent.parent / ".env")


class Assistant(Agent):
    name = "assistant"
    description = "Helpful assistant — single agent demo"
    model = Model.mock(latency_min=1, latency_max=3, lorem_length=800, pricing_tier="high")
    system_prompt = "You are a helpful assistant. Be concise."
    budget = Budget(max_cost=0.5, rate_limits=RateLimit(hour=10, day=100, week=700))


if __name__ == "__main__":
    agent = Assistant()
    agent.serve(port=8000, enable_playground=True, debug=True)
