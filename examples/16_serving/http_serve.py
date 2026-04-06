"""HTTP Serving Example.

Demonstrates:
- agent.as_router() — mount on existing FastAPI app
- agent.serve() — run uvicorn with one agent
- Routes: /chat, /stream, /health, /ready, /budget, /describe
- Playground: GET /playground (when enable_playground=True)

Requires: uv pip install syrin[serve]

Run: python -m examples.serving.http_serve
Visit: http://localhost:8000/playground (when enable_playground=True)
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
    description = "Helpful assistant for questions and tasks"
    model = Model.mock(latency_min=1, latency_max=3, lorem_length=800, pricing_tier="high")
    system_prompt = "You are a helpful assistant. Be concise."
    budget = Budget(max_cost=0.5, rate_limits=RateLimit(hour=10, day=100, week=700))


if __name__ == "__main__":
    agent = Assistant()
    # Run HTTP server on port 8000 with playground
    # Visit http://localhost:8000/playground to chat, see cost, budget
    agent.serve(port=8000, enable_playground=True, debug=True)
