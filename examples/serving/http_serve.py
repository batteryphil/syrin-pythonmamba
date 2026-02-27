"""HTTP Serving Example.

Demonstrates:
- agent.as_router() — mount on existing FastAPI app
- agent.serve() — run uvicorn with one agent
- Routes: /chat, /stream, /health, /ready, /budget, /describe

Requires: uv pip install syrin[serve]

Run: python -m examples.serving.http_serve
"""

from __future__ import annotations

import sys
from pathlib import Path

_root = Path(__file__).resolve().parents[2]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from dotenv import load_dotenv

from examples.models.models import almock
from syrin import Agent, Budget, RateLimit

load_dotenv(Path(__file__).resolve().parent.parent / ".env")


class Assistant(Agent):
    name = "assistant"
    description = "Helpful assistant for questions and tasks"
    model = almock
    system_prompt = "You are a helpful assistant. Be concise."
    budget = Budget(run=0.5, per=RateLimit(hour=10, day=100, week=700))


if __name__ == "__main__":
    agent = Assistant()
    # Run HTTP server on port 8000
    # Visit http://localhost:8000/health, POST to /chat with {"message": "Hi"}
    agent.serve(port=8000)
