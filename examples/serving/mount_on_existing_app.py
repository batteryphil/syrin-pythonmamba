"""Mount Agent on Existing FastAPI App Example.

Demonstrates:
- agent.as_router() — returns APIRouter for mounting
- Mount on your own FastAPI app with custom prefix

Requires: uv pip install syrin[serve]

Run: python -m examples.16_serving.mount_on_existing_app
"""

from __future__ import annotations

import sys
from pathlib import Path

_root = Path(__file__).resolve().parents[2]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from dotenv import load_dotenv

from examples.models.models import almock
from syrin import Agent

load_dotenv(Path(__file__).resolve().parent.parent / ".env")


class Assistant(Agent):
    name = "assistant"
    description = "Helpful assistant"
    model = almock
    system_prompt = "You are a helpful assistant."


if __name__ == "__main__":
    from fastapi import FastAPI

    agent = Assistant()
    app = FastAPI(title="My API", description="Custom API with Syrin agent")
    app.include_router(agent.as_router(), prefix="/agent")
    # Run: uvicorn examples.serving.mount_on_existing_app:app --reload
    # Visit /agent/health, POST /agent/chat with {"message": "Hi"}
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
