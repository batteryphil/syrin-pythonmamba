"""Stream Async — Async streaming with astream().

Demonstrates:
- agent.astream(input) for async token-by-token output
- async for chunk in agent.astream(...)

Run: python -m examples.08_streaming.stream_async
"""

import asyncio
from pathlib import Path

from dotenv import load_dotenv

from examples.models.models import almock
from syrin import Agent

load_dotenv(Path(__file__).resolve().parent.parent / ".env")


async def _run() -> None:
    class AsyncStreamAgent(Agent):
        model = almock
        system_prompt = "You are a helpful assistant."

    agent = AsyncStreamAgent()
    full_text = ""
    async for chunk in agent.astream("Explain machine learning in one sentence"):
        full_text += chunk.text
    print(full_text)
    print(f"Length: {len(full_text)}")


asyncio.run(_run())
