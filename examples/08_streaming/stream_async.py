"""Async Streaming -- Token-by-token output with astream().

Demonstrates:
- agent.astream(input) for async streaming
- async for chunk in agent.astream(...)
- Collecting chunks into full text

Run: python examples/08_streaming/stream_async.py
"""

import asyncio

from syrin import Agent, Model


# --- Define an async streaming agent ---

class AsyncStreamAgent(Agent):
    _agent_name = "async-stream-agent"
    _agent_description = "Async token-by-token streaming"
    model = Model.Almock()
    system_prompt = "You are a helpful assistant."


# --- Stream asynchronously and collect output ---

async def main() -> None:
    agent = AsyncStreamAgent()

    print("Async streaming response:")
    print("-" * 40)

    full_text = ""
    async for chunk in agent.astream("Explain machine learning in one sentence"):
        full_text += chunk.text

    print(full_text)
    print("-" * 40)
    print(f"Length: {len(full_text)} chars")


if __name__ == "__main__":
    asyncio.run(main())

    # Optional: serve with playground UI
    # agent = AsyncStreamAgent()
    # agent.serve(port=8000, enable_playground=True, debug=True)
