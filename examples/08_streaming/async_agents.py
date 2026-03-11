"""Async Agents Example -- Parallel and sequential async patterns.

Demonstrates:
- agent.arun() for async execution
- asyncio.gather() to run multiple agents in parallel
- Sequential async calls where each depends on the previous
- Async with timeout protection

Run: python examples/08_streaming/async_agents.py
"""

from __future__ import annotations

import asyncio

from syrin import Agent, Model

# --- 1. Basic async call ---


async def example_basic_arun() -> None:
    """Single async agent call with arun()."""
    print("=" * 50)
    print("1. agent.arun() -- basic async call")
    print("=" * 50)

    agent = Agent(model=Model.Almock(), system_prompt="You are a helpful assistant.")
    result = await agent.arun("What is Python?")
    print(f"  Answer: {result.content[:80]}...")
    print(f"  Cost: ${result.cost:.6f}")


# --- 2. Parallel agents with asyncio.gather ---


async def example_parallel_agents() -> None:
    """Run three agents in parallel and collect results."""
    print("\n" + "=" * 50)
    print("2. asyncio.gather() -- parallel agents")
    print("=" * 50)

    class Researcher(Agent):
        model = Model.Almock()
        system_prompt = "You are a researcher."

    class Writer(Agent):
        model = Model.Almock()
        system_prompt = "You are a writer."

    class Reviewer(Agent):
        model = Model.Almock()
        system_prompt = "You are a reviewer."

    researcher = Researcher()
    writer = Writer()
    reviewer = Reviewer()

    results = await asyncio.gather(
        researcher.arun("Research AI trends"),
        writer.arun("Write about machine learning"),
        reviewer.arun("Review the code quality"),
    )

    for i, result in enumerate(results):
        agent_name = ["Researcher", "Writer", "Reviewer"][i]
        print(f"  {agent_name}: {result.content[:60]}...")
        print(f"    Cost: ${result.cost:.6f}, Tokens: {result.tokens.total_tokens}")


# --- 3. Sequential async (dependent calls) ---


async def example_sequential_async() -> None:
    """Chain async calls where each depends on the previous result."""
    print("\n" + "=" * 50)
    print("3. Sequential async -- dependent calls")
    print("=" * 50)

    agent = Agent(model=Model.Almock(), system_prompt="You are a helpful assistant.")

    r1 = await agent.arun("What is Python?")
    print(f"  Step 1: {r1.content[:60]}...")

    r2 = await agent.arun(f"Summarize this: {r1.content[:50]}")
    print(f"  Step 2: {r2.content[:60]}...")

    total_cost = r1.cost + r2.cost
    print(f"  Total cost: ${total_cost:.6f}")


# --- 4. Async with timeout ---


async def example_async_with_timeout() -> None:
    """Protect an async call with a timeout."""
    print("\n" + "=" * 50)
    print("4. Async with timeout")
    print("=" * 50)

    agent = Agent(model=Model.Almock())

    try:
        result = await asyncio.wait_for(agent.arun("Hello!"), timeout=10.0)
        print(f"  Result: {result.content[:60]}...")
    except asyncio.TimeoutError:
        print("  Timed out!")


# --- Run all examples ---


async def main() -> None:
    await example_basic_arun()
    await example_parallel_agents()
    await example_sequential_async()
    await example_async_with_timeout()


if __name__ == "__main__":
    asyncio.run(main())

    # Optional: serve with playground UI
    # agent = Agent(model=Model.Almock(), system_prompt="You are a helpful assistant.")
    # agent.serve(port=8000, enable_playground=True, debug=True)
