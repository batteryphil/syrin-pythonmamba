"""Parallel swarm — independent agents run simultaneously on a shared goal.

All agents receive the same goal and run concurrently. The result combines
every output into a single SwarmResult. Use this topology when agents are
independent and all outputs are valuable.

Run: python examples/07_multi_agent/swarm_parallel.py
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from syrin import Agent, Budget, Model
from syrin.enums import SwarmTopology
from syrin.swarm import Swarm, SwarmConfig

_MODEL = Model.OpenAI("gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"))


class MarketResearcher(Agent):
    model = _MODEL
    system_prompt = "You analyse market size, growth rates, and dominant players."


class CompetitiveAnalyst(Agent):
    model = _MODEL
    system_prompt = "You surface competitive dynamics, pricing pressure, and threats."


class ExecutiveBriefer(Agent):
    model = _MODEL
    system_prompt = "You distil findings into a crisp C-suite recommendation."


async def main() -> None:
    budget = Budget(max_cost=0.50)

    swarm = Swarm(
        agents=[MarketResearcher, CompetitiveAnalyst, ExecutiveBriefer],
        goal="AI developer tooling market — 2025 outlook",
        budget=budget,
        config=SwarmConfig(topology=SwarmTopology.PARALLEL),
    )
    result = await swarm.run()

    print(result.content[:300])
    total = sum(result.cost_breakdown.values())
    print(f"\nTotal cost: ${total:.4f}")
    print("Per-agent cost breakdown:")
    for agent_name, cost in result.cost_breakdown.items():
        print(f"  {agent_name}: ${cost:.4f}")


if __name__ == "__main__":
    asyncio.run(main())
