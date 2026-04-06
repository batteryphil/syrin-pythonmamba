"""Basic swarm — multiple agents on a shared goal.

The simplest way to use a Swarm: declare agent classes, pass them to Swarm,
and call run(). Each agent tackles the same goal from its specialist angle.

Run: python examples/07_multi_agent/team.py
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


class Researcher(Agent):
    model = _MODEL
    system_prompt = "You research topics and summarise findings concisely."


class Writer(Agent):
    model = _MODEL
    system_prompt = "You write clear, engaging summaries."


class FactChecker(Agent):
    model = _MODEL
    system_prompt = "You verify claims and flag any inaccuracies."


async def main() -> None:
    # Pass class references — Swarm instantiates them automatically
    swarm = Swarm(
        agents=[Researcher, Writer, FactChecker],  # classes, not instances
        goal="Summarise the impact of AI on software engineering",
        budget=Budget(max_cost=1.00),
        config=SwarmConfig(topology=SwarmTopology.PARALLEL),
    )
    result = await swarm.run()
    print(result.content[:200])
    total = sum(result.cost_breakdown.values())
    print(f"Cost: ${total:.4f}")


if __name__ == "__main__":
    asyncio.run(main())
