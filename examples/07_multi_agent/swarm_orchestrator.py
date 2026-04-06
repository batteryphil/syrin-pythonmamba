"""Swarm ORCHESTRATOR topology — LLM-driven task delegation.

The first agent in the list acts as orchestrator: it analyses the shared goal,
decides which workers to invoke, and synthesises their outputs. Unlike a fixed
workflow, the delegation logic lives in the LLM and adapts per goal.

Convention: first agent = orchestrator, remaining agents = workers.

Run: python examples/07_multi_agent/swarm_orchestrator.py
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


class ResearchDirector(Agent):
    """Coordinates the team and synthesises a final research briefing."""

    model = _MODEL
    system_prompt = (
        "You are a senior research director. Delegate sub-tasks to your team "
        "(MarketAnalyst, CompetitiveStrategist, StrategicAdvisor) and synthesise "
        "their outputs into a structured briefing: Market Overview, Competitive "
        "Landscape, Strategic Recommendation."
    )


class MarketAnalyst(Agent):
    model = _MODEL
    system_prompt = "You report market size, CAGR, and top-3 player revenue share."


class CompetitiveStrategist(Agent):
    model = _MODEL
    system_prompt = "You map competitive moats, pricing pressure, and disruptive threats."


class StrategicAdvisor(Agent):
    model = _MODEL
    system_prompt = "You translate findings into a crisp investment verdict for board readers."


async def main() -> None:
    swarm = Swarm(
        agents=[ResearchDirector, MarketAnalyst, CompetitiveStrategist, StrategicAdvisor],
        goal="Generative AI infrastructure market — investment thesis 2025",
        budget=Budget(max_cost=0.50),
        config=SwarmConfig(topology=SwarmTopology.ORCHESTRATOR),
    )
    result = await swarm.run()

    print(result.content[:400])
    total = sum(result.cost_breakdown.values())
    print(f"\nCost: ${total:.4f}")
    print("Per-agent breakdown:")
    for agent_name, cost in result.cost_breakdown.items():
        print(f"  {agent_name}: ${cost:.4f}")


if __name__ == "__main__":
    asyncio.run(main())
