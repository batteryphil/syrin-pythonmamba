"""Consensus swarm — majority vote for high-stakes decisions.

Multiple independent agents evaluate the same question and cast votes. The
swarm reaches a decision when a configurable fraction of agents agree.

Use this topology when a single-model answer carries unacceptable risk and
you want a built-in redundancy against hallucination or bias.

Run: python examples/07_multi_agent/swarm_consensus.py
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from syrin import Agent, Budget, Model
from syrin.enums import ConsensusStrategy, SwarmTopology
from syrin.swarm import ConsensusConfig, Swarm, SwarmConfig

_MODEL = Model.OpenAI("gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"))

CLAUSE = (
    "Clause 14.3 — Limitation of Liability: In no event shall either party be "
    "liable for consequential or indirect damages arising from use of the software."
)


class USCounsel(Agent):
    model = _MODEL
    system_prompt = (
        "You are a US software contracts attorney. Assess the clause and end your "
        "response with exactly one of: ENFORCEABLE, UNENFORCEABLE, or JURISDICTION-DEPENDENT."
    )


class EUCounsel(Agent):
    model = _MODEL
    system_prompt = (
        "You are an EU technology law specialist. Assess the clause and end your "
        "response with exactly one of: ENFORCEABLE, UNENFORCEABLE, or JURISDICTION-DEPENDENT."
    )


class CommonwealthCounsel(Agent):
    model = _MODEL
    system_prompt = (
        "You are a Commonwealth jurisdiction solicitor (UK/AU/CA). Assess the clause and "
        "end your response with exactly one of: ENFORCEABLE, UNENFORCEABLE, or JURISDICTION-DEPENDENT."
    )


async def main() -> None:
    swarm = Swarm(
        agents=[USCounsel, EUCounsel, CommonwealthCounsel],
        goal=CLAUSE,
        budget=Budget(max_cost=0.10),
        config=SwarmConfig(
            topology=SwarmTopology.CONSENSUS,
            consensus=ConsensusConfig(
                min_agreement=0.67,  # two of three must agree
                strategy=ConsensusStrategy.MAJORITY,
            ),
        ),
    )
    result = await swarm.run()

    print(result.content[:300])
    print(f"\nCost: ${sum(result.cost_breakdown.values()):.4f}")


if __name__ == "__main__":
    asyncio.run(main())
