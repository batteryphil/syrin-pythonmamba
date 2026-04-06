"""REFLECTION topology — iterative writer + critic loop.

The REFLECTION topology runs a producer–critic loop: a WriterAgent drafts
content, a CriticAgent scores it and gives structured feedback, and the writer
revises. This repeats up to max_rounds or until the critic's score meets the
configured threshold — whichever comes first.

Use this topology when:
  - Output quality matters more than latency.
  - Errors caught by a specialist critic are less costly than human review.
  - You want self-improving outputs without manual iteration.

Key concepts:
  - ReflectionConfig(producer, critic, max_rounds, stop_when=lambda ro: ro.score >= 0.8)
  - SwarmConfig(topology=SwarmTopology.REFLECTION)
  - ReflectionConfig.producer and .critic reference agent *classes*, not strings.

Requires:
    OPENAI_API_KEY — set in your environment before running.

Run:
    OPENAI_API_KEY=sk-... uv run python examples/07_multi_agent/swarm_reflection.py
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from syrin import Agent, Budget, Model
from syrin.enums import SwarmTopology
from syrin.swarm import ReflectionConfig, Swarm, SwarmConfig

_MODEL = Model.OpenAI("gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"))


# ── Agent definitions ─────────────────────────────────────────────────────────
#
# The producer writes; the critic evaluates. System prompts must be precise:
# the critic's scoring format drives the reflection loop's stop condition.
# ReflectionConfig references the *classes*, not instances.


class TechWriterAgent(Agent):
    """Drafts and iteratively refines technical explanations."""

    model = _MODEL
    system_prompt = (
        "You are a senior technical writer specialising in AI and distributed systems. "
        "When given a topic, write a clear, accurate 3-5 sentence explanation suitable "
        "for a technically literate audience (engineers and architects). "
        "When given prior feedback, incorporate it precisely into a revised draft. "
        "Prioritise clarity, accuracy, and concrete examples over abstract generalisations."
    )


class TechnicalEditorAgent(Agent):
    """Evaluates drafts for accuracy, clarity, and conciseness; provides scored feedback."""

    model = _MODEL
    system_prompt = (
        "You are a technical editor who reviews AI and engineering content. "
        "Evaluate the given draft on three criteria:\n"
        "  1. Technical accuracy — is every claim correct and precise?\n"
        "  2. Clarity — is the explanation easy to follow for engineers?\n"
        "  3. Conciseness — is every sentence earning its place?\n\n"
        "Provide 1-3 specific improvement suggestions. "
        "End your response with exactly this line: 'Score: <value>' where <value> "
        "is a decimal between 0.0 (unacceptable) and 1.0 (publication-ready). "
        "Example: 'Score: 0.85'"
    )


# ── Example 1: Technical explanation refined through critic feedback ───────────


async def example_technical_explanation() -> None:
    print("\n── Example 1: Technical explanation refined through critic rounds ─")

    swarm = Swarm(
        agents=[TechWriterAgent(), TechnicalEditorAgent()],
        goal="Explain how vector embeddings enable semantic search in RAG systems",
        budget=Budget(max_cost=0.20),
        config=SwarmConfig(
            topology=SwarmTopology.REFLECTION,
            reflection=ReflectionConfig(
                producer=TechWriterAgent,
                critic=TechnicalEditorAgent,
                max_rounds=3,
                stop_when=lambda ro: ro.score >= 0.80,
            ),
        ),
    )
    result = await swarm.run()

    print(result.content)
    if result.budget_report:
        print(f"\nTotal spent: ${result.budget_report.total_spent:.4f}")


# ── Example 2: Aggressive quality bar — higher threshold, more rounds ─────────
#
# Raise the stop_when threshold to demand near-perfect output. The loop continues until
# the critic is satisfied or max_rounds is reached. Useful for compliance-critical
# or customer-facing content where quality is non-negotiable.


async def example_high_quality_bar() -> None:
    print("\n── Example 2: High-quality bar (threshold=0.90, max_rounds=4) ───")

    swarm = Swarm(
        agents=[TechWriterAgent(), TechnicalEditorAgent()],
        goal="Explain the CAP theorem and its practical implications for distributed databases",
        budget=Budget(max_cost=0.30),
        config=SwarmConfig(
            topology=SwarmTopology.REFLECTION,
            reflection=ReflectionConfig(
                producer=TechWriterAgent,
                critic=TechnicalEditorAgent,
                max_rounds=4,
                stop_when=lambda ro: ro.score >= 0.90,
            ),
        ),
    )
    result = await swarm.run()

    print(result.content[:400])
    if result.budget_report:
        print(f"\nTotal spent: ${result.budget_report.total_spent:.4f}")


# ── Example 3: Domain-specific writer/critic pair ─────────────────────────────
#
# The topology is reusable for any writer/critic pair — here repurposed for
# financial analysis with a different producer and critic.


class FinancialAnalystAgent(Agent):
    """Writes structured financial analysis with investment-grade precision."""

    model = _MODEL
    system_prompt = (
        "You are a CFA charterholder writing for institutional investors. "
        "When given a financial topic or company, write a 3-5 sentence analysis "
        "covering: the core thesis, the key financial metric supporting it, "
        "the primary downside risk, and a time horizon. "
        "When given prior feedback, revise the analysis to address it exactly."
    )


class RiskReviewAgent(Agent):
    """Reviews financial analysis for logical consistency and risk completeness."""

    model = _MODEL
    system_prompt = (
        "You are a risk manager reviewing investment analysis for an institutional fund. "
        "Evaluate the given analysis on:\n"
        "  1. Logical consistency — does the thesis follow from the data cited?\n"
        "  2. Risk completeness — are the material downside risks identified?\n"
        "  3. Quantitative rigour — are claims backed by specific metrics?\n\n"
        "Provide 1-2 specific revisions required. "
        "End with exactly: 'Score: <value>' (0.0–1.0)."
    )


async def example_financial_analysis() -> None:
    print("\n── Example 3: Financial analysis writer/critic pair ─────────────")

    swarm = Swarm(
        agents=[FinancialAnalystAgent(), RiskReviewAgent()],
        goal="Nvidia: infrastructure play in the sovereign AI buildout cycle",
        budget=Budget(max_cost=0.20),
        config=SwarmConfig(
            topology=SwarmTopology.REFLECTION,
            reflection=ReflectionConfig(
                producer=FinancialAnalystAgent,
                critic=RiskReviewAgent,
                max_rounds=3,
                stop_when=lambda ro: ro.score >= 0.82,
            ),
        ),
    )
    result = await swarm.run()

    print(result.content)
    if result.budget_report:
        print(f"\nTotal spent: ${result.budget_report.total_spent:.4f}")


# ── Main ──────────────────────────────────────────────────────────────────────


async def main() -> None:
    await example_technical_explanation()
    await example_high_quality_bar()
    await example_financial_analysis()
    print("\nAll reflection swarm examples completed.")


if __name__ == "__main__":
    asyncio.run(main())
