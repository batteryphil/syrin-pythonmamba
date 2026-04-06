"""Workflow sequential steps — research → extract → write.

The simplest Workflow pattern: a chain of `.step()` calls where each agent
receives the previous agent's output as its task input via HandoffContext.

Key concepts shown here:
- Workflow("name").step(A).step(B).step(C) builder API
- await wf.run("initial task") returns a Response
- result.content — final step's output text
- result.cost   — total USD spend across all steps
- Lifecycle hooks: WORKFLOW_STEP_START / WORKFLOW_STEP_END
- wf.estimate() — pre-flight cost estimation without LLM calls

Run:
    OPENAI_API_KEY=sk-... uv run python examples/07_multi_agent/workflow_sequential.py
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

from dotenv import load_dotenv

from syrin import Agent, Budget, Model
from syrin.enums import Hook
from syrin.workflow import Workflow

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

_MODEL = Model.OpenAI("gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"))

# ── Agent definitions ─────────────────────────────────────────────────────────
#
# Three agents in a research pipeline.  Each receives the previous agent's
# output automatically via HandoffContext.content.


class ResearchAgent(Agent):
    """Finds and summarises raw sources on the given topic."""

    model = _MODEL
    system_prompt = (
        "You are a research analyst. Find and summarise the most relevant sources, "
        "statistics, and key data points on the topic. Be concise and factual."
    )


class ExtractAgent(Agent):
    """Extracts structured key facts from the research findings."""

    model = _MODEL
    system_prompt = (
        "You extract and structure key facts, statistics, and data points from "
        "research text. Output a bullet-point list of the most important findings."
    )


class WriteAgent(Agent):
    """Drafts a polished executive brief from extracted facts."""

    model = _MODEL
    system_prompt = (
        "You write concise executive briefs for C-suite audiences. "
        "Synthesise provided data into a 2-3 paragraph report with a clear recommendation."
    )


# ── Example 1: Basic sequential workflow ─────────────────────────────────────
#
# Build a three-step pipeline with the fluent `.step()` API.
# Each step passes its output forward as the next step's input.


async def example_basic() -> None:
    print("\n── Example 1: Basic sequential workflow ─────────────────────────")

    wf = Workflow("research-pipeline").step(ResearchAgent).step(ExtractAgent).step(WriteAgent)

    result = await wf.run("AI market trends in 2025")
    print(result.content)
    print(f"\nTotal cost: ${result.cost:.6f}")


# ── Example 2: Per-step task overrides ───────────────────────────────────────
#
# Pass an explicit task string to any step.  When task is set, HandoffContext
# content is still available but the agent gets the explicit task string
# combined with the spawn content.


async def example_step_tasks() -> None:
    print("\n── Example 2: Per-step task overrides ───────────────────────────")

    wf = (
        Workflow("research-pipeline-v2")
        .step(ResearchAgent, task="Focus on generative AI adoption rates in enterprise")
        .step(ExtractAgent, task="Extract only statistics and percentages")
        .step(WriteAgent, task="Write a 3-sentence investor memo")
    )

    result = await wf.run("AI market trends 2025")
    print(f"Content: {result.content[:300]}")
    print(f"\nCost: ${result.cost:.6f}")


# ── Example 3: Workflow with budget ──────────────────────────────────────────
#
# A Budget limits total spend across all steps.  If any step would push spend
# over the limit, the workflow raises BudgetExceededError.


async def example_with_budget() -> None:
    print("\n── Example 3: Workflow with budget ─────────────────────────────")

    wf = (
        Workflow("research-pipeline-budget", budget=Budget(max_cost=1.00))
        .step(ResearchAgent)
        .step(ExtractAgent)
        .step(WriteAgent)
    )

    result = await wf.run("Renewable energy market trends")
    print(f"Cost ${result.cost:.6f} (budget $1.00)")
    print(f"Content preview: {result.content[:200]}...")


# ── Example 4: Lifecycle hooks ────────────────────────────────────────────────
#
# Register hook handlers to observe every step start/end with cost and content.


async def example_hooks() -> None:
    print("\n── Example 4: Lifecycle hooks ───────────────────────────────────")

    wf = Workflow("research-pipeline-hooks").step(ResearchAgent).step(ExtractAgent).step(WriteAgent)

    step_costs: list[float] = []
    wf.events.on(
        Hook.WORKFLOW_STEP_END,
        lambda ctx: step_costs.append(float(ctx.get("cost", 0.0))),
    )

    result = await wf.run("Cloud computing market trends")
    print(f"Per-step costs: {[f'${c:.6f}' for c in step_costs]}")
    print(f"Total:          ${result.cost:.6f}")


# ── Example 5: Cost estimation (no LLM calls) ────────────────────────────────
#
# wf.estimate() walks all steps and returns a p50/p95 cost report
# without making any actual LLM calls.


async def example_estimate() -> None:
    print("\n── Example 5: Pre-flight cost estimation ────────────────────────")

    wf = (
        Workflow("research-pipeline-estimate", budget=Budget(max_cost=5.00))
        .step(ResearchAgent)
        .step(ExtractAgent)
        .step(WriteAgent)
    )

    report = wf.estimate("AI market trends")
    print(f"p50 estimate:      ${report.total_p50:.4f}")
    print(f"p95 estimate:      ${report.total_p95:.4f}")
    print(f"Budget sufficient: {report.budget_sufficient}")


# ── Main ──────────────────────────────────────────────────────────────────────


async def main() -> None:
    await example_basic()
    await example_step_tasks()
    await example_with_budget()
    await example_hooks()
    await example_estimate()
    print("\nAll sequential workflow examples completed.")


if __name__ == "__main__":
    asyncio.run(main())
