"""Workflow parallel fan-out — multi-source research → merge → summarize.

Use `.parallel([A, B])` when multiple agents can work independently on the
same input.  All agents in the parallel block receive the same HandoffContext
and run concurrently.  Their combined output is concatenated and forwarded to
the next `.step()`.

Key concepts shown here:
- wf.parallel([A, B]) — concurrent fan-out
- MergeAgent receives a single string with all parallel outputs concatenated
- SummaryAgent runs after the merge as a normal sequential step
- Mixing parallel and sequential steps in one workflow
- Per-step budget limits applied to the parallel block

Run:
    OPENAI_API_KEY=sk-... uv run python examples/07_multi_agent/workflow_parallel.py
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


class PrimaryResearchAgent(Agent):
    """Searches primary academic sources and tier-1 industry reports."""

    model = _MODEL
    system_prompt = (
        "You research primary academic papers and tier-1 industry reports on the given topic. "
        "Summarise the 3 most significant findings with sources."
    )


class SecondaryResearchAgent(Agent):
    """Searches secondary sources: news, blogs, and analyst commentary."""

    model = _MODEL
    system_prompt = (
        "You research secondary sources: news articles, analyst blogs, and social signals. "
        "Summarise the 3 most notable recent developments."
    )


class MergeAgent(Agent):
    """Deduplicates and merges outputs from multiple research streams."""

    model = _MODEL
    system_prompt = (
        "You are a research editor. You receive multiple research streams as input. "
        "Merge, deduplicate, and harmonise them into a single coherent brief without losing key facts."
    )


class SummaryAgent(Agent):
    """Produces the final executive summary from the merged brief."""

    model = _MODEL
    system_prompt = (
        "You write precise, one-paragraph executive summaries. "
        "Synthesise the provided brief into a single paragraph with a clear recommendation."
    )


# ── Example 1: Basic parallel fan-out ────────────────────────────────────────
#
# PrimaryResearchAgent and SecondaryResearchAgent run simultaneously.
# MergeAgent then processes their combined output, followed by SummaryAgent.


async def example_basic_parallel() -> None:
    print("\n── Example 1: Basic parallel fan-out ────────────────────────────")

    wf = (
        Workflow("parallel-research")
        .parallel([PrimaryResearchAgent, SecondaryResearchAgent])
        .step(MergeAgent)
        .step(SummaryAgent)
    )

    result = await wf.run("AI agent framework adoption in enterprise")
    print(result.content)
    print(f"\nTotal cost: ${result.cost:.6f}")


# ── Example 2: Parallel with per-step budget ──────────────────────────────────
#
# Apply a Budget to the parallel step to cap total spend across all concurrent
# agents in that step.


async def example_parallel_budget() -> None:
    print("\n── Example 2: Parallel step with budget cap ─────────────────────")

    wf = (
        Workflow("parallel-research-budget", budget=Budget(max_cost=2.00))
        .parallel(
            [PrimaryResearchAgent, SecondaryResearchAgent],
            budget=Budget(max_cost=0.50),
        )
        .step(MergeAgent)
        .step(SummaryAgent)
    )

    result = await wf.run("Renewable energy storage market")
    print(f"Cost: ${result.cost:.6f} (parallel step capped at $0.50)")
    print(f"Summary: {result.content[:200]}...")


# ── Example 3: Three-way parallel fan-out ────────────────────────────────────
#
# You can fan out to any number of agents in a single parallel block.


class PatentSignalAgent(Agent):
    """Scans patent filings and regulatory documents for market signals."""

    model = _MODEL
    system_prompt = (
        "You analyse patent filings and regulatory documents for signals about the given topic. "
        "Report the 3 most significant patent or regulatory developments."
    )


async def example_three_way_parallel() -> None:
    print("\n── Example 3: Three-way parallel fan-out ────────────────────────")

    wf = (
        Workflow("three-source-research")
        .parallel([PrimaryResearchAgent, SecondaryResearchAgent, PatentSignalAgent])
        .step(MergeAgent)
        .step(SummaryAgent)
    )

    result = await wf.run("AI patent and regulatory landscape")
    print(f"Content: {result.content[:300]}")
    print(f"Cost:    ${result.cost:.6f}")


# ── Example 4: Lifecycle hooks on parallel step ──────────────────────────────
#
# WORKFLOW_STEP_START fires once before the parallel block; WORKFLOW_STEP_END
# fires once after all parallel agents finish.


async def example_hooks() -> None:
    print("\n── Example 4: Parallel step lifecycle hooks ─────────────────────")

    wf = (
        Workflow("parallel-hooks")
        .parallel([PrimaryResearchAgent, SecondaryResearchAgent])
        .step(MergeAgent)
        .step(SummaryAgent)
    )

    steps_logged: list[str] = []
    wf.events.on(
        Hook.WORKFLOW_STEP_START,
        lambda ctx: steps_logged.append(f"START step={ctx.get('step_index')}"),
    )
    wf.events.on(
        Hook.WORKFLOW_STEP_END,
        lambda ctx: steps_logged.append(
            f"END   step={ctx.get('step_index')} cost=${float(ctx.get('cost', 0)):.6f}"
        ),
    )

    await wf.run("Quantum computing investment landscape")
    for entry in steps_logged:
        print(f"  {entry}")


# ── Main ──────────────────────────────────────────────────────────────────────


async def main() -> None:
    await example_basic_parallel()
    await example_parallel_budget()
    await example_three_way_parallel()
    await example_hooks()
    print("\nAll parallel workflow examples completed.")


if __name__ == "__main__":
    asyncio.run(main())
