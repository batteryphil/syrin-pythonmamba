"""Workflow nesting — workflow as a step, and agent spawning a workflow.

Two patterns for composing Workflows:

Part 1 — Workflow as a step in another workflow:
    An inner Workflow can be passed directly to the outer workflow's `.step()`.
    The inner workflow runs as a single step: its run() is called with the
    HandoffContext.content and its final Response is forwarded to the next step.

Part 2 — Agent spawning a workflow via self.spawn():
    Inside a Swarm (or any agent with access to a budget pool), an agent can
    call `await self.spawn(inner_workflow, task="...", budget=0.50)` to kick
    off an entire sub-workflow and receive a SpawnResult.

Key concepts shown here:
- outer_wf.step(inner_wf)  — nesting a Workflow as a step
- agent.spawn(workflow, task, budget)  — spawning a workflow from an agent
- SpawnResult.content / .cost / .budget_remaining
- wf.visualize(expand_nested=True)  — shows nested steps inline

Run:
    OPENAI_API_KEY=sk-... uv run python examples/07_multi_agent/workflow_nested.py
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

from dotenv import load_dotenv

from syrin import Agent, Budget, Model
from syrin.response import Response
from syrin.swarm import SpawnResult, Swarm
from syrin.workflow import Workflow

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

_MODEL = Model.OpenAI("gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"))

# ── Shared agent definitions ──────────────────────────────────────────────────


class PlannerAgent(Agent):
    """Creates a concise 3-step research plan for the given topic."""

    model = _MODEL
    system_prompt = (
        "You create concise research plans. "
        "Output a numbered 3-step plan describing: what to gather, what to extract, and what to write."
    )


class GatherAgent(Agent):
    """Gathers primary data sources and key references for the research topic."""

    model = _MODEL
    system_prompt = (
        "You gather primary data sources on a topic. "
        "List 3-5 key sources with a brief description of what each provides."
    )


class ExtractAgent(Agent):
    """Extracts structured key facts from gathered sources."""

    model = _MODEL
    system_prompt = (
        "You extract structured key facts from research sources. "
        "Output 5 bullet points covering the most important statistics and insights."
    )


class WriteAgent(Agent):
    """Drafts a polished executive report from extracted facts."""

    model = _MODEL
    system_prompt = (
        "You write executive reports for investment committees. "
        "Synthesise provided facts into a 2-paragraph report with a clear recommendation."
    )


class ReviewAgent(Agent):
    """Reviews and quality-checks a report, then passes it on approved."""

    model = _MODEL
    system_prompt = (
        "You are a senior editor. Review the provided report for accuracy, clarity, and completeness. "
        "If it passes, prepend 'APPROVED: ' and output the full report. "
        "If it needs improvement, prepend 'REVISION NEEDED: ' and explain what to fix."
    )


# ═══════════════════════════════════════════════════════════════════════════════
# PART 1 — Workflow as a step inside another workflow
# ═══════════════════════════════════════════════════════════════════════════════


async def example_nested_workflow() -> None:
    print("\n── Part 1: Inner workflow as a step ─────────────────────────────")

    inner_wf = (
        Workflow("inner-research", budget=Budget(max_cost=1.00))
        .step(GatherAgent)
        .step(ExtractAgent)
        .step(WriteAgent)
    )

    outer_wf = (
        Workflow("outer-pipeline", budget=Budget(max_cost=3.00))
        .step(PlannerAgent)
        .step(inner_wf)
        .step(ReviewAgent)
    )

    result = await outer_wf.run("AI infrastructure investment opportunities 2025")
    print(result.content[:500])
    print(f"\nTotal cost: ${result.cost:.6f}")


async def example_nested_visualize() -> None:
    print("\n── Part 1b: Visualize nested workflow ───────────────────────────")

    inner_wf = Workflow("inner-research").step(GatherAgent).step(ExtractAgent).step(WriteAgent)
    outer_wf = Workflow("outer-pipeline").step(PlannerAgent).step(inner_wf).step(ReviewAgent)

    print("Collapsed view:")
    outer_wf.visualize(expand_nested=False)
    print("\nExpanded view:")
    outer_wf.visualize(expand_nested=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PART 2 — Agent spawning a workflow via self.spawn()
# ═══════════════════════════════════════════════════════════════════════════════


class OrchestratingAgent(Agent):
    """High-level orchestrator that delegates deep research to a sub-workflow."""

    model = _MODEL
    system_prompt = (
        "You are a strategic orchestrator. You delegate deep research to a specialised "
        "sub-workflow and incorporate the findings into a concise strategic brief."
    )

    async def arun(self, input_text: str) -> Response[str]:
        research_wf = (
            Workflow("research-sub-wf").step(GatherAgent).step(ExtractAgent).step(WriteAgent)
        )

        spawn_result: SpawnResult = await self.spawn(
            research_wf,
            task=input_text,
            budget=1.00,
        )

        return Response(
            content=(
                f"Strategic Brief (Orchestrator)\n\n"
                f"Sub-workflow completed in ${spawn_result.cost:.4f}.\n\n"
                f"Findings:\n{spawn_result.content}"
            ),
            cost=0.0,
        )


async def example_agent_spawns_workflow() -> None:
    print("\n── Part 2: Agent spawning a workflow ────────────────────────────")

    swarm = Swarm(
        agents=[OrchestratingAgent()],
        goal="Analyse AI market opportunities — delegate research to sub-workflow",
        budget=Budget(
            max_cost=3.00,
        ),
    )

    result = await swarm.run()
    print(result.content[:500])
    print(f"\nSwarm cost breakdown: {result.cost_breakdown}")


async def example_multiple_spawns() -> None:
    print("\n── Part 2b: Agent spawning two independent sub-workflows ────────")

    class DualResearchOrchestrator(Agent):
        """Spawns two separate sub-workflows covering different research angles."""

        model = _MODEL
        system_prompt = (
            "You coordinate research across two market segments and synthesise the findings "
            "into a comparative investment brief."
        )

        async def arun(self, input_text: str) -> Response[str]:
            enterprise_wf = Workflow("enterprise-research").step(GatherAgent).step(WriteAgent)
            startup_wf = Workflow("startup-research").step(GatherAgent).step(WriteAgent)

            enterprise_result: SpawnResult = await self.spawn(
                enterprise_wf, task="Enterprise AI adoption and spending", budget=0.50
            )
            startup_result: SpawnResult = await self.spawn(
                startup_wf, task="AI startup ecosystem and funding rounds", budget=0.50
            )

            combined = (
                f"Comparative Brief\n\n"
                f"[Enterprise Segment]\n{enterprise_result.content}\n\n"
                f"[Startup Ecosystem]\n{startup_result.content}\n\n"
                f"Sub-workflow costs: enterprise=${enterprise_result.cost:.4f}, "
                f"startups=${startup_result.cost:.4f}"
            )
            return Response(content=combined, cost=0.0)

    swarm = Swarm(
        agents=[DualResearchOrchestrator()],
        goal="Dual-segment AI market research",
        budget=Budget(
            max_cost=3.00,
        ),
    )

    result = await swarm.run()
    print(result.content[:500])


# ── Main ──────────────────────────────────────────────────────────────────────


async def main() -> None:
    await example_nested_workflow()
    await example_nested_visualize()
    await example_agent_spawns_workflow()
    await example_multiple_spawns()
    print("\nAll nested workflow examples completed.")


if __name__ == "__main__":
    asyncio.run(main())
