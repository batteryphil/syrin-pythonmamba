"""Workflow visualization — Rich tree, Mermaid export, and live show_graph.

Three ways to see your workflow graph:

1. wf.visualize()          — Print a static Rich ASCII tree to stdout.
2. wf.to_mermaid()         — Return a Mermaid ``graph TD`` string for embedding
                             in docs, GitHub README, or a web UI.
3. wf.run(show_graph=True) — Live Rich table that updates as steps complete,
                             showing status (PENDING/RUNNING/COMPLETE/FAILED),
                             cost, and elapsed time per step.

This example also shows how Pipeline and Swarm have their own visualize() methods
for comparison.

Run:
    uv run python examples/07_multi_agent/workflow_visualization.py
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from syrin import Agent, Model
from syrin.enums import SwarmTopology
from syrin.response import Response
from syrin.swarm import Swarm, SwarmConfig
from syrin.workflow import Workflow

_MODEL = Model.OpenAI("gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"))

# ── Agent definitions ─────────────────────────────────────────────────────────


class PlannerAgent(Agent):
    model = _MODEL
    system_prompt = "You create research plans."

    async def arun(self, input_text: str) -> Response[str]:
        return Response(content="Plan: 1) research, 2) analyse, 3) write.", cost=0.01)


class ResearchAgent(Agent):
    model = _MODEL
    system_prompt = "You gather market research."

    async def arun(self, input_text: str) -> Response[str]:
        return Response(content="Research: AI market $207B, CAGR 44%.", cost=0.02)


class AnalysisAgent(Agent):
    model = _MODEL
    system_prompt = "You analyse market research."

    async def arun(self, input_text: str) -> Response[str]:
        return Response(content="Analysis: strong growth, manageable competition.", cost=0.02)


class WriterAgent(Agent):
    model = _MODEL
    system_prompt = "You write executive reports."

    async def arun(self, input_text: str) -> Response[str]:
        return Response(content="Report: invest in AI. Growth trajectory robust.", cost=0.01)


class EditorAgent(Agent):
    model = _MODEL
    system_prompt = "You review and edit reports."

    async def arun(self, input_text: str) -> Response[str]:
        return Response(content=f"APPROVED. {input_text}", cost=0.005)


# ── Example 1: Static visualize() ────────────────────────────────────────────
#
# wf.visualize() prints a Rich ASCII tree to stdout.
# No run is needed — it works on the workflow definition alone.


async def example_static_visualize() -> None:
    print("\n── Example 1: wf.visualize() — static Rich tree ─────────────────")

    wf = (
        Workflow("research-pipeline")
        .step(PlannerAgent)
        .parallel([ResearchAgent, AnalysisAgent])
        .step(WriterAgent)
        .step(EditorAgent)
    )

    # Prints a Rich-formatted tree showing each step and its type
    wf.visualize()

    # expand_nested=True shows nested sub-workflows inline when present
    print("\n  (expand_nested=True — same result when no nesting):")
    wf.visualize(expand_nested=True)


# ── Example 2: Mermaid export ─────────────────────────────────────────────────
#
# wf.to_mermaid() returns a Mermaid diagram string.
# Paste it into https://mermaid.live or embed in a GitHub README.


async def example_mermaid_export() -> None:
    print("\n── Example 2: wf.to_mermaid() — Mermaid string ──────────────────")

    wf = (
        Workflow("research-pipeline")
        .step(PlannerAgent)
        .parallel([ResearchAgent, AnalysisAgent])
        .step(WriterAgent)
        .step(EditorAgent)
    )

    # Top-down Mermaid diagram (default direction="TD")
    mermaid_td = wf.to_mermaid()
    print("Top-down (TD):")
    print(mermaid_td)

    # Left-right variant
    mermaid_lr = wf.to_mermaid(direction="LR")
    print("\nLeft-right (LR):")
    print(mermaid_lr)


# ── Example 3: to_dict() — JSON-serialisable graph ────────────────────────────
#
# wf.to_dict() returns {"nodes": [...], "edges": [...]} for custom rendering.


async def example_to_dict() -> None:
    print("\n── Example 3: wf.to_dict() — JSON-serialisable graph ────────────")

    wf = (
        Workflow("research-pipeline")
        .step(PlannerAgent)
        .parallel([ResearchAgent, AnalysisAgent])
        .step(WriterAgent)
    )

    graph = wf.to_dict()
    print("Nodes:")
    for node in graph.get("nodes", []):
        print(f"  {node}")
    print("Edges:")
    for edge in graph.get("edges", []):
        print(f"  {edge}")


# ── Example 4: Live show_graph=True ──────────────────────────────────────────
#
# Pass show_graph=True to wf.run() to see a live Rich table that refreshes
# after each step: PENDING → RUNNING → COMPLETE (with cost and elapsed time).
# Failed steps appear as FAILED; steps that did not run appear as SKIPPED.


async def example_live_graph() -> None:
    print("\n── Example 4: wf.run(show_graph=True) — live table ─────────────")

    wf = (
        Workflow("live-graph-demo")
        .step(PlannerAgent)
        .parallel([ResearchAgent, AnalysisAgent])
        .step(WriterAgent)
        .step(EditorAgent)
    )

    # The table is rendered to the terminal as steps complete.
    # In non-TTY environments it still runs but the table may appear all at once.
    result = await wf.run("AI market trends", show_graph=True)
    print(f"\nFinal result: {result.content[:100]}...")
    print(f"Cost: ${result.cost:.6f}")


# ── Example 5: Compare Workflow vs Pipeline vs Swarm visualize() ──────────────
#
# Each multi-agent primitive has its own visualize() with a topology-appropriate
# output:
#   - Workflow: Rich tree with step types (sequential, parallel, branch, dynamic)
#   - Pipeline: agent → agent → agent chain
#   - Swarm: topology name + agent pool + budget + MemoryBus (if set)


async def example_compare_visualizations() -> None:
    print("\n── Example 5: Compare Workflow / Pipeline / Swarm visualize() ───")

    # Workflow
    print("\n[Workflow.visualize()]")
    wf = (
        Workflow("research-workflow")
        .step(PlannerAgent)
        .parallel([ResearchAgent, AnalysisAgent])
        .step(WriterAgent)
    )
    wf.visualize()

    # Pipeline — list agent steps manually (Pipeline.visualize() not available;
    # use a Workflow wrapper to get visual output for linear chains)
    print("\n[Pipeline as Workflow.visualize()]")
    pipeline_wf = Workflow("pipeline-view")
    pipeline_wf.step(PlannerAgent).step(WriterAgent).step(EditorAgent)
    pipeline_wf.visualize()

    # Swarm — shows topology + agent pool
    print("\n[Swarm.visualize()]")
    swarm = Swarm(
        agents=[ResearchAgent(), AnalysisAgent(), WriterAgent()],
        goal="AI market analysis",
        config=SwarmConfig(topology=SwarmTopology.PARALLEL),
    )
    swarm.visualize()


# ── Example 6: Branch and dynamic steps in the Mermaid output ─────────────────


class DetailAgent(Agent):
    model = _MODEL
    system_prompt = "You write detailed reports."

    async def arun(self, input_text: str) -> Response[str]:
        return Response(content="Detailed: " + input_text, cost=0.02)


class BriefAgent(Agent):
    model = _MODEL
    system_prompt = "You write brief summaries."

    async def arun(self, input_text: str) -> Response[str]:
        return Response(content="Brief: " + input_text, cost=0.01)


async def example_complex_mermaid() -> None:
    print("\n── Example 6: Complex workflow Mermaid (branch + dynamic) ───────")

    wf = (
        Workflow("complex-workflow")
        .step(PlannerAgent)
        .branch(
            condition=lambda ctx: len(ctx.content) > 20,
            if_true=DetailAgent,
            if_false=BriefAgent,
        )
        .dynamic(
            lambda ctx: [(WriterAgent, ctx.content[:50], 0.10)],
            max_agents=5,
            label="per-section",
        )
        .step(EditorAgent)
    )

    print(wf.to_mermaid())


# ── Main ──────────────────────────────────────────────────────────────────────


async def main() -> None:
    await example_static_visualize()
    await example_mermaid_export()
    await example_to_dict()
    await example_live_graph()
    await example_compare_visualizations()
    await example_complex_mermaid()
    print("\nAll visualization examples completed.")


if __name__ == "__main__":
    asyncio.run(main())
