"""Workflow conditional routing — branch() based on previous step output.

Use `.branch(condition, if_true, if_false)` when the next agent to run depends
on the output of the previous step.  The condition receives the current
HandoffContext and should return a truthy or falsy value.

Key concepts shown here:
- wf.branch(condition_fn, TrueAgent, FalseAgent) — conditional routing
- HandoffContext.content — the text from the previous step
- HandoffContext.budget_remaining — remaining budget at this point
- HandoffContext.step_index — zero-based index of the current step
- Chaining multiple branches in one workflow
- Branching on content length, keywords, and budget thresholds

Run:
    OPENAI_API_KEY=sk-... uv run python examples/07_multi_agent/workflow_conditional.py
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

from dotenv import load_dotenv

from syrin import Agent, Budget, Model
from syrin.workflow import Workflow

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

_MODEL = Model.OpenAI("gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"))

# ── Agent definitions ─────────────────────────────────────────────────────────


class ResearchAgent(Agent):
    """Gathers comprehensive market intelligence on the given topic."""

    model = _MODEL
    system_prompt = (
        "You are a research analyst. Gather and summarise comprehensive market intelligence "
        "including statistics, trends, key players, and growth drivers. Be thorough."
    )


class BriefResearchAgent(Agent):
    """Returns a concise, one-paragraph research summary."""

    model = _MODEL
    system_prompt = (
        "You are a research analyst. Give a concise, one-paragraph summary "
        "of the most important fact about the topic. Be brief."
    )


class DetailedReportAgent(Agent):
    """Writes a comprehensive multi-section market report."""

    model = _MODEL
    system_prompt = (
        "You write comprehensive, multi-section market reports for investment committees. "
        "Structure with sections: Overview, Key Findings, Risks, and Recommendation."
    )


class ExecutiveBriefAgent(Agent):
    """Writes a concise one-paragraph executive brief."""

    model = _MODEL
    system_prompt = (
        "You write concise executive briefs — a single paragraph distilling the most "
        "important insight and a clear recommendation. C-suite audience."
    )


class UrgentAlertAgent(Agent):
    """Writes a short action-oriented alert for time-sensitive findings."""

    model = _MODEL
    system_prompt = (
        "You write urgent, action-oriented executive alerts. "
        "Lead with the risk, state the immediate action required, and keep it under 100 words."
    )


class ConclusionAgent(Agent):
    """Appends a one-sentence strategic conclusion to any report."""

    model = _MODEL
    system_prompt = (
        "You append a single-sentence strategic conclusion to a report. "
        "Focus on the most important strategic implication for leadership."
    )


# ── Example 1: Branch on content length ──────────────────────────────────────
#
# Route to a detailed report writer when the research output is substantial
# (>300 chars), otherwise use a brief writer.


async def example_branch_on_length() -> None:
    print("\n── Example 1: Branch on content length ──────────────────────────")

    wf = (
        Workflow("conditional-research")
        .step(ResearchAgent)
        .branch(
            condition=lambda ctx: len(ctx.content) > 300,
            if_true=DetailedReportAgent,
            if_false=ExecutiveBriefAgent,
        )
    )

    result = await wf.run("AI semiconductor market investment landscape 2025")
    print(result.content[:400])
    print(f"\nCost: ${result.cost:.6f}")


# ── Example 2: Branch on keyword in content ───────────────────────────────────
#
# Route to an urgent alert when the research mentions risk-related terms,
# otherwise produce a standard detailed report.


async def example_branch_on_keyword() -> None:
    print("\n── Example 2: Branch on keyword in content ──────────────────────")

    risk_keywords = ("risk", "decline", "disruption", "threat", "warning", "concern")

    wf = (
        Workflow("keyword-branch")
        .step(ResearchAgent)
        .branch(
            condition=lambda ctx: any(kw in ctx.content.lower() for kw in risk_keywords),
            if_true=UrgentAlertAgent,
            if_false=DetailedReportAgent,
        )
    )

    result = await wf.run("Cybersecurity market threats 2025")
    routed_to = "UrgentAlert" if len(result.content) < 400 else "DetailedReport"
    print(f"Routed to: {routed_to}")
    print(result.content[:300])
    print(f"\nCost: ${result.cost:.6f}")


# ── Example 3: Branch on budget remaining ─────────────────────────────────────
#
# HandoffContext.budget_remaining lets you route to a cheaper agent when the
# remaining budget is tight.


async def example_branch_on_budget() -> None:
    print("\n── Example 3: Branch on budget remaining ────────────────────────")

    wf = (
        Workflow("budget-branch", budget=Budget(max_cost=1.00))
        .step(ResearchAgent)
        .branch(
            condition=lambda ctx: ctx.budget_remaining > 0.50,
            if_true=DetailedReportAgent,
            if_false=ExecutiveBriefAgent,
        )
    )

    result = await wf.run("Cloud infrastructure spending trends")
    print(f"Result ({len(result.content)} chars): {result.content[:200]}...")
    print(f"Budget remaining at completion: ${result.budget_remaining:.4f}")


# ── Example 4: Chained branches with final step ───────────────────────────────
#
# Chain multiple `.branch()` calls into a decision tree, then always run a
# final conclusion step regardless of which path was taken.


async def example_chained_branches() -> None:
    print("\n── Example 4: Chained branches + final step ─────────────────────")

    risk_keywords = ("risk", "decline", "disruption", "threat", "warning")

    wf = (
        Workflow("chained-branches")
        .step(ResearchAgent)
        .branch(
            condition=lambda ctx: any(kw in ctx.content.lower() for kw in risk_keywords),
            if_true=UrgentAlertAgent,
            if_false=DetailedReportAgent,
        )
        .step(ConclusionAgent)
    )

    result = await wf.run("Fintech regulatory changes 2025")
    print(result.content[:400])
    print(f"\nSteps completed: {wf.step_count}")
    print(f"Cost: ${result.cost:.6f}")


# ── Example 5: Brief research → always takes the short path ──────────────────
#
# Use BriefResearchAgent (short output) and verify the branch routes correctly.


async def example_brief_path() -> None:
    print("\n── Example 5: Brief research → ExecutiveBrief path ─────────────")

    wf = (
        Workflow("brief-path")
        .step(BriefResearchAgent)
        .branch(
            condition=lambda ctx: len(ctx.content) > 500,
            if_true=DetailedReportAgent,
            if_false=ExecutiveBriefAgent,
        )
    )

    result = await wf.run("AI market one-liner")
    print(result.content)
    print(f"\nCost: ${result.cost:.6f}")


# ── Main ──────────────────────────────────────────────────────────────────────


async def main() -> None:
    await example_branch_on_length()
    await example_branch_on_keyword()
    await example_branch_on_budget()
    await example_chained_branches()
    await example_brief_path()
    print("\nAll conditional workflow examples completed.")


if __name__ == "__main__":
    asyncio.run(main())
