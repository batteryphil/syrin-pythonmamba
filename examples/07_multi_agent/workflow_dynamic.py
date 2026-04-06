"""Workflow dynamic fan-out — spawn N agents determined at runtime.

Use `.dynamic(fn)` when the number or identity of agents cannot be known until
the previous step's output is available.  The factory function receives the
current HandoffContext and returns a list of ``(agent_class, task, budget_usd)``
tuples — one tuple per agent to spawn.

Key concepts shown here:
- wf.dynamic(fn, max_agents=N) — runtime fan-out
- Factory function signature: (HandoffContext) → [(agent_class, task, budget_usd)]
- max_agents enforces an upper bound; exceeding it raises DynamicFanoutError
- All dynamically-spawned agents run concurrently
- Their concatenated output is forwarded to the next .step()
- DynamicFanoutError is raised before any LLM calls when the limit is exceeded

Run:
    OPENAI_API_KEY=sk-... uv run python examples/07_multi_agent/workflow_dynamic.py
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

from dotenv import load_dotenv

from syrin import Agent, Budget, Model
from syrin.workflow import Workflow
from syrin.workflow.exceptions import DynamicFanoutError

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

_MODEL = Model.OpenAI("gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"))

# ── Agent definitions ─────────────────────────────────────────────────────────


class TopicPlannerAgent(Agent):
    """Identifies sub-topics for analysis, one per line."""

    model = _MODEL
    system_prompt = (
        "You identify exactly 3 key sub-topics that require separate analysis. "
        "Output ONLY a plain list with one sub-topic per line. No numbering, no bullets, "
        "no extra text — just 3 lines."
    )


class TopicAnalystAgent(Agent):
    """Performs a focused deep-dive on a single market sub-topic."""

    model = _MODEL
    system_prompt = (
        "You perform a focused deep-dive analysis on a single market sub-topic. "
        "Provide 2-3 key findings with supporting data points."
    )


class SynthesisAgent(Agent):
    """Synthesises parallel analysis results into a unified view."""

    model = _MODEL
    system_prompt = (
        "You synthesise multiple parallel analysis streams into a unified market view. "
        "Identify the cross-cutting theme and provide a single strategic recommendation."
    )


# ── Dynamic factory functions ─────────────────────────────────────────────────
#
# The factory receives the HandoffContext from the previous step and returns
# one (agent_class, task, budget_usd) tuple per topic.


def _topic_factory(ctx: object) -> list[tuple[type[TopicAnalystAgent], str, float]]:
    """Parse one sub-topic per line from ctx.content, spawn one analyst per topic."""
    from syrin.workflow._context import HandoffContext  # noqa: PLC0415

    content: str = ctx.content if isinstance(ctx, HandoffContext) else str(ctx)  # type: ignore[union-attr]
    topics = [line.strip() for line in content.splitlines() if line.strip()]
    return [(TopicAnalystAgent, topic, 0.10) for topic in topics]


# ── Example 1: Basic dynamic fan-out ─────────────────────────────────────────
#
# TopicPlannerAgent outputs 3 sub-topics (one per line).
# The dynamic step parses those topics and spawns one analyst per topic.
# max_agents=5 protects against runaway fan-out.


async def example_basic_dynamic() -> None:
    print("\n── Example 1: Basic dynamic fan-out ─────────────────────────────")

    wf = (
        Workflow("dynamic-market-analysis")
        .step(TopicPlannerAgent)
        .dynamic(_topic_factory, max_agents=5, label="per-topic-analyst")
        .step(SynthesisAgent)
    )

    result = await wf.run("AI infrastructure investment landscape — identify sub-topics to analyse")
    print(result.content)
    print(f"\nCost: ${result.cost:.6f}")


# ── Example 2: Inline lambda factory ─────────────────────────────────────────
#
# Write the factory as an inline lambda for simple parsing logic.


class TwoTopicPlannerAgent(Agent):
    """Produces exactly two research topics separated by a pipe character."""

    model = _MODEL
    system_prompt = (
        "You identify exactly 2 research topics for the given domain. "
        "Output ONLY the two topics separated by a pipe character (|). "
        "Example format: market sizing | competitive dynamics"
    )


class ResearchAgent(Agent):
    """Researches a single focused topic."""

    model = _MODEL
    system_prompt = (
        "You research a single market topic. "
        "Provide 3 key data points and a one-sentence conclusion."
    )


async def example_inline_lambda() -> None:
    print("\n── Example 2: Inline lambda factory ────────────────────────────")

    wf = (
        Workflow("lambda-dynamic")
        .step(TwoTopicPlannerAgent)
        .dynamic(
            lambda ctx: [
                (ResearchAgent, topic.strip(), 0.10)
                for topic in ctx.content.split("|")
                if topic.strip()
            ],
            max_agents=10,
            label="topic-researchers",
        )
        .step(SynthesisAgent)
    )

    result = await wf.run("Cloud computing market")
    print(f"Result: {result.content[:300]}")
    print(f"\nCost: ${result.cost:.6f}")


# ── Example 3: DynamicFanoutError — factory exceeds max_agents ───────────────
#
# If the factory returns more tuples than max_agents, the workflow raises
# DynamicFanoutError BEFORE running any agents.


class BroadPlannerAgent(Agent):
    """Returns 6 topics to analyse — more than the capped workflow allows."""

    model = _MODEL
    system_prompt = (
        "You identify exactly 6 distinct sub-topics for market analysis. "
        "Output ONLY one sub-topic per line with no extra text — just 6 lines."
    )


async def example_fanout_error() -> None:
    print("\n── Example 3: DynamicFanoutError (max_agents exceeded) ──────────")

    wf = (
        Workflow("capped-dynamic")
        .step(BroadPlannerAgent)
        # max_agents=3 but the planner returns 6 topics → DynamicFanoutError
        .dynamic(_topic_factory, max_agents=3, label="capped-analyst")
        .step(SynthesisAgent)
    )

    try:
        await wf.run("Identify topics for a broad market study")
    except DynamicFanoutError as exc:
        print(f"Caught DynamicFanoutError: {exc}")
        print(f"  Actual agents requested: {exc.actual}")
        print(f"  Configured max_agents:   {exc.maximum}")


# ── Example 4: Budget-aware dynamic factory ───────────────────────────────────
#
# The factory reads ctx.budget_remaining to allocate per-agent budget
# dynamically and avoid blowing past the workflow ceiling.


async def example_budget_aware_factory() -> None:
    print("\n── Example 4: Budget-aware dynamic factory ──────────────────────")

    def _budget_aware_factory(
        ctx: object,
    ) -> list[tuple[type[TopicAnalystAgent], str, float]]:
        from syrin.workflow._context import HandoffContext  # noqa: PLC0415

        content: str = ctx.content if isinstance(ctx, HandoffContext) else str(ctx)  # type: ignore[union-attr]
        budget_remaining: float = (
            ctx.budget_remaining if isinstance(ctx, HandoffContext) else 1.0  # type: ignore[union-attr]
        )
        topics = [line.strip() for line in content.splitlines() if line.strip()]
        per_agent = budget_remaining / max(len(topics), 1)
        return [(TopicAnalystAgent, topic, per_agent) for topic in topics]

    wf = (
        Workflow("budget-aware-dynamic", budget=Budget(max_cost=1.00))
        .step(TopicPlannerAgent)
        .dynamic(_budget_aware_factory, max_agents=10, label="budget-analyst")
        .step(SynthesisAgent)
    )

    result = await wf.run("SaaS market competitive landscape — identify sub-topics")
    print(f"Result: {result.content[:300]}")
    print(f"Cost:   ${result.cost:.6f}")


# ── Main ──────────────────────────────────────────────────────────────────────


async def main() -> None:
    await example_basic_dynamic()
    await example_inline_lambda()
    await example_fanout_error()
    await example_budget_aware_factory()
    print("\nAll dynamic workflow examples completed.")


if __name__ == "__main__":
    asyncio.run(main())
