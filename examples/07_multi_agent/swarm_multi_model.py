"""Multi-model swarm — heterogeneous LLM panel on the same goal.

Different LLMs have different strengths, training biases, and failure modes.
Running Claude, GPT-4o, and a third model on the same question gives you a
diverse panel: each agent brings a different perspective, and the merged result
captures reasoning that a single model might miss or distort.

Use this topology when:
  - Single-model blind spots are unacceptable (legal, medical, financial).
  - You want to benchmark model quality on your specific domain.
  - You need a cheap cross-check before committing to a high-stakes answer.

Requires:
    ANTHROPIC_API_KEY — for ClaudeAgent (claude-haiku-4-5-20251001)
    OPENAI_API_KEY    — for GPTAgent (gpt-4o-mini)

Both keys must be set. The example will not fall back to mocks — the whole
point is to compare real model outputs side by side.

Run:
    ANTHROPIC_API_KEY=sk-ant-... OPENAI_API_KEY=sk-... \\
        uv run python examples/07_multi_agent/swarm_multi_model.py
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

from dotenv import load_dotenv

from syrin import Agent, Budget, Model
from syrin.enums import SwarmTopology
from syrin.swarm import Swarm, SwarmConfig

load_dotenv(Path(__file__).resolve().parent.parent / ".env")


# ── Agent definitions ─────────────────────────────────────────────────────────
#
# Each agent uses a different provider. They receive the same goal, run in
# parallel, and each contributes a labelled response to the merged output.
# System prompts are identical so differences in output reflect model behaviour.

_SHARED_SYSTEM_PROMPT = (
    "You are a senior technology analyst. "
    "Given a technology or market topic, write a concise 3-5 sentence analysis "
    "covering the most important insight, the key risk, and a concrete recommendation. "
    "Be direct. Use specific numbers or examples where possible."
)


class ClaudeAnalystAgent(Agent):
    """Technology analyst powered by Claude claude-haiku-4-5-20251001."""

    model = Model.Anthropic(
        "claude-haiku-4-5-20251001",
        api_key=os.environ["ANTHROPIC_API_KEY"],
    )
    system_prompt = _SHARED_SYSTEM_PROMPT


class GPTAnalystAgent(Agent):
    """Technology analyst powered by GPT-4o-mini."""

    model = Model.OpenAI(
        "gpt-4o-mini",
        api_key=os.environ["OPENAI_API_KEY"],
    )
    system_prompt = _SHARED_SYSTEM_PROMPT


# ── Example 1: Side-by-side model comparison on the same topic ────────────────
#
# Both models receive the exact same goal. The SwarmResult.agent_results list
# preserves per-agent responses so you can compare outputs directly.


async def example_model_comparison() -> None:
    print("\n── Example 1: Side-by-side model comparison ─────────────────────")

    swarm = Swarm(
        agents=[ClaudeAnalystAgent(), GPTAnalystAgent()],
        goal="Large language model inference infrastructure — market opportunity 2025",
        config=SwarmConfig(topology=SwarmTopology.PARALLEL),
        budget=Budget(
            max_cost=0.10,
        ),
    )
    result = await swarm.run()

    # Print each model's response separately for comparison
    agent_classes = [ClaudeAnalystAgent, GPTAnalystAgent]
    for agent_cls, response in zip(agent_classes, result.agent_results, strict=False):
        print(f"\n[{agent_cls.__name__}]")
        print(response.content)

    if result.budget_report:
        print("\nCost breakdown:")
        for entry in result.budget_report.per_agent:
            print(f"  {entry.agent_name}: ${entry.spent:.4f}")


# ── Example 2: Model panel on a contentious technical question ────────────────
#
# Where models diverge is often where the interesting signal lives. Disagreement
# on a technical topic reveals model-specific biases or training data differences.


async def example_technical_panel() -> None:
    print("\n── Example 2: Multi-model technical panel ───────────────────────")

    swarm = Swarm(
        agents=[ClaudeAnalystAgent(), GPTAnalystAgent()],
        goal=(
            "Is retrieval-augmented generation (RAG) a transitional architecture "
            "that will be superseded by longer context windows, or a durable pattern "
            "that will remain central to production AI systems?"
        ),
        config=SwarmConfig(topology=SwarmTopology.PARALLEL),
        budget=Budget(
            max_cost=0.10,
        ),
    )
    result = await swarm.run()

    for agent_cls, response in zip(
        [ClaudeAnalystAgent, GPTAnalystAgent], result.agent_results, strict=False
    ):
        print(f"\n[{agent_cls.__name__}]")
        print(response.content)


# ── Example 3: Cost comparison across providers ───────────────────────────────
#
# Multi-model swarms let you measure cost efficiency per provider for a given
# task. Use budget_report.per_agent to build cost-per-quality benchmarks.


async def example_cost_comparison() -> None:
    print("\n── Example 3: Cost comparison across providers ───────────────────")

    swarm = Swarm(
        agents=[ClaudeAnalystAgent(), GPTAnalystAgent()],
        goal="Open-source LLM ecosystem — top 3 models to watch in 2025",
        config=SwarmConfig(topology=SwarmTopology.PARALLEL),
        budget=Budget(
            max_cost=0.10,
        ),
    )
    result = await swarm.run()

    if result.budget_report:
        print(f"{'Agent':<30}  {'Cost':>8}  {'Tokens approx':>14}")
        print("-" * 58)
        for entry in result.budget_report.per_agent:
            # Cost-per-token estimate is illustrative (actual depends on model pricing)
            estimated_tokens = int(entry.spent / 0.00000015)  # rough gpt-4o-mini rate
            print(f"{entry.agent_name:<30}  ${entry.spent:>7.5f}  {estimated_tokens:>14,}")
        print(f"\nTotal: ${result.budget_report.total_spent:.5f}")


# ── Main ──────────────────────────────────────────────────────────────────────


async def main() -> None:
    await example_model_comparison()
    await example_technical_panel()
    await example_cost_comparison()
    print("\nAll multi-model swarm examples completed.")


if __name__ == "__main__":
    asyncio.run(main())
