"""Budget delegation — spawn child agents with per-spawn budget slices.

Shows how an orchestrator agent can spawn sub-agents inside a Swarm,
each getting a specific portion of the shared budget pool.

Key concepts:
  - Agent.spawn(AgentClass, task="...", budget=0.25) — sync spawn with budget
  - Agent.spawn_many([SpawnSpec(...)]) — concurrent spawn of multiple children
  - SpawnResult.content, .cost, .budget_remaining, .child_agent_id
  - BudgetAllocationError — raised when pool is exhausted
  - Budget(max_cost=N) — pool configuration; sharing is automatic in Swarm

Run:
    OPENAI_API_KEY=sk-... uv run python examples/07_multi_agent/budget_delegation.py
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

from dotenv import load_dotenv

from syrin import Agent, Budget, Model
from syrin.response import Response
from syrin.swarm import SpawnSpec, Swarm

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

_MODEL = Model.OpenAI("gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"))

# ── Sub-agent definitions ─────────────────────────────────────────────────────


class ResearchSubAgent(Agent):
    """Performs a focused deep-dive on a specific market research topic."""

    model = _MODEL
    system_prompt = (
        "You are a market research specialist. Provide detailed findings with "
        "3 key data points and supporting evidence for the given topic."
    )


class SummarySubAgent(Agent):
    """Condenses research findings into a concise executive brief."""

    model = _MODEL
    system_prompt = (
        "You are an executive summariser. Distil the provided research into "
        "a 2-sentence brief with one clear investment recommendation."
    )


class FactCheckSubAgent(Agent):
    """Verifies the accuracy of research claims against known market data."""

    model = _MODEL
    system_prompt = (
        "You are a fact-checker. Review the provided research claims, "
        "flag any that lack sufficient evidence, and provide an accuracy rating 1-10."
    )


# ── Orchestrator: spawns children from inside arun() ─────────────────────────
#
# An orchestrator inside a Swarm can call self.spawn() / self.spawn_many()
# to launch child agents that draw budget from the shared pool.


class ResearchOrchestratorAgent(Agent):
    """Orchestrates research, summarisation, and fact-checking children."""

    model = _MODEL
    system_prompt = (
        "You are a research orchestrator. You coordinate specialised sub-agents "
        "to produce verified, summarised research briefs."
    )

    async def arun(self, input_text: str) -> Response[str]:
        print(f"  [Orchestrator] starting research for: '{input_text[:60]}'")

        # Spawn a focused researcher with a dedicated budget slice
        research_result = await self._pool_spawn(
            ResearchSubAgent,
            task=input_text,
            budget_amount=0.20,
        )
        print(
            f"  [Orchestrator] research done  "
            f"cost=${research_result.cost:.4f}  "
            f"pool_remaining=${research_result.budget_remaining:.4f}"
        )

        # Concurrently spawn a summariser and a fact-checker
        concurrent_results = await self.spawn_many(
            [
                SpawnSpec(agent=SummarySubAgent, task=research_result.content, budget=0.10),
                SpawnSpec(agent=FactCheckSubAgent, task=research_result.content, budget=0.10),
            ]
        )
        for sr in concurrent_results:
            print(
                f"  [Orchestrator] child={sr.child_agent_id.split('::')[-1]}  cost=${sr.cost:.4f}"
            )

        total_child_cost = research_result.cost + sum(r.cost for r in concurrent_results)
        combined = (
            f"Research Brief\n\n"
            f"Findings:\n{research_result.content}\n\n"
            f"Summary:\n{concurrent_results[0].content if concurrent_results else 'n/a'}\n\n"
            f"Fact-Check:\n"
            f"{concurrent_results[1].content if len(concurrent_results) > 1 else 'n/a'}\n\n"
            f"Total child agent cost: ${total_child_cost:.4f}"
        )
        return Response(content=combined, cost=0.0)


# ── Example 1: Orchestrator with shared budget pool ───────────────────────────


async def example_basic_spawn() -> None:
    print("\n── Example 1: Orchestrator spawns children via shared pool ──────")

    budget = Budget(
        max_cost=1.00,
    )

    swarm = Swarm(
        agents=[ResearchOrchestratorAgent()],
        goal="Research AI agent framework adoption in enterprise software and produce a verified brief",
        budget=budget,
    )

    result = await swarm.run()

    print(f"\nFinal result:\n{result.content[:500]}")
    if result.budget_report:
        print(f"\nTotal spent: ${result.budget_report.total_spent:.4f}")


# ── Example 2: BudgetAllocationError when pool is exhausted ──────────────────


class TightBudgetOrchestrator(Agent):
    """Demonstrates graceful handling of pool exhaustion."""

    model = _MODEL
    system_prompt = "You coordinate a research task with tight budget constraints."

    async def arun(self, input_text: str) -> Response[str]:
        try:
            # Pool only has $0.05 total; child's work may exceed this
            result = await self._pool_spawn(
                ResearchSubAgent,
                task=input_text,
                budget_amount=0.04,
            )
            return Response(content=result.content, cost=result.cost)
        except Exception as exc:
            print(f"  [TightBudgetOrchestrator] budget error: {type(exc).__name__}: {exc}")
            return Response(
                content="Run aborted: budget exhausted before research could complete.",
                cost=0.0,
            )


async def example_budget_exhaustion() -> None:
    print("\n── Example 2: Budget exhaustion handling ────────────────────────")

    budget = Budget(
        max_cost=0.05,
    )

    swarm = Swarm(
        agents=[TightBudgetOrchestrator()],
        goal="Research quantum computing market with a very tight budget",
        budget=budget,
    )

    result = await swarm.run()
    print(f"Result: {result.content}")


# ── Example 3: Inspect all SpawnResult fields ─────────────────────────────────


class InspectorOrchestrator(Agent):
    """Spawns a single child and prints all SpawnResult fields."""

    model = _MODEL
    system_prompt = "You spawn a research child and inspect its result."

    async def arun(self, input_text: str) -> Response[str]:
        sr = await self._pool_spawn(
            ResearchSubAgent,
            task="AI model pricing trends and cost reduction rates",
            budget_amount=0.15,
        )

        print(f"  SpawnResult.content:           {sr.content[:80]}...")
        print(f"  SpawnResult.cost:              ${sr.cost:.4f}")
        print(f"  SpawnResult.budget_remaining:  ${sr.budget_remaining:.4f}")
        print(f"  SpawnResult.stop_reason:       {sr.stop_reason}")
        print(f"  SpawnResult.child_agent_id:    {sr.child_agent_id}")

        return Response(content=sr.content, cost=sr.cost)


async def example_spawn_result_fields() -> None:
    print("\n── Example 3: SpawnResult field inspection ──────────────────────")

    budget = Budget(
        max_cost=0.50,
    )

    swarm = Swarm(
        agents=[InspectorOrchestrator()],
        goal="Inspect spawn result attributes",
        budget=budget,
    )

    await swarm.run()


# ── Main ──────────────────────────────────────────────────────────────────────


async def main() -> None:
    await example_basic_spawn()
    await example_budget_exhaustion()
    await example_spawn_result_fields()
    print("\nAll budget delegation examples completed.")


if __name__ == "__main__":
    asyncio.run(main())
