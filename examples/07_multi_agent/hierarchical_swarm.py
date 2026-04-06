"""Hierarchical swarm — org-chart-style agent delegation.

The ``Agent.team`` ClassVar lets you declare direct-report relationships between
agent classes. When a Swarm initialises, it automatically expands the team tree
into the full agent pool and sets ``_supervisor_id`` on each team member. All
agents in the hierarchy then run concurrently under the PARALLEL topology.

This mirrors how real organisations work: the CEO sets the goal, the leadership
team translates it into functional plans, and the individual contributors execute
in parallel — all reporting back to their direct supervisor.

Use this pattern when:
  - You want to model a real org structure in code with typed class relationships.
  - Different parts of the hierarchy need domain-specific prompts without coupling
    the top-level goal to implementation details.
  - You want concurrent execution across all levels with budget isolation.

Requires:
    OPENAI_API_KEY — set in your environment before running.

Run:
    OPENAI_API_KEY=sk-... uv run python examples/07_multi_agent/hierarchical_swarm.py
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import ClassVar

from dotenv import load_dotenv

from syrin import Agent, Budget, Model
from syrin.enums import SwarmTopology
from syrin.swarm import Swarm, SwarmConfig

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

_MODEL = Model.OpenAI("gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"))


# ── Leaf workers ──────────────────────────────────────────────────────────────
#
# Individual contributors at the bottom of the hierarchy. Their system prompts
# are narrowly scoped to a single function.


class BackendEngineer(Agent):
    """Designs and implements the server-side components for a product feature."""

    model = _MODEL
    system_prompt = (
        "You are a senior backend engineer. Given a product feature goal, describe in "
        "2-3 sentences what backend work is required: API design, data model changes, "
        "and the key technical risk. Be specific and use concrete technology names."
    )


class FrontendEngineer(Agent):
    """Builds the user interface components and API integration layer."""

    model = _MODEL
    system_prompt = (
        "You are a senior frontend engineer. Given a product feature goal, describe in "
        "2-3 sentences what frontend work is required: component structure, state "
        "management approach, and UX considerations. Be specific."
    )


class DataEngineer(Agent):
    """Designs data schemas, pipelines, and the analytics instrumentation."""

    model = _MODEL
    system_prompt = (
        "You are a senior data engineer. Given a product feature goal, describe in "
        "2-3 sentences the data infrastructure required: schema design, pipeline "
        "changes, and instrumentation for analytics. Be specific."
    )


class ContentWriter(Agent):
    """Produces the marketing copy, onboarding content, and help documentation."""

    model = _MODEL
    system_prompt = (
        "You are a senior content strategist. Given a product feature goal, describe "
        "in 2-3 sentences the content deliverables required: launch copy, onboarding "
        "flow, and documentation structure. Focus on user outcomes."
    )


class SEOSpecialist(Agent):
    """Handles keyword strategy, metadata, and organic search positioning."""

    model = _MODEL
    system_prompt = (
        "You are an SEO specialist. Given a product feature goal, describe in 2-3 "
        "sentences the SEO strategy: target keywords, meta structure, and the organic "
        "growth lever this feature creates. Be specific."
    )


# ── Middle management ─────────────────────────────────────────────────────────
#
# Team leads translate the C-suite goal into functional plans and coordinate
# their direct reports. Each lead's team ClassVar declares the direct reports
# that Swarm will expand into the agent pool.


class EngineeringLead(Agent):
    """Translates product goals into engineering plans across backend, frontend, and data."""

    model = _MODEL
    system_prompt = (
        "You are an engineering team lead. Given a product feature goal, write a "
        "2-3 sentence technical plan: the core architectural decision, the cross-team "
        "dependency that needs early alignment, and the delivery risk."
    )

    team: ClassVar[list[type[Agent]]] = [BackendEngineer, FrontendEngineer, DataEngineer]


class MarketingLead(Agent):
    """Drives the go-to-market strategy: messaging, content, and organic acquisition."""

    model = _MODEL
    system_prompt = (
        "You are a marketing team lead. Given a product feature goal, write a 2-3 "
        "sentence go-to-market plan: the target segment, the core message, and the "
        "primary distribution channel for launch."
    )

    team: ClassVar[list[type[Agent]]] = [ContentWriter, SEOSpecialist]


# ── C-suite ────────────────────────────────────────────────────────────────────
#
# Executive layer: set strategic direction and delegate to functional leads.


class CTO(Agent):
    """Owns the technical vision and ensures engineering execution aligns with strategy."""

    model = _MODEL
    system_prompt = (
        "You are the CTO. Given a product feature goal, write a 2-3 sentence technical "
        "strategy statement: the architectural principle guiding the build, the platform "
        "bet being made, and the engineering investment required."
    )

    team: ClassVar[list[type[Agent]]] = [EngineeringLead]


class CMO(Agent):
    """Owns growth, brand, and market positioning across all channels."""

    model = _MODEL
    system_prompt = (
        "You are the CMO. Given a product feature goal, write a 2-3 sentence marketing "
        "strategy statement: the positioning angle, the target buyer persona, and the "
        "metric that will signal successful adoption."
    )

    team: ClassVar[list[type[Agent]]] = [MarketingLead]


class CEO(Agent):
    """Sets the company vision and ensures all functions are aligned and executing."""

    model = _MODEL
    system_prompt = (
        "You are the CEO. Given a product feature goal, write a 2-3 sentence strategic "
        "framing for the company: why this matters for the long-term position, what "
        "customer problem it solves definitively, and the one number that will prove success."
    )

    # The CEO's direct reports — Swarm expands the full tree recursively
    team: ClassVar[list[type[Agent]]] = [CTO, CMO]


# ── Example 1: Full company hierarchy — one call, entire org executes ─────────
#
# Pass only the CEO to Swarm. The team expansion resolves the full hierarchy
# automatically: CEO → CTO + CMO → EngineeringLead + MarketingLead → workers.


async def example_full_hierarchy() -> None:
    print("\n── Example 1: Full company hierarchy swarm ──────────────────────")
    print("   CEO → [CTO, CMO] → [EngineeringLead, MarketingLead] → workers\n")

    swarm = Swarm(
        agents=[CEO()],
        goal="Launch an AI-powered code review feature that surfaces security vulnerabilities "
        "in pull requests before merge — targeting enterprise engineering teams.",
        config=SwarmConfig(topology=SwarmTopology.PARALLEL),
        budget=Budget(
            max_cost=0.50,
        ),
    )
    print(f"Agent pool after hierarchy expansion: {swarm.agent_count} agents\n")

    result = await swarm.run()

    print("Combined output from all levels:\n")
    print(result.content)

    if result.budget_report:
        print("\nCost breakdown:")
        for entry in result.budget_report.per_agent:
            print(f"  {entry.agent_name:<30}  ${entry.spent:.4f}")
        print(f"  {'TOTAL':<30}  ${result.budget_report.total_spent:.4f}")


# ── Example 2: Inspect the supervisor chain ───────────────────────────────────
#
# After Swarm expands the hierarchy, each agent carries a _supervisor_id that
# points to its direct supervisor. Introspect the chain to verify the structure.


async def example_supervisor_chain() -> None:
    print("\n── Example 2: Supervisor chain inspection ───────────────────────")

    swarm = Swarm(agents=[CEO()], goal="inspect hierarchy")

    print("  Agent                          Supervisor")
    print("  " + "-" * 54)
    for agent in swarm._agents:
        supervisor_id = getattr(agent, "_supervisor_id", None)
        name = type(agent).__name__
        print(f"  {name:<30}  {supervisor_id or '(root)'}")


# ── Example 3: Agent statuses after run ───────────────────────────────────────


async def example_agent_statuses() -> None:
    print("\n── Example 3: Agent status snapshot after run ───────────────────")

    swarm = Swarm(
        agents=[CEO()],
        goal="Build a self-serve analytics dashboard for enterprise customers.",
        config=SwarmConfig(topology=SwarmTopology.PARALLEL),
        budget=Budget(
            max_cost=0.50,
        ),
    )
    await swarm.run()

    entries = swarm.agent_statuses()
    print(f"  Status snapshot ({len(entries)} agents):")
    for entry in entries:
        print(f"  {entry.agent_name:<30}  {entry.state}")


# ── Main ──────────────────────────────────────────────────────────────────────


async def main() -> None:
    await example_full_hierarchy()
    await example_supervisor_chain()
    await example_agent_statuses()
    print("\nAll hierarchical swarm examples completed.")


if __name__ == "__main__":
    asyncio.run(main())
