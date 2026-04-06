---
title: Hierarchical Agent Composition
description: Agent.team, parent-child hierarchies, supervisor_id, authority chain inheritance, and the company org-chart pattern
weight: 77
---

## Overview

syrin supports hierarchical agent composition through `Agent.team`.  Setting `team` on an agent class declares that it manages a group of child agents.  When placed in a `Swarm`, the parent automatically spawns its team members and gains `CONTROL` + `CONTEXT` permissions over them.

## `Agent.team` — Declare a Team

```python
from syrin import Agent, Model
from syrin.enums import AgentRole

class BackendEngineer(Agent):
    model = Model.mock()
    system_prompt = "Implement backend services."

class FrontendEngineer(Agent):
    model = Model.mock()
    system_prompt = "Implement frontend components."

class EngineeringLead(Agent):
    model = Model.mock()
    system_prompt = "Coordinate engineering tasks."
    team = [BackendEngineer, FrontendEngineer]   # ClassVar
```

`team` is a `ClassVar[list[type[Agent]]]`.  It is read by the `Swarm` at construction time.

## How Swarm Expands the Team

When a `Swarm` is constructed with `agents=[EngineeringLead()]`, it:

1. Detects `EngineeringLead.team = [BackendEngineer, FrontendEngineer]`.
2. Instantiates each team member.
3. Appends them to `swarm._agents`.
4. Sets `member._supervisor_id = parent.agent_id` on each member.
5. Records `swarm._team_map[parent_id] = [member_ids]` for the authority system.

Nesting is recursive: if `BackendEngineer.team = [DatabaseEngineer]`, `DatabaseEngineer` is also expanded and its `_supervisor_id` points to `BackendEngineer`.

## Company Org-Chart Example

```python
import asyncio
from syrin import Agent, Budget, Model
from syrin.response import Response
from syrin.swarm import Swarm

class DatabaseEngineer(Agent):
    model = Model.mock()
    system_prompt = "Design and implement database schemas."

class BackendEngineer(Agent):
    model = Model.mock()
    system_prompt = "Implement API endpoints and business logic."
    team = [DatabaseEngineer]

class FrontendEngineer(Agent):
    model = Model.mock()
    system_prompt = "Build UI components and user flows."

class CTO(Agent):
    model = Model.mock()
    system_prompt = "Define technical strategy and architecture."
    team = [BackendEngineer, FrontendEngineer]

class CEO(Agent):
    model = Model.mock()
    system_prompt = "Set company direction and priorities."
    team = [CTO]

async def main() -> None:
    # Swarm expands: CEO → CTO → [BackendEngineer → DatabaseEngineer, FrontendEngineer]
    swarm = Swarm(
        agents=[CEO()],
        goal="Build the new product feature",
        budget=Budget(max_cost=10.00),
    )
    result = await swarm.run()
    print(result.content)

asyncio.run(main())
```

The swarm runs all expanded agents concurrently (PARALLEL topology).  Each agent in the hierarchy has its `_supervisor_id` set.

## Automatic Permission Grants

When the swarm expands the team, `SwarmAuthorityGuard` is configured automatically:

- A parent agent gets `AgentRole.SUPERVISOR` (or `ORCHESTRATOR` if it has a `team`).
- Team members get `AgentRole.WORKER`.
- The parent can issue `CONTROL` and `CONTEXT` actions on its direct reports.

```python
from syrin.swarm import SwarmAuthorityGuard
from syrin.enums import AgentPermission

guard = SwarmAuthorityGuard(
    roles={
        "ceo": AgentRole.ORCHESTRATOR,
        "cto": AgentRole.SUPERVISOR,
        "backend_engineer": AgentRole.WORKER,
    },
    teams={
        "ceo": ["cto"],
        "cto": ["backend_engineer"],
    },
)

# CEO can control CTO
guard.require("ceo", AgentPermission.CONTROL, "cto")   # passes

# CTO can control BackendEngineer
guard.require("cto", AgentPermission.CONTROL, "backend_engineer")  # passes

# BackendEngineer cannot control CTO
guard.require("backend_engineer", AgentPermission.CONTROL, "cto")  # raises AgentPermissionError
```

## Reading `supervisor_id` from `AgentStateSnapshot`

`AgentStateSnapshot.supervisor_id` tells you which agent manages this one:

```python
from syrin.swarm import SwarmController, SwarmAuthorityGuard, AgentStateSnapshot
from syrin.enums import AgentRole, AgentStatus

state = {
    "backend_engineer": AgentStateSnapshot(
        agent_id="backend_engineer",
        status=AgentStatus.RUNNING,
        role=AgentRole.WORKER,
        last_output_summary="Implementing API...",
        cost_spent=0.10,
        task="Build /api/users endpoint",
        context_override=None,
        supervisor_id="cto",         # ← set by Swarm._expand_team_agents()
    )
}
```

Access it to build observability dashboards or log the authority chain.

## Delegation

A parent can temporarily grant permissions to a peer agent via `guard.delegate()`:

```python
from syrin.enums import AgentPermission, DelegationScope

# CEO delegates CONTROL to CMO for this run
guard.delegate(
    delegator_id="ceo",
    delegate_id="cmo",
    permissions=[AgentPermission.CONTROL],
    scope=DelegationScope.CURRENT_RUN,
)

# CMO can now control CTO
guard.require("cmo", AgentPermission.CONTROL, "cto")  # passes

# Revoke later
guard.revoke_delegation(delegator_id="ceo", delegate_id="cmo")
```

> **Note:** `DelegationScope.PERMANENT` raises `NotImplementedError` in v0.11.0.  Permanent delegation arrives in v0.12.0.

Delegation rules:
- Only `ADMIN` role can delegate `AgentPermission.ADMIN`.
- Any agent can delegate permissions they currently hold (except `ADMIN`).

## Authority Chain Inheritance

The authority chain flows top-down through `supervisor_id`:

```
CEO (ORCHESTRATOR)
  └── CTO (SUPERVISOR)
        ├── BackendEngineer (WORKER, supervisor_id="cto")
        │     └── DatabaseEngineer (WORKER, supervisor_id="backend_engineer")
        └── FrontendEngineer (WORKER, supervisor_id="cto")
```

A supervisor can act on direct reports.  Skip-level control requires delegation or an `ADMIN` role.  Unknown actors are treated as `WORKER` (minimal permissions).

## Reporting Results Upward

When a child agent finishes, its output travels back to the parent through `SpawnResult.content`.  Whatever the child returns as `Response.content` is the report the parent reads — no polling, no subscriptions.

```python
import asyncio
import json
from syrin import Agent, Model
from syrin.response import Response
from syrin.swarm._spawn import SpawnResult, SpawnSpec

class SoftwareEngineer(Agent):
    model = Model.mock()
    system_prompt = "You implement backend services and APIs."

class Designer(Agent):
    model = Model.mock()
    system_prompt = "You design interfaces and user flows."

class CTOAgent(Agent):
    model = Model.mock()
    system_prompt = "You decide workforce composition and technical direction."

    async def arun(self, input_text: str) -> Response[str]:
        # Spawn the workforce and collect their outputs
        results: list[SpawnResult] = await self.spawn_many([
            SpawnSpec(agent=SoftwareEngineer, task=input_text, budget=1.00),
            SpawnSpec(agent=Designer,          task=input_text, budget=0.50),
        ])

        # Package a structured report for the CEO
        report = json.dumps({
            "headcount": {"engineers": 1, "designers": 1},
            "total_spent": sum(r.cost for r in results),
            "engineer_output": results[0].content[:200],
            "designer_output": results[1].content[:200],
        })
        return Response(content=report, cost=sum(r.cost for r in results))


class CEOAgent(Agent):
    model = Model.mock()
    system_prompt = "You set company direction and synthesise all reports."

    async def arun(self, input_text: str) -> Response[str]:
        # Allocate budget to the CTO and wait for the workforce report
        cto_result: SpawnResult = await self.spawn(
            CTOAgent, task=input_text, budget=2.00
        )

        # Deserialise and act on the structured report
        report = json.loads(cto_result.content)
        summary = (
            f"Workforce: {report['headcount']}\n"
            f"Spent: ${report['total_spent']:.4f} / $2.00 allocated\n"
            f"Engineer: {report['engineer_output'][:80]}\n"
            f"Designer: {report['designer_output'][:80]}"
        )
        return Response(content=summary, cost=cto_result.cost)


async def main() -> None:
    ceo = CEOAgent()
    result = await ceo.arun("Launch a new product dashboard feature")
    print(result.content)

asyncio.run(main())
```

**Key rule:** the child's `Response.content` becomes `SpawnResult.content` on the parent side.  If you need machine-readable data (headcount, costs, statuses), return JSON from the child and `json.loads()` it in the parent.

For unsolicited mid-run signals — e.g. a worker escalating a blocker without being awaited — use [A2A messaging](/multi-agent/a2a) instead.

## The Virtual Office Pattern

A full three-level hierarchy: CEO sets budgets, CTO decides the workforce, workers execute.

```python
import asyncio
import json
from syrin import Agent, Budget, Model
from syrin.response import Response
from syrin.enums import AgentRole
from syrin.swarm import Swarm, SwarmConfig
from syrin.swarm._spawn import SpawnResult, SpawnSpec
from syrin.enums import SwarmTopology

class SoftwareEngineer(Agent):
    model = Model.mock()
    system_prompt = "You implement features and write code."

class MarketingManager(Agent):
    model = Model.mock()
    system_prompt = "You run growth campaigns and track KPIs."

class CTOAgent(Agent):
    role = AgentRole.SUPERVISOR
    model = Model.mock()
    system_prompt = (
        "You decide how many engineers to hire and what to build. "
        "Spawn engineers via spawn_many(), collect their outputs, "
        "and return a JSON report: {headcount, outputs, total_spent}."
    )

    async def arun(self, input_text: str) -> Response[str]:
        results: list[SpawnResult] = await self.spawn_many([
            SpawnSpec(agent=SoftwareEngineer, task=f"Build: {input_text}", budget=0.80),
            SpawnSpec(agent=SoftwareEngineer, task=f"Test: {input_text}",  budget=0.40),
        ])
        report = json.dumps({
            "department": "Engineering",
            "headcount": 2,
            "total_spent": sum(r.cost for r in results),
            "outputs": [r.content[:150] for r in results],
        })
        return Response(content=report, cost=sum(r.cost for r in results))


class CMOAgent(Agent):
    role = AgentRole.SUPERVISOR
    model = Model.mock()
    system_prompt = (
        "You run marketing for company growth goals. "
        "Spawn marketing workers and return a JSON report."
    )

    async def arun(self, input_text: str) -> Response[str]:
        result: SpawnResult = await self.spawn(
            MarketingManager,
            task=f"Growth campaign: {input_text}",
            budget=0.50,
        )
        report = json.dumps({
            "department": "Marketing",
            "headcount": 1,
            "total_spent": result.cost,
            "outputs": [result.content[:150]],
        })
        return Response(content=report, cost=result.cost)


class CEOAgent(Agent):
    role = AgentRole.ORCHESTRATOR
    model = Model.mock()
    system_prompt = (
        "You allocate budget to CTO and CMO, wait for their reports, "
        "and synthesise a company-wide executive summary."
    )

    async def arun(self, input_text: str) -> Response[str]:
        # CEO decides each department's budget and delegates concurrently
        dept_results: list[SpawnResult] = await self.spawn_many([
            SpawnSpec(agent=CTOAgent, task=input_text, budget=2.00),
            SpawnSpec(agent=CMOAgent, task=input_text, budget=1.00),
        ])

        # Collect structured reports from both departments
        reports = [json.loads(r.content) for r in dept_results]
        total_headcount = sum(d["headcount"] for d in reports)
        total_spent = sum(r.cost for r in dept_results)

        summary = (
            f"Company Goal: {input_text}\n\n"
            + "\n".join(
                f"[{d['department']}] {d['headcount']} staff  "
                f"${d['total_spent']:.4f} spent"
                for d in reports
            )
            + f"\n\nTotal headcount: {total_headcount}  "
            f"Total spent: ${total_spent:.4f}"
        )
        return Response(content=summary, cost=total_spent)


async def main() -> None:
    swarm = Swarm(
        agents=[CEOAgent()],   # Swarm entry point is the CEO instance
        goal="Drive growth by shipping our Q3 product roadmap",
        budget=Budget(max_cost=5.00),
        config=SwarmConfig(topology=SwarmTopology.ORCHESTRATOR),
    )
    result = await swarm.run()
    print(result.content)

asyncio.run(main())
```

What this demonstrates:

- **CEO controls all budgets** — `SpawnSpec(budget=2.00)` for CTO, `budget=1.00` for CMO.  Neither can overspend their allocation.
- **CTO decides the workforce** — spawns 2 engineers dynamically; CEO never touches engineer agents directly.
- **Reports flow upward through `SpawnResult`** — CTO returns JSON, CMO returns JSON, CEO reads both and synthesises the executive summary.
- **Three levels of hierarchy** — CEO → CTO/CMO → workers.  The pattern extends to any depth.

## Team Expansion Details

`Swarm._expand_team_agents()` processes the queue iteratively (breadth-first) to handle arbitrary nesting depth.  It sets `_supervisor_id` using `object.__setattr__` to work with frozen dataclasses or agents without arbitrary `__setattr__`.

Duplicate agent IDs are detected via `processed_ids` to prevent infinite loops in circular team definitions.

## See Also

- [Budget Delegation](/multi-agent/budget-delegation) — `spawn()`, `spawn_many()`, `SpawnResult`, budget pools
- [Authority](/multi-agent/authority) — permission model, roles, delegation
- [Swarm](/multi-agent/swarm) — topologies, shared budget, `SwarmResult`
- [A2A Messaging](/multi-agent/a2a) — unsolicited mid-run signals between agents
- [MemoryBus](/multi-agent/memory-bus) — shared knowledge board for cross-agent findings
