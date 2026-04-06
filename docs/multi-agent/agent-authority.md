---
title: Agent Authority
description: Declare agent roles at the class level and build authority guards without free strings
weight: 78
---

This page covers the class-level `role` and `team` attributes, `build_guard_from_agents()`, and the `ControlAction` enum. For the full permissions model (guard setup, delegation, hooks), see [Authority](/multi-agent/authority).

## Agent.role ClassVar

Every `Agent` subclass has a `role` class attribute of type `AgentRole`. It defaults to `AgentRole.WORKER`. Declare it at the class level to assign authority:

```python
from syrin import Agent, Model
from syrin.enums import AgentRole

class Supervisor(Agent):
    role = AgentRole.SUPERVISOR
    team = [WorkerA, WorkerB]       # agents this supervisor can control
    model = Model.mock()
    system_prompt = "You coordinate the team."

class WorkerA(Agent):
    # role defaults to AgentRole.WORKER — no declaration needed
    model = Model.mock()
    system_prompt = "You handle data collection."

class WorkerB(Agent):
    model = Model.mock()
    system_prompt = "You handle data analysis."
```

## Agent.team ClassVar

`team` is a `ClassVar[list[type[Agent]]]`. It lists the agent classes that this agent has authority over. Used by `build_guard_from_agents()` to wire up team membership automatically.

```python
class OrchestratorAgent(Agent):
    role = AgentRole.ORCHESTRATOR
    team = [ResearchAgent, WriterAgent, EditorAgent]
    model = Model.mock()
    system_prompt = "You manage the pipeline."
```

## build_guard_from_agents()

Build a `SwarmAuthorityGuard` from a list of instantiated agents. Reads `role` and `team` from each agent's class definition — no string IDs to manage:

```python
from syrin.swarm import build_guard_from_agents, Swarm

supervisor = Supervisor()
worker_a = WorkerA()
worker_b = WorkerB()

guard = build_guard_from_agents([supervisor, worker_a, worker_b])

swarm = Swarm(
    agents=[supervisor, worker_a, worker_b],
    goal="Collect and analyse quarterly data",
    authority_guard=guard,
)
```

Each `Agent()` instance is assigned a unique `agent_id` (`ClassName-<hex>`) at creation time. `build_guard_from_agents()` uses these IDs internally — you never need to reference them directly.

## SwarmController with agent objects

Get the controller from a live swarm handle and pass agent objects directly:

```python
handle = swarm.play()
ctrl = handle.controller

await ctrl.pause_agent(worker_a)
await ctrl.change_context(worker_b, "Summarise in three bullet points")
await ctrl.resume_agent(worker_a)
await ctrl.kill_agent(worker_b)

snap = await ctrl.read_agent_state(worker_a)
print(snap.status, snap.cost_spent)

result = await handle.wait()
```

All controller methods accept an `Agent` instance (recommended) or a string `agent_id`. Passing the object is preferred — the ID is resolved for you.

## ControlAction enum

`AuditEntry.action` is typed as `ControlAction` — a `StrEnum` — instead of a raw string:

```python
from syrin.enums import ControlAction

guard.record_action(supervisor.agent_id, worker_a.agent_id, ControlAction.PAUSE)
log = guard.audit_log()

for entry in log:
    if entry.action == ControlAction.KILL:
        alert(f"Agent {entry.target_id} was killed by {entry.actor_id}")
```

| Value | String | When used |
|-------|--------|-----------|
| `PAUSE` | `"pause"` | Agent paused by a control action |
| `RESUME` | `"resume"` | Agent resumed from PAUSED state |
| `SKIP` | `"skip"` | Agent's current task was skipped |
| `KILL` | `"kill"` | Agent was forcibly terminated |
| `CHANGE_CONTEXT` | `"change_context"` | Agent context was overridden |
| `DELEGATE` | `"delegate"` | Permission delegated to another agent |
| `REVOKE` | `"revoke"` | Delegation revoked |

## See Also

- [Authority](/multi-agent/authority) — Full permissions model, delegation, hooks, audit log
- [Swarm](/multi-agent/swarm) — Parallel, consensus, and reflection topologies
- [Enums Reference](/reference/enums#controlaction) — `ControlAction`, `AgentRole`, `AgentPermission`
