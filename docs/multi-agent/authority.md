---
title: Agent Authority
description: Role-based access control and permission delegation for multi-agent swarms
weight: 77
---

## Who's in Charge?

In a multi-agent system, you don't want every agent to be able to pause, kill, or reprogram every other agent. That's chaos. The authority system gives you role-based access control: some agents can control others, some can only signal, some can read state. You define the rules, and the guard enforces them.

## Roles

Every agent in a swarm has an `AgentRole`. There are four:

`AgentRole.ADMIN` has full control over all agents and swarm settings. Nothing is off-limits.

`AgentRole.ORCHESTRATOR` can spawn agents, change context, and issue control commands — but only on agents in its assigned team.

`AgentRole.SUPERVISOR` can pause and resume agents in its assigned team. It can't change context or spawn.

`AgentRole.WORKER` is the standard role. Workers can execute tasks and send A2A signals, but they can't control other agents. Unknown actors are treated as WORKER by default.

## Permissions

Roles map to sets of `AgentPermission` values:

`AgentPermission.READ` — Read another agent's state and output.

`AgentPermission.SIGNAL` — Send A2A messages to another agent. Every role has this.

`AgentPermission.CONTROL` — Issue control commands: pause, resume, kill, skip.

`AgentPermission.CONTEXT` — Read another agent's full context window.

`AgentPermission.SPAWN` — Spawn child agents on behalf of another agent.

`AgentPermission.ADMIN` — All of the above.

The default rules: ADMIN has every permission on every agent. ORCHESTRATOR has CONTROL, CONTEXT, and SPAWN on agents in its team. SUPERVISOR has CONTROL on agents in its team. WORKER has SIGNAL only.

## Declaring Roles on Agent Classes

The recommended way to assign roles is via a `role` class attribute on the `Agent` subclass:

```python
from syrin import Agent, Model
from syrin.enums import AgentRole

class SupervisorAgent(Agent):
    role = AgentRole.SUPERVISOR
    team = [WorkerAgent]          # agents this supervisor can control
    model = Model.mock()
    system_prompt = "You coordinate the team."

class WorkerAgent(Agent):
    # role defaults to AgentRole.WORKER
    model = Model.mock()
    system_prompt = "You execute tasks."
```

`role` and `team` are `ClassVar` fields. `team` is a list of `Agent` subclasses that this agent has authority over. The `WORKER` role is the default and does not need to be declared explicitly.

## build_guard_from_agents()

Build the `SwarmAuthorityGuard` directly from instantiated agents — no free strings, no manual ID management:

```python
from syrin.swarm import build_guard_from_agents

supervisor = SupervisorAgent()
worker = WorkerAgent()

guard = build_guard_from_agents([supervisor, worker])
```

`build_guard_from_agents()` reads each agent's `role` and `team` class attributes and maps `agent_id` values automatically. Each `Agent()` instance is assigned a unique `agent_id` (`ClassName-<hex>`) at creation time.

## SwarmAuthorityGuard (manual setup)

You can still build the guard manually when you need explicit control:

```python
from syrin.swarm import SwarmAuthorityGuard
from syrin.enums import AgentRole, AgentPermission

guard = SwarmAuthorityGuard(
    roles={
        supervisor.agent_id: AgentRole.SUPERVISOR,
        worker.agent_id:     AgentRole.WORKER,
    },
    teams={
        supervisor.agent_id: [worker.agent_id],
    },
)

# Check without raising — returns True/False
if guard.check(supervisor.agent_id, AgentPermission.CONTROL, worker.agent_id):
    print("Permission granted")

# Require — raises AgentPermissionError if denied
guard.require(supervisor.agent_id, AgentPermission.CONTROL, worker.agent_id)
```

`check()` returns `True` or `False`. Use it when you want to branch on permission. `require()` raises `AgentPermissionError` if the permission is denied — use it when a denied permission should stop execution.

## Permission Delegation

Temporarily grant permissions to another agent for the current run:

```python
from syrin.enums import DelegationScope

# Grant the CMO the ability to issue control commands for this run
guard.delegate(
    delegator_id="ceo",
    delegate_id="cmo",
    permissions=[AgentPermission.CONTROL],
    scope=DelegationScope.CURRENT_RUN,
)

# Revoke when done
guard.revoke_delegation(delegator_id="ceo", delegate_id="cmo")
```

Two delegation rules: only an `ADMIN` can delegate `AgentPermission.ADMIN`. Any agent can delegate permissions it already holds (except ADMIN).

`DelegationScope.PERMANENT` raises `NotImplementedError` in v0.11.0 — permanent delegation arrives in v0.12.0.

## Hooks

Three hooks fire for authority events:

`Hook.AGENT_CONTROL_ACTION` fires after a successful `record_action()` call.

`Hook.AGENT_PERMISSION_DENIED` fires when `require()` denies a permission.

`Hook.AGENT_DELEGATION` fires when `delegate()` succeeds.

## Audit Log

Every successful control action is recorded and retrievable. `AuditEntry.action` is a `ControlAction` enum value — not a free string:

```python
from syrin.enums import ControlAction

guard.record_action(supervisor.agent_id, worker.agent_id, ControlAction.PAUSE)
log = guard.audit_log()
# [AuditEntry(actor_id="SupervisorAgent-a1b2", target_id="WorkerAgent-c3d4",
#             action=ControlAction.PAUSE, timestamp=...)]

for entry in log:
    print(f"{entry.actor_id} → {entry.target_id}: {entry.action}")
```

`ControlAction` values: `PAUSE`, `RESUME`, `SKIP`, `KILL`, `CHANGE_CONTEXT`, `DELEGATE`, `REVOKE`. See the [Enums reference](/reference/enums#controlaction) for the full table.

Use the audit log for compliance, debugging, or tracing how agents interacted.

## SwarmController

`SwarmController` is the high-level API for taking control actions — it combines the guard check with the actual state modification.

### Getting the controller from a live swarm

The easiest way to get a controller is via `handle.controller` after calling `swarm.play()`:

```python
from syrin.swarm import Swarm, build_guard_from_agents

supervisor = SupervisorAgent()
worker = WorkerAgent()

swarm = Swarm(
    agents=[supervisor, worker],
    goal="...",
    authority_guard=build_guard_from_agents([supervisor, worker]),
)
handle = swarm.play()

# Bound to the live swarm — no manual wiring needed
ctrl = handle.controller

# Pass agent objects directly
await ctrl.pause_agent(worker)
await ctrl.change_context(worker, "Be more concise")
await ctrl.resume_agent(worker)
snap = await ctrl.read_agent_state(worker)

result = await handle.wait()
```

### Manual wiring

```python
from syrin.swarm import SwarmController, SwarmAuthorityGuard, AgentStateSnapshot
from syrin.enums import AgentRole, AgentStatus

supervisor = SupervisorAgent()
worker = WorkerAgent()
guard = build_guard_from_agents([supervisor, worker])

state = {
    worker.agent_id: AgentStateSnapshot(
        agent_id=worker.agent_id,
        status=AgentStatus.RUNNING,
        role=AgentRole.WORKER,
        last_output_summary="Processing...",
        cost_spent=0.10,
        task="analyse data",
        context_override=None,
        supervisor_id=supervisor.agent_id,
    )
}

ctrl = SwarmController(
    actor_id=supervisor.agent_id,
    guard=guard,
    state_registry=state,
    task_registry={},
)

await ctrl.pause_agent(worker)
await ctrl.change_context(worker, "Be more concise")
await ctrl.resume_agent(worker)
```

### Control methods

`pause_agent(agent)` requires CONTROL permission and sets the agent's status to PAUSED. `resume_agent(agent)` requires CONTROL and returns the agent to RUNNING. `skip_agent(agent)` requires CONTROL and sets status to IDLE, cancelling the current task. `kill_agent(agent)` requires CONTROL and sets status to KILLED. `change_context(agent, ctx)` requires CONTROL and sets the agent's `context_override` string. `read_agent_state(agent)` requires READ permission and returns an `AgentStateSnapshot`.

All methods accept an `Agent` instance (recommended) or a string `agent_id`. The string form still works but using objects is preferred — IDs are managed for you.

The `AgentStateSnapshot` has these fields: `agent_id`, `status` (an `AgentStatus` value), `role` (an `AgentRole` value), `last_output_summary` (always 500 characters or fewer), `cost_spent`, `task`, `context_override`, and `supervisor_id`.

## See Also

- [Swarm](/multi-agent/swarm) — Parallel, consensus, and reflection topologies
- [MonitorLoop](/multi-agent/monitor-loop) — Async supervisor loop
- [Broadcast](/multi-agent/broadcast) — Publish events across the swarm
