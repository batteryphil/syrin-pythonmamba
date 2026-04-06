"""Agent authority — role-based control with agent objects.

Shows how agents use role class attributes and build_guard_from_agents()
to set up authority without free strings. Control actions pass agent
objects — no "orchestrator-1" string IDs anywhere.

Key concepts:
  - Agent.role = AgentRole.ORCHESTRATOR / SUPERVISOR / WORKER
  - Agent.team = [WorkerClass]  — declares managed sub-agents
  - build_guard_from_agents([...]) — builds guard from class metadata
  - ctrl.pause_agent(worker)     — pass agent object, not a string
  - guard.delegate(actor, delegate, permissions)

Run:
    uv run python examples/07_multi_agent/agent_authority.py
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from syrin import Agent, Model
from syrin.enums import AgentPermission, AgentRole, AgentStatus, DelegationScope
from syrin.swarm import (
    AgentPermissionError,
    AgentStateSnapshot,
    SwarmController,
    build_guard_from_agents,
)

_MODEL = Model.OpenAI("gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"))

# ── Agent class definitions with roles ───────────────────────────────────────


class WorkerAgent(Agent):
    model = _MODEL
    system_prompt = "You process tasks."
    role = AgentRole.WORKER


class OrchestratorAgent(Agent):
    model = _MODEL
    system_prompt = "You delegate tasks to workers."
    role = AgentRole.ORCHESTRATOR
    team = [WorkerAgent]


class SupervisorAgent(Agent):
    model = _MODEL
    system_prompt = "You supervise workers."
    role = AgentRole.SUPERVISOR
    team = [WorkerAgent]


# ── Helper ────────────────────────────────────────────────────────────────────


def _state_for(agent: Agent) -> AgentStateSnapshot:
    return AgentStateSnapshot(
        agent_id=agent.agent_id,
        status=AgentStatus.RUNNING,
        role=type(agent).role,
        last_output_summary="",
        cost_spent=0.0,
        task="process task",
        context_override=None,
        supervisor_id=None,
    )


# ── Example 1: Orchestrator controls its team ────────────────────────────────


async def example_orchestrator_controls_worker() -> None:
    print("\n── Example 1: Orchestrator controls workers ─────────────────────")

    orch = OrchestratorAgent()
    w1 = WorkerAgent()
    w2 = WorkerAgent()

    guard = build_guard_from_agents([orch, w1, w2])
    state = {a.agent_id: _state_for(a) for a in (orch, w1, w2)}

    ctrl = SwarmController(actor_id=orch, guard=guard, state_registry=state, task_registry={})

    await ctrl.pause_agent(w1)
    print(f"  w1 after pause:  {state[w1.agent_id].status}")

    await ctrl.resume_agent(w1)
    print(f"  w1 after resume: {state[w1.agent_id].status}")

    await ctrl.change_context(w2, "Focus on healthcare sector only")
    snap = await ctrl.read_agent_state(w2)
    print(f"  w2 context: {snap.context_override!r}")

    for entry in guard.audit_log():
        print(f"  [{entry.action}] {entry.actor_id} → {entry.target_id}")


# ── Example 2: Worker denied when trying to control orchestrator ──────────────


async def example_worker_denied() -> None:
    print("\n── Example 2: Worker cannot control orchestrator ────────────────")

    orch = OrchestratorAgent()
    worker = WorkerAgent()

    guard = build_guard_from_agents([orch, worker])
    state = {orch.agent_id: _state_for(orch)}

    worker_ctrl = SwarmController(
        actor_id=worker, guard=guard, state_registry=state, task_registry={}
    )

    try:
        await worker_ctrl.pause_agent(orch)
    except AgentPermissionError as e:
        print(f"  DENIED: {e.reason}")

    try:
        await worker_ctrl.read_agent_state(orch)
    except AgentPermissionError as e:
        print(f"  DENIED (read): {e.reason}")


# ── Example 3: guard.check() without raising ─────────────────────────────────


async def example_permission_check() -> None:
    print("\n── Example 3: guard.check() — non-raising permission test ──────")

    orch = OrchestratorAgent()
    sup = SupervisorAgent()
    worker = WorkerAgent()

    guard = build_guard_from_agents([orch, sup, worker])

    for actor, perm, target, expected in [
        (orch, AgentPermission.CONTROL, worker, True),
        (sup, AgentPermission.CONTROL, worker, True),
        (worker, AgentPermission.CONTROL, orch, False),
        (worker, AgentPermission.SIGNAL, orch, True),
    ]:
        result = guard.check(actor, perm, target)
        mark = "✓" if result == expected else "✗"
        print(f"  {mark} {type(actor).__name__} → {perm} {type(target).__name__}: {result}")


# ── Example 4: Delegation of authority ───────────────────────────────────────


async def example_delegation() -> None:
    print("\n── Example 4: Delegation ────────────────────────────────────────")

    orch = OrchestratorAgent()
    sup = SupervisorAgent()
    worker = WorkerAgent()

    guard = build_guard_from_agents([orch, sup, worker])

    before = guard.check(sup, AgentPermission.CONTEXT, worker)
    print(f"  Before delegation — supervisor CONTEXT worker: {before}")

    guard.delegate(
        delegator_id=orch,
        delegate_id=sup,
        permissions=[AgentPermission.CONTEXT],
        scope=DelegationScope.CURRENT_RUN,
    )
    after = guard.check(sup, AgentPermission.CONTEXT, worker)
    print(f"  After delegation  — supervisor CONTEXT worker: {after}")

    guard.revoke_delegation(orch, sup)
    revoked = guard.check(sup, AgentPermission.CONTEXT, worker)
    print(f"  After revocation  — supervisor CONTEXT worker: {revoked}")


# ── Main ──────────────────────────────────────────────────────────────────────


async def main() -> None:
    await example_orchestrator_controls_worker()
    await example_worker_denied()
    await example_permission_check()
    await example_delegation()
    print("\nAll agent authority examples completed.")


if __name__ == "__main__":
    asyncio.run(main())
