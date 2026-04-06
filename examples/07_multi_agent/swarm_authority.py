"""Swarm authority — role-based control without free strings.

Declare roles on agent classes, build the authority guard automatically,
then use agent objects (not string IDs) for all control actions.

Run: python examples/07_multi_agent/swarm_authority.py
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from syrin import Agent, Model
from syrin.enums import AgentPermission, AgentRole, AgentStatus
from syrin.swarm import (
    AgentPermissionError,
    AgentStateSnapshot,
    SwarmController,
    build_guard_from_agents,
)

_MODEL = Model.OpenAI("gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"))

# Step 1: Declare roles on each class.
# Define Worker first so Supervisor.team can reference it directly.


class Worker(Agent):
    model = _MODEL
    system_prompt = "You do the research work."
    role = AgentRole.WORKER


class Supervisor(Agent):
    model = _MODEL
    system_prompt = "You supervise the research team."
    role = AgentRole.SUPERVISOR  # declare role as class attribute
    team = [Worker]  # declare managed workers as class attribute


async def main() -> None:
    supervisor = Supervisor()
    worker = Worker()

    # Step 2: Build guard from class metadata — no free strings
    guard = build_guard_from_agents([supervisor, worker])

    # Step 3: Check permissions using agent objects
    can_control = guard.check(supervisor, AgentPermission.CONTROL, worker)
    print(f"Supervisor can control worker: {can_control}")

    # Step 4: Create controller with agent object as actor
    state = {
        worker.agent_id: AgentStateSnapshot(
            agent_id=worker.agent_id,
            status=AgentStatus.RUNNING,
            role=AgentRole.WORKER,
            last_output_summary="Analysing all market segments...",
            cost_spent=0.01,
            task="research task",
            context_override=None,
            supervisor_id=supervisor.agent_id,
        )
    }
    ctrl = SwarmController(
        actor_id=supervisor,  # pass agent object, not "supervisor-1"
        guard=guard,
        state_registry=state,
        task_registry={},
    )

    # Step 5: Control actions use agent objects
    await ctrl.pause_agent(worker)  # not ctrl.pause_agent("worker-1")
    snap = await ctrl.read_agent_state(worker)
    print(f"Status after pause:  {snap.status}")

    await ctrl.change_context(worker, "Focus on Q4 only")
    await ctrl.resume_agent(worker)
    snap = await ctrl.read_agent_state(worker)
    print(f"Status after resume: {snap.status}")

    # Step 6: Audit log is fully typed — action is a ControlAction StrEnum
    for entry in guard.audit_log():
        print(f"  [{entry.action}] {entry.actor_id} -> {entry.target_id}")

    # Workers cannot control each other
    worker2 = Worker()
    state2 = {
        worker2.agent_id: AgentStateSnapshot(
            agent_id=worker2.agent_id,
            status=AgentStatus.RUNNING,
            role=AgentRole.WORKER,
            last_output_summary="",
            cost_spent=0.0,
            task="other task",
            context_override=None,
            supervisor_id=None,
        )
    }
    guard2 = build_guard_from_agents([supervisor, worker, worker2])
    ctrl2 = SwarmController(
        actor_id=worker,
        guard=guard2,
        state_registry=state2,
        task_registry={},
    )
    try:
        await ctrl2.pause_agent(worker2)
    except AgentPermissionError as e:
        print(f"DENIED: {e.reason}")


if __name__ == "__main__":
    asyncio.run(main())
