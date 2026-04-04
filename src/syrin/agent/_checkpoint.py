"""Checkpoint use case: save, load, list checkpoints and auto-checkpoint.

Agent delegates to functions here. Public API stays on Agent.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from syrin.agent import Agent

from syrin.checkpoint import CheckpointTrigger
from syrin.enums import Hook
from syrin.events import EventContext
from syrin.response import AgentReport


def save_checkpoint(agent: Agent, name: str | None = None, reason: str | None = None) -> str | None:
    """Save agent state to checkpoint. Returns checkpoint ID or None if disabled."""
    if agent._checkpointer is None:
        return None

    agent_name = name or agent._agent_name
    messages_serialized: list[dict[str, object]] = []
    for msg in agent.messages:
        messages_serialized.append(msg.model_dump())
    context_snapshot: dict[str, object] | None = None
    if hasattr(agent, "_context") and hasattr(agent._context, "snapshot"):
        snap = agent._context.snapshot()
        if snap is not None:
            context_snapshot = snap.to_dict(include_raw_messages=False)
    state = {
        "iteration": agent.iteration,
        "messages": messages_serialized,
        "memory_data": {},
        "budget_state": (
            {
                "remaining": agent._budget.remaining,
                "spent": agent._budget._spent,
                "tracker_state": agent._budget_tracker.get_state(),
            }
            if agent._budget is not None
            else None
        ),
        "checkpoint_reason": reason,
        "context_snapshot": context_snapshot,
    }

    checkpoint_id = cast("str | None", agent._checkpointer.save(agent_name, state))
    agent._run_report.checkpoints.saves += 1
    agent._emit_event(Hook.CHECKPOINT_SAVE, EventContext(checkpoint_id=checkpoint_id))
    return checkpoint_id


def maybe_checkpoint(agent: Agent, reason: str) -> None:
    """Automatically checkpoint based on trigger configuration."""
    if agent._checkpointer is None or agent._checkpoint_config is None:
        return

    trigger = agent._checkpoint_config.trigger if agent._checkpoint_config else None
    if trigger == CheckpointTrigger.MANUAL:
        return

    should_checkpoint = (trigger == CheckpointTrigger.STEP and reason in ("step", "tool")) or (
        trigger is not None and trigger.value == reason
    )
    if should_checkpoint:
        save_checkpoint(agent, reason=reason)


def load_checkpoint(agent: Agent, checkpoint_id: str) -> bool:
    """Restore agent state from checkpoint. Returns True if loaded."""
    if agent._checkpointer is None:
        return False

    state = agent._checkpointer.load(checkpoint_id)
    if state is None:
        return False

    if state.messages and agent._persistent_memory is not None:
        agent._persistent_memory.load_conversation_messages(
            cast(list[dict[str, object]], list(state.messages))  # type: ignore[arg-type]
        )
    iter_val = getattr(state, "iteration", None)
    if iter_val is not None and isinstance(iter_val, (int, float)):
        object.__setattr__(agent, "_last_iteration", int(iter_val))

    budget_state = getattr(state, "budget_state", None)
    if budget_state is not None and agent._budget is not None:
        tracker_state = budget_state.get("tracker_state")
        if tracker_state is not None:
            agent._budget_tracker.load_state(tracker_state)
        spent = budget_state.get("spent")
        if spent is not None:
            agent._budget._set_spent(spent)

    agent._run_report.checkpoints.loads += 1
    agent._emit_event(Hook.CHECKPOINT_LOAD, EventContext(checkpoint_id=checkpoint_id))
    return True


def list_checkpoints(agent: Agent, name: str | None = None) -> list[str]:
    """List checkpoint IDs for this agent, optionally filtered by name."""
    if agent._checkpointer is None:
        return []

    agent_name = name or agent._agent_name
    return agent._checkpointer.list_checkpoints(agent_name)


def get_checkpoint_report(agent: Agent) -> AgentReport:
    """Return the full agent report including checkpoint stats."""
    return agent._run_report
