"""PromptContext for dynamic system prompt resolution.

Passed to callable system prompts and @system_prompt methods.
Full access to agent, memory, budget, and built-in vars.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime


@dataclass
class PromptContext:
    """Context passed to callable system prompts at resolution time.

    Enables dynamic prompts that access agent state, memory, budget,
    and built-in values (date, agent_id, conversation_id).
    """

    agent: object
    """The Agent instance. Use for ctx.agent.memory, ctx.agent.name, etc."""

    agent_id: str
    """Agent name (for routing, logging). Same as agent.name or class name."""

    conversation_id: str | None
    """Current conversation ID for state isolation (per-user or per-session). None if not set."""

    memory: object
    """Persistent memory backend (Memory) or None. Use for ctx.memory.recall(...)."""

    budget_state: object
    """BudgetState or None. Use for remaining, spent, etc."""

    date: datetime
    """Current UTC datetime. For prompts that need \"today's date\"."""

    builtins: dict[str, object] = field(default_factory=dict)
    """Built-in vars (date, agent_id, conversation_id) that would be injected. For introspection."""

    def __post_init__(self) -> None:
        if self.builtins is None:
            self.builtins = {}


def make_prompt_context(
    agent: object,
    *,
    conversation_id: str | None = None,
    inject_template_vars: bool = True,
) -> PromptContext:
    """Build PromptContext from an Agent instance.

    Args:
        agent: The Agent instance.
        conversation_id: Current conversation ID for state isolation.
        inject_template_vars: If True (default), populate builtins with date, agent_id,
            conversation_id. These are available as template vars in system prompts
            (e.g. {date}, {agent_id}, {conversation_id}). Set False to skip.
    """
    agent_id = (
        getattr(agent, "_agent_name", None)
        or getattr(agent, "name", None)
        or agent.__class__.__name__
    )
    memory = getattr(agent, "_persistent_memory", None) or getattr(agent, "_memory_backend", None)
    budget_state = None
    if hasattr(agent, "_budget") and agent._budget is not None:
        budget_state = getattr(agent._budget, "state", None) or {
            "remaining": getattr(agent._budget, "remaining", None),
            "spent": getattr(agent._budget, "_spent", 0),
        }
    date_val = datetime.now(UTC)
    builtins: dict[str, object] = {}
    if inject_template_vars:
        builtins = {
            "date": date_val,
            "agent_id": agent_id,
            "conversation_id": conversation_id,
        }
    return PromptContext(
        agent=agent,
        agent_id=str(agent_id),
        conversation_id=conversation_id,
        memory=memory,
        budget_state=budget_state,
        date=date_val,
        builtins=builtins,
    )
