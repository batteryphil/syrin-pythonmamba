"""Run context for dependency injection into tools.

Tools that declare a first parameter named ``ctx`` with type RunContext[SomeDeps]
receive this context when executed. Use for injecting DB, search clients,
user preferences, etc. Enables testing (mock deps) and multi-tenant (different
deps per user).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, TypeVar

from syrin.budget import BudgetState

DepsT = TypeVar("DepsT")


@dataclass
class RunContext(Generic[DepsT]):
    """Context passed to tools when dependency injection is configured.

    Available inside @syrin.tool methods when the first parameter is
    ``ctx: syrin.RunContext[YourDeps]``. Tools receive this context
    instead of hardcoding backends — enables testing and multi-tenant.

    Attributes:
        deps: The injected dependencies (Agent.deps).
        agent_name: Current agent class name (e.g. "Researcher").
        conversation_id: Current conversation ID for state isolation, or None.
        budget_state: Current budget state (limit, remaining, spent, percent_used)
            or None when no budget.
        retry_count: Current retry attempt (for output validation loops).

    Example:
        >>> from dataclasses import dataclass
        >>> @dataclass
        ... class MyDeps:
        ...     db: Database
        ...     user_id: str
        >>>
        >>> class MyAgent(Agent):
        ...     @syrin.tool
        ...     def get_data(self, ctx: syrin.RunContext[MyDeps], query: str) -> str:
        ...         return ctx.deps.db.query(user_id=ctx.deps.user_id, q=query)
        >>>
        >>> agent = MyAgent(deps=MyDeps(db=db, user_id="alice"))
        >>> agent.response("What's in my data?")
    """

    deps: DepsT
    """The injected dependencies (Agent.deps)."""

    agent_name: str
    """Current agent class name (e.g. 'Researcher')."""

    conversation_id: str | None
    """Current conversation ID for state isolation (per-user or per-session), or None."""

    budget_state: BudgetState | None
    """Current budget state or None when no budget."""

    retry_count: int = 0
    """Current retry attempt (for output validation loops)."""


__all__ = ["RunContext"]
