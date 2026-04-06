"""Spawn use case: create child agents, optionally run task.

Agent delegates to functions here. Public API stays on Agent.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import TYPE_CHECKING, cast

from syrin.budget import Budget
from syrin.enums import Hook
from syrin.events import EventContext
from syrin.response import Response

if TYPE_CHECKING:
    from syrin.agent import Agent


def update_parent_budget(agent: Agent, cost: float) -> None:
    """Update parent's budget when child spends (borrow mechanism)."""
    if agent._budget is not None:
        from syrin.types import CostInfo

        model_id = (
            agent._model_config.model_id
            if hasattr(agent, "_model_config") and agent._model_config
            else "unknown"
        )
        cost_info = CostInfo(cost_usd=cost, model_name=model_id)
        agent._budget_tracker.record(cost_info)
        agent._budget._set_spent(agent._budget_tracker.current_run_cost)


def spawn(
    agent: Agent,
    agent_class: type[Agent],
    task: str | None = None,
    *,
    budget: Budget | None = None,
    max_child_agents: int | None = None,
) -> Agent | Response[str]:
    """Create child agent. Optionally run task and return response. Else return agent."""
    use_instance_limit = max_child_agents is None
    limit = getattr(agent, "_max_child_agents", 10) if use_instance_limit else max_child_agents

    current_children = getattr(agent, "_child_count", 0)

    if limit and current_children >= limit:
        raise RuntimeError(f"Cannot spawn: max child agents ({limit}) reached")

    child_name = agent_class.__name__
    child_task = task or ""
    child_budget = budget

    from syrin.context.snapshot import ContextSnapshot

    parent_snapshot = (
        agent._context.snapshot() if hasattr(agent._context, "snapshot") else ContextSnapshot()
    )
    parent_context_tokens = parent_snapshot.total_tokens

    start_ctx = EventContext(
        {
            "source_agent": type(agent).__name__,
            "child_agent": child_name,
            "child_task": child_task,
            "child_budget": child_budget,
            "context_inherited": False,
            "initial_context_tokens": 0,
            "parent_context_tokens": parent_context_tokens,
        }
    )
    agent._emit_event(Hook.SPAWN_START, start_ctx)

    if not hasattr(agent, "_child_count"):
        agent._child_count = 0
    if use_instance_limit:
        agent._child_count += 1

    agent_kwargs: dict[str, object] = {}

    if budget is not None:
        if agent._budget is not None and agent._budget.max_cost is not None:
            parent_remaining = agent._budget.remaining
            if (
                parent_remaining is not None
                and budget.max_cost is not None
                and budget.max_cost > parent_remaining
            ):
                raise ValueError(
                    f"Child budget (${budget.max_cost:.2f}) cannot exceed parent's "
                    f"remaining budget (${parent_remaining:.2f}). "
                    "Pocket money must be less than or equal to parent's available funds."
                )
        agent_kwargs["budget"] = budget
    elif agent._budget is not None:
        borrowed_budget = Budget(
            max_cost=agent._budget.remaining,
            rate_limits=agent._budget.rate_limits,
            exceed_policy=agent._budget.exceed_policy,
            thresholds=agent._budget.thresholds,
        )
        borrowed_budget._parent_budget = agent._budget
        agent_kwargs["budget"] = borrowed_budget

    child_agent = agent_class(**agent_kwargs)  # type: ignore[arg-type]
    child_agent._conversation_id = agent._conversation_id  # B6: propagate session_id

    # Track spawned children on parent for observability tools (Pry, tracers).
    spawned_list: list[object] = getattr(agent, "_spawned_children", [])
    spawned_list.append(child_agent)
    object.__setattr__(agent, "_spawned_children", spawned_list)

    # Propagate parent's event bus to child's context manager so hooks like
    # context.snapshot bubble up to any debugger/tracer attached to the parent.
    if hasattr(child_agent, "_context") and hasattr(child_agent._context, "set_emit_fn"):
        _parent_emit = agent._emit_event
        _child_ctx_emit = getattr(child_agent._context, "_emit_fn", None)

        def _make_bubble(
            child_fn: Callable[[str, dict[str, object]], None] | None,
            parent_fn: Callable[[str, dict[str, object]], None],
        ) -> Callable[[str, dict[str, object]], None]:
            def _bubble(event_str: str, ctx: dict[str, object]) -> None:
                if child_fn:
                    child_fn(event_str, ctx)
                parent_fn(event_str, ctx)

            return _bubble

        if _child_ctx_emit is not None:
            child_agent._context.set_emit_fn(_make_bubble(_child_ctx_emit, _parent_emit))

    borrowed = agent_kwargs.get("budget")
    if borrowed is not None and getattr(borrowed, "_parent_budget", None) is not None:
        child_agent._parent_agent = agent
        child_agent._budget_tracker = agent._budget_tracker
        child_agent._runtime.budget_tracker_shared = True

    if task:
        t0 = time.perf_counter()
        try:
            result = child_agent.run(task)
        finally:
            if use_instance_limit and agent._child_count > 0:
                agent._child_count -= 1
        duration = time.perf_counter() - t0
        if not child_agent._runtime.budget_tracker_shared:
            update_parent_budget(agent, result.cost)
        end_ctx = EventContext(
            {
                "source_agent": type(agent).__name__,
                "child_agent": child_name,
                "child_task": task,
                "cost": result.cost,
                "duration": duration,
            }
        )
        agent._emit_event(Hook.SPAWN_END, end_ctx)
        return result

    return child_agent


def spawn_parallel(
    agent: Agent,
    agents_spec: list[tuple[type[Agent], str]],
) -> list[Response[str]]:
    """Run multiple agents via spawn(), each with its own task."""
    results: list[Response[str]] = []
    for ac, t in agents_spec:
        r = spawn(agent, ac, task=t)
        results.append(cast(Response[str], r))
    return results
