"""Budget operations use case: pre-call check, apply budget, record cost.

Agent delegates to functions here. Public API stays on Agent.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from syrin.agent import Agent

from syrin.budget import (
    BudgetExceededContext,
    BudgetLimitType,
    BudgetStatus,
    CheckBudgetResult,
)
from syrin.cost import calculate_cost
from syrin.enums import Hook
from syrin.events import EventContext
from syrin.exceptions import BudgetExceededError
from syrin.types import CostInfo, TokenUsage


def _apply_budget_exceeded(
    handler: Callable[[BudgetExceededContext], None] | None,
    ctx: BudgetExceededContext,
) -> bool:
    """Call the exceed handler with ctx and return whether execution should continue.

    Contract:
    - If handler is None: raise BudgetExceededError (default stop behaviour).
    - If handler raises: the exception propagates (caller stops).
    - If handler returns: execution may continue (WARN/IGNORE policy).

    Returns True if execution should continue (handler returned without raising).
    Never returns False — either returns True or raises.
    """
    if handler is None:
        raise BudgetExceededError(
            ctx.message,
            current_cost=ctx.current_cost,
            limit=ctx.limit,
            budget_type=ctx.budget_type.value,
        )
    handler(ctx)  # may raise; if it returns, execution continues
    return True


def pre_call_budget_check(
    agent: Agent,
    messages: list[object],
    max_output_tokens: int = 1024,
) -> None:
    """If run budget would be exceeded after an estimated call, invoke exceed handler."""
    estimate = agent.estimate_cost(messages, max_output_tokens=max_output_tokens)
    # Store estimate on agent so Response.cost_estimated can be populated.
    agent._last_cost_estimated = estimate
    if agent._budget is None or agent._budget.max_cost is None:
        return
    effective_run = agent._budget.effective_run_limit
    run_usage = agent._budget_tracker.run_usage_with_reserved
    if run_usage + estimate < effective_run:  # type: ignore[operator]
        return
    handler = agent._budget._handler
    limit = effective_run
    current = run_usage + estimate
    msg = (
        f"Budget would be exceeded: estimated run cost ${current:.4f} >= ${limit:.4f} "
        "(pre-call estimate)"
    )
    ctx = BudgetExceededContext(
        current_cost=current,
        limit=limit,  # type: ignore[arg-type]
        budget_type=BudgetLimitType.RUN,
        message=msg,
    )
    _apply_budget_exceeded(handler, ctx)


def check_and_apply_budget(agent: Agent) -> None:
    """Raise if budget or token limits exceeded; apply threshold actions (switch, warn)."""
    if agent._budget is None and agent._token_limits is None:
        return
    result: CheckBudgetResult = agent._budget_tracker.check_budget(
        agent._budget,
        token_limits=agent._token_limits,
        parent=agent,
    )
    if result.status == BudgetStatus.THRESHOLD:
        current = agent._budget_tracker.current_run_cost
        limit = (
            agent._budget.effective_run_limit
            if agent._budget is not None and agent._budget.effective_run_limit is not None
            else 0.0
        )
        pct = int((current / limit) * 100) if limit and limit > 0 else 0
        agent._emit_event(
            Hook.BUDGET_THRESHOLD,
            EventContext(
                threshold_percent=pct,
                current_value=current,
                limit_value=limit,
                metric="cost",
            ),
        )
        return
    if result.status != BudgetStatus.EXCEEDED:
        return
    limit_key = result.exceeded_limit or BudgetLimitType.RUN
    handler = agent._budget._handler if agent._budget is not None else None
    # Route to token _handler when the exceeded limit is a token limit type,
    # or fall back to token _handler when budget has no handler.
    if (
        limit_key
        in (
            BudgetLimitType.RUN_TOKENS,
            BudgetLimitType.HOUR_TOKENS,
            BudgetLimitType.DAY_TOKENS,
            BudgetLimitType.WEEK_TOKENS,
            BudgetLimitType.MONTH_TOKENS,
        )
        and agent._token_limits is not None
        and agent._token_limits._handler is not None
    ) or (handler is None and agent._token_limits is not None):
        handler = agent._token_limits._handler
    if limit_key == BudgetLimitType.RUN:
        current = agent._budget_tracker.current_run_cost
        limit = 0.0 if agent._budget is None else agent._budget.effective_run_limit or 0.0
        msg = f"Budget exceeded: run cost ${current:.4f} >= ${limit:.4f}"
    elif limit_key == BudgetLimitType.RUN_TOKENS:
        current = agent._budget_tracker.current_run_tokens
        run_tok = agent._token_limits.max_tokens if agent._token_limits is not None else None
        limit = float(run_tok or 0)
        msg = f"Budget exceeded: run tokens {current} >= {int(limit)}"
    elif limit_key in (
        BudgetLimitType.HOUR_TOKENS,
        BudgetLimitType.DAY_TOKENS,
        BudgetLimitType.WEEK_TOKENS,
        BudgetLimitType.MONTH_TOKENS,
    ):
        token_per = agent._token_limits.rate_limits if agent._token_limits is not None else None
        if limit_key == BudgetLimitType.HOUR_TOKENS and token_per is not None:
            current = float(agent._budget_tracker.hourly_tokens)
            limit = float(token_per.hour or 0)
        elif limit_key == BudgetLimitType.DAY_TOKENS and token_per is not None:
            current = float(agent._budget_tracker.daily_tokens)
            limit = float(token_per.day or 0)
        elif limit_key == BudgetLimitType.WEEK_TOKENS and token_per is not None:
            current = float(agent._budget_tracker.weekly_tokens)
            limit = float(token_per.week or 0)
        elif limit_key == BudgetLimitType.MONTH_TOKENS and token_per is not None:
            current = float(agent._budget_tracker.monthly_tokens)
            limit = float(token_per.month or 0)
        else:
            current, limit = 0.0, 0.0
        msg = f"Budget exceeded: {limit_key.value} {int(current)} >= {int(limit)}"
    else:
        rate = agent._budget.rate_limits if agent._budget else None
        if limit_key == BudgetLimitType.HOUR and rate:
            current, limit = agent._budget_tracker.hourly_cost, (rate.hour or 0)
        elif limit_key == BudgetLimitType.DAY and rate:
            current, limit = agent._budget_tracker.daily_cost, (rate.day or 0)
        elif limit_key == BudgetLimitType.WEEK and rate:
            current, limit = agent._budget_tracker.weekly_cost, (rate.week or 0)
        elif limit_key == BudgetLimitType.MONTH and rate:
            current, limit = agent._budget_tracker.monthly_cost, (rate.month or 0)
        else:
            current, limit = agent._budget_tracker.current_run_cost, 0.0
        msg = f"Budget exceeded: {limit_key.value} cost ${current:.4f} >= ${limit:.4f}"
    ctx = BudgetExceededContext(
        current_cost=current,
        limit=limit,
        budget_type=limit_key,
        message=msg,
    )
    _apply_budget_exceeded(handler, ctx)


def record_cost(agent: Agent, token_usage: TokenUsage, model_id: str) -> None:
    """Compute cost, build CostInfo, record on tracker, sync Budget._spent, re-check budget."""
    pricing = getattr(agent._model, "pricing", None) if agent._model is not None else None
    cost_usd = calculate_cost(model_id, token_usage, pricing_override=pricing)
    cost_info = CostInfo(
        token_usage=token_usage,
        cost_usd=cost_usd,
        model_name=model_id,
    )
    record_cost_info(agent, cost_info)


def make_budget_consume_callback(agent: Agent) -> Callable[[float], None]:
    """Return a callback for Budget.consume() so guardrails can record cost."""

    def _consume(amount: float) -> None:
        model_id = (
            agent._model_config.model_id
            if hasattr(agent, "_model_config") and agent._model_config
            else "unknown"
        )
        record_cost_info(agent, CostInfo(cost_usd=amount, model_name=model_id))

    return _consume


def record_cost_info(agent: Agent, cost_info: CostInfo) -> None:
    """Record a CostInfo. Syncs spent and checks budget."""
    agent._budget_tracker.record(cost_info)
    if agent._budget is not None:
        agent._budget._set_spent(agent._budget_tracker.current_run_cost)
    agent._budget_component.save()
    check_and_apply_budget(agent)
