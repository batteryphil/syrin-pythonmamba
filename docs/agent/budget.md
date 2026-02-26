# Budget

> **BudgetStore & use case:** For BudgetStore, cost utilities, and the full use-case guide, see [Budget Control](../budget-control.md).

Control and track agent spending via budgets.

## Basic Usage

```python
from syrin import Agent, Budget

agent = Agent(
    model=model,
    budget=Budget(run=1.0),
)

response = agent.response("Hello")
print(agent.budget_state)
print(response.budget_remaining)
```

## Budget Parameters

```python
Budget(
    run=1.0,                    # Per-run limit (USD)
    per=RateLimit(hour=10),     # Per-period limits
    on_exceeded=raise_on_exceeded,  # Callback when exceeded
    thresholds=[...],
    shared=False,
)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `run` | `float \| None` | Max USD per run |
| `reserve` | `float` | Amount to reserve; effective run limit is `run - reserve`. Default 0. |
| `per` | `RateLimit \| None` | Per-period limits (USD) |
| `on_exceeded` | `Callable[[BudgetExceededContext], None] \| None` | Called when a limit is exceeded. Raise to stop; return to continue. Use `raise_on_exceeded` or `warn_on_exceeded`. |
| `thresholds` | `list[Threshold]` | Switch/warn/stop at % (only when agent has a Budget) |
| `shared` | `bool` | Share with child agents |

## on_exceeded callback

Pass a function that receives `BudgetExceededContext`. Built-in helpers:

| Helper | Behavior |
|--------|----------|
| `raise_on_exceeded` | Raise `BudgetExceededError` and stop |
| `warn_on_exceeded` | Log a warning and continue |
| `stop_on_exceeded` | Raise `BudgetThresholdError` and stop |

## Budget Store

Persist budget across runs:

```python
from syrin.budget_store import FileBudgetStore

agent = Agent(
    model=model,
    budget=Budget(run=1.0),
    budget_store=FileBudgetStore("/tmp/budget.json"),
    budget_store_key="user_123",
)
```

## Response Fields

- `response.budget_remaining` — Remaining run budget (may be `None` if the agent has only TokenLimits and no Budget)
- `response.budget_used` — Used this run (may be `None` if the agent has only TokenLimits and no Budget)
- `response.budget` — Budget info on the response

## Budget Tracker and Reservation

**`agent.get_budget_tracker()`** returns the tracker when the agent has a **budget or token_limits**; otherwise `None`. Use it for reservation before a call and commit or rollback after:

```python
tracker = agent.get_budget_tracker()
if tracker:
    token = tracker.reserve(estimated_cost)
    try:
        response = await agent.complete(messages, tools)
        token.commit(actual_cost, response.token_usage)
    except Exception:
        token.rollback()
```

## Token limits (on Context)

Budget is **USD only**. To cap **token usage** (e.g. per run or per hour), use **TokenLimits** on **Context** and pass `context=Context(budget=...)` on the agent:

```python
from syrin import Agent, Budget, Context, TokenLimits, TokenRateLimit
from syrin.budget import raise_on_exceeded

agent = Agent(
    model=model,
    budget=Budget(run=1.0),
    context=Context(
        budget=TokenLimits(
            run=10_000,
            per=TokenRateLimit(hour=50_000, day=200_000),
            on_exceeded=raise_on_exceeded,
        )
    ),
)
```

You can use **Budget only**, **Context.budget only**, or **both**. Full details: [Budget Control](../budget-control.md). Runnable example: `python -m examples.core.budget_rate_limits_and_tokens` (see `example_budget_plus_token_limits`).

For type-safe limit checks in callbacks, use **`BudgetLimitType`** and **`ThresholdWindow`**. See [Budget Control](../budget-control.md).

## See Also

- [Budget Control](../budget-control.md) — BudgetStore, cost utilities, full use-case guide
