# Use Case 4: Budget Control & Cost Management

> **Agent integration:** For the agent `budget=` constructor param and `budget_state`, see [Agent: Budget](agent/budget.md).

## What You'll Learn

How to **keep your AI costs under control**. Perfect for:
- Production applications
- Limiting spending
- Testing without surprises
- Enterprise deployments

AI API calls cost money. Syrin helps you manage and monitor every dollar.

**Budget = real money (USD).** All limits and thresholds here are about **spend**. Token usage caps live on **Context**: use **TokenLimits** (run, per, on_exceeded) and pass `context=Context(budget=TokenLimits(...))` on the agent, so Budget stays strictly about dollars.

## The Idea

```
You: "Run this agent but don't spend more than $1"
  ↓
Agent: "I'll do everything within budget"
  ↓
Agent runs... costs $0.50
  ↓
Agent: ✓ "Stayed within budget!"

OR

Agent runs... costs would be $2
  ↓
Agent: ❌ "STOP! Would exceed budget!"
```

## Complete Example: Copy & Paste This!

```python
"""
Budget Control Example
Keep your AI spending under control!
"""

import os
from syrin import Agent, Budget, RateLimit, raise_on_exceeded, warn_on_exceeded
from syrin.model import Model
from syrin.threshold import BudgetThreshold
from syrin.enums import ThresholdWindow


class BudgetAwareAgent(Agent):
    """An agent with a spending budget."""

    # model = Model.OpenAI("gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"))
    model = Model.Almock()  # No API Key needed
    system_prompt = "You are a helpful assistant."

    def __init__(self):
        super().__init__()
        # Set a budget of $1.00 per run; thresholds at 80% and 95% (run window)
        self.budget = Budget(
            run=1.00,  # Max $1 per single request
            on_exceeded=warn_on_exceeded,
            thresholds=[
                BudgetThreshold(at=80, action=lambda ctx: print(f"80% budget used: ${ctx.current_value:.2f}")),
                BudgetThreshold(at=95, action=lambda ctx: print(f"95% budget used: ${ctx.current_value:.2f}")),
            ],
        )


def main():
    agent = BudgetAwareAgent()
    
    questions = [
        "What is Python?",
        "Explain machine learning",
        "Tell me a joke",
    ]
    
    total_spent = 0.0
    
    for question in questions:
        print(f"\n❓ {question}")
        
        response = agent.response(question)
        cost = response.cost
        total_spent += cost
        
        print(f"✓ Answer: {response.content[:100]}...")
        print(f"💰 Cost: ${cost:.4f}")
        print(f"📊 Total spent: ${total_spent:.4f}")
        print(f"📈 Budget remaining: ${1.00 - total_spent:.4f}")


if __name__ == "__main__":
    main()
```

## Budget Types

You can set different budget limits:

### 1. Per-Run Budget

Max spending for a single request:

```python
import os
from syrin import Agent, Budget, RateLimit, raise_on_exceeded, warn_on_exceeded
from syrin.model import Model

class Agent1(Agent):
    # model = Model.OpenAI("gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"))
    model = Model.Almock()  # No API Key needed

    def __init__(self):
        super().__init__()
        self.budget = Budget(
            run=0.10  # Max $0.10 per single request
        )
```

### 2. Rate Limits (Hourly, Daily, Weekly, Monthly)

Max spending per time period via `RateLimit`:

```python
import os
from syrin import Agent, Budget, RateLimit
from syrin.model import Model

class Agent2(Agent):
    # model = Model.OpenAI("gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"))
    model = Model.Almock()  # No API Key needed

    def __init__(self):
        super().__init__()
        self.budget = Budget(
            per=RateLimit(hour=5.00)  # Max $5 per hour
        )
```

### 3. Daily Budget

```python
class Agent3(Agent):
    # model = Model.OpenAI("gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"))
    model = Model.Almock()  # No API Key needed

    def __init__(self):
        super().__init__()
        self.budget = Budget(
            per=RateLimit(day=50.00)  # Max $50 per day
        )
```

### 4. Monthly Budget and configurable month length

The "month" window is the last **N** days (wall-clock). Default is 30; you can set `month_days` (1–31):

```python
class Agent4(Agent):
    # model = Model.OpenAI("gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"))
    model = Model.Almock()  # No API Key needed

    def __init__(self):
        super().__init__()
        self.budget = Budget(
            per=RateLimit(month=500.00, month_days=30)  # Last 30 days (default)
            # Or month_days=7 for a "rolling week" style month
        )
```

### 4b. Calendar month

Use **calendar month** (current month, e.g. 1–30 Nov) instead of "last N days":

```python
self.budget = Budget(
    per=RateLimit(month=500.00, calendar_month=True)  # Current calendar month only
)
```

### 5. Token limits (separate from Budget)

**Budget = real money (USD) only.** Token usage caps are on **Context**: **TokenLimits**. Use `context=Context(budget=TokenLimits(...))` on the agent so you don't mix spend and usage.

```python
from syrin import Agent, Budget, Context, TokenLimits, Model, TokenRateLimit, raise_on_exceeded

agent = Agent(
    model=Model("openai/gpt-4o-mini"),
    budget=Budget(run=0.10, on_exceeded=raise_on_exceeded),  # USD only
    context=Context(budget=TokenLimits(
        run=10_000,
        per=TokenRateLimit(hour=50_000, day=200_000),
        on_exceeded=raise_on_exceeded,
    )),
)
```

When a token limit is exceeded, the callback receives `BudgetExceededContext` with `budget_type` (e.g. `BudgetLimitType.HOUR_TOKENS`). You can use **Budget only** (USD), **Context.budget only** (usage), or **both**.

**Example:** Run `python -m examples.core.budget_rate_limits_and_tokens` — the first example uses Budget + Context.budget.

### 6. All Combined

You can combine run limit and rate limits:

```python
import os
from syrin import Agent, Budget, RateLimit, raise_on_exceeded, warn_on_exceeded
from syrin.model import Model

class MultibudgetAgent(Agent):
    # model = Model.OpenAI("gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"))
    model = Model.Almock()  # No API Key needed

    def __init__(self):
        super().__init__()
        self.budget = Budget(
            run=0.10,                              # Each run max $0.10
            per=RateLimit(
                hour=5.00,
                day=50.00,
                week=100.00,
                month=500.00,
            ),
            on_exceeded=raise_on_exceeded
        )
```

## What Happens When Budget is Exceeded

You control what happens by passing a callback to `on_exceeded`. Use `warn_on_exceeded` to log and continue, or `raise_on_exceeded` to stop.

### 1. Warn and continue

```python
import os
from syrin import Agent, Budget, warn_on_exceeded
from syrin.model import Model

class WarnAgent(Agent):
    # model = Model.OpenAI("gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"))
    model = Model.Almock()  # No API Key needed

    def __init__(self):
        super().__init__()
        self.budget = Budget(
            run=0.10,
            on_exceeded=warn_on_exceeded  # Log warning and continue
        )


# Result: Agent will still work, but warns you
```

### 2. OnExceeded.ERROR

Stop and raise an error:

```python
class StrictAgent(Agent):
    # model = Model.OpenAI("gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"))
    model = Model.Almock()  # No API Key needed

    def __init__(self):
        super().__init__()
        self.budget = Budget(
            run=0.10,
            on_exceeded=raise_on_exceeded  # Stop immediately
        )


# Result: If budget would be exceeded, agent raises an error
agent = StrictAgent()

try:
    response = agent.response("Answer 100 questions")
except Exception as e:
    print(f"Budget exceeded: {e}")
```

## Budget Thresholds (spend alerts)

Thresholds run only when the agent has a **Budget**. They run an action when **spend** (or token usage, if using TokenLimits and `metric=ThresholdMetric.TOKENS`) reaches a percentage of a limit. The limit is either "this request" (run), or "this hour", "this day", or "this month."

- **`at=80`** means "when we've used **at or above 80%** of that limit" (e.g. 80% of this request's budget, or 80% of today's budget).
- If several thresholds are crossed, by default only the **closest** one runs (e.g. at 88% usage, only the 80% action runs, not the 50% one). Use `threshold_fallthrough=True` on the budget to run all crossed thresholds.

**Simple example — thresholds on this request's spend:**

```python
import os
from syrin import Agent, Budget, raise_on_exceeded, warn_on_exceeded
from syrin.model import Model
from syrin.threshold import BudgetThreshold

class ThresholdAgent(Agent):
    # model = Model.OpenAI("gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"))
    model = Model.Almock()  # No API Key needed

    def __init__(self):
        super().__init__()
        self.budget = Budget(
            run=1.00,  # $1 max per request
            thresholds=[
                BudgetThreshold(at=50, action=lambda ctx: print(f"Halfway: {ctx.percentage}% of run budget")),
                BudgetThreshold(at=80, action=lambda ctx: print(f"Getting close: {ctx.percentage}%")),
                BudgetThreshold(at=95, action=lambda ctx: ctx.parent.switch_model(Model("openai/gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY")))),
            ],
        )
```

**Example — thresholds on run spend and daily spend:**

Use `window=` to choose which limit the percentage is based on: this request (`ThresholdWindow.RUN`), this hour (`HOUR`), this day (`DAY`), or this month (`MONTH`). All are **spend in USD**.

```python
import os
from syrin import Agent, Budget, RateLimit, warn_on_exceeded
from syrin.model import Model
from syrin.threshold import BudgetThreshold
from syrin.enums import ThresholdWindow

class SpendThresholdAgent(Agent):
    # model = Model.OpenAI("gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"))
    model = Model.Almock()  # No API Key needed
    system_prompt = "Be concise."

    def __init__(self):
        super().__init__()
        self.budget = Budget(
            run=1.00,
            per=RateLimit(hour=5.00, day=20.00),  # $5/hour, $20/day — all dollars
            on_exceeded=warn_on_exceeded,
            thresholds=[
                BudgetThreshold(at=80, action=lambda ctx: print(f"80% of this request's budget used")),
                BudgetThreshold(at=90, action=lambda ctx: print(f"90% of today's budget used"), window=ThresholdWindow.DAY),
            ],
        )
```

**Optional: token usage alerts** — If you use **TokenLimits** (separate from Budget), you can run a threshold when **token usage** hits a percentage of that cap. Use `metric=ThresholdMetric.TOKENS` and the same window enums. For a narrow band use `at_range=(70, 75)`. See the [Budget reference](#budget-reference-whats-available) table for `ThresholdMetric` and `ThresholdWindow`.

## Cost Tracking

Check how much you've spent:

```python
agent = BudgetAwareAgent()

response1 = agent.response("Question 1")
print(f"Cost: ${response1.cost:.4f}")

response2 = agent.response("Question 2")
print(f"Cost: ${response2.cost:.4f}")

# Check budget status
print(f"Budget: {agent.budget}")
print(f"Remaining: {response2.budget_remaining}")
print(f"Used: {response2.budget_used}")
```

## Real-World Example: Production Agent

```python
"""
Production-Grade Agent with Budget Management
Safe to deploy with cost controls
"""

from syrin import Agent, Budget, RateLimit, raise_on_exceeded, warn_on_exceeded
from syrin.model import Model
from syrin.threshold import BudgetThreshold
from syrin.tool import tool
from syrin.enums import ThresholdWindow


@tool
def search_database(query: str) -> dict:
    """Search internal database."""
    return {"results": ["result1", "result2"]}


class ProductionAgent(Agent):
    """
    A production agent with careful budget management.
    
    Budget breakdown:
    - $0.01 per request (gpt-4o-mini is very cheap)
    - $10 per hour maximum
    - $100 per day maximum
    - $1000 per month maximum
    """
    
    # model = Model.OpenAI("gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"))  # Cheapest reliable model
    model = Model.Almock()  # No API Key needed

    system_prompt = """
    You are a production assistant.
    Be efficient and concise.
    Help users quickly without unnecessary details.
    """
    
    tools = [search_database]
    
    def __init__(self):
        super().__init__()
        
        # Production budget with multiple safeguards
        self.budget = Budget(
            run=0.01,                               # Each request max $0.01
            per=RateLimit(
                hour=10.00,
                day=100.00,
                month=1000.00,
            ),
            on_exceeded=raise_on_exceeded,  # Fail hard if exceeded
            thresholds=[
                BudgetThreshold(at=75, action=lambda ctx: print(f"Run budget at {ctx.percentage}%")),
                BudgetThreshold(at=90, action=lambda ctx: print(f"Run budget at {ctx.percentage}%")),
            ],
        )


def main():
    agent = ProductionAgent()
    
    # Simulate production usage
    for i in range(5):
        try:
            response = agent.response(f"Help with request {i+1}")
            print(f"Request {i+1}: ${response.cost:.4f}")
        except Exception as e:
            print(f"Request {i+1}: Budget exceeded - {e}")
            break


if __name__ == "__main__":
    main()
```

## Using Different Models by Cost

Pass `api_key` (e.g. `os.getenv("OPENAI_API_KEY")`) when creating models. Cheaper models:

```python
import os
# ~$0.00005 per 1K tokens
Model.OpenAI("gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"))

# ~$0.0001 per 1K tokens (Anthropic)
Model.Anthropic("claude-3-haiku", api_key=os.getenv("ANTHROPIC_API_KEY"))
```

Mid-range models:
```python
Model.OpenAI("gpt-4o", api_key=os.getenv("OPENAI_API_KEY"))
Model.Anthropic("claude-3-sonnet", api_key=os.getenv("ANTHROPIC_API_KEY"))
```

Expensive (but better) models:
```python
Model.OpenAI("gpt-4", api_key=os.getenv("OPENAI_API_KEY"))
Model.Anthropic("claude-3-opus", api_key=os.getenv("ANTHROPIC_API_KEY"))
```

## Cost Monitoring Dashboard

Track costs over time:

```python
"""
Simple Cost Dashboard
Monitor spending trends
"""

from syrin import Agent, Budget, RateLimit, raise_on_exceeded, warn_on_exceeded
from syrin.model import Model
import json
from datetime import datetime


class MonitoredAgent(Agent):
    # model = Model.OpenAI("gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"))
    model = Model.Almock()  # No API Key needed

    def __init__(self):
        super().__init__()
        self.budget = Budget(run=0.10, per=RateLimit(hour=10.00))
        self.costs_log = []
    
    def response(self, prompt: str):
        response = super().response(prompt)
        
        # Log the cost (response.cost in USD; response.tokens is TokenUsage)
        self.costs_log.append({
            "timestamp": datetime.now().isoformat(),
            "cost": response.cost,
            "tokens": response.tokens.total_tokens,
            "model": str(response.model)
        })
        
        return response
    
    def print_stats(self):
        if not self.costs_log:
            print("No requests made yet")
            return
        
        total_cost = sum(log["cost"] for log in self.costs_log)
        total_tokens = sum(log["tokens"] for log in self.costs_log)
        avg_cost = total_cost / len(self.costs_log)
        
        print("\n📊 Cost Statistics")
        print(f"Total requests: {len(self.costs_log)}")
        print(f"Total cost: ${total_cost:.4f}")
        print(f"Total tokens: {total_tokens}")
        print(f"Average cost per request: ${avg_cost:.4f}")
        print(f"Requests: {json.dumps(self.costs_log, indent=2)}")


def main():
    agent = MonitoredAgent()
    
    # Make some requests
    for i in range(3):
        response = agent.response(f"Answer question {i+1}")
        print(f"Request {i+1}: Completed")
    
    # Show statistics
    agent.print_stats()


if __name__ == "__main__":
    main()
```

## Budget Best Practices

### 1. Start Conservative

```python
# ✓ Good - start small
budget = Budget(run=0.05, per=RateLimit(hour=5.00, day=50.00))

# ❌ Not good - too generous
budget = Budget(run=1.00, per=RateLimit(hour=100.00, day=1000.00))
```

### 2. Use Cheap Models for Testing

```python
import os
# ✓ Good - cheap for testing
test_agent = Agent()
# test_agent.model = Model.OpenAI("gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"))
test_agent.model = Model.Almock()  # No API Key needed

# ❌ Not good - expensive for testing
# test_agent.model = Model.OpenAI("gpt-4", api_key=os.getenv("OPENAI_API_KEY"))
test_agent.model = Model.Almock()  # No API Key needed
```

### 3. Monitor Costs Regularly

```python
# ✓ Good - track spending
total_cost = 0.0
for i in range(10):
    response = agent.response(prompt)
    total_cost += response.cost
    print(f"Cost so far: ${total_cost:.4f}")

# ❌ Not good - don't check
for i in range(10):
    response = agent.response(prompt)
```

### 4. Set Multiple Budget Levels

```python
# ✓ Good - multiple safeguards
budget = Budget(
    run=0.01,
    per=RateLimit(hour=10, day=100, month=1000),
    on_exceeded=raise_on_exceeded
)

# ❌ Not good - only one limit
budget = Budget(run=100.00)
```

## Common Questions

**Q: How accurate is the cost estimate?**
A: Very accurate! It's based on actual token counts from the API.

**Q: What if the budget is exceeded?**
A: It depends on your `on_exceeded` callback:
- Pass `warn_on_exceeded`: logs a warning and continues
- Pass `raise_on_exceeded`: raises `BudgetExceededError` and stops

When a limit is exceeded, the error message and `BudgetExceededError.budget_type` (a string) tell you which limit was hit. `BudgetExceededContext.budget_type` and `CheckBudgetResult.exceeded_limit` use the `BudgetLimitType` enum (e.g. `BudgetLimitType.RUN`, `BudgetLimitType.HOUR_TOKENS`) for type-safe checks. Use `result.status == BudgetStatus.EXCEEDED` when checking `check_budget()` results (the result is a `CheckBudgetResult`, not a `BudgetStatus`).

**Q: Can I see all my costs?**
A: Yes! Check `response.cost` and track them over time.

**Q: What's the cheapest model?**
A: `gpt-4o-mini` (~$0.00005 per 1K tokens)

**Q: Can I use custom pricing?**
A: Yes. Pass `pricing_override` (a `ModelPricing` instance) when constructing your model, or use a `pricing_resolver` callable with `syrin.cost.calculate_cost()` for pluggable pricing (e.g. from a database or config).

## Streaming and budget

When you use `agent.stream()` or `agent.astream()`, cost is recorded **per chunk** and the budget (and token limits, if `context.budget` is set) is checked after each chunk. If the run limit or a token limit is exceeded mid-stream, the stream stops and `BudgetExceededError` is raised so you get partial output and accurate cost so far.

## FileBudgetStore and concurrency

**BudgetTracker** is **thread-safe**: all public methods (e.g. `record`, `check_budget`, `get_state`, `load_state`, cost/token properties) use an internal lock so concurrent use from multiple threads is safe.

`FileBudgetStore` uses **file locking** when saving so that multiple threads or processes can safely write to the same file (e.g. different keys in `single_file=True` mode): **fcntl** on Unix and **msvcrt** on Windows. This helps prevent corruption when several processes or threads save at once.

## Pre-call estimate and reservation

**Pre-call estimate:** Before each LLM call, the built-in loops call a **best-effort** estimate of the next call's cost (from message token counts and model pricing). If `current_run_cost + reserved + estimate` would exceed the run limit, `on_exceeded` is invoked and the call is skipped (no API request). Use `agent.estimate_cost(messages, max_output_tokens=1024)` yourself for custom loops. Accuracy depends on tokenizer and pricing; actual cost may differ.

**Reservation and rollback:** When the agent has a budget or token_limits, use the tracker to reserve before a call and commit actual cost or roll back on failure:

```python
tracker = agent.get_budget_tracker()  # None if agent has neither budget nor context.budget
if tracker:
    token = tracker.reserve(estimated_cost)
    try:
        response = await agent.complete(messages, tools)
        token.commit(actual_cost, response.token_usage)
    except Exception:
        token.rollback()
```

`check_budget` treats reserved amount as used for the run limit, so concurrent reservations cannot over-commit.

**BudgetSummary** includes `hourly_tokens`, `daily_tokens`, `weekly_tokens`, and `monthly_tokens` (from the tracker) in addition to cost fields.

## Implementing a custom BudgetStore

To use Redis, Postgres, or another backend for budget state:

1. **Implement the interface:** Your class must provide:
   - `load(key: str) -> BudgetTracker | None` — return a `BudgetTracker` restored from storage, or `None` if the key is missing.
   - `save(key: str, tracker: BudgetTracker) -> None` — persist `tracker.get_state()` under `key`.

2. **State shape:** `tracker.get_state()` returns a dict with `version` (schema version), `cost_history` (list of entries with `cost_usd`, `timestamp`, `model_name`, `total_tokens`), `run_start`, `month_days`, and `use_calendar_month`. Use `tracker.load_state(state)` to restore. Older state without `version` is still accepted.

3. **Concurrency:** For a single file or key, use exclusive locking when writing (e.g. fcntl/msvcrt or Redis SET with a single key) so concurrent writers do not corrupt state.

4. **Usage:** Pass your store and a key when creating the agent: `Agent(..., budget_store=MyStore(), budget_store_key="user_123")`.

## Org/user-level budget pattern

Budget is per agent by default. To enforce a **cap per org or per user** across multiple agents:

- Use **one** `BudgetStore` (e.g. `FileBudgetStore` or a custom Redis store).
- Set **one key per org or user**, e.g. `budget_store_key="org:123"` or `budget_store_key="user:456"`.
- All agents that use the same key share the same cost/token history, so hour/day/week/month limits apply across them.

Example: `Agent(..., budget_store=FileBudgetStore("/data/budget.json"), budget_store_key=f"org:{org_id}")`.

## Budget reference (what’s available)

**Budget** = spend in USD. **Token caps** are usage limits on Context; use **TokenLimits** and pass `context=Context(budget=...)` on the agent.

| Feature | Where | Description |
|--------|--------|--------------|
| **Run limit** | `Budget(run=..., reserve=...)` | Max **USD** per request; effective limit is `run - reserve`. |
| **Rate limits (spend)** | `Budget(per=RateLimit(...))` | Max **USD** per period: `hour`, `day`, `week`, `month`. |
| **Token limits (Context)** | `Agent(..., context=Context(budget=TokenLimits(...)))` | Token caps live on Context. Use `TokenLimits(run=..., per=TokenRateLimit(...))` on Context. |
| **month_days** | `RateLimit(month=..., month_days=N)` | Month = last N days (1–31). Default 30. |
| **calendar_month** | `RateLimit(month=..., calendar_month=True)` | Month = current calendar month only (e.g. 1–30 Nov). |
| **on_exceeded** | `Budget(on_exceeded=...)` | Callback when a limit is exceeded; receives `BudgetExceededContext` with `budget_type` (`BudgetLimitType`). |
| **BudgetLimitType** | `syrin.enums` | Enum for type-safe checks: `RUN`, `RUN_TOKENS`, `HOUR`, `DAY`, `WEEK`, `MONTH`, `HOUR_TOKENS`, `DAY_TOKENS`, `WEEK_TOKENS`, `MONTH_TOKENS`. |
| **Thresholds** | `Budget(thresholds=[...])` | `BudgetThreshold(at=..., action=..., window=..., metric=...)`; optional `at_range=(lo, hi)`. |
| **ThresholdWindow** | `syrin.enums` | `RUN`, `HOUR`, `DAY`, `WEEK`, `MONTH` — use for `window=` (no free strings). |
| **ThresholdMetric** | `syrin.enums` | `COST` (default) or `TOKENS` for threshold metric. |
| **BudgetState** | `agent.budget_state` | `limit`, `remaining`, `spent`, `percent_used`. Use `agent.get_budget_tracker().get_summary()` for full rolling-window stats. |
| **get_budget_tracker()** | `agent.get_budget_tracker()` | Returns the tracker when the agent has a budget or context.budget; otherwise `None`. Use for reservation or inspection. |
| **Reservation** | `tracker.reserve(amount)` | Returns a token; call `token.commit(actual_cost, token_usage)` or `token.rollback()`. |
| **Pre-call estimate** | `agent.estimate_cost(messages, max_output_tokens=...)` | Best-effort cost estimate; used by built-in loops to skip calls that would exceed run limit. |
| **State** | `tracker.get_state()` / `tracker.load_state(state)` | State includes `version`, `cost_history`, `run_start`, `month_days`, `use_calendar_month`. State without `version` is accepted for loading. |
| **BudgetStore** | `Agent(..., budget_store=..., budget_store_key=...)` | Persist tracker across restarts (e.g. `FileBudgetStore`). |
| **Thread safety** | BudgetTracker | All public tracker methods are thread-safe (internal lock). |
| **Streaming** | `agent.stream()` / `agent.astream()` | Cost recorded per chunk; run limit (and token limits if context.budget set) checked after each chunk; can raise `BudgetExceededError` mid-stream. |
| **Response** | `response.cost`, `response.budget_remaining`, `response.budget_used`, `response.budget` | Cost in USD and budget status. With context.budget only (no Budget), `budget_remaining` and `budget_used` may be `None`. |

## Limitations

- **Rate windows:** Hour, day, and week use fixed lengths. The month window is the last **N** days (configurable via `month_days`, default 30). Custom or sliding windows beyond that are not supported.
- **Rate limits and persistence:** Hour/day/week/month limits use the in-memory cost and token history of the current `BudgetTracker`. Without a `budget_store`, that state is lost when the process exits. To enforce rate limits **across restarts**, pass a `BudgetStore` (e.g. `FileBudgetStore`) and a `budget_store_key`.
- **Check timing:** A **pre-call** estimate (best-effort) can skip an LLM call if it would exceed the run limit; otherwise budget is checked **after** each LLM call (and after each chunk when streaming). You can overshoot by at most one call's cost (or one chunk when streaming).
- **Pricing:** For custom or new models, set `pricing_override` on the model so budget reflects actual cost; otherwise built-in pricing may be wrong or zero.
- **Scope:** Budget is per agent (or shared with child agents via `shared=True`). For org/user caps, use the pattern above (one store, one key per org/user).
- **Distributed:** No built-in shared store for multi-instance deployments; implement a custom `BudgetStore` (e.g. Redis-backed) for cross-process or cross-machine budget.

## Next Steps

- **Build Teams** → Learn [Use Case 5: Multi-Agent Orchestration](multi-agent.md)
- **Get Real-Time Updates** → Learn [Use Case 6: Streaming](streaming.md)
- **Advanced Features** → See [Feature Reference Guide](reference.md)

---

Questions? See the [FAQ in Getting Started](getting-started.md#common-questions)
