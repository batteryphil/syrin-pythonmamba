# Event Bus — Typed Domain Events

The **event bus** (`EventBus`) lets you subscribe to **typed domain events** such as `BudgetThresholdReached` and `ContextCompacted`. Use it when you need structured, type-safe event handling for metrics, observability, or custom pipelines.

## When to Use the Event Bus

| Scenario | Event bus | Hooks (`agent.events`) |
|----------|-----------|------------------------|
| **Quick logging** | Optional | ✅ Use Hooks |
| **Type-safe handlers** | ✅ Use bus | Hooks use dicts |
| **Metrics / dashboards** | ✅ Use bus | Possible but manual |
| **Observability pipelines** | ✅ Use bus | Possible but manual |
| **IDE autocomplete** | ✅ Typed events | Dict access only |
| **Multiple subscribers** | ✅ Subscribe per type | One handler per hook |
| **With default hooks** | Add-on | Default API |

**Use the event bus when:**
- You need typed event payloads (e.g. `e.percentage`, `e.tokens_before`) instead of dict keys
- You're building metrics, tracing, or analytics pipelines
- You want multiple handlers for the same event type
- You prefer structured events over generic `EventContext` dicts

**Use Hooks when:**
- You're doing simple logging or debugging
- You don't need typed payloads
- You want the default, zero-config API

## How It Works

1. Create an `EventBus`.
2. Subscribe handlers to specific event types.
3. Pass the bus to the agent via `bus=bus`.
4. The agent emits typed events when they occur; your handlers are called.

```
Agent runs → budget threshold crossed → emits BudgetThresholdReached
                                        ↓
                              bus.emit(event)
                                        ↓
                              Subscribed handlers run
                                        ↓
                              e.g. metrics.increment(...)
```

## Quick Start

```python
from syrin import Agent, Budget, Model
from syrin.domain_events import (
    EventBus,
    AgentRunStarted,
    AgentRunEnded,
    BudgetThresholdReached,
    ToolCallCompleted,
)

bus = EventBus()

# Subscribe to typed events — IDE autocomplete works on all fields
bus.subscribe(AgentRunStarted, lambda e: print(f"Starting with model={e.model}"))
bus.subscribe(AgentRunEnded, lambda e: print(f"Done! cost=${e.cost:.4f}, tokens={e.tokens}"))
bus.subscribe(BudgetThresholdReached, lambda e: print(f"Budget at {e.percentage}%"))
bus.subscribe(ToolCallCompleted, lambda e: print(f"Tool {e.tool_name} took {e.duration_ms:.0f}ms"))

agent = Agent(
    # model=Model("openai/gpt-4o-mini"),
    model=Model.Almock(),  # No API Key needed
    budget=Budget(run=0.50),
    bus=bus,
)

agent.response("Hello")
# Handlers run when events occur
```

## Using the Builder

```python
agent = (
    Agent.builder(Model("openai/gpt-4o-mini"))
    .with_budget(Budget(run=0.5))
    .with_event_bus(bus)
    .build()
)
```

## Event Types

All domain events are immutable `@dataclass(frozen=True)` classes. Every field has IDE autocomplete.

### AgentRunStarted

Emitted when an agent begins processing user input.

| Field | Type | Description |
| ----- | ---- | ----------- |
| `input` | str | User input text |
| `model` | str | Model ID being used |
| `iteration` | int | Current iteration count |

### AgentRunEnded

Emitted when an agent finishes and returns a response.

| Field | Type | Description |
| ----- | ---- | ----------- |
| `content` | str | Response content text |
| `cost` | float | Total cost in USD |
| `tokens` | int | Total tokens consumed |
| `duration` | float | Wall-clock time in seconds |
| `stop_reason` | str | Why the run stopped (end_turn, tool_use, etc.) |
| `iteration` | int | Final iteration count |

### LLMRequestStarted

Emitted before an LLM API call is sent.

| Field | Type | Description |
| ----- | ---- | ----------- |
| `iteration` | int | Current iteration |
| `tool_count` | int | Number of tools available |

### LLMRequestCompleted

Emitted after an LLM API response is received.

| Field | Type | Description |
| ----- | ---- | ----------- |
| `content` | str | Response content |
| `iteration` | int | Current iteration |

### ToolCallCompleted

Emitted after a tool executes successfully.

| Field | Type | Description |
| ----- | ---- | ----------- |
| `tool_name` | str | Name of the tool |
| `duration_ms` | float | Execution time in milliseconds |

### ToolCallFailed

Emitted when a tool execution raises an error.

| Field | Type | Description |
| ----- | ---- | ----------- |
| `tool_name` | str | Name of the tool |
| `error` | str | Error message |
| `iteration` | int | Current iteration |

### BudgetThresholdReached

Emitted when a budget threshold is crossed (e.g. 80% of run budget).

| Field | Type | Description |
| ----- | ---- | ----------- |
| `percentage` | int | Utilization (0-100) that triggered the threshold |
| `current_value` | float | Current cost or token count |
| `limit_value` | float | Limit or cap |
| `metric` | str | `"cost"` or `"tokens"` |
| `action_taken` | str \| None | Action from threshold (e.g. `"warn"`, `"switch_model"`) |

```python
bus.subscribe(BudgetThresholdReached, lambda e: log(f"Budget at {e.percentage}%: ${e.current_value:.2f}"))
```

### BudgetExceeded

Emitted when a hard budget limit is exceeded.

| Field | Type | Description |
| ----- | ---- | ----------- |
| `used` | float | Amount used (cost or tokens) |
| `limit` | float | The limit that was exceeded |
| `exceeded_by` | float | How much over the limit |

### GuardrailBlocked

Emitted when a guardrail blocks input or output.

| Field | Type | Description |
| ----- | ---- | ----------- |
| `stage` | str | `"input"` or `"output"` |
| `reason` | str | Human-readable reason |
| `guardrail_names` | list[str] | Names of guardrails that triggered |

### HandoffStarted

Emitted when an agent begins delegating to another agent.

| Field | Type | Description |
| ----- | ---- | ----------- |
| `target_agent` | str | Name of the target agent |
| `task` | str | Task being delegated |

### HandoffCompleted

Emitted when a handoff finishes and the result is received.

| Field | Type | Description |
| ----- | ---- | ----------- |
| `target_agent` | str | Name of the target agent |
| `success` | bool | Whether the handoff succeeded |

### ContextCompacted

Emitted when the context window is compacted (truncation or summarization).

| Field | Type | Description |
| ----- | ---- | ----------- |
| `method` | str | Compaction method (e.g. `"middle_out"`, `"truncate"`) |
| `tokens_before` | int | Token count before compaction |
| `tokens_after` | int | Token count after compaction |
| `messages_before` | int | Messages before compaction |
| `messages_after` | int | Messages after compaction |

```python
bus.subscribe(ContextCompacted, lambda e: metrics.histogram("context.tokens_saved", e.tokens_before - e.tokens_after))
```

## Scenarios

### Metrics and Dashboards

Track budget utilization and context compaction in your monitoring system:

```python
import prometheus_client

budget_usage = prometheus_client.Gauge("syrin_budget_percent", "Budget utilization %")
tokens_saved = prometheus_client.Histogram("syrin_context_tokens_saved", "Tokens saved by compaction")

bus = EventBus()
bus.subscribe(BudgetThresholdReached, lambda e: budget_usage.set(e.percentage))
bus.subscribe(ContextCompacted, lambda e: tokens_saved.observe(e.tokens_before - e.tokens_after))

agent = Agent(model=..., budget=..., bus=bus)
```

### Observability Pipelines

Forward events to your observability backend:

```python
bus = EventBus()

def forward_to_otel(event):
    # OpenTelemetry, Datadog, etc.
    tracer.span("syrin.event", attributes={
        "event.type": type(event).__name__,
        **event.__dict__,
    })

bus.subscribe(BudgetThresholdReached, forward_to_otel)
bus.subscribe(ContextCompacted, forward_to_otel)

agent = Agent(model=..., bus=bus)
```

### Alerts and Notifications

Notify when budgets or context hit thresholds:

```python
bus = EventBus()

def alert_on_high_budget(e: BudgetThresholdReached):
    if e.percentage >= 90:
        slack.send(f"⚠️ Budget at {e.percentage}% (${e.current_value:.2f})")

bus.subscribe(BudgetThresholdReached, alert_on_high_budget)
agent = Agent(model=..., budget=..., bus=bus)
```

### Subscribing to Base Types

You can subscribe to `DomainEvent` to receive all domain events:

```python
bus.subscribe(DomainEvent, lambda e: print(type(e).__name__, e))
```

## Relationship to Hooks

- **Hooks** (`agent.events`) are the default API. They use string event names and `EventContext` dicts.
- **Event bus** is optional. It uses typed dataclasses and `bus.subscribe(event_type, handler)` or `bus.on(event_type, handler)` (alias).

Both can be used together. When you pass `bus=bus`, the agent emits to the bus **in addition to** running hook handlers. Hooks remain for simple use cases and per-agent handlers.

## See Also

- [Events & Hooks](agent/events-hooks.md) — Default event API
- [Budget Control](budget-control.md) — Budget and thresholds
- [Context](context.md) — Context window and compaction
