# Agent Complete API Reference

Exhaustive reference for the `Agent` class public API.

---

## Constructor

```python
Agent(
    model=None,
    system_prompt=None,
    tools=None,
    budget=None,
    *,
    output=None,
    max_tool_iterations=10,
    budget_store=None,
    budget_store_key="default",
    memory=None,
    loop_strategy=LoopStrategy.REACT,
    loop=None,
    guardrails=None,
    context=None,
    rate_limit=None,
    checkpoint=None,
    debug=False,
    tracer=None,
)
```

See [Constructor Reference](constructor.md) for full parameter details.

---

## Execution Methods

| Method | Signature | Returns |
|--------|-----------|---------|
| `response` | `(user_input: str)` | `Response[str]` |
| `arun` | `(user_input: str)` | `Response[str]` (async) |
| `stream` | `(user_input: str)` | `Iterator[StreamChunk]` |
| `astream` | `(user_input: str)` | `AsyncIterator[StreamChunk]` |

---

## Memory Methods

| Method | Signature | Returns |
|--------|-----------|---------|
| `remember` | `(content, memory_type=EPISODIC, importance=1.0, **metadata)` | `str` (memory ID) |
| `recall` | `(query=None, memory_type=None, limit=10)` | `list[MemoryEntry]` |
| `forget` | `(memory_id=None, query=None, memory_type=None)` | `int` |

---

## Checkpoint Methods

| Method | Signature | Returns |
|--------|-----------|---------|
| `save_checkpoint` | `(name=None, reason=None)` | `str | None` |
| `load_checkpoint` | `(checkpoint_id: str)` | `bool` |
| `list_checkpoints` | `(name=None)` | `list[str]` |
| `get_checkpoint_report` | `()` | `AgentReport` |

---

## Multi-Agent Methods

| Method | Signature | Returns |
|--------|-----------|---------|
| `handoff` | `(target_agent, task, transfer_context=True, transfer_budget=False)` | `Response[str]` |
| `spawn` | `(agent_class, task=None, budget=None, max_children=None)` | `Agent | Response[str]` |
| `spawn_parallel` | `(agents: list[tuple[type[Agent], str]])` | `list[Response[str]]` |

---

## State Methods

| Method | Signature | Returns |
|--------|-----------|---------|
| `switch_model` | `(model: Model | ModelConfig)` | `None` |
| `complete` | `(messages, tools=None)` | `ProviderResponse` (async) |
| `execute_tool` | `(name: str, arguments: dict)` | `str` (async) |

---

## Properties

| Property | Type |
|----------|------|
| `budget_state` | `BudgetState \| None` |
| `tools` | `list[ToolSpec]` |
| `model_config` | `ModelConfig \| None` |
| `memory` | `ConversationMemory | Memory | None` — Active memory (default: BufferMemory). |
| `conversation_memory` | `ConversationMemory | None` — Read-only; set via `memory=`. |
| `persistent_memory` | `Memory | None` |
| `context` | `Context` |
| `context_stats` | `ContextStats` |
| `rate_limit` | `APIRateLimit | None` |
| `rate_limit_stats` | `RateLimitStats` |
| `report` | `AgentReport` |
| `events` | `Events` |

---

## Response Object

| Field | Type |
|-------|------|
| `content` | `T` |
| `raw` | `str` |
| `cost` | `float` |
| `tokens` | `TokenUsage` |
| `model` | `str` |
| `duration` | `float` |
| `budget_remaining` | `float | None` |
| `budget_used` | `float | None` |
| `trace` | `list[TraceStep]` |
| `tool_calls` | `list` |
| `stop_reason` | `StopReason` |
| `structured` | `StructuredOutput | None` |
| `iterations` | `int` |
| `report` | `AgentReport` |

---

## Enums

### LoopStrategy
- `REACT`, `SINGLE_SHOT`, `PLAN_EXECUTE`, `CODE_ACTION`

### StopReason
- `END_TURN`, `BUDGET`, `MAX_ITERATIONS`, `TIMEOUT`, `TOOL_ERROR`, `HANDOFF`, `GUARDRAIL`, `CANCELLED`

### on_exceeded (callback)
- Pass `raise_on_exceeded`, `warn_on_exceeded`, or `stop_on_exceeded`; or any callable receiving `BudgetExceededContext`.

### MemoryType
- `CORE`, `EPISODIC`, `SEMANTIC`, `PROCEDURAL`

### Hook
- See [Events & Hooks](events-hooks.md) for full list.

---

## Exceptions

| Exception | When |
|-----------|------|
| `TypeError` | Model not provided |
| `ValidationError` | Invalid input (e.g. `handoff` with empty `task`, wrong `target_agent` type) |
| `BudgetExceededError` | Budget exceeded (e.g. when using `raise_on_exceeded`) |
| `BudgetThresholdError` | Threshold action (e.g. STOP) |
| `ToolExecutionError` | Tool failed or unknown tool |
| `HandoffBlockedError` | Handoff blocked by before-handler; see [Handoff & Spawn](handoff-spawn.md) |
| `HandoffRetryRequested` | Target signals invalid data; caller should retry with `format_hint` |
