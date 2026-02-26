# Running Agents

Four entry points control how you run an agent: sync, async, and streaming. **Parity:** `response` and `arun` behave the same (sync vs async); use `stream` / `astream` for token-by-token output.

## `response(user_input)` — Sync

Runs the agent synchronously and returns a `Response`.

```python
response = agent.response("What is the capital of France?")
print(response.content)
print(response.cost)
```

**Returns:** `Response[str]`  
**Use when:** Scripts, CLI, or blocking flows.

---

## `arun(user_input)` — Async

Same as `response` but async. Use with `await`.

```python
response = await agent.arun("What is the capital of France?")
print(response.content)
```

**Returns:** `Response[str]`  
**Use when:** Async apps (FastAPI, etc.).

---

## `stream(user_input)` — Sync Streaming

Streams chunks as they are generated. Returns an iterator.

```python
for chunk in agent.stream("Write a short poem"):
    print(chunk.text, end="", flush=True)
```

**Returns:** `Iterator[StreamChunk]`  
**Use when:** You want token-by-token output in sync code.

---

## `astream(user_input)` — Async Streaming

Same as `stream` but async. Returns an async iterator.

```python
async for chunk in agent.astream("Write a short poem"):
    print(chunk.text, end="", flush=True)
```

**Returns:** `AsyncIterator[StreamChunk]`  
**Use when:** Async streaming (e.g. WebSockets).

---

## StreamChunk

Each streaming chunk has:

| Field | Type | Description |
|-------|------|-------------|
| `text` | `str` | New text in this chunk |
| `accumulated_text` | `str` | Full text so far |
| `cost_so_far` | `float` | Cumulative cost |
| `tokens_so_far` | `TokenUsage` | Cumulative token counts |

```python
for chunk in agent.stream("Hello"):
    print(f"New: {chunk.text!r}")
    print(f"Total: {chunk.accumulated_text}")
    print(f"Cost: ${chunk.cost_so_far:.4f}")
```

---

## Execution Flow

All four methods follow the same flow:

1. Reset `AgentReport` and (if set) run budget
2. Input guardrails
3. Build messages (system prompt + memory + history + user input)
4. Run the loop (e.g. ReactLoop):
   - LLM call
   - Execute tools if present
   - Repeat until no more tool calls or max iterations
5. Output guardrails on final text
6. Optional structured output validation
7. Return `Response` (or yield `StreamChunk`s)

---

## Error Handling

- **BudgetExceededError** — Budget limit reached (when `on_exceeded=raise_on_exceeded` or your callback raises)
- **BudgetThresholdError** — Threshold action (e.g. STOP) triggered
- **ToolExecutionError** — Tool raised or failed
- Other provider errors propagate as usual

```python
from syrin.exceptions import BudgetExceededError, ToolExecutionError

try:
    response = agent.response("Complex task")
except BudgetExceededError as e:
    print(f"Budget exceeded: ${e.current_cost:.2f}")
except ToolExecutionError as e:
    print(f"Tool failed: {e}")
```

---

## Budget Reset

`response()`, `arun()`, `stream()`, and `astream()` all reset the run budget before execution when `budget` is set.

---

## Next Steps

- [Response Object](response.md) — Fields and usage
- [Loop Strategies](loop-strategies.md) — Behavior of the main loop
