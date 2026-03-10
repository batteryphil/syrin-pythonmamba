# Running Agents

Four entry points control how you run an agent: sync, async, and streaming. **Parity:** `response` and `arun` behave the same (sync vs async); `stream` and `astream` behave the same (sync vs async).

| Method | Output | Runs tools? | Use when |
|--------|--------|-------------|----------|
| `response()` | `Response[str]` | Yes | Scripts, CLI, blocking flows |
| `arun()` | `Response[str]` | Yes | Async apps (FastAPI, etc.) |
| `stream()` | `Iterator[StreamChunk]` | No | Token-by-token output, sync, no tools |
| `astream()` | `AsyncIterator[StreamChunk]` | No | Token-by-token output, async, no tools |

**Important:** If your agent has tools, use `response()` or `arun()`. `stream()` and `astream()` perform a single LLM completion only—they do not run the tool loop. If the model returns tool calls, you'll get a fallback message instead of the real reply. For integrations (voice, WebSockets, etc.) that need streaming *and* tools, call `arun()` and emit the returned text.

---

## `response(user_input)` — Sync

Runs the agent synchronously (full REACT loop, including tools) and returns a `Response`.

```python
response = agent.response("What is the capital of France?")
print(response.content)
print(response.cost)
```

**Output example:**
```
Paris.
0.0001
```

**Returns:** `Response[str]` with fields: `content`, `cost`, `tokens`, `model`, `stop_reason`, `duration`, `trace`, `report`, etc. See [Response Object](response.md).

**Use when:** Scripts, CLI, or blocking flows. **Use for agents with tools.**

---

## `arun(user_input)` — Async

Same as `response` but async. Use with `await`.

```python
response = await agent.arun("What is the capital of France?")
print(response.content)
```

**Output example:** Same as `response()` — `"Paris."` (content), `0.0001` (cost), etc.

**Returns:** `Response[str]` — same shape as `response()`.

**Use when:** Async apps (FastAPI, voice pipelines, etc.). **Use for agents with tools.**

---

## `stream(user_input)` — Sync Streaming

Streams chunks as they are generated. **Single completion only—no tool execution.** Use only for agents without tools.

```python
for chunk in agent.stream("Write a short poem"):
    print(chunk.text, end="", flush=True)
```

**Output example (chunks yielded):**
```
Here's     # chunk.text
a haiku    # chunk.text
about code # chunk.text
...
```

**Returns:** `Iterator[StreamChunk]` — each chunk has `text`, `accumulated_text`, `cost_so_far`, `tokens_so_far`.

**Use when:** Token-by-token output in sync code, agent has no tools.

---

## `astream(user_input)` — Async Streaming

Same as `stream` but async. **Single completion only—no tool execution.** Use only for agents without tools.

```python
async for chunk in agent.astream("Write a short poem"):
    print(chunk.text, end="", flush=True)
```

**Output example:** Same as `stream()` — yields `StreamChunk` objects with `text`, `accumulated_text`, etc.

**Returns:** `AsyncIterator[StreamChunk]` — same shape as `stream()`.

**Use when:** Async streaming (e.g. WebSockets), agent has no tools.

---

## Output Reference

### Response (from `response()` / `arun()`)

| Field | Type | Example |
|-------|------|---------|
| `content` | `str` | `"Paris."` |
| `cost` | `float` | `0.0001` |
| `tokens` | `TokenUsage` | `TokenUsage(input_tokens=42, output_tokens=5, total_tokens=47)` |
| `model` | `str` | `"openai/gpt-4o-mini"` |
| `stop_reason` | `StopReason` | `StopReason.END_TURN` |
| `duration` | `float` | `0.32` |

### StreamChunk (from `stream()` / `astream()`)

| Field | Type | Description | Example |
|-------|------|-------------|---------|
| `text` | `str` | New text in this chunk (delta) | `"Paris."` |
| `accumulated_text` | `str` | Full text so far | `"The capital of France is Paris."` |
| `cost_so_far` | `float` | Cumulative cost (USD) | `0.0001` |
| `tokens_so_far` | `TokenUsage` | Cumulative token counts | `TokenUsage(input_tokens=42, output_tokens=12, total_tokens=54)` |
| `index` | `int` | Zero-based chunk index | `0`, `1`, `2`, ... |

```python
for chunk in agent.stream("Hello"):
    print(f"New: {chunk.text!r}")
    print(f"Total: {chunk.accumulated_text}")
    print(f"Cost: ${chunk.cost_so_far:.4f}")
# Output: New: 'Hello'  Total: Hello  Cost: $0.0000
#         New: '!'      Total: Hello! Cost: $0.0001
```

---

## Execution Flow

**`response()` and `arun()`:**

1. Reset `AgentReport` and (if set) run budget
2. Input guardrails
3. Build messages (system prompt + memory + history + user input)
4. Run the loop (e.g. ReactLoop): LLM call → execute tools if present → repeat until done
5. Output guardrails on final text
6. Return `Response`

**`stream()` and `astream()`:**

1. Reset `AgentReport` and (if set) run budget
2. Input guardrails
3. Build messages
4. Single LLM completion (no tool loop)
5. Yield `StreamChunk`s

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
- [Serving](../serving.md) — Serve via HTTP, CLI, or playground
