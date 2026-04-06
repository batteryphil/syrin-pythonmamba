---
title: Response Object
description: Every field on the Response — cost, tokens, stop reason, trace, and more
weight: 65
---

## More Than Just Text

Most agent libraries give you a string. Syrin gives you a `Response` object with a full picture of what just happened: what it said, what it cost, how many tokens it burned, why it stopped, and whether anything went wrong along the way.

You asked your agent a question. Before you use the answer, you might want to know: did it finish? How much did it cost? Did a tool fail? The Response object has all of that.

## The Basics

```python
from syrin import Agent, Model, Budget
from syrin.enums import ExceedPolicy

class Assistant(Agent):
    model = Model.mock()
    system_prompt = "You are a helpful assistant."
    budget = Budget(max_cost=1.00, exceed_policy=ExceedPolicy.WARN)

agent = Assistant()
response = agent.run("What is 2 + 2?")

print(response.content)         # The text response
print(f"${response.cost:.6f}")  # Cost in USD
print(response.tokens)          # TokenUsage object
print(response.model)           # Model ID used
print(response.stop_reason)     # Why it stopped
print(f"{response.duration:.2f}s")  # How long it took
print(response.iterations)      # How many LLM calls
print(response.budget_remaining)  # Remaining budget
```

Output:

```
Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod...
$0.000041
input_tokens=7 output_tokens=25 total_tokens=32 cached_tokens=0 reasoning_tokens=0
mock/default
end_turn
2.48s
1
0.999959
```

## Field Reference

### Content

**`response.content`** is the main text the agent produced. It is always a string.

**`response.model`** is the model ID that generated the response, e.g. `"openai/gpt-4o-mini"` or `"mock/default"`.

**`response.model_used`** is the actual model that ran (may differ from `model` when using routing or fallback).

```python
print(response.content[:60])
print(str(response))  # Same as response.content — the Response is stringable
print(bool(response))  # True if stop_reason == END_TURN, False otherwise
```

### Cost

**`response.cost`** is the total USD cost of this run (all LLM calls, all tool iterations combined).

**`response.budget_remaining`** is how much budget is left, or `None` if no budget was set.

**`response.budget_used`** is how much budget was consumed in this run.

**`response.budget`** is a `BudgetStatus` object with `.remaining`, `.used`, and `.total`.

```python
print(f"Cost: ${response.cost:.6f}")
print(f"Remaining: ${response.budget_remaining:.4f}")
print(f"Budget: {response.budget}")
# Budget: BudgetStatus(remaining=$1.0000, used=$0.000041, total=$1.0000)
```

### Tokens

**`response.tokens`** is a `TokenUsage` object with:

- `.input_tokens` — tokens in the prompt
- `.output_tokens` — tokens in the response
- `.total_tokens` — the sum
- `.cached_tokens` — tokens served from the provider's cache (if any)
- `.reasoning_tokens` — tokens used for chain-of-thought (o1/o3 models)

```python
t = response.tokens
print(f"In: {t.input_tokens}, Out: {t.output_tokens}, Total: {t.total_tokens}")
```

**`response.total_tokens`** `int` — Total tokens used (input + output). Convenience for `response.tokens.total_tokens`.

Tokens are money. If `input_tokens` is growing across calls, your context is accumulating. Watch it.

### Stop Reason

**`response.stop_reason`** is a `StopReason` enum value telling you why the agent stopped:

- `StopReason.END_TURN` — Completed successfully. The model decided it was done.
- `StopReason.BUDGET` — Hit the cost limit before finishing.
- `StopReason.MAX_ITERATIONS` — The tool loop hit `max_tool_iterations`.
- `StopReason.TIMEOUT` — The run timed out.
- `StopReason.TOOL_ERROR` — A tool raised an exception and the agent couldn't recover.
- `StopReason.HANDOFF` — The agent handed off to another agent.
- `StopReason.GUARDRAIL` — Input or output was blocked by a guardrail.
- `StopReason.CANCELLED` — The run was cancelled.

```python
from syrin.enums import StopReason

if response.stop_reason == StopReason.END_TURN:
    print("Completed normally")
elif response.stop_reason == StopReason.BUDGET:
    print(f"Stopped at ${response.cost:.4f} — increase the budget")
elif response.stop_reason == StopReason.MAX_ITERATIONS:
    print(f"Tool loop ran {response.iterations} iterations — something looping?")
```

### Performance

**`response.duration`** is wall-clock seconds from the start of `run()` to return.

**`response.iterations`** is how many LLM calls were made. A simple question is 1. A task that requires two tool calls is typically 3 (ask → call tool → resume → finish).

A high iteration count isn't necessarily bad, but if it's much higher than expected, a tool might be failing or the agent might be confused.

### Tool Calls

**`response.tool_calls`** is the list of tool calls the model made (as request objects, not results).

```python
for call in response.tool_calls:
    print(f"Tool: {call.name}, Args: {call.arguments}")
```

### Structured Output

When you configure `Output(MyModel)`, the validated Python object is in:

**`response.output`** — the main output of this response. For structured agents (`Output(MyModel)` configured): the validated Python object (e.g. a `UserInfo` instance). For plain text agents: the text string (same as `response.content`).

**`response.structured`** — validation state object with `.is_valid`, `.all_errors`, etc.

```python
if response.output:
    print(response.output.name)  # Direct field access on your model
```

### Attachments

For agents that generate or return files:

**`response.attachments`** — list of `MediaAttachment` objects with `.type`, `.content_type`, `.url`, `.content_bytes`.

**`response.citations`** — parsed citations if citation formatting is enabled.

**`response.file`** — file path if the output was written to disk.

---

## The Report Object

`response.report` gives you subsystem-level metrics from the entire run:

```python
report = response.report

# Guardrail results
print(f"Input passed: {report.guardrail.input_passed}")
print(f"Output passed: {report.guardrail.output_passed}")
print(f"Blocked: {report.guardrail.blocked}")

# Context usage
print(f"Initial tokens: {report.context.initial_tokens}")
print(f"Final tokens: {report.context.final_tokens}")
print(f"Compressions: {report.context.compressions}")

# Memory operations
print(f"Recalls: {report.memory.recalls}")
print(f"Stores: {report.memory.stores}")
print(f"Forgets: {report.memory.forgets}")

# Tokens breakdown
print(f"Cost: ${report.tokens.cost_usd:.6f}")

# Rate limiting
print(f"Throttles: {report.ratelimits.throttles}")
```

The sub-objects on `report`:

- `report.guardrail` — GuardrailReport with input/output pass state and blocked flag
- `report.context` — ContextReport with token counts and compression events
- `report.memory` — MemoryReport with recall/store/forget counts
- `report.tokens` — TokenReport with a detailed cost breakdown
- `report.output` — OutputReport with structured output validation state
- `report.ratelimits` — RateLimitReport with throttle counts
- `report.checkpoints` — CheckpointReport with save/load counts
- `report.grounding` — GroundingReport (when fact grounding is enabled)

---

## The Execution Trace

`response.trace` is the step-by-step flight recorder of the run. It starts empty for simple runs and populates when there are tools or multi-step interactions:

```python
for step in response.trace:
    print(f"Step: {step.step_type}")
    print(f"  Cost: ${step.cost_usd:.6f}")
    print(f"  Latency: {step.latency_ms}ms")
    if step.extra:
        print(f"  Extra: {step.extra}")
```

Use `response.trace` for post-hoc debugging and auditing. Use `agent.events.on(Hook.XXX, handler)` for real-time observation during the run.

---

## Quick Patterns

**Check if it completed successfully:**

```python
response = agent.run("Task")
if bool(response):  # True when stop_reason == END_TURN
    process(response.content)
else:
    print(f"Stopped early: {response.stop_reason}")
```

**Log cost and tokens:**

```python
response = agent.run("Task")
print(f"${response.cost:.6f} | {response.tokens.total_tokens} tokens | {response.duration:.1f}s")
```

**Alert on high iteration count:**

```python
response = agent.run("Task with tools")
if response.iterations > 5:
    print(f"Warning: {response.iterations} iterations — check tool behavior")
```

**Check guardrail state:**

```python
response = agent.run("User input")
if response.report.guardrail.blocked:
    print(f"Blocked at stage: {response.report.guardrail.blocked_stage}")
```

---

## What's Next

- [Streaming](/agent-kit/agent/streaming) — Get tokens as they arrive
- [Structured Output](/agent-kit/agent/structured-output) — Get typed Python objects back
- [Hooks Reference](/agent-kit/debugging/hooks-reference) — Subscribe to events during execution
- [Budget](/agent-kit/core/budget) — Control what each run can spend
