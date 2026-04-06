---
title: Loop Strategies
description: Choose how your agent reasons — single-shot, ReAct, plan-execute, code-action, or human-in-the-loop
weight: 65
---

## What a Loop Controls

Every agent run is driven by a loop. The loop decides how many times the LLM is called, whether tools are executed between calls, and when execution stops. Choosing the right loop is the single biggest lever you have on cost, latency, and task quality.

Set it with `loop=`:

```python
from syrin import Agent, Model
from syrin.loop import ReactLoop

class MyAgent(Agent):
    model = Model.mock()
    loop = ReactLoop(max_iterations=5)
```

Or at construction time:

```python
agent = Agent(model=Model.mock(), loop=ReactLoop(max_iterations=5))
```

## The Five Loops

### SingleShotLoop — one call, no tools

One LLM call. No tool execution. Returns immediately.

```python
from syrin import Agent, Model
from syrin.loop import SingleShotLoop

class SummaryAgent(Agent):
    model = Model.mock()
    system_prompt = "Summarize the given text in three sentences."
    loop = SingleShotLoop()

result = SummaryAgent().run("Long article text here...")
print(result.iterations)  # Always 1
```

**When to use:** Questions, translations, summaries, classifications — anything that can be answered in a single LLM call. No tools needed. `response.iterations` is always 1.

**Tradeoffs:** Fastest. Cheapest. Cannot use tools. Cannot self-correct.

---

### ReactLoop — think, act, observe (default)

The default loop. Calls the LLM → if it calls tools, executes them and feeds results back → repeats until `end_turn` or `max_iterations`.

```python
from syrin import Agent, Model
from syrin.loop import ReactLoop
from syrin.tool import tool

@tool
def search_web(query: str) -> str:
    return f"Results for '{query}'..."

class ResearchAgent(Agent):
    model = Model.mock()
    system_prompt = "Research the topic using available tools."
    tools = [search_web]
    loop = ReactLoop(max_iterations=10)  # default
```

**Constructor:**

```python
ReactLoop(max_iterations: int = 10)
```

`max_iterations` caps the number of LLM calls per `agent.run()`. If the agent calls tools repeatedly and hits the cap, it stops with `stop_reason = "max_iterations"`.

**When to use:** Any agent with tools. Research, multi-step workflows, agentic tasks. This is the right loop for 90% of use cases.

**Tradeoffs:** More expensive than SingleShotLoop (multiple LLM calls). Latency grows with tool depth. Adjusting `max_iterations` is your primary cost control.

---

### PlanExecuteLoop — plan first, then act

First phase: the agent generates a numbered plan for the full task. Second phase: the agent executes each step. Final phase: reviews the results.

```python
from syrin import Agent, Model
from syrin.loop import PlanExecuteLoop
from syrin.tool import tool

@tool
def read_file(path: str) -> str:
    with open(path) as f:
        return f.read()

@tool
def write_file(path: str, content: str) -> str:
    with open(path, "w") as f:
        f.write(content)
    return "written"

class ProjectAgent(Agent):
    model = Model.mock()
    system_prompt = "You decompose and execute complex multi-step tasks."
    tools = [read_file, write_file]
    loop = PlanExecuteLoop(
        max_plan_iterations=3,    # Max attempts to generate the plan
        max_execution_iterations=20,  # Max tool calls during execution
    )

result = ProjectAgent().run("Refactor the utils module to add type annotations")
```

**Constructor:**

```python
PlanExecuteLoop(
    max_plan_iterations: int = 5,
    max_execution_iterations: int = 20,
)
```

**When to use:** Complex multi-step tasks where upfront planning improves quality — code refactoring, document generation, research reports. Produces more coherent output than ReactLoop for tasks that require ordered steps.

**Tradeoffs:** One extra LLM call for planning. Better quality for structured tasks. Not useful for simple queries.

---

### CodeActionLoop — write code, run it, interpret results

The LLM generates Python code to solve the problem. The code runs in the current process via `exec()`. Output is fed back to the LLM for interpretation.

```python
from syrin import Agent, Model
from syrin.loop import CodeActionLoop

class DataAgent(Agent):
    model = Model.mock()
    system_prompt = """You write and execute Python to answer data questions.
    Use print() to show results. pandas and numpy are available."""
    loop = CodeActionLoop(max_iterations=5, timeout_seconds=30)

result = DataAgent().run("What is the standard deviation of [1, 4, 9, 16, 25]?")
print(result.content)
```

**Constructor:**

```python
CodeActionLoop(
    max_iterations: int = 10,
    timeout_seconds: int = 60,
)
```

`timeout_seconds` limits each individual `exec()` call, not the total loop duration.

> **Warning: In-Process Execution**
>
> Generated code runs directly in the calling Python process with the same permissions and file-system access as your application. There is no sandbox, no container, and no resource limit beyond `timeout_seconds`.
>
> Only use `CodeActionLoop` for trusted inputs — internal tooling, developer-written tasks, offline notebooks. Do not expose it to arbitrary user input in production.

**When to use:** Mathematical computations, data analysis, internal automation where the input is fully trusted.

**Tradeoffs:** Powerful for numeric/algorithmic tasks. Requires trusting the LLM to generate safe code. Full sandbox coming in a future release.

---

### HumanInTheLoop — pause for approval before every tool call

Every time the LLM attempts a tool call, execution pauses and waits for human approval. If approved, the tool runs. If rejected (or timeout), the tool is skipped.

```python
from syrin import Agent, Model, ApprovalGate
from syrin.hitl import HumanInTheLoop
from syrin.tool import tool

@tool
def delete_file(path: str) -> str:
    import os
    os.remove(path)
    return f"Deleted {path}"

# Simple CLI approval gate
async def cli_approve(message: str, timeout: int, ctx: dict) -> bool:
    tool_name = ctx.get("tool_name", "?")
    args = ctx.get("arguments", {})
    answer = input(f"\nApprove {tool_name}({args})? [y/n]: ")
    return answer.strip().lower() == "y"

gate = ApprovalGate(callback=cli_approve)

class SafeFileAgent(Agent):
    model = Model.mock()
    system_prompt = "You manage files safely with human oversight."
    tools = [delete_file]
    loop = HumanInTheLoop(approval_gate=gate, timeout=120)
```

**Constructor:**

```python
HumanInTheLoop(
    approval_gate: ApprovalGate,  # Preferred — async callback-based gate
    approve: Callable | None = None,  # Legacy: async (tool_name, args) -> bool
    timeout: int = 300,           # Seconds to wait; rejects on timeout
    max_iterations: int = 10,
)
```

`ApprovalGate` is the standard interface. Its `callback` signature is `async (message: str, timeout: int, ctx: dict) -> bool`. The `ctx` dict contains `tool_name` and `arguments`.

```python
from syrin import ApprovalGate

# Async approval via webhook / Slack / UI
async def webhook_approve(message: str, timeout: int, ctx: dict) -> bool:
    # Send to Slack, wait for reaction, return True/False
    ...

gate = ApprovalGate(callback=webhook_approve)
```

**When to use:** Safety-critical operations (file deletion, database writes, external API calls with side effects) where a human must verify before execution. Required for regulated environments.

**Tradeoffs:** High latency (human in critical path). Ensures no unintended side effects.

---

## Quick Comparison

| Loop | LLM Calls | Tools | Best For | Cost |
|------|-----------|-------|----------|------|
| `SingleShotLoop` | 1 | No | Simple queries, summaries | Lowest |
| `ReactLoop` | 1–N | Yes | Most tool-using agents | Medium |
| `PlanExecuteLoop` | 2–N | Yes | Complex multi-step tasks | Medium+ |
| `CodeActionLoop` | 1–N | exec() | Data analysis, computation | Medium |
| `HumanInTheLoop` | 1–N | Yes + approval | Safety-critical operations | Medium |

## Writing a Custom Loop

Every loop implements the `Loop` protocol. Subclass `Loop` and implement `run()`:

```python
from syrin.loop import Loop, LoopResult

class MyLoop(Loop):
    name = "my_loop"

    def __init__(self, max_iterations: int = 5):
        self.max_iterations = max_iterations

    async def run(self, ctx, user_input):
        # ctx: AgentRunContext — has ctx.complete(), ctx.tools, ctx.build_messages()
        # user_input: str or list of message dicts

        messages = ctx.build_messages(user_input)
        response = await ctx.complete(messages, ctx.tools)

        return LoopResult(
            content=response.content or "",
            stop_reason=response.stop_reason or "end_turn",
            iterations=1,
            latency_ms=0,
            cost_usd=0,
            token_usage={},
            tool_calls=[],
            tools_used=[],
        )

class CustomAgent(Agent):
    model = Model.mock()
    loop = MyLoop(max_iterations=3)
```

`LoopResult` fields: `content`, `stop_reason`, `iterations`, `latency_ms`, `cost_usd`, `token_usage` (dict with `input`/`output`/`total` keys), `tool_calls`, `tools_used`.

## What's Next

- [Running Agents](/agent-kit/agent/running-agents) — sync, async, streaming
- [Tools](/agent-kit/agent/tools) — Give agents tools to call
- [Budget Control](/agent-kit/core/budget) — Cap how many iterations are affordable
- [Hooks Reference](/agent-kit/debugging/hooks-reference) — AGENT_RUN_START, LLM_REQUEST_START, TOOL_CALL_START, etc.
