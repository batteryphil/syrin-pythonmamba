---
title: AgentRouter
description: LLM-driven dynamic multi-agent orchestration — let the LLM decide which agents to spawn
weight: 73
---

## Let the LLM Plan the Work

With a `Workflow`, you decide upfront which agents run and in what order. That's great when the task structure is known. But what if it isn't?

`AgentRouter` flips the script. You give it a pool of available agent classes and a task. The orchestrator LLM figures out which agents to use, what sub-task to give each one, and runs them. The LLM is the planner, not you.

This is the right tool when you don't know at design time what agents a task will need — the task could be "write a market report" (needs research + writing) or "analyze sentiment" (just one agent) or "build a pipeline" (needs five different specialists). The LLM decides.

> **v0.11.0:** `AgentRouter` replaces the old `DynamicPipeline`. `DynamicPipeline` has been removed. If you used it, switch to `AgentRouter` — the API is identical.

## Basic Usage

```python
from syrin import Agent, Budget, Model
from syrin.agent.agent_router import AgentRouter
from syrin.enums import ExceedPolicy

class ResearchAgent(Agent):
    model = Model.mock()
    system_prompt = "Search for relevant information on the given topic."

class AnalystAgent(Agent):
    model = Model.mock()
    system_prompt = "Analyse data and identify patterns."

class WriterAgent(Agent):
    model = Model.mock()
    system_prompt = "Write a clear, concise report."

# NOTE: AgentRouter takes CLASSES, not instances
router = AgentRouter(
    agents=[ResearchAgent, AnalystAgent, WriterAgent],
    model=Model.OpenAI("gpt-4o-mini", api_key="your-key"),  # The orchestrator LLM
    budget=Budget(max_cost=2.00, exceed_policy=ExceedPolicy.WARN),
)

result = router.run("Research the AI chip market and write an executive summary")
print(result.content)
print(f"Cost: ${result.cost:.4f}")
```

The orchestrator LLM reads the agent descriptions, picks the ones needed for this task, assigns sub-tasks, and runs them in parallel. The result is the combined output.

**Important:** `agents` takes **classes**, not instances (`ResearchAgent`, not `ResearchAgent()`). The router instantiates them internally.

## How It Works

When you call `router.run(task)`:

1. **Planning call** — an internal planning agent (using your `model`) reads the available agents and their descriptions, then returns a JSON plan: which agents to spawn and what sub-task to give each.
2. **Execution** — the selected agents are instantiated and run in parallel (or sequentially, if you specify `mode="sequential"`).
3. **Consolidation** — the outputs are joined into a single `Response`.

The agent name used in planning comes from `name` if set, otherwise the lowercased class name. Give your agents clear names:

```python
class MarketResearchAgent(Agent):
    name = "market_research"
    description = "Searches market databases and news for current data"
    model = Model.mock()
    system_prompt = "Research market data."
```

The `description` is sent to the planner LLM to help it decide when to use this agent.

## Execution Modes

**Parallel (default):** all LLM-selected agents run simultaneously via `asyncio.gather`. Results are joined with newlines.

```python
result = router.run("Analyze earnings and write a report")          # parallel
result = router.run("Analyze earnings and write a report", mode="parallel")  # explicit
```

**Sequential:** agents run one after another. Each agent receives its own task plus the previous agent's output as context.

```python
result = router.run("Research, then analyze, then write", mode="sequential")
```

Use sequential when later agents need to build on earlier agents' outputs.

## Constructor Parameters

- `agents` — list of agent classes (not instances) available to the planner
- `model` — the orchestrator Model that plans which agents to spawn. Required. Must be a real model capable of JSON output — `Model.mock()` won't plan correctly.
- `budget` — optional `Budget` applied across all spawned agents
- `max_parallel` — cap on concurrent agents (default: 10)
- `debug` — print hook events to the console (default: `False`)
- `output_format` — `"clean"` joins outputs; `"verbose"` adds per-agent headers and costs

## Lifecycle Hooks

```python
from syrin.enums import Hook

router.events.on(Hook.DYNAMIC_PIPELINE_START, lambda ctx: print(f"Planning for: {ctx.get('task', '')[:40]}"))
router.events.on(Hook.DYNAMIC_PIPELINE_PLAN, lambda ctx: print(f"Plan: {ctx.get('plan')}"))
router.events.on(Hook.DYNAMIC_PIPELINE_AGENT_SPAWN, lambda ctx: print(f"Spawning: {ctx.get('agent_type')}"))
router.events.on(Hook.DYNAMIC_PIPELINE_AGENT_COMPLETE, lambda ctx: print(f"Done: {ctx.get('agent_type')} cost=${ctx.get('cost', 0):.4f}"))
router.events.on(Hook.DYNAMIC_PIPELINE_END, lambda ctx: print(f"Total cost: ${ctx.get('total_cost', 0):.4f}"))
```

## When to Use AgentRouter vs. Workflow vs. Swarm

Use **Workflow** when you know the steps upfront. Sequential, parallel, and conditional steps are all supported with `.step()`, `.parallel()`, and `.branch()`. Predictable, testable, debuggable.

Use **Swarm** when you want a standard topology (parallel execution, orchestrator, consensus, reflection). High-level and opinionated.

Use **AgentRouter** when the task structure is unknown at design time and you want the LLM to figure out what to do. Most flexible, least predictable.

## Migrating from DynamicPipeline

If you're on v0.10.x:

```python
# Before (v0.10.x)
from syrin import DynamicPipeline
pipeline = DynamicPipeline(agents=[ResearchAgent, WriterAgent], model=model)
result = pipeline.run("task")

# After (v0.11.0+)
from syrin.agent.agent_router import AgentRouter
router = AgentRouter(agents=[ResearchAgent, WriterAgent], model=model)
result = router.run("task")
```

The constructor parameters and `run()` signature are identical. Only the class name changed.

## What's Next

- [Workflow](/agent-kit/multi-agent/workflow) — Declarative step-by-step execution
- [Swarm](/agent-kit/multi-agent/swarm) — Parallel, orchestrator, consensus, reflection topologies
- [Hooks Reference](/agent-kit/debugging/hooks-reference) — DYNAMIC_PIPELINE_* hooks
