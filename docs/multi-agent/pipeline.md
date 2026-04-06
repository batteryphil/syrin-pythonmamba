---
title: Sequential & Parallel Helpers
description: Lightweight one-off agent chaining with sequential() and parallel()
weight: 92
---

## When to Use These Helpers

For production multi-step workflows, use [`Workflow`](/agent-kit/multi-agent/workflow) — it supports conditional branching, visualization, and lifecycle hooks.

For quick, one-off agent chains where you don't need the full Workflow API, syrin provides two utility functions:

- `sequential()` — run agents one after another, each receiving the previous output as context
- `parallel()` — run agents simultaneously and collect all results

## Sequential Execution

`sequential()` takes a list of `(agent_instance, task)` tuples. It runs them in order and returns the last agent's `Response`.

```python
from syrin import Agent, Model
from syrin.agent.pipeline import sequential

class ResearchAgent(Agent):
    model = Model.mock()
    system_prompt = "You research topics thoroughly."

class WriterAgent(Agent):
    model = Model.mock()
    system_prompt = "You write clear, engaging summaries."

class EditorAgent(Agent):
    model = Model.mock()
    system_prompt = "You edit and polish content for publication."

result = sequential([
    (ResearchAgent(), "Research the history of Python"),
    (WriterAgent(),   "Write a summary of Python's history"),
    (EditorAgent(),   "Polish and finalize the article"),
])

print(result.content)
print(f"Cost: ${result.cost:.6f}")
```

Each step receives the previous step's output appended to its task. The final `Response` reflects the total cost across all agents.

**Empty list** → returns an empty `Response` with no content and zero cost.

## Parallel Execution

`parallel()` is an async function. It runs all agents simultaneously and returns a **list** of `Response` objects — one per agent, in the same order as the input.

```python
import asyncio
from syrin import Agent, Model
from syrin.agent.pipeline import parallel

class NewsAgent(Agent):
    model = Model.mock()
    system_prompt = "Summarize AI news."

class StockAgent(Agent):
    model = Model.mock()
    system_prompt = "Report on AI stock performance."

class ResearchAgent(Agent):
    model = Model.mock()
    system_prompt = "Find recent AI research papers."

async def main():
    results = await parallel([
        (NewsAgent(),     "AI and tech news today"),
        (StockAgent(),    "AI sector stocks"),
        (ResearchAgent(), "Latest AI papers"),
    ])
    for r in results:
        print(r.content[:80])

asyncio.run(main())
```

Wall time equals the **slowest** agent, not the sum of all agents.

Note: `parallel()` takes **instances** (not classes). Construct your agents before passing them.

## Mix Sequential and Parallel

A common pattern: parallel research, sequential synthesis.

```python
import asyncio
from syrin import Agent, Model
from syrin.agent.pipeline import sequential, parallel

class SourceA(Agent):
    model = Model.mock()
    system_prompt = "Research source A."

class SourceB(Agent):
    model = Model.mock()
    system_prompt = "Research source B."

class Writer(Agent):
    model = Model.mock()
    system_prompt = "Synthesize research into a report."

async def run():
    # Gather from both sources simultaneously
    gathered = await parallel([
        (SourceA(), "Research AI trends from source A"),
        (SourceB(), "Research AI trends from source B"),
    ])

    combined = "\n\n".join(r.content for r in gathered)

    # Synthesize sequentially
    final = sequential([
        (Writer(), f"Write a unified report from:\n{combined}"),
    ])
    return final

result = asyncio.run(run())
print(result.content)
```

## When to Use These vs. Workflow vs. Swarm

| Need | Use |
|------|-----|
| Fixed steps, conditional logic, branching | `Workflow` |
| Multiple agents on the same goal | `Swarm(topology=PARALLEL)` |
| Quick one-off sequential chain | `sequential()` |
| Quick one-off parallel run | `parallel()` |
| LLM decides which agents to use | `AgentRouter` |

`sequential()` and `parallel()` are intentionally minimal. No lifecycle hooks, no budget enforcement, no visualization. If you need those features, use `Workflow` or `Swarm`.

## What's Next

- [Workflow](/agent-kit/multi-agent/workflow) — Conditional routing, parallel steps, visualization
- [Swarm](/agent-kit/multi-agent/swarm) — Five topologies: PARALLEL, CONSENSUS, REFLECTION, ORCHESTRATOR, WORKFLOW
- [Budget Delegation](/agent-kit/multi-agent/budget-delegation) — Cost control across multi-agent systems
