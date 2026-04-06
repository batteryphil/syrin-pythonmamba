---
title: Swarm
description: Multiple agents, one shared goal, one shared budget — the v0.11.0 flagship feature
weight: 95
---

## What Is a Swarm?

A swarm is when multiple agents work together toward a single goal. Instead of one agent doing everything, you break the work into specialties. A research agent finds facts. A writer turns them into prose. An editor refines the prose. Each is excellent at its one thing.

The `Swarm` class coordinates all of this: it runs agents concurrently (or in sequence, depending on topology), manages a shared budget pool so costs are automatically distributed, merges results, and handles failures gracefully so one bad agent does not bring everything down.

This is the feature built in direct response to the $47,000 incident. Shared budgets with per-agent limits mean that cost overruns are contained — not catastrophic.

## Your First Swarm

```python
import asyncio
from syrin import Agent, Budget, Model
from syrin.enums import ExceedPolicy, AgentRole
from syrin.swarm import Swarm, SwarmConfig
from syrin.enums import SwarmTopology

class ResearchAgent(Agent):
    model = Model.mock()
    system_prompt = "You research topics and provide facts."

class WriterAgent(Agent):
    model = Model.mock()
    system_prompt = "You write clear summaries based on research."

class EditorAgent(Agent):
    model = Model.mock()
    system_prompt = "You edit and refine written content."

async def main():
    swarm = Swarm(
        agents=[ResearchAgent, WriterAgent, EditorAgent],  # class references, auto-instantiated
        goal="Write a brief overview of Python programming language",
        budget=Budget(max_cost=1.00, exceed_policy=ExceedPolicy.WARN),
        config=SwarmConfig(topology=SwarmTopology.PARALLEL),
    )
    
    result = await swarm.run()
    print(f"Content: {result.content[:80]}")
    print(f"\nCost breakdown:")
    for agent_name, cost in result.cost_breakdown.items():
        print(f"  {agent_name}: ${cost:.6f}")

asyncio.run(main())
```

Output:

```
Content: Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor i

Cost breakdown:
  ResearchAgent: $0.000048
  WriterAgent: $0.000050
  EditorAgent: $0.000048
```

All three agents ran in parallel. Each received the same `goal` text as their input. Their outputs were merged into `result.content`. The budget was shared across all three.

## Swarm Topologies

The `topology` setting controls how agents are coordinated. You pick the topology based on whether your agents' work is independent or sequential.

**`SwarmTopology.ORCHESTRATOR`** *(default)* — The first agent in the list is the orchestrator. It reads the goal, reasons about which specialists to call, and coordinates their work. Best when the delegation strategy should be dynamic.

**`SwarmTopology.PARALLEL`** — All agents run at the same time with the same goal. Best when agents are independent specialists. Fastest topology because there is no waiting.

**`SwarmTopology.CONSENSUS`** — All agents analyze the goal independently. Their answers are synthesized into a consensus result. Best for high-stakes decisions where multiple perspectives matter.

**`SwarmTopology.REFLECTION`** — One agent produces an answer, another critiques it, the first refines based on the critique. Best for improving output quality through self-correction.

**`SwarmTopology.WORKFLOW`** — The swarm is backed by a `Workflow` instance (sequential + parallel steps). Best for complex multi-step pipelines.

## Shared Budget Pool

The most important multi-agent cost feature: a shared pool that all agents draw from automatically.

```python
import asyncio
from syrin import Agent, Budget, Model
from syrin.enums import ExceedPolicy
from syrin.swarm import Swarm, SwarmConfig
from syrin.enums import SwarmTopology

class ResearchAgent(Agent):
    model = Model.mock()
    system_prompt = "You research topics."

class WriterAgent(Agent):
    model = Model.mock()
    system_prompt = "You write summaries."

async def main():
    swarm = Swarm(
        agents=[ResearchAgent, WriterAgent],  # class references, auto-instantiated
        goal="Write an overview of machine learning",
        budget=Budget(
            max_cost=1.00,         # Total pool: $1.00 for all agents
            exceed_policy=ExceedPolicy.WARN,
        ),
        config=SwarmConfig(topology=SwarmTopology.PARALLEL),
    )
    result = await swarm.run()
    
    print(f"Content: {result.content[:60]}")
    print(f"\nCost breakdown:")
    for name, cost in result.cost_breakdown.items():
        print(f"  {name}: ${cost:.6f}")
    
    if result.budget_report:
        print(f"\nTotal spent: ${result.budget_report.total_spent:.6f}")
        for entry in result.budget_report.per_agent:
            print(f"  {entry.agent_name}: spent=${entry.spent:.6f} allocated=${entry.allocated:.6f}")

asyncio.run(main())
```

Output:

```
Content: Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed

Cost breakdown:
  ResearchAgent: $0.000045
  WriterAgent: $0.000045

Total spent: $0.000090
  WriterAgent: spent=$0.000045 allocated=$0.500000
  ResearchAgent: spent=$0.000045 allocated=$0.500000
```

Each agent was allocated up to $0.50 from the $1.00 pool. The `budget_report` gives you a complete accounting of where every cent went. If one agent had spent its $0.50 cap, it would stop while the other continued.

## Lifecycle Hooks

Subscribe to swarm events for logging, alerting, and observability:

```python
import asyncio
from syrin import Agent, Model
from syrin.swarm import Swarm, SwarmConfig
from syrin.enums import SwarmTopology, Hook

class AnalystAgent(Agent):
    model = Model.mock()
    system_prompt = "You analyze data and provide insights."

class SummarizerAgent(Agent):
    model = Model.mock()
    system_prompt = "You create concise summaries."

async def main():
    swarm = Swarm(
        agents=[AnalystAgent, SummarizerAgent],  # class references
        goal="Analyze Q4 sales data and summarize findings",
        config=SwarmConfig(topology=SwarmTopology.PARALLEL),
    )
    
    events_seen = []
    swarm.events.on(Hook.SWARM_STARTED, lambda ctx: events_seen.append("Swarm started"))
    swarm.events.on(Hook.AGENT_JOINED_SWARM, lambda ctx: events_seen.append(f"Agent joined: {ctx.get('agent_name', '?')}"))
    swarm.events.on(Hook.SWARM_ENDED, lambda ctx: events_seen.append("Swarm ended"))
    
    result = await swarm.run()
    
    print("Events:")
    for e in events_seen:
        print(f"  {e}")
    print(f"\nResult: {result.content[:60]}")
    print(f"Agents: {len(result.agent_results)} completed")

asyncio.run(main())
```

Output:

```
Events:
  Swarm started
  Agent joined: AnalystAgent
  Agent joined: SummarizerAgent
  Swarm ended

Result: Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed
Agents: 2 completed
```

Key swarm hooks:
- `Hook.SWARM_STARTED` — swarm is about to begin
- `Hook.AGENT_JOINED_SWARM` — an agent started running (fires per agent)
- `Hook.AGENT_LEFT_SWARM` — an agent finished (fires per agent)
- `Hook.AGENT_FAILED` — an agent raised an exception
- `Hook.SWARM_BUDGET_LOW` — the shared budget pool is running low
- `Hook.SWARM_ENDED` — all agents completed (or were cancelled)

## Handling Agent Failures

When one agent fails, what happens to the others? You control this with `on_agent_failure`:

```python
from syrin.enums import FallbackStrategy
from syrin.swarm import SwarmConfig

# Skip the failed agent, continue with the others (default)
config = SwarmConfig(on_agent_failure=FallbackStrategy.SKIP_AND_CONTINUE)

# Stop the entire swarm immediately on any failure
config = SwarmConfig(on_agent_failure=FallbackStrategy.ABORT_SWARM)

# Remove the failed agent from the pool, let the rest continue
config = SwarmConfig(on_agent_failure=FallbackStrategy.ISOLATE_AND_CONTINUE)
```

When agents fail with `SKIP_AND_CONTINUE`, their responses go into `result.partial_results` (not `result.agent_results`). You can inspect them to understand what went wrong.

## Per-Agent Timeouts

Prevent one slow agent from blocking the whole swarm:

```python
from syrin.swarm import SwarmConfig

config = SwarmConfig(
    agent_timeout=30.0,    # Each agent has 30 seconds max
    max_parallel_agents=5, # Run at most 5 agents at a time
)
```

If an agent exceeds `agent_timeout`, it is treated as a failure and handled according to your `on_agent_failure` strategy.

## The SwarmResult

After `swarm.run()` returns, here is what you have:

```python
result.content           # Merged output from all successful agents (newline-separated)
result.cost_breakdown    # dict: agent_name → cost in USD
result.agent_results     # list of Response objects from successful agents
result.partial_results   # list of Response objects when some agents failed
result.budget_report     # SwarmBudgetReport (total_spent + per_agent entries)
```

`result.content` concatenates the successful responses. For the ORCHESTRATOR topology, it is the orchestrator's synthesized output. For CONSENSUS, it is the consensus verdict.

## Running Synchronously

If you are in a synchronous context (a script, a CLI tool, a sync web framework), use `run_sync()` instead of `await swarm.run()`:

```python
from syrin import Agent, Budget, Model
from syrin.enums import ExceedPolicy
from syrin.swarm import Swarm, SwarmConfig
from syrin.enums import SwarmTopology

class ResearchAgent(Agent):
    model = Model.mock()
    system_prompt = "You research topics and provide facts."

class WriterAgent(Agent):
    model = Model.mock()
    system_prompt = "You write clear summaries based on research."

swarm = Swarm(
    agents=[ResearchAgent, WriterAgent],  # class references, auto-instantiated
    goal="Write a brief overview of Python programming language",
    budget=Budget(max_cost=1.00, exceed_policy=ExceedPolicy.WARN),
    config=SwarmConfig(topology=SwarmTopology.PARALLEL),
)

result = swarm.run_sync()  # No asyncio.run() needed
print(f"Content: {result.content[:80]}")
for agent_name, cost in result.cost_breakdown.items():
    print(f"  {agent_name}: ${cost:.6f}")
```

`run_sync()` is equivalent to `asyncio.run(swarm.run())` — it creates an event loop internally and blocks until the swarm finishes. Use `await swarm.run()` when you are already inside an async context.

## Pause, Resume, and Cancel

For long-running swarms, use `play()` instead of `run()` to get lifecycle controls:

```python
import asyncio
from syrin import Agent, Model
from syrin.swarm import Swarm

class ResearchAgent(Agent):
    model = Model.mock()
    system_prompt = "You research topics."

class WriterAgent(Agent):
    model = Model.mock()
    system_prompt = "You write summaries."

async def main():
    swarm = Swarm(
        agents=[ResearchAgent, WriterAgent],
        goal="Research and summarize recent AI news",
    )
    
    handle = swarm.play()       # Starts in background, returns immediately
    
    await asyncio.sleep(2)
    await swarm.pause()         # Agents finish their current step, then pause
    
    await asyncio.sleep(1)
    await swarm.resume()        # Resume all paused agents
    
    result = await handle.wait()  # Wait for completion
    print(result.content[:60])
```

### SwarmController via handle.controller

Get a `SwarmController` bound to the live swarm without any manual wiring:

```python
async def main():
    researcher = ResearchAgent()
    writer = WriterAgent()
    
    swarm = Swarm(agents=[researcher, writer], goal="...")
    handle = swarm.play()
    
    # Get the controller directly from the handle
    ctrl = handle.controller
    
    # Pass agent objects — no string IDs needed
    await ctrl.pause_agent(researcher)
    await ctrl.change_context(writer, "Focus on key bullet points only")
    await ctrl.resume_agent(researcher)
    
    result = await handle.wait()
```

Each `Agent()` instance is automatically assigned a unique `agent_id` (`ClassName-<hex>`) at creation time. Pass the object directly to controller methods instead of managing string IDs manually.

You can also cancel a specific agent while letting the others continue:

```python
await swarm.cancel_agent("ResearchAgent")  # Cancel by class name
```

The cancelled agent is terminated immediately; its partial output (if any) goes into `result.partial_results`. The remaining agents continue running.

Check the status of each agent mid-run:

```python
for entry in swarm.agent_statuses():
    print(f"{entry.agent_name}: {entry.state}")
    # entry.agent_name — class name of the agent
    # entry.state      — AgentStatus (IDLE, RUNNING, PAUSED, KILLED, ...)
```

## Serving the Swarm

Give your swarm an HTTP interface:

```python
swarm = Swarm(
    agents=[ResearchAgent, WriterAgent],  # class references
    goal="AI research assistant",
)
swarm.serve(port=8000)
```

This exposes:
- `POST /chat` — send a goal, get back the swarm result
- `GET /graph` — returns the execution graph as Mermaid diagram (for WORKFLOW topology)

## Agent Roles

Declare an agent's authority role at the class level using the `role` class attribute (default: `AgentRole.WORKER`):

```python
from syrin import Agent, Model
from syrin.enums import AgentRole

class SupervisorAgent(Agent):
    role = AgentRole.SUPERVISOR
    team = [ResearchAgent, WriterAgent]
    model = Model.mock()
    system_prompt = "You coordinate the research and writing team."

class ResearchAgent(Agent):
    model = Model.mock()
    system_prompt = "You research topics."

class WriterAgent(Agent):
    model = Model.mock()
    system_prompt = "You write summaries."
```

To build the authority guard from class metadata — no string IDs required:

```python
from syrin.swarm import build_guard_from_agents

supervisor = SupervisorAgent()
researcher = ResearchAgent()
writer = WriterAgent()

guard = build_guard_from_agents([supervisor, researcher, writer])

swarm = Swarm(
    agents=[supervisor, researcher, writer],
    goal="Research and write a report on quantum computing",
    authority_guard=guard,
)
```

`build_guard_from_agents()` reads each agent's `role` and `team` class attributes to construct the `SwarmAuthorityGuard` automatically. See [Agent Authority](/multi-agent/authority) for the full permissions model.

## What's Next

- [Multi-Agent Overview](/agent-kit/multi-agent/overview) — Pattern selection guide
- [Budget Delegation](/agent-kit/multi-agent/budget-delegation) — Shared pools, per-agent caps, reallocation
- [MemoryBus](/agent-kit/multi-agent/memory-bus) — Shared memory across agents in a swarm
- [A2A Communication](/agent-kit/multi-agent/a2a) — Typed agent-to-agent messaging
- [Pipeline](/agent-kit/multi-agent/pipeline) — Sequential agent chains
- [Workflow](/agent-kit/multi-agent/workflow) — Conditional and parallel execution graphs
