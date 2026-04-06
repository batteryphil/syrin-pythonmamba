---
title: Multi-Agent Overview
description: Why multiple agents beat one big agent, when to use each pattern, and what to expect
weight: 90
---

## The Problem With One Big Agent

Imagine you are building an AI research assistant. It needs to search the web, analyze financial data, write professional prose, check legal compliance, and generate charts. You could put all of that into one agent.

But here is what happens: the agent becomes mediocre at everything. Its system prompt gets so long and contradictory that the LLM starts picking and choosing what to follow. Tool calls from one task pollute the context of another. When something goes wrong, you have no idea which part failed.

There is also a mathematical reality that every multi-agent builder learns the hard way. If each agent has 85% accuracy and you chain 10 agents together, your end-to-end success rate is 0.85 to the power of 10 — about 20%. The longer the chain, the worse it gets.

Syrin's multi-agent system is designed around these realities. Multiple specialized agents, each excellent at one thing, coordinated in ways that contain failures and share costs responsibly.

## The Five Patterns

Syrin gives you five multi-agent patterns. Each is right for a specific situation.

**Parallel Swarm** — All agents run at the same time toward the same goal. Best when you want multiple independent perspectives on the same question, or when each agent handles a different slice of the work that does not depend on what the others do.

**Orchestrator Swarm** — One lead agent decides which specialists to call and in what order. The lead agent reads the goal, reasons about it, and delegates. Best when the strategy should be dynamic — different goals require different specialists.

**Workflow** — Agents run in a deterministic sequence with optional conditional branching and parallel sections. Each step's output becomes the next step's input. Best for multi-step workflows and business logic: "if the sentiment is negative, route to the complaints specialist; otherwise route to the upsell specialist."

**Consensus** — Multiple agents independently analyze the same question and their answers are synthesized into one. Best for high-stakes decisions where you want multiple viewpoints before committing.

## Choosing the Right Pattern

Start simple and add complexity only when you need it.

If you have **independent tasks** (research, translation, summarization can all happen at the same time), use **Parallel Swarm**.

If you have a **dynamic goal** where you do not know in advance which agents will be needed, use **Orchestrator Swarm**.

If you have a **linear workflow** (research → draft → edit), use **Workflow** or the `sequential()` helper for lightweight one-off chains.

If you have **conditional logic** (route to specialist A or B based on content), use **Workflow**.

If you need **agreement before action** (three agents must all agree before a recommendation is made), use **Consensus**.

## The Cost Problem in Multi-Agent Systems

Multi-agent systems use roughly 15 times more tokens than single-agent interactions. That is not a bug — it is the nature of coordination. But it means budget control is not optional in a swarm.

The $47,000 incident described in the [introduction](/agent-kit/introduction) involved four agents in an infinite loop. A shared budget with a per-agent maximum would have capped the damage at $100.

Every Syrin multi-agent pattern supports shared budgets:

```python
from syrin import Budget
from syrin.enums import ExceedPolicy

shared_budget = Budget(
    max_cost=10.00,      # Total pool for all agents
    exceed_policy=ExceedPolicy.WARN,
)
```

This budget is passed to the `Swarm`, `AgentRouter`, or `Workflow`. All agents draw from the same pool. The hard `max_cost` cap ensures total spend never exceeds the limit, regardless of how many agents are running.

## Memory Across Agents

The other classic multi-agent problem: agents do not know what the others did. One agent builds a background. Another agent builds an incompatible asset. Neither knows about the other's implicit design decisions.

Syrin solves this with `MemoryBus` — a shared publish/subscribe memory layer. Agents publish facts they discover. Other agents subscribe to topics they need. When Agent A discovers something important, Agent B learns about it without being coupled to Agent A's code.

```python
from syrin.swarm import MemoryBus
from syrin.enums import MemoryType

memory_bus = MemoryBus()
# Agent A publishes: "Customer is in the EU, so GDPR applies"
# Agent B subscribes to "compliance" and learns this before running
```

More on this in [MemoryBus](/agent-kit/multi-agent/memory-bus).

## Agent Identity and Trust

In a swarm, agents communicate with each other. Agent A tells Agent B what to do. How does Agent B know the message actually came from Agent A and not from a malicious injection?

Syrin gives every agent a cryptographic Ed25519 identity. Every inter-agent message is signed. Every control command (pause, resume, kill) is verified. No agent can impersonate another.

This is enterprise-grade security for multi-agent systems that were, until recently, completely without it.

## What's Next

Choose your pattern and dive in:

- [Swarm](/agent-kit/multi-agent/swarm) — All topologies (PARALLEL, ORCHESTRATOR, CONSENSUS, REFLECTION) with full examples
- [Sequential & Parallel Helpers](/agent-kit/multi-agent/pipeline) — Lightweight one-off `sequential()` and `parallel()` chains
- [Workflow](/agent-kit/multi-agent/workflow) — Conditional and parallel workflow execution
- [A2A Communication](/agent-kit/multi-agent/a2a) — Typed agent-to-agent messaging
- [MemoryBus](/agent-kit/multi-agent/memory-bus) — Shared memory across agents
- [Budget Delegation](/agent-kit/multi-agent/budget-delegation) — Shared budgets, per-agent caps, reallocation
- [When to Use Multi-Agent](/agent-kit/multi-agent/when-to-use) — Detailed pattern selection guide
