---
title: When to Use Multi-Agent
description: A decision framework for when one agent is enough vs. when you need multiple
weight: 91
---

## Start with One Agent

Before you add a second agent, make sure you actually need one. A single agent with the right tools and a well-written system prompt can handle surprising complexity. Multi-agent systems are more expensive, harder to debug, and harder to test.

Ask yourself these questions first:

Could better tools solve this? If the agent needs to search the web, call an API, or run code, give it tools — not more agents.

Could a better system prompt solve this? A clear, specific prompt with examples often eliminates the need for orchestration.

Could memory solve this? If the agent forgets context across sessions, add `Memory()`. That's one agent, not two.

If none of these work, then you probably need multiple agents.

## Multi-Agent Patterns at a Glance

| Pattern | When to use |
|---------|-------------|
| `Workflow` | Known, fixed agent sequence with optional branching |
| `Swarm` (PARALLEL) | Independent subtasks that run concurrently |
| `Swarm` (CONSENSUS) | High-stakes decisions needing multiple independent opinions |
| `Swarm` (REFLECTION) | Iterative quality improvement via producer–critic loop |
| `Swarm` (ORCHESTRATOR) | Dynamic task decomposition via LLM |
| `AgentRouter` | LLM decides which agents to invoke at runtime |
| `agent.spawn()` | Imperative parent-child delegation |
| `agent.handoff()` | Transfer control to a specialist agent |

## When Multiple Agents Make Sense

### Different tasks need different expertise

Some work naturally divides into areas that benefit from specialized agents. Research requires different behavior than writing.

```python
from syrin import Agent, Model
from syrin.workflow import Workflow

class Researcher(Agent):
    model = Model.mock()
    system_prompt = "You find and summarize information. Be factual, cite sources."

class Writer(Agent):
    model = Model.mock()
    system_prompt = "You write compelling content. Use the research as your foundation."

# Sequential: Writer receives Researcher's output as context
wf = (
    Workflow("research-and-write")
    .step(Researcher, task="Find key facts about AI adoption in healthcare")
    .step(Writer, task="Write a 500-word article based on the research")
)
result = await wf.run("AI in healthcare")
```

The researcher and writer have genuinely different jobs. One is optimized for gathering; the other for crafting.

### Different tasks justify different models

Using GPT-4o for every step is expensive. Use cheaper models for extraction; expensive ones only for synthesis.

```python
from syrin import Agent, Model
from syrin.workflow import Workflow

class ExtractorAgent(Agent):
    model = Model.OpenAI("gpt-4o-mini")  # Cheap + fast
    system_prompt = "Extract key data points from the text. Return JSON."

class SynthesizerAgent(Agent):
    model = Model.OpenAI("gpt-4o")       # Best quality
    system_prompt = "Synthesize extracted data into a strategic recommendation."

wf = (
    Workflow("extract-and-synthesize")
    .step(ExtractorAgent, task="Extract from this earnings call transcript: ...")
    .step(SynthesizerAgent, task="Based on these data points, what should we do?")
)
result = await wf.run("Q3 earnings")
```

This costs roughly half as much as running both steps with GPT-4o.

### Independent work can run in parallel

If multiple tasks don't depend on each other, run them at the same time with `Swarm(topology=PARALLEL)`.

```python
from syrin import Agent, Model
from syrin.swarm import Swarm, SwarmConfig
from syrin.enums import SwarmTopology

class NewsAgent(Agent):
    model = Model.mock()
    system_prompt = "Summarize AI news."

class StockAgent(Agent):
    model = Model.mock()
    system_prompt = "Report on AI stock performance."

class ResearchAgent(Agent):
    model = Model.mock()
    system_prompt = "Find recent AI research papers."

# All three run simultaneously
swarm = Swarm(
    agents=[NewsAgent(), StockAgent(), ResearchAgent()],
    goal="AI morning briefing",
    config=SwarmConfig(topology=SwarmTopology.PARALLEL),
)
result = await swarm.run()
```

Wall time is the max of the three tasks, not their sum.

### You need verification or consensus

For high-stakes outputs — content moderation, fact-checking, financial recommendations — multiple independent agents looking at the same thing gives you corroboration.

```python
from syrin.swarm import Swarm, SwarmConfig, ConsensusConfig
from syrin.enums import SwarmTopology

swarm = Swarm(
    agents=[ReviewerA(), ReviewerB(), ReviewerC()],
    goal="Is this content appropriate for the platform?",
    config=SwarmConfig(
        topology=SwarmTopology.CONSENSUS,
        consensus=ConsensusConfig(min_agreement=0.67),
    ),
)
result = await swarm.run()
```

### You need iterative quality improvement

For content that needs to be refined, use the REFLECTION topology: a producer drafts, a critic scores, the producer revises.

```python
from syrin.swarm import Swarm, SwarmConfig, ReflectionConfig
from syrin.enums import SwarmTopology

swarm = Swarm(
    agents=[WriterAgent(), EditorAgent()],
    goal="Write a technical explanation of vector embeddings",
    config=SwarmConfig(
        topology=SwarmTopology.REFLECTION,
        reflection=ReflectionConfig(
            producer=WriterAgent,
            critic=EditorAgent,
            max_rounds=3,
            score_threshold=0.85,
        ),
    ),
)
result = await swarm.run()
```

## When to Stick with One Agent

**Chat-like interactions.** A conversation flows naturally. There's no pipeline — just messages and responses.

**Low latency requirements.** Every agent adds overhead. If you need responses in under 200ms, fewer agents is better.

**Simple, contained tasks.** "Translate this" or "summarize this paragraph" doesn't improve with three agents.

**Prototypes.** Start simple. Add complexity only when you've proven a single agent isn't enough.

## The Migration Path

Most successful multi-agent systems start as single agents and evolve:

**Stage 1: Single agent with tools.** This handles most use cases.

```python
class Assistant(Agent):
    model = Model.mock()
    system_prompt = "You are a technical support specialist."
    tools = [search, lookup, calculate]
```

**Stage 2: Better prompting + memory.** Before adding agents, optimize the single-agent setup.

```python
class Assistant(Agent):
    model = Model.mock()
    memory = Memory()
    system_prompt = """You are a senior technical support specialist with 10 years
    of experience. You have access to search and lookup tools. Always check the
    knowledge base before asking clarifying questions."""
```

**Stage 3: Multiple agents only when needed.**

```python
class TriageAgent(Agent):
    system_prompt = "Classify the issue and route to the right specialist."

class BillingAgent(Agent):
    system_prompt = "Handle billing and subscription questions."

class TechAgent(Agent):
    system_prompt = "Handle technical debugging and configuration."
```

## Quick Decision Guide

- Different expertise needed → `Workflow` with specialized agents
- Known sequence, no branching → `Workflow`
- Independent subtasks → `Swarm(topology=PARALLEL)`
- High-stakes decision needing corroboration → `Swarm(topology=CONSENSUS)`
- Output needs iterative refinement → `Swarm(topology=REFLECTION)`
- Task complexity requires LLM planning → `Swarm(topology=ORCHESTRATOR)`
- Runtime agent selection by LLM → `AgentRouter`
- Simple imperative delegation → `agent.spawn()`
- Transfer control to specialist → `agent.handoff()`
- It's a conversation → one agent
- Prototyping → one agent
- Latency matters most → one agent

## What's Next

- [Workflow](/agent-kit/multi-agent/workflow) — Deterministic sequential/branching execution
- [Swarm](/agent-kit/multi-agent/swarm) — Parallel, consensus, reflection, and orchestrator topologies
- [AgentRouter](/agent-kit/multi-agent/agent-router) — LLM-driven dynamic orchestration
