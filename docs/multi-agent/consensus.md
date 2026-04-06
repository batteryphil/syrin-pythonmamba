---
title: Consensus Topology
description: Run multiple agents independently and surface the most agreed-upon answer
weight: 85
---

## When One Opinion Isn't Enough

You're building an AI system to flag potentially dangerous content. One model says "safe." Should you trust it?

What if three independent models look at the same content and all say "safe"? That's much more convincing. And if two say "flagged" and one says "safe," you know there's ambiguity worth escalating.

That's the CONSENSUS topology. Multiple agents answer the same question independently. Their answers are compared. The result is whatever the majority agreed on — or whichever approach your chosen strategy dictates.

## Basic Consensus

```python
import asyncio
from syrin import Agent, Model
from syrin.enums import ConsensusStrategy, SwarmTopology
from syrin.swarm import Swarm, SwarmConfig
from syrin.swarm.topologies._consensus import ConsensusConfig

class FactCheckerA(Agent):
    model = Model.mock()
    system_prompt = "You fact-check claims and respond with TRUE or FALSE."

class FactCheckerB(Agent):
    model = Model.mock()
    system_prompt = "You fact-check claims and respond with TRUE or FALSE."

class FactCheckerC(Agent):
    model = Model.mock()
    system_prompt = "You fact-check claims and respond with TRUE or FALSE."

async def main():
    swarm = Swarm(
        agents=[FactCheckerA(), FactCheckerB(), FactCheckerC()],
        goal="Is the Earth flat?",
        config=SwarmConfig(topology=SwarmTopology.CONSENSUS),
        consensus_config=ConsensusConfig(strategy=ConsensusStrategy.MAJORITY),
    )
    result = await swarm.run()
    print(f"Answer: {result.content[:50]}")

    cr = result.consensus_result
    print(f"Consensus reached: {cr.consensus_reached}")
    print(f"Agreement: {cr.agreement_fraction:.0%}")
    print(f"Winning answer: {cr.winning_answer[:30]}")
    for vote in cr.votes:
        print(f"  {vote.agent_name}: {vote.answer[:20]} (weight={vote.weight})")

asyncio.run(main())
```

Output:

```
Answer: Lorem ipsum dolor sit amet, consectetur adipiscing
Consensus reached: True
Agreement: 100%
Winning answer: Lorem ipsum dolor sit amet, co
  FactCheckerA: Lorem ipsum dolor si (weight=1.0)
  FactCheckerB: Lorem ipsum dolor si (weight=1.0)
  FactCheckerC: Lorem ipsum dolor si (weight=1.0)
```

With the mock model, all three agents return the same lorem ipsum, so they always agree 100%. With real models and real prompts, they would return varied answers like "TRUE" vs "FALSE" and the consensus machinery would count the votes.

## The ConsensusResult

After `swarm.run()`, inspect `result.consensus_result`:

- `cr.consensus_reached` — `True` if the strategy found a winner
- `cr.winning_answer` — the answer that won
- `cr.votes` — list of `ConsensusVote` objects, one per agent
- `cr.agreement_fraction` — fraction of agents that agreed (1.0 = 100%)

Each `ConsensusVote` has:
- `vote.agent_name` — which agent cast this vote
- `vote.answer` — what they answered
- `vote.weight` — how much this vote counts (for weighted strategies)

## Consensus Strategies

**`ConsensusStrategy.MAJORITY`** — the answer with the most votes wins. If no majority exists, the result is the most common answer.

**`ConsensusStrategy.UNANIMITY`** — all agents must agree. If they don't, `consensus_reached=False`.

**`ConsensusStrategy.WEIGHTED`** — votes are multiplied by their `weight`. Use this when some agents (e.g. a more capable model) should have more influence.

```python
from syrin.swarm.topologies._consensus import ConsensusConfig
from syrin.enums import ConsensusStrategy

# Simple majority
consensus_config = ConsensusConfig(strategy=ConsensusStrategy.MAJORITY)

# All must agree
consensus_config = ConsensusConfig(strategy=ConsensusStrategy.UNANIMITY)
```

## When to Use Consensus

Use consensus when you need corroboration — facts, safety checks, high-stakes decisions. Multiple independent agents looking at the same question and agreeing is a much stronger signal than any single agent.

It's especially useful when:
- Hallucination risk is high and you need a sanity check
- The cost of a wrong answer is high (medical, legal, financial)
- You want to surface uncertainty (disagreement = flag for human review)
- You're using multiple models from different providers and want a "committee" view

## What's Next

- [Swarm](/agent-kit/multi-agent/swarm) — All swarm topologies
- [Reflection](/agent-kit/multi-agent/reflection) — Writer-editor loops that improve quality iteratively
- [Parallel Swarm](/agent-kit/multi-agent/swarm) — Run agents in parallel without voting
