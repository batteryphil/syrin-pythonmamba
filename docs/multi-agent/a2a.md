---
title: Agent-to-Agent Messaging (A2A)
description: Typed, async message passing between agents in a swarm
weight: 120
---

## Why A2A?

In a swarm, agents usually communicate through the orchestrator or by chaining outputs. But sometimes you want direct communication: one agent sends a message to another with a specific type, and the receiver processes it explicitly.

Agent-to-agent (A2A) messaging gives you typed, asynchronous message passing between agents. You define a Pydantic model as your message type. One agent sends it. Another receives it. The message is validated at both ends.

This is useful for assignment systems, status updates, coordination protocols, and anywhere the agents need to exchange structured information without going through a shared context.

## Basic Setup

```python
import asyncio
from pydantic import BaseModel
from syrin.swarm._a2a import A2ARouter

class TaskAssignment(BaseModel):
    task_id: str
    description: str

async def main():
    router = A2ARouter()
    router.register_agent("orchestrator")
    router.register_agent("worker")

    # Orchestrator sends an assignment to worker
    await router.send(
        from_agent="orchestrator",
        to_agent="worker",
        message=TaskAssignment(task_id="t1", description="Summarize the report"),
    )

    # Worker receives the message
    envelope = await router.receive(agent_id="worker", timeout=2.0)
    task = envelope.payload
    print(f"task_id: {task.task_id}")
    print(f"description: {task.description}")

asyncio.run(main())
```

Output:

```
task_id: t1
description: Summarize the report
```

`envelope.payload` is the typed `TaskAssignment` instance. Not a dict, not a string — the actual Pydantic object.

## The Envelope

Every message is wrapped in an `A2AMessageEnvelope` with routing metadata:

- `envelope.payload` — the typed message (your Pydantic model)
- `envelope.from_agent` — sender's agent ID
- `envelope.to_agent` — recipient's agent ID
- `envelope.message_id` — unique UUID for this message
- `envelope.message_type` — the message class name as a string
- `envelope.timestamp` — when the message was sent
- `envelope.channel` — `"direct"`, `"broadcast"`, or `"topic"`

## Delivery Channels

### Direct (Default)

Point-to-point delivery to exactly one agent:

```python
from syrin.enums import A2AChannel

await router.send(
    from_agent="orchestrator",
    to_agent="worker",
    message=TaskAssignment(task_id="t2", description="Process batch"),
    channel=A2AChannel.DIRECT,  # This is the default
)
```

### Broadcast

Send to all registered agents (excluding the sender):

```python
from pydantic import BaseModel
from syrin.enums import A2AChannel

class StatusUpdate(BaseModel):
    status: str
    progress: float

await router.send(
    from_agent="orchestrator",
    to_agent="broadcast",
    message=StatusUpdate(status="Phase 1 complete", progress=0.33),
    channel=A2AChannel.BROADCAST,
)

# All registered agents except "orchestrator" receive this
```

### Topic (Pub/Sub)

Subscribe agents to named topics, then fan out messages to all subscribers at once:

```python
from syrin.enums import A2AChannel

# Subscribe agents to a topic
router.subscribe("researcher-1", "findings")
router.subscribe("researcher-2", "findings")
router.subscribe("researcher-3", "findings")

# Send to all subscribers of "findings"
await router.send_topic(
    from_agent="orchestrator",
    topic="findings",
    message=StatusUpdate(status="Phase complete", progress=1.0),
)

# All three researchers receive it
envelope = await router.receive(agent_id="researcher-1", timeout=2.0)
```

`router.subscribe(agent, topic)` registers an agent as a subscriber to a topic name. `router.send_topic(from_agent, topic, message)` delivers the message to all subscribed agents. The sender is not excluded from receiving its own topic broadcasts (unlike `BROADCAST` which excludes the sender).

## Acknowledgment

For critical messages, require acknowledgment before the sender continues:

```python
await router.send(
    from_agent="orchestrator",
    to_agent="worker",
    message=TaskAssignment(task_id="t3", description="Critical task"),
    requires_ack=True,
)

# Sender waits; worker must ack
envelope = await router.receive(agent_id="worker")
await router.ack(envelope.message_id, agent_id="worker")
```

## Typed Message Retrieval

Use `get_typed_payload()` to get the message as a specific type with validation:

```python
envelope = await router.receive(agent_id="worker")
task = envelope.get_typed_payload(TaskAssignment)
print(f"Task: {task.task_id}")
```

This is useful when an agent handles multiple message types — receive the envelope first, check `envelope.message_type`, then cast to the appropriate type.

## Hooks

A2A messaging fires hooks you can subscribe to:

```python
from syrin.enums import Hook

# (on an agent that has access to the router's event bus)
agent.events.on(Hook.A2A_MESSAGE_SENT, lambda ctx: print(f"Sent to {ctx['to']}"))
agent.events.on(Hook.A2A_MESSAGE_RECEIVED, lambda ctx: print(f"From {ctx['from']}"))
agent.events.on(Hook.A2A_MESSAGE_TIMEOUT, lambda ctx: print(f"Timeout sending to {ctx['to']}"))
```

## When to Use A2A vs. Swarm

Use **A2A** when agents need to exchange structured data explicitly — task assignments, status reports, typed commands. It's a coordination primitive.

Use a **Swarm** when you want Syrin to handle the coordination for you — parallel execution, orchestration, consensus voting. Swarms are higher-level.

A2A is the lower-level building block that swarm topologies use internally. You reach for it directly when you're building custom multi-agent coordination that doesn't fit a standard topology.

## What's Next

- [Swarm](/agent-kit/multi-agent/swarm) — High-level multi-agent topologies
- [MemoryBus](/agent-kit/multi-agent/memory-bus) — Shared memory between agents
- [Hooks Reference](/agent-kit/debugging/hooks-reference) — A2A hook details
