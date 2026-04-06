---
title: MonitorLoop
description: Real-time monitoring and bounded intervention for swarm agents
weight: 110
---

`MonitorLoop` is an async context manager that continuously monitors agents in a swarm, yielding structured events and allowing bounded interventions.

## Overview

```python
from syrin.swarm import MonitorLoop
from syrin.enums import MonitorEventType, InterventionAction

# Pass agent instances — IDs are resolved automatically
async with MonitorLoop(targets=[worker_1, worker_2], poll_interval=1.0) as monitor:
    async for event in monitor:
        print(f"[{event.event_type}] {event.agent_id}: {event.data}")

        if event.event_type == MonitorEventType.OUTPUT_READY:
            # Respond to output
            break
```

## Event types

Two event types are emitted. `HEARTBEAT` fires periodically at `poll_interval` seconds. `OUTPUT_READY` fires when `notify_agent_output()` is called externally.

## MonitorEvent

```python
@dataclass
class MonitorEvent:
    agent_id: str
    event_type: MonitorEventType
    data: dict[str, object]
```

## Interventions

```python
async with MonitorLoop(
    targets=[worker],           # agent instance
    poll_interval=1.0,
    max_interventions=3,        # raises MaxInterventionsExceeded on 4th
) as monitor:
    async for event in monitor:
        await monitor.intervene(
            worker,             # agent instance
            InterventionAction.CHANGE_CONTEXT_AND_RERUN,
            context="Be more concise",
        )
        break
```

When `max_interventions` is exceeded, `MaxInterventionsExceeded` is raised and `Hook.AGENT_ESCALATION` fires automatically.

### Intervention actions

Two intervention actions are available. `PAUSE_AND_WAIT` pauses the agent and waits for input. `CHANGE_CONTEXT_AND_RERUN` injects new context and re-runs the agent.

## Releasing agents

```python
monitor.release(worker_1)  # stop heartbeats for worker_1
```

## Injecting output events externally

When a swarm executor detects that an agent has produced output, call:

```python
monitor.notify_agent_output(worker_1, output_text)
```

This enqueues an `OUTPUT_READY` event immediately.

## Constructor reference

```python
MonitorLoop(
    targets: list[Agent | str],      # agent instances (recommended) or string IDs
    poll_interval: float = 1.0,      # seconds between heartbeats
    max_interventions: int = 0,       # 0 = unlimited
    fire_event_fn: Callable | None = None,
)
```

## Hook events

`Hook.AGENT_ESCALATION` fires when `MaxInterventionsExceeded` is raised.
