---
title: MonitorLoop
description: Async supervisor loop that polls agents, delivers events, and performs bounded interventions with escalation
weight: 76
---

## What Is MonitorLoop?

`MonitorLoop` is an async supervisor that watches over a set of agents, emitting periodic heartbeat events and letting you step in when things go wrong.

The pattern it implements is: **monitor → assess → intervene**. The loop emits `HEARTBEAT` events on a timer. When an agent produces output, you assess it. If quality is poor, you call `monitor.intervene()` to correct it. If you intervene too many times, the loop raises `MaxInterventionsExceeded` and fires `AGENT_ESCALATION` — a signal that the agent needs human attention, not more automatic nudging.

## Quickstart

```python
import asyncio
from syrin.swarm._monitor import MonitorLoop, MaxInterventionsExceeded
from syrin.enums import MonitorEventType, InterventionAction

async def main():
    async with MonitorLoop(
        targets=["worker-1", "worker-2"],
        poll_interval=1.0,
        max_interventions=3,
    ) as monitor:
        async for event in monitor:
            if event.event_type == MonitorEventType.HEARTBEAT:
                print(f"[heartbeat] {event.agent_id}")

            elif event.event_type == MonitorEventType.OUTPUT_READY:
                output = event.data.get("output", "")
                print(f"[output] {event.agent_id}: {output}")

                if "error" in output.lower():
                    await monitor.intervene(
                        event.agent_id,
                        InterventionAction.CHANGE_CONTEXT_AND_RERUN,
                        context="Focus on accuracy, avoid hallucinations.",
                    )

            # Break after first output for demo
            if event.event_type == MonitorEventType.OUTPUT_READY:
                break

asyncio.run(main())
```

## Constructor

```python
MonitorLoop(
    targets=["worker-1", "worker-2"],  # Agent IDs to watch
    poll_interval=1.0,                  # Seconds between HEARTBEAT events
    max_interventions=5,                # 0 = unlimited
    fire_event_fn=None,                 # Optional hook emitter
)
```

`targets` is required — it's the list of agent IDs the loop will supervise. `poll_interval` controls how often `HEARTBEAT` events fire per agent. `max_interventions` sets the cap on how many times `intervene()` can be called; set to `0` for unlimited. `fire_event_fn` is an optional hook emitter `(Hook, dict) -> None` — pass `swarm._fire_event` to connect monitor events to your swarm's lifecycle hooks.

## Event Types

Two event types flow through the `async for event in monitor` loop.

`MonitorEventType.HEARTBEAT` fires every `poll_interval` seconds for each active target. Use it to check on agents proactively — poll their state, update a dashboard, log that they're alive.

`MonitorEventType.OUTPUT_READY` fires when you call `monitor.notify_agent_output(agent_id, output)` from outside the loop. Use it to deliver agent outputs to the monitor for assessment.

Every event has three fields: `agent_id` (which agent this event relates to), `event_type` (`HEARTBEAT` or `OUTPUT_READY`), and `data` (a dict with event metadata — for `OUTPUT_READY`, it includes `{"output": "..."`}).

## Interventions

Call `monitor.intervene(agent_id, action)` to take action on a target agent:

```python
# Pause the agent and wait for manual instruction
await monitor.intervene("worker-1", InterventionAction.PAUSE_AND_WAIT)

# Replace the agent's context and run it again
await monitor.intervene(
    "worker-1",
    InterventionAction.CHANGE_CONTEXT_AND_RERUN,
    context="Be more concise and focus on facts only.",
)
```

`InterventionAction.PAUSE_AND_WAIT` pauses the target agent and waits for a manual instruction before continuing. `InterventionAction.CHANGE_CONTEXT_AND_RERUN` replaces the agent's context string and reruns the current step with the new context.

### Escalation

When `max_interventions > 0` and the limit is reached, `intervene()` raises `MaxInterventionsExceeded` and fires the `AGENT_ESCALATION` hook before raising. This is the moment to alert a human operator:

```python
from syrin.swarm._monitor import MaxInterventionsExceeded

try:
    await monitor.intervene("w1", InterventionAction.PAUSE_AND_WAIT)
except MaxInterventionsExceeded as e:
    print(f"Escalating: {e.count} interventions attempted, limit was {e.limit}")
    # AGENT_ESCALATION hook has already fired at this point
    # Notify your on-call team here
```

`MaxInterventionsExceeded` has two attributes: `limit` (the configured maximum) and `count` (how many interventions were attempted).

## Notifying the Monitor of Agent Output

When an agent outside the monitor loop produces output, tell the monitor about it:

```python
monitor.notify_agent_output("worker-1", "The analysis shows...")
```

This enqueues an `OUTPUT_READY` event that your `async for` loop will receive on the next iteration.

## Releasing an Agent

Stop monitoring a specific agent without shutting down the whole loop:

```python
monitor.release("worker-1")
```

After `release()`, no more `HEARTBEAT` events are produced for `worker-1`. Events already queued are unaffected and will still be delivered.

## The Full Assess → Intervene Pattern

Here's the complete supervisor pattern with quality assessment:

```python
import asyncio
from syrin.swarm._monitor import MonitorLoop
from syrin.enums import AssessmentResult, InterventionAction, MonitorEventType

def assess_output(output: str) -> AssessmentResult:
    if "excellent" in output:
        return AssessmentResult.EXCELLENT
    if len(output) < 50:
        return AssessmentResult.POOR
    return AssessmentResult.ACCEPTABLE

async def supervisor_loop(worker_ids: list[str]) -> None:
    async with MonitorLoop(
        targets=worker_ids,
        poll_interval=2.0,
        max_interventions=5,
    ) as monitor:
        async for event in monitor:
            if event.event_type != MonitorEventType.OUTPUT_READY:
                continue

            output = str(event.data.get("output", ""))
            result = assess_output(output)

            if result == AssessmentResult.POOR:
                await monitor.intervene(
                    event.agent_id,
                    InterventionAction.CHANGE_CONTEXT_AND_RERUN,
                    context="Please provide a more detailed response (at least 100 words).",
                )
            elif result == AssessmentResult.EXCELLENT:
                monitor.release(event.agent_id)
```

`AssessmentResult` has four values. `EXCELLENT` means the output quality is high — no intervention needed, optionally release the agent. `ACCEPTABLE` means the output is good enough — continue without intervention. `POOR` means quality is low — request an intervention. `FAILED` means the agent has failed — escalate to a human.

## Integration with Swarm

Run the monitor alongside a swarm. Use `swarm.play()` to get an async handle, then drive the monitor loop independently:

```python
from syrin.swarm import Swarm
from syrin.swarm._monitor import MonitorLoop

swarm = Swarm(agents=[...], goal="...")

async def run_with_monitor():
    handle = swarm.play()

    async with MonitorLoop(
        targets=[worker_agent],   # agent instances
        poll_interval=1.0,
        max_interventions=3,
    ) as monitor:
        async for event in monitor:
            # Assess and intervene here
            if handle.status.value in ("completed", "failed", "cancelled"):
                break

    result = await handle.wait()
```

## Hooks

`Hook.AGENT_ESCALATION` fires when `max_interventions` is exceeded, just before `MaxInterventionsExceeded` is raised. The hook context includes: `agent_id`, `action`, `intervention_count`, and `max_interventions`.

## See Also

- [Swarm](/multi-agent/swarm) — Parallel, consensus, reflection topologies
- [Broadcast](/multi-agent/broadcast) — Publish events to multiple agents
- [Authority](/multi-agent/authority) — Permission model for agent actions
