"""MonitorLoop — QAAgent monitors WriterAgent output quality.

The MonitorLoop lets a supervisor agent poll targets for heartbeats and
receive output events. When quality is poor, the supervisor can intervene
by injecting new context or pausing the agent. After too many interventions
MaxInterventionsExceeded is raised and Hook.AGENT_ESCALATION fires.

Key concepts:
  - MonitorLoop(targets=[agent, ...], poll_interval=1.0, max_interventions=3)
  - async with MonitorLoop(...) as monitor: async for event in monitor
  - MonitorEventType.HEARTBEAT, .OUTPUT_READY
  - monitor.notify_agent_output(agent, output) — feed output into monitor
  - monitor.intervene(agent, InterventionAction.CHANGE_CONTEXT_AND_RERUN)
  - MaxInterventionsExceeded — raised when limit exceeded
  - Hook.AGENT_ESCALATION — fired on max intervention breach

Run:
    uv run python examples/07_multi_agent/monitor_loop.py
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from syrin import Agent, Model
from syrin.enums import Hook, InterventionAction, MonitorEventType
from syrin.swarm import MaxInterventionsExceeded, MonitorEvent, MonitorLoop

_MODEL = Model.OpenAI("gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"))


class WriterAgent(Agent):
    model = _MODEL
    system_prompt = "You write clear, engaging summaries."


# ── Example 1: Basic heartbeat monitoring ────────────────────────────────────
#
# MonitorLoop polls targets at poll_interval seconds and emits HEARTBEAT
# events to show the agent is alive.  Pass agent instances, not string IDs.


async def example_heartbeat() -> None:
    print("\n── Example 1: Basic heartbeat monitoring ────────────────────────")

    writer = WriterAgent()

    async with MonitorLoop(
        targets=[writer],  # agent object, not "writer-agent"
        poll_interval=0.05,  # fast poll for the example
    ) as monitor:
        beats = 0
        async for event in monitor:
            if event.event_type == MonitorEventType.HEARTBEAT:
                beats += 1
                print(f"  HEARTBEAT from {event.agent_id}  (beat #{beats})")
            if beats >= 3:
                break  # stop after 3 beats

    print(f"  Received {beats} heartbeats")


# ── Example 2: OUTPUT_READY event via notify_agent_output() ──────────────────
#
# External code (e.g. a swarm executor or test) calls
# monitor.notify_agent_output() to push an output event into the queue.


async def example_output_ready() -> None:
    print("\n── Example 2: OUTPUT_READY events ───────────────────────────────")

    writer = WriterAgent()
    output_events: list[MonitorEvent] = []

    async with MonitorLoop(
        targets=[writer],
        poll_interval=10.0,  # long poll — we will trigger outputs manually
    ) as monitor:
        # Simulate the writer agent producing 2 outputs
        monitor.notify_agent_output(writer, "Draft 1: The AI landscape is changing.")
        monitor.notify_agent_output(writer, "Draft 2: Revised with more clarity.")

        async for event in monitor:
            if event.event_type == MonitorEventType.OUTPUT_READY:
                output_events.append(event)
                print(
                    f"  OUTPUT_READY from {event.agent_id}: "
                    f"{str(event.data.get('output', ''))[:60]}"
                )
            if len(output_events) >= 2:
                break


# ── Example 3: Intervention — change context and rerun ───────────────────────
#
# When the QA observer decides a draft is poor, it calls monitor.intervene().
# The intervention is recorded. If max_interventions is set, further calls
# raise MaxInterventionsExceeded.


async def example_intervention() -> None:
    print("\n── Example 3: Intervention on poor quality output ───────────────")

    writer = WriterAgent()
    interventions_done: list[str] = []

    async with MonitorLoop(
        targets=[writer],
        poll_interval=10.0,
        max_interventions=3,  # cap at 3 interventions before escalation
    ) as monitor:
        # Simulate poor output
        monitor.notify_agent_output(writer, "Bad draft: vague, unclear, too short.")

        async for event in monitor:
            if event.event_type == MonitorEventType.OUTPUT_READY:
                output = str(event.data.get("output", ""))
                print(f"  Received output: {output[:60]}")

                # QA assessment: bad draft — intervene
                if "bad" in output.lower() or "vague" in output.lower():
                    await monitor.intervene(
                        writer,
                        InterventionAction.CHANGE_CONTEXT_AND_RERUN,
                        context="Be more specific. Focus on enterprise AI adoption.",
                    )
                    interventions_done.append(event.agent_id)
                    print(
                        f"  Intervened on {event.agent_id}: "
                        f"action={InterventionAction.CHANGE_CONTEXT_AND_RERUN}"
                    )
                break

    print(f"\n  Interventions recorded: {len(interventions_done)}")


# ── Example 4: MaxInterventionsExceeded + AGENT_ESCALATION ───────────────────
#
# When the intervention count reaches max_interventions, the next intervene()
# call raises MaxInterventionsExceeded and fires Hook.AGENT_ESCALATION.


async def example_max_interventions_exceeded() -> None:
    print("\n── Example 4: MaxInterventionsExceeded and AGENT_ESCALATION ────")

    writer = WriterAgent()
    escalation_events: list[dict[str, object]] = []

    def fire_fn(hook: Hook, data: dict[str, object]) -> None:
        if hook == Hook.AGENT_ESCALATION:
            escalation_events.append(data)

    async with MonitorLoop(
        targets=[writer],
        poll_interval=10.0,
        max_interventions=2,  # only 2 allowed
        fire_event_fn=fire_fn,
    ) as monitor:
        # Use up both intervention slots
        await monitor.intervene(writer, InterventionAction.PAUSE_AND_WAIT)
        await monitor.intervene(writer, InterventionAction.PAUSE_AND_WAIT)

        # Third intervention exceeds limit
        try:
            await monitor.intervene(writer, InterventionAction.CHANGE_CONTEXT_AND_RERUN)
            print("  ERROR: should have raised MaxInterventionsExceeded")
        except MaxInterventionsExceeded as e:
            print(f"  MaxInterventionsExceeded: limit={e.limit}  count={e.count}")

    if escalation_events:
        evt = escalation_events[0]
        print(
            f"  AGENT_ESCALATION hook fired:"
            f"\n    agent_id:           {evt.get('agent_id')}"
            f"\n    intervention_count: {evt.get('intervention_count')}"
            f"\n    max_interventions:  {evt.get('max_interventions')}"
        )


# ── Example 5: release() — stop monitoring an agent mid-run ──────────────────


async def example_release() -> None:
    print("\n── Example 5: monitor.release() — stop monitoring an agent ─────")

    writer = WriterAgent()
    heartbeats_before: int = 0
    heartbeats_after: int = 0
    released = False

    async with MonitorLoop(
        targets=[writer],
        poll_interval=0.05,
    ) as monitor:
        async for event in monitor:
            if event.event_type == MonitorEventType.HEARTBEAT:
                if not released:
                    heartbeats_before += 1
                    if heartbeats_before >= 2:
                        monitor.release(writer)  # pass agent object
                        released = True
                        # Drain any already-queued event then stop
                        break
                else:
                    heartbeats_after += 1

    print(f"  Heartbeats before release: {heartbeats_before}")
    print(f"  Heartbeats after release:  {heartbeats_after} (may be 0 or 1 from queue)")


# ── Main ──────────────────────────────────────────────────────────────────────


async def main() -> None:
    await example_heartbeat()
    await example_output_ready()
    await example_intervention()
    await example_max_interventions_exceeded()
    await example_release()
    print("\nAll MonitorLoop examples completed.")


if __name__ == "__main__":
    asyncio.run(main())
