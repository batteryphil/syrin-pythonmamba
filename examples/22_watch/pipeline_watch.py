"""Agent Watch — Trigger an Agent (or Workflow) via cron schedule.

Demonstrates:
- agent.watch(protocol=CronProtocol(...)) — Agent reacts to scheduled events
- watch_handler() routes each trigger through the agent
- on_trigger / on_result / on_error callbacks at the agent level
- Workflow.watch() pattern note for multi-step pipelines

Run:
    python examples/22_watch/pipeline_watch.py

The agent fires once immediately (run_on_start=True) via CronProtocol.
"""

from __future__ import annotations

import asyncio
import contextlib
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from syrin import Agent, Model  # noqa: E402
from syrin.watch import CronProtocol, TriggerEvent  # noqa: E402

# ---------------------------------------------------------------------------
# Agent used for the cron demo
# ---------------------------------------------------------------------------


class ResearchAgent(Agent):
    name = "researcher"
    model = Model.mock(latency_min=1, latency_max=3, lorem_length=800, pricing_tier="high")
    system_prompt = "You research topics and return key findings in 1-2 sentences."


# ---------------------------------------------------------------------------
# Demo 1: Agent.watch() with CronProtocol
# ---------------------------------------------------------------------------


async def demo_agent_cron() -> None:
    print("=" * 60)
    print("Agent.watch() with CronProtocol")
    print("=" * 60)

    agent = ResearchAgent()

    run_count = 0
    stop_event = asyncio.Event()

    def on_trigger(event: TriggerEvent) -> None:
        print(f"  → agent trigger: {event.input!r}")

    def on_result(event: TriggerEvent, result: object) -> None:
        nonlocal run_count
        run_count += 1
        content = getattr(result, "content", str(result))
        cost = getattr(result, "cost", 0.0)
        print(f"  [{run_count}] result (${cost:.4f}): {content[:120]}")
        stop_event.set()  # Stop after first successful result

    def on_error(event: TriggerEvent, exc: Exception) -> None:
        print(f"  ✗ error: {exc}")
        stop_event.set()

    # CronProtocol: run_on_start=True fires once immediately without waiting
    protocol = CronProtocol(
        schedule="* * * * *",
        input="Summarize the latest AI developments",
        timezone="UTC",
        run_on_start=True,
    )

    # agent.watch() — Agent inherits Watchable
    agent.watch(
        protocol=protocol,
        on_trigger=on_trigger,
        on_result=on_result,
        on_error=on_error,
    )

    handler = agent.watch_handler(
        concurrency=1,
        timeout=30.0,
        on_result=on_result,
        on_error=on_error,
    )

    proto_task = asyncio.create_task(protocol.start(handler))
    await stop_event.wait()
    await protocol.stop()
    proto_task.cancel()
    with contextlib.suppress(asyncio.CancelledError, Exception):
        await proto_task

    print(f"\n  Agent processed {run_count} run(s)\n")


# ---------------------------------------------------------------------------
# Demo 2: Workflow.watch() usage pattern note
# ---------------------------------------------------------------------------


def demo_workflow_watch_note() -> None:
    print("=" * 60)
    print("Workflow.watch() (usage pattern)")
    print("=" * 60)
    print("""
For multi-step agent chains, use Workflow instead of a single Agent:

    from syrin.workflow import Workflow
    from syrin.watch import WebhookProtocol

    wf = (
        Workflow("research-pipeline")
        .step(ResearchAgent, "Research the topic")
        .step(SummaryAgent, "Summarize the findings")
    )

    protocol = WebhookProtocol(
        path="/workflow/trigger",
        port=9090,
        input_field="task",
        secret="my-hmac-secret",   # HMAC validation — rejects tampered POSTs
    )

    wf.watch(
        protocol=protocol,
        concurrency=3,
        timeout=120.0,
        on_trigger=lambda e: print(f"Triggered: {e.input[:60]}"),
        on_result=lambda e, r: print(f"Done: {getattr(r, 'content', '')[:80]}"),
        on_error=lambda e, exc: print(f"Error: {exc}"),
    )

    handler = wf.watch_handler(concurrency=3, timeout=120.0)
    await protocol.start(handler)

Test with curl:
    curl -X POST http://localhost:9090/workflow/trigger \\
        -H "Content-Type: application/json" \\
        -d '{"task": "Research and summarize AI chip trends"}'

AgentRouter also inherits Watchable for LLM-driven dynamic routing:

    from syrin import AgentRouter
    from syrin.watch import QueueProtocol

    router = AgentRouter(agents=[ResearchAgent, SummaryAgent], model=model)
    router.watch(protocol=QueueProtocol(...), on_result=on_result)
""")


async def main() -> None:
    await demo_agent_cron()
    demo_workflow_watch_note()

    print("=" * 60)
    print("Agent, Workflow, and AgentRouter all inherit Watchable.")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
