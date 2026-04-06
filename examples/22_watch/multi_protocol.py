"""Multiple Protocols — Agent watches cron + webhook simultaneously.

Demonstrates:
- agent.watch() registering multiple protocols
- All protocols share the same agent instance and budget
- Custom WatchProtocol: implement your own trigger source
- Protocol.start(handler) to run each protocol concurrently

Run:
    python examples/22_watch/multi_protocol.py

Then also POST to http://localhost:9090/trigger while cron fires every minute.
"""

from __future__ import annotations

import asyncio
import signal
import sys
import uuid
from collections.abc import Awaitable, Callable
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from syrin import Agent, Model  # noqa: E402
from syrin.watch import CronProtocol, TriggerEvent, WebhookProtocol  # noqa: E402

# ---------------------------------------------------------------------------
# Custom protocol — fires once on startup then idles
# ---------------------------------------------------------------------------


class StartupProtocol:
    """Fires once immediately when started; useful for warm-up runs."""

    def __init__(self, input: str = "Warm-up run") -> None:  # noqa: A002
        self.input = input
        self._stopped = False

    async def start(
        self,
        handler: Callable[[TriggerEvent], Awaitable[None]],
    ) -> None:
        event = TriggerEvent(
            input=self.input,
            source="startup",
            metadata={"reason": "process_start"},
            trigger_id=str(uuid.uuid4()),
        )
        await handler(event)
        while not self._stopped:
            await asyncio.sleep(0.5)

    async def stop(self) -> None:
        self._stopped = True


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------


class OpsAgent(Agent):
    name = "ops_agent"
    model = Model.mock(latency_min=1, latency_max=3, lorem_length=800, pricing_tier="high")
    system_prompt = "You are an operations agent. Respond with a brief status update."


async def main() -> None:
    agent = OpsAgent()

    def on_trigger(event: TriggerEvent) -> None:
        print(f"\n[{event.source}] trigger: {event.input[:60]}")

    def on_result(event: TriggerEvent, result: object) -> None:
        content = getattr(result, "content", "")
        print(f"  → {content[:120]}")

    protocols: list[StartupProtocol | CronProtocol | WebhookProtocol] = [
        StartupProtocol(input="System startup — initial health check"),
        CronProtocol(schedule="* * * * *", input="Periodic health check", run_on_start=False),
        WebhookProtocol(path="/trigger", port=9090, input_field="message"),
    ]

    # Register all protocols via watch()
    agent.watch(
        protocols=protocols,  # type: ignore[arg-type]
        concurrency=2,
        on_trigger=on_trigger,
        on_result=on_result,
    )

    # Get shared dispatch handler
    handler = agent.watch_handler(concurrency=2, on_result=on_result)

    print("Agent watching 3 protocols simultaneously:")
    print("  • Startup: fires once immediately")
    print("  • Cron:    every minute")
    print("  • Webhook: POST http://localhost:9090/trigger")
    print("Press Ctrl+C to stop.\n")

    stop_event = asyncio.Event()
    signal.signal(signal.SIGINT, lambda *_: stop_event.set())

    # Start all protocols concurrently
    tasks = [asyncio.create_task(p.start(handler)) for p in protocols]

    await stop_event.wait()

    for p in protocols:
        await p.stop()
    for task in tasks:
        task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)

    print("\nAll protocols stopped.")


if __name__ == "__main__":
    asyncio.run(main())
