"""Cron Trigger — Run agent on a schedule.

Demonstrates:
- CronProtocol: fire agent on a POSIX cron schedule
- run_on_start=True: run once immediately when the protocol starts
- on_trigger / on_result / on_error callbacks
- Hook.WATCH_TRIGGER to observe each trigger via agent hooks
- Clean shutdown with Ctrl+C

Run (fires immediately then every minute):
    python examples/22_watch/cron_trigger.py
"""

from __future__ import annotations

import asyncio
import contextlib
import signal
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from syrin import Agent, Model  # noqa: E402
from syrin.enums import Hook  # noqa: E402
from syrin.watch import CronProtocol, TriggerEvent  # noqa: E402


class DailyReportAgent(Agent):
    name = "daily_reporter"
    description = "Generates a brief daily status report"
    model = Model.mock(latency_min=1, latency_max=3, lorem_length=800, pricing_tier="high")
    system_prompt = "You generate concise daily status reports. Keep each report to 2-3 sentences."


async def main() -> None:
    agent = DailyReportAgent()

    # Observe each trigger via agent hooks
    def on_watch_trigger(ctx: object) -> None:
        import datetime

        ts = datetime.datetime.now().strftime("%H:%M:%S")
        print(f"\n[{ts}] WATCH_TRIGGER — source={getattr(ctx, 'source', '?')}")

    agent.events.on(Hook.WATCH_TRIGGER, on_watch_trigger)

    # Callbacks called by the watch_handler dispatch
    def on_trigger(event: TriggerEvent) -> None:
        print(f"  trigger_id={event.trigger_id[:8]}  input={event.input!r}")

    def on_result(event: TriggerEvent, result: object) -> None:
        content = getattr(result, "content", str(result))
        cost = getattr(result, "cost", 0.0)
        print(f"  → result (${cost:.4f}): {content[:120]}")

    def on_error(event: TriggerEvent, exc: Exception) -> None:
        print(f"  ✗ error: {exc}")

    # 1. Register protocol + callbacks via watch()
    protocol = CronProtocol(
        schedule="* * * * *",  # Every minute
        input="Generate the hourly status report",
        timezone="UTC",
        run_on_start=True,  # Fire once right now too
    )

    agent.watch(
        protocol=protocol,
        on_trigger=on_trigger,
        on_result=on_result,
        on_error=on_error,
    )

    # 2. Get the dispatch handler and start the protocol
    handler = agent.watch_handler(concurrency=1, timeout=30.0)

    print("Cron agent starting — fires immediately then every minute.")
    print("Press Ctrl+C to stop.\n")

    stop_event = asyncio.Event()
    signal.signal(signal.SIGINT, lambda *_: stop_event.set())

    # 3. Run protocol — blocks until stopped
    proto_task = asyncio.create_task(protocol.start(handler))

    await stop_event.wait()
    await protocol.stop()
    proto_task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await proto_task

    print("Done.")


if __name__ == "__main__":
    asyncio.run(main())
