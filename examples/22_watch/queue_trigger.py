"""Queue Trigger — Consume agent tasks from a Redis queue.

Demonstrates:
- QueueProtocol: consume messages from Redis (BLPOP) and fire agent per message
- QueueBackend Protocol: implement a custom in-memory backend (no Redis needed)
- ack_on_success / nack_on_error for reliable message processing
- agent.watch(protocol=QueueProtocol(...)) — trigger an Agent from a queue
- Hook.WATCH_TRIGGER to observe each consumed message

Run (uses the in-memory backend — no Redis required):
    python examples/22_watch/queue_trigger.py

With a real Redis server:
    python examples/22_watch/queue_trigger.py --redis redis://localhost:6379/0

Requires redis for the real-Redis demo:
    pip install redis
"""

from __future__ import annotations

import asyncio
import sys
from collections.abc import AsyncIterator
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from syrin import Agent, Model  # noqa: E402
from syrin.enums import Hook  # noqa: E402
from syrin.watch import QueueProtocol, TriggerEvent  # noqa: E402
from syrin.watch._queue import QueueBackend  # noqa: E402

# ---------------------------------------------------------------------------
# In-memory queue backend — no Redis required for this demo
# ---------------------------------------------------------------------------


class InMemoryQueueBackend:
    """Simple in-memory queue for demo/testing.

    In production, replace with Redis, SQS, RabbitMQ, or any QueueBackend impl.
    """

    def __init__(self, messages: list[str]) -> None:
        self._queue: asyncio.Queue[str] = asyncio.Queue()
        self._messages = messages
        self._running = False

    async def connect(self) -> None:
        self._running = True
        # Pre-load messages into the queue
        for msg in self._messages:
            await self._queue.put(msg)
        print(f"  [queue] connected — {len(self._messages)} message(s) pre-loaded")

    async def disconnect(self) -> None:
        self._running = False
        print("  [queue] disconnected")

    async def receive(self) -> AsyncIterator[tuple[str, object]]:
        while self._running:
            try:
                text = await asyncio.wait_for(self._queue.get(), timeout=0.5)
                yield text, text
            except TimeoutError:
                if self._queue.empty() and self._running:
                    # No more messages — signal done
                    self._running = False
                    break

    async def ack(self, message_id: object) -> None:
        print(f"  [queue] ✓ ack: {str(message_id)[:60]}")

    async def nack(self, message_id: object) -> None:
        print(f"  [queue] ✗ nack (will retry): {str(message_id)[:60]}")


# Confirm InMemoryQueueBackend satisfies the QueueBackend Protocol
assert isinstance(InMemoryQueueBackend([]), QueueBackend)


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------


class TaskAgent(Agent):
    name = "task_agent"
    description = "Processes queued tasks"
    model = Model.mock(latency_min=1, latency_max=3, lorem_length=800, pricing_tier="high")
    system_prompt = (
        "You are an operations task processor. "
        "Respond concisely with a 1-sentence status for each task."
    )


# ---------------------------------------------------------------------------
# Demo 1: Agent consuming from an in-memory queue
# ---------------------------------------------------------------------------


async def demo_agent_queue() -> None:
    print("=" * 60)
    print("Demo 1 — Agent consuming from in-memory queue")
    print("=" * 60)

    agent = TaskAgent()

    # Observe triggers via hook
    def on_watch_trigger(ctx: object) -> None:
        src = getattr(ctx, "source", "?")
        print(f"\n  [Hook.WATCH_TRIGGER] source={src}")

    agent.events.on(Hook.WATCH_TRIGGER, on_watch_trigger)

    results: list[str] = []

    def on_trigger(event: TriggerEvent) -> None:
        print(f"  → consuming: {event.input!r}")

    def on_result(event: TriggerEvent, result: object) -> None:
        content = getattr(result, "content", str(result))
        cost = getattr(result, "cost", 0.0)
        results.append(content)
        print(f"     processed (${cost:.4f}): {content[:100]}")

    def on_error(event: TriggerEvent, exc: Exception) -> None:
        print(f"     ✗ error: {exc}")

    tasks_to_process = [
        "Run daily database backup",
        "Send weekly usage digest to admin@company.com",
        "Rotate API keys for production services",
    ]

    backend = InMemoryQueueBackend(tasks_to_process)
    protocol = QueueProtocol(
        source=backend,  # Custom backend — pass instance directly
        queue="agent_tasks",
        concurrency=2,  # Process up to 2 messages simultaneously
        ack_on_success=True,
        nack_on_error=True,
    )

    agent.watch(
        protocol=protocol,
        on_trigger=on_trigger,
        on_result=on_result,
        on_error=on_error,
    )

    handler = agent.watch_handler(
        concurrency=2,
        on_result=on_result,
        on_error=on_error,
    )

    # Run until queue is drained (backend sets _running=False when empty)
    await protocol.start(handler)

    print(f"\n  Processed {len(results)}/{len(tasks_to_process)} tasks\n")


# ---------------------------------------------------------------------------
# Demo 2: Note on real Redis usage
# ---------------------------------------------------------------------------


def demo_redis_note(redis_url: str | None) -> None:
    print("=" * 60)
    print("Demo 2 — Real Redis usage")
    print("=" * 60)

    if redis_url:
        print(f"  Would connect to: {redis_url}")
        print("  (skipped in this demo — no live agent call)")
    else:
        print("""
To use a real Redis queue:

    from syrin.watch import QueueProtocol

    protocol = QueueProtocol(
        source="redis://localhost:6379/0",  # URL string → Redis backend
        queue="agent_tasks",
        concurrency=3,
        ack_on_success=True,
        nack_on_error=True,
    )

    agent.watch(protocol=protocol, on_result=on_result)
    handler = agent.watch_handler(concurrency=3, on_result=on_result)
    await protocol.start(handler)

Push messages to the queue:
    redis-cli LPUSH agent_tasks "Summarize today's AI news"
    redis-cli LPUSH agent_tasks "Run weekly cost report"

For SQS, RabbitMQ, or any other backend, implement QueueBackend protocol:

    class SQSBackend:
        async def connect(self) -> None: ...
        async def disconnect(self) -> None: ...
        async def receive(self) -> AsyncIterator[tuple[str, object]]:
            async for msg in poll_sqs():
                yield msg["Body"], msg["ReceiptHandle"]
        async def ack(self, handle: object) -> None:
            await sqs.delete_message(ReceiptHandle=handle)
        async def nack(self, handle: object) -> None:
            pass  # SQS visibility timeout handles redelivery

    protocol = QueueProtocol(source=SQSBackend(), queue="my-queue")
""")


async def main() -> None:
    redis_url = None
    if "--redis" in sys.argv:
        idx = sys.argv.index("--redis")
        if idx + 1 < len(sys.argv):
            redis_url = sys.argv[idx + 1]

    await demo_agent_queue()
    demo_redis_note(redis_url)


if __name__ == "__main__":
    asyncio.run(main())
