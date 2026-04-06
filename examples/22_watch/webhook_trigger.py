"""Webhook Trigger — Run agent on HTTP POST requests.

Demonstrates:
- WebhookProtocol: start HTTP server, trigger agent on POST
- HMAC signature validation (GitHub webhook-compatible)
- input_field: extract agent input from a JSON field
- concurrency=3: process up to 3 webhooks simultaneously
- agent.trigger(): one-shot fire from existing framework code

Run (starts server on port 8080):
    python examples/22_watch/webhook_trigger.py

Test with curl (no auth):
    curl -X POST http://localhost:8080/trigger \\
        -H "Content-Type: application/json" \\
        -d '{"message": "Summarize the latest AI news"}'

One-shot (no server):
    python examples/22_watch/webhook_trigger.py --one-shot

Requires: pip install aiohttp  (for production HTTP server; falls back to stdlib)
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
from syrin.watch import TriggerEvent, WebhookProtocol  # noqa: E402


class WebhookAgent(Agent):
    name = "webhook_agent"
    description = "Processes webhook payloads"
    model = Model.mock(latency_min=1, latency_max=3, lorem_length=800, pricing_tier="high")
    system_prompt = "You process incoming webhook requests and provide concise responses."


async def main() -> None:
    agent = WebhookAgent()

    def on_trigger(event: TriggerEvent) -> None:
        print(f"\n→ Webhook [{event.trigger_id[:8]}]: {event.input[:80]}")

    def on_result(event: TriggerEvent, result: object) -> None:
        content = getattr(result, "content", str(result))
        cost = getattr(result, "cost", 0.0)
        print(f"  ✓ Response (${cost:.4f}): {content[:200]}")

    def on_error(event: TriggerEvent, exc: Exception) -> None:
        print(f"  ✗ Error: {exc}")

    protocol = WebhookProtocol(
        path="/trigger",
        port=8080,
        secret=None,  # Set to "my-secret" to require HMAC validation
        input_field="message",  # Extract from {"message": "..."} — None = whole payload
    )

    # Register protocol and callbacks
    agent.watch(
        protocol=protocol,
        concurrency=3,
        timeout=30.0,
        on_trigger=on_trigger,
        on_result=on_result,
        on_error=on_error,
    )

    # Get dispatch handler, then start the protocol
    handler = agent.watch_handler(
        concurrency=3, timeout=30.0, on_result=on_result, on_error=on_error
    )

    print("Webhook server starting on http://localhost:8080/trigger")
    print('POST JSON with {"message": "..."} to trigger the agent.')
    print("Press Ctrl+C to stop.\n")

    stop_event = asyncio.Event()
    signal.signal(signal.SIGINT, lambda *_: stop_event.set())

    proto_task = asyncio.create_task(protocol.start(handler))

    await stop_event.wait()
    await protocol.stop()
    proto_task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await proto_task

    print("Stopped.")


async def demo_one_shot() -> None:
    """Use agent.trigger() — no protocol, no server, just one call."""
    agent = WebhookAgent()

    result = await agent.trigger(
        input="Summarize recent trends in AI infrastructure",
        source="my-app",
        metadata={"user_id": "u_123"},
    )
    print(f"One-shot result: {result.content[:200]}")
    print(f"Cost: ${result.cost:.4f}")


if __name__ == "__main__":
    if "--one-shot" in sys.argv:
        asyncio.run(demo_one_shot())
    else:
        asyncio.run(main())
