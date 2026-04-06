"""Agent broadcast — pub-sub messaging between agents.

The BroadcastBus enables topic-based publish-subscribe communication
inside a swarm. Any agent can broadcast on a topic; any agent that has
subscribed to a matching pattern receives the message.

Key concepts:
  - BroadcastBus(config=BroadcastConfig(...))
  - bus.subscribe(agent_id, topic, handler)   — exact topic or glob pattern
  - await bus.broadcast(sender_id, topic, payload)
  - "research.*" wildcard — matches any subtopic under "research"
  - "*" global wildcard — receives every broadcast
  - BroadcastEvent.sender_id, .topic, .payload

Run:
    uv run python examples/agent_broadcast.py
"""

from __future__ import annotations

import asyncio

from syrin.enums import Hook
from syrin.swarm import (
    BroadcastBus,
    BroadcastConfig,
    BroadcastEvent,
    BroadcastPayloadTooLarge,
)

# ── Example 1: Basic pub-sub between three agents ─────────────────────────────
#
# ResearchAgent broadcasts its findings on "research.complete".
# SummaryAgent and WriterAgent both subscribe to that exact topic.


async def example_basic_pubsub() -> None:
    print("\n── Example 1: Basic pub-sub (exact topic) ───────────────────────")

    bus = BroadcastBus()

    received_summary: list[BroadcastEvent] = []
    received_writer: list[BroadcastEvent] = []

    # SummaryAgent subscribes to exact topic "research.complete"
    bus.subscribe(
        agent_id="summary-agent",
        topic="research.complete",
        handler=lambda evt: received_summary.append(evt),
    )

    # WriterAgent subscribes to the same exact topic
    bus.subscribe(
        agent_id="writer-agent",
        topic="research.complete",
        handler=lambda evt: received_writer.append(evt),
    )

    # ResearchAgent broadcasts its findings
    findings = {
        "topic": "AI agent market",
        "summary": "340% YoY growth confirmed",
        "sources": 14,
        "confidence": "high",
    }
    subscriber_count = await bus.broadcast(
        sender="research-agent",
        topic="research.complete",
        payload=findings,
    )

    print(f"  Broadcast delivered to {subscriber_count} subscribers")
    print(f"  SummaryAgent received: {received_summary[0].payload['summary']}")
    print(f"  WriterAgent received:  {received_writer[0].payload['summary']}")
    print(f"  Sender:                {received_summary[0].sender_id}")
    print(f"  Topic:                 {received_summary[0].topic}")


# ── Example 2: Wildcard subscription "research.*" ────────────────────────────
#
# A downstream agent subscribes to "research.*" and receives any message
# published under the "research." prefix (research.started, research.complete,
# research.error, etc.).


async def example_wildcard_subtopic() -> None:
    print("\n── Example 2: Wildcard 'research.*' subscription ────────────────")

    bus = BroadcastBus()
    all_research_events: list[BroadcastEvent] = []

    # Monitor agent subscribes to all research-related topics
    bus.subscribe(
        agent_id="monitor-agent",
        topic="research.*",
        handler=lambda evt: all_research_events.append(evt),
    )

    # Also subscribe a specific handler for research.error only
    error_events: list[BroadcastEvent] = []
    bus.subscribe(
        agent_id="alert-agent",
        topic="research.error",
        handler=lambda evt: error_events.append(evt),
    )

    # Broadcast multiple research lifecycle events
    for topic, payload in [
        ("research.started", {"query": "AI market trends", "agent": "research-agent-1"}),
        ("research.progress", {"pct": 45, "sources_found": 6}),
        ("research.complete", {"summary": "Growth confirmed", "sources": 14}),
        ("research.error", {"reason": "Source timeout", "retryable": True}),
    ]:
        await bus.broadcast("research-agent-1", topic, payload)

    print(f"  Monitor (research.*) received {len(all_research_events)} events:")
    for evt in all_research_events:
        print(f"    [{evt.topic}]")

    print(f"\n  Alert agent (research.error only) received: {len(error_events)} events")


# ── Example 3: Global wildcard "*" ────────────────────────────────────────────
#
# An audit agent subscribes to "*" and receives every broadcast from every agent.


async def example_global_wildcard() -> None:
    print("\n── Example 3: Global wildcard '*' subscription ──────────────────")

    bus = BroadcastBus()
    all_events: list[BroadcastEvent] = []

    # Audit agent receives everything
    bus.subscribe(
        agent_id="audit-agent",
        topic="*",
        handler=lambda evt: all_events.append(evt),
    )

    # Various agents broadcast on different topics
    await bus.broadcast("agent-research", "research.complete", {"status": "done"})
    await bus.broadcast("agent-analyst", "analysis.ready", {"confidence": 0.95})
    await bus.broadcast("agent-writer", "draft.submitted", {"word_count": 450})

    print(f"  Audit agent captured {len(all_events)} messages:")
    for evt in all_events:
        print(f"    sender={evt.sender_id:<20} topic={evt.topic}")


# ── Example 4: BroadcastConfig limits ─────────────────────────────────────────
#
# BroadcastConfig.max_payload_bytes limits payload size.
# BroadcastPayloadTooLarge is raised if the limit is exceeded.


async def example_payload_size_limit() -> None:
    print("\n── Example 4: Payload size limit ────────────────────────────────")

    config = BroadcastConfig(max_payload_bytes=256)
    bus = BroadcastBus(config=config)

    bus.subscribe("consumer-agent", "*", lambda _evt: None)

    # Small payload — OK
    small_payload = {"data": "hello"}
    count = await bus.broadcast("sender-agent", "topic.a", small_payload)
    print(f"  Small payload: delivered to {count} subscriber(s)")

    # Large payload — raises BroadcastPayloadTooLarge
    large_payload = {"data": "x" * 1000}
    try:
        await bus.broadcast("sender-agent", "topic.b", large_payload)
    except BroadcastPayloadTooLarge as e:
        print(f"  BroadcastPayloadTooLarge: size={e.size_bytes} bytes  limit={e.max_bytes} bytes")


# ── Example 5: Lifecycle hooks (AGENT_BROADCAST) ──────────────────────────────


async def example_broadcast_hooks() -> None:
    print("\n── Example 5: Hook.AGENT_BROADCAST lifecycle event ─────────────")

    broadcast_events: list[dict[str, object]] = []

    def fire_fn(hook: Hook, data: dict[str, object]) -> None:
        if hook == Hook.AGENT_BROADCAST:
            broadcast_events.append(data)

    bus = BroadcastBus(fire_event_fn=fire_fn)
    bus.subscribe("receiver-1", "updates.*", lambda _evt: None)
    bus.subscribe("receiver-2", "updates.*", lambda _evt: None)

    await bus.broadcast("sender-1", "updates.daily", {"items": 42})

    if broadcast_events:
        evt = broadcast_events[0]
        print("  Hook.AGENT_BROADCAST fired:")
        print(f"    sender_id:        {evt.get('sender_id')}")
        print(f"    topic:            {evt.get('topic')}")
        print(f"    subscriber_count: {evt.get('subscriber_count')}")
        print(f"    payload_size:     {evt.get('payload_size')} bytes")


# ── Main ──────────────────────────────────────────────────────────────────────


async def main() -> None:
    await example_basic_pubsub()
    await example_wildcard_subtopic()
    await example_global_wildcard()
    await example_payload_size_limit()
    await example_broadcast_hooks()
    print("\nAll broadcast pub-sub examples completed.")


if __name__ == "__main__":
    asyncio.run(main())
