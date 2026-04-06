"""Tests for MemoryBus — P3-T1."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from unittest.mock import patch

import pytest

from syrin.enums import Hook, MemoryType
from syrin.events import EventContext, Events
from syrin.memory.config import MemoryEntry
from syrin.swarm._memory_bus import MemoryBus

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_entry(
    content: str = "test content",
    memory_type: MemoryType = MemoryType.KNOWLEDGE,
    importance: float = 0.8,
    keywords: list[str] | None = None,
) -> MemoryEntry:
    """Create a minimal MemoryEntry for testing."""
    import uuid
    from datetime import datetime

    return MemoryEntry(
        id=str(uuid.uuid4()),
        content=content,
        type=memory_type,
        importance=importance,
        keywords=keywords or [],
        created_at=datetime.now(),
    )


def _events_with_capture(captured: list[tuple[Hook, EventContext]]) -> Events:
    """Return an Events instance that records all hooks via on() handlers."""
    events = Events(lambda _h, _c: None)

    for hook in Hook:
        # Capture hook via closure
        def _make_handler(h: Hook) -> Callable[[EventContext], None]:
            def _handler(ctx: EventContext) -> None:
                captured.append((h, ctx))

            return _handler

        events.on(hook, _make_handler(hook))

    return events


# ---------------------------------------------------------------------------
# P3-T1-1: allow_types filter (type rejection)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_memory_bus_rejects_disallowed_type() -> None:
    """MemoryBus with allow_types=[SEMANTIC] silently rejects EPISODIC entries."""
    captured: list[tuple[Hook, EventContext]] = []
    events = _events_with_capture(captured)

    bus = MemoryBus(allow_types=[MemoryType.KNOWLEDGE], swarm_events=events)
    entry = _make_entry(memory_type=MemoryType.HISTORY)

    stored = await bus.publish(entry, agent_id="a1")

    assert stored is False, "Rejected entry should return False"
    results = await bus.read(query="test", agent_id="a2")
    assert results == [], "No entries should be readable after rejection"


@pytest.mark.asyncio
async def test_memory_bus_rejects_disallowed_fires_filtered_hook() -> None:
    """MEMORY_BUS_FILTERED hook fires when an entry is rejected by allow_types."""
    captured: list[tuple[Hook, EventContext]] = []
    events = _events_with_capture(captured)

    bus = MemoryBus(allow_types=[MemoryType.KNOWLEDGE], swarm_events=events)
    entry = _make_entry(memory_type=MemoryType.HISTORY)

    await bus.publish(entry, agent_id="a1")

    hooks_fired = [h for h, _ in captured]
    assert Hook.MEMORY_BUS_FILTERED in hooks_fired, "MEMORY_BUS_FILTERED should fire"
    filtered_ctx = next(ctx for h, ctx in captured if h == Hook.MEMORY_BUS_FILTERED)
    assert "filter_reason" in filtered_ctx


# ---------------------------------------------------------------------------
# P3-T1-2: publish stores entry and fires PUBLISHED hook
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_memory_bus_publish_stores_and_fires_hook() -> None:
    """bus.publish(entry, agent_id) stores the entry and fires MEMORY_BUS_PUBLISHED."""
    captured: list[tuple[Hook, EventContext]] = []
    events = _events_with_capture(captured)

    bus = MemoryBus(swarm_events=events)
    entry = _make_entry(content="shared knowledge")

    stored = await bus.publish(entry, agent_id="a1")

    assert stored is True, "Accepted entry should return True"
    hooks_fired = [h for h, _ in captured]
    assert Hook.MEMORY_BUS_PUBLISHED in hooks_fired, "MEMORY_BUS_PUBLISHED should fire"


# ---------------------------------------------------------------------------
# P3-T1-3: read returns matching entries
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_memory_bus_read_returns_matching_entries() -> None:
    """bus.read(query) returns entries whose content matches the query substring."""
    bus = MemoryBus()
    entry_ai = _make_entry(content="ai safety is important")
    entry_other = _make_entry(content="dogs are cute")

    await bus.publish(entry_ai, agent_id="a1")
    await bus.publish(entry_other, agent_id="a1")

    results = await bus.read(query="ai", agent_id="a2")
    contents = [e.content for e in results]

    assert "ai safety is important" in contents
    assert "dogs are cute" not in contents


@pytest.mark.asyncio
async def test_memory_bus_read_fires_hook() -> None:
    """MEMORY_BUS_READ hook fires when read() is called."""
    captured: list[tuple[Hook, EventContext]] = []
    events = _events_with_capture(captured)

    bus = MemoryBus(swarm_events=events)
    await bus.publish(_make_entry(content="ai topic"), agent_id="a1")
    await bus.read(query="ai", agent_id="a2")

    hooks_fired = [h for h, _ in captured]
    assert Hook.MEMORY_BUS_READ in hooks_fired


# ---------------------------------------------------------------------------
# P3-T1-4: custom filter blocks entries with matching keywords
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_memory_bus_custom_filter_blocks_private_entries() -> None:
    """filter=lambda e: 'private' not in e.keywords blocks private entries."""
    captured: list[tuple[Hook, EventContext]] = []
    events = _events_with_capture(captured)

    bus = MemoryBus(
        filter=lambda e: "private" not in e.keywords,
        swarm_events=events,
    )
    private_entry = _make_entry(content="secret info", keywords=["private"])
    public_entry = _make_entry(content="public knowledge")

    stored_private = await bus.publish(private_entry, agent_id="a1")
    stored_public = await bus.publish(public_entry, agent_id="a1")

    assert stored_private is False, "Private entry should be filtered"
    assert stored_public is True, "Public entry should pass"

    results = await bus.read(query="", agent_id="a2")
    assert all("secret" not in e.content for e in results)


@pytest.mark.asyncio
async def test_memory_bus_custom_filter_fires_filtered_hook_with_reason() -> None:
    """MEMORY_BUS_FILTERED fires with filter_reason when custom filter blocks entry."""
    captured: list[tuple[Hook, EventContext]] = []
    events = _events_with_capture(captured)

    bus = MemoryBus(
        filter=lambda e: "private" not in e.keywords,
        swarm_events=events,
    )
    private_entry = _make_entry(content="secret", keywords=["private"])
    await bus.publish(private_entry, agent_id="a1")

    hooks_fired = [h for h, _ in captured]
    assert Hook.MEMORY_BUS_FILTERED in hooks_fired
    filtered_ctx = next(ctx for h, ctx in captured if h == Hook.MEMORY_BUS_FILTERED)
    assert "filter_reason" in filtered_ctx


# ---------------------------------------------------------------------------
# P3-T1-5: TTL — entry disappears after expiry
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_memory_bus_ttl_entry_expires() -> None:
    """entry disappears from bus after ttl seconds; MEMORY_BUS_EXPIRED fires."""
    captured: list[tuple[Hook, EventContext]] = []
    events = _events_with_capture(captured)

    bus = MemoryBus(ttl=60, swarm_events=events)
    entry = _make_entry(content="ephemeral data")

    await bus.publish(entry, agent_id="a1")

    # Simulate time passing beyond TTL by patching time.time
    future_time = time.time() + 120  # 2 minutes later
    with patch("time.time", return_value=future_time):
        results = await bus.read(query="ephemeral", agent_id="a2")

    assert results == [], "Expired entry should not be returned"


@pytest.mark.asyncio
async def test_memory_bus_ttl_expired_hook_fires() -> None:
    """MEMORY_BUS_EXPIRED fires when expired entries are cleared."""
    captured: list[tuple[Hook, EventContext]] = []
    events = _events_with_capture(captured)

    bus = MemoryBus(ttl=60, swarm_events=events)
    entry = _make_entry(content="transient")

    await bus.publish(entry, agent_id="a1")

    future_time = time.time() + 120
    with patch("time.time", return_value=future_time):
        await bus.expire_now()

    hooks_fired = [h for h, _ in captured]
    assert Hook.MEMORY_BUS_EXPIRED in hooks_fired


# ---------------------------------------------------------------------------
# P3-T1-6: Concurrent writes — all 20 stored
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_memory_bus_concurrent_writes_all_stored() -> None:
    """20 concurrent publish() calls all succeed and all entries are stored."""
    bus = MemoryBus()
    entries = [_make_entry(content=f"entry {i}") for i in range(20)]

    results = await asyncio.gather(*[bus.publish(e, agent_id="a1") for e in entries])

    assert all(results), "All 20 publishes should succeed"
    all_entries = await bus.read(query="entry", agent_id="a2")
    assert len(all_entries) == 20, f"Expected 20 entries, got {len(all_entries)}"
