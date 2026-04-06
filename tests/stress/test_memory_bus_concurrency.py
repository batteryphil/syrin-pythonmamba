"""MemoryBus concurrency stress tests."""

from __future__ import annotations

import asyncio

import pytest

from syrin.enums import MemoryType
from syrin.memory.config import MemoryEntry
from syrin.swarm._memory_bus import MemoryBus


def _make_entry(i: int) -> MemoryEntry:
    """Create a test MemoryEntry."""
    return MemoryEntry(
        id=f"entry-{i}",
        content=f"Entry {i}: synthetic memory content for agent {i}",
        type=MemoryType.KNOWLEDGE,
        importance=0.8,
    )


@pytest.mark.asyncio
async def test_20_agents_concurrent_publish() -> None:
    """20 agents publishing to MemoryBus concurrently — no dropped writes, no deadlock."""
    bus = MemoryBus()
    n = 20
    entries = [_make_entry(i) for i in range(n)]

    async def publish_one(idx: int) -> bool:
        return await bus.publish(entries[idx], agent_id=f"agent-{idx}")

    results = await asyncio.gather(*[publish_one(i) for i in range(n)])
    assert all(results), "Some entries were dropped"

    # All entries should be readable
    all_entries = await bus.read(query="", agent_id="reader")
    assert len(all_entries) == n


@pytest.mark.asyncio
async def test_memory_bus_ttl_under_load() -> None:
    """Entries expire correctly even with concurrent publish/read."""
    bus = MemoryBus(ttl=0.05)  # 50ms TTL
    entries = [_make_entry(i) for i in range(10)]

    # Publish all
    await asyncio.gather(*[bus.publish(entries[i], agent_id=f"a-{i}") for i in range(10)])

    # Wait for TTL expiry
    await asyncio.sleep(0.1)

    # Force expiry sweep
    expired = await bus.expire_now()
    assert len(expired) == 10, f"Expected 10 expired entries, got {len(expired)}"

    # No entries remaining
    remaining = await bus.read(query="", agent_id="reader")
    assert len(remaining) == 0
