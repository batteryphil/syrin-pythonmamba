"""Tests for MemoryBus backends — P3-T2."""

from __future__ import annotations

import contextlib
import os
import tempfile
import uuid
from datetime import datetime

import pytest

from syrin.enums import MemoryType
from syrin.memory.config import MemoryEntry
from syrin.swarm.backends._memory import InMemoryBusBackend
from syrin.swarm.backends._protocol import MemoryBusBackend
from syrin.swarm.backends._sqlite import SqliteBusBackend

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_entry(
    content: str = "test content",
    memory_type: MemoryType = MemoryType.KNOWLEDGE,
) -> MemoryEntry:
    return MemoryEntry(
        id=str(uuid.uuid4()),
        content=content,
        type=memory_type,
        importance=0.8,
        keywords=[],
        created_at=datetime.now(),
    )


# ---------------------------------------------------------------------------
# P3-T2-1: InMemoryBusBackend — all operations work
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_in_memory_backend_store_and_query() -> None:
    """InMemoryBusBackend stores and retrieves entries."""
    backend = InMemoryBusBackend()
    entry = _make_entry(content="machine learning techniques")

    await backend.store(entry, agent_id="a1", ttl=None)
    results = await backend.query(query="machine", agent_id="a2")

    assert len(results) == 1
    assert results[0].content == "machine learning techniques"


@pytest.mark.asyncio
async def test_in_memory_backend_query_no_match() -> None:
    """InMemoryBusBackend returns empty list when no entries match."""
    backend = InMemoryBusBackend()
    entry = _make_entry(content="dogs are great")
    await backend.store(entry, agent_id="a1", ttl=None)

    results = await backend.query(query="cats", agent_id="a2")
    assert results == []


@pytest.mark.asyncio
async def test_in_memory_backend_all_entries() -> None:
    """InMemoryBusBackend.all_entries returns all stored (entry, agent_id, ttl) tuples."""
    backend = InMemoryBusBackend()
    e1 = _make_entry(content="first")
    e2 = _make_entry(content="second")

    await backend.store(e1, agent_id="a1", ttl=60.0)
    await backend.store(e2, agent_id="a2", ttl=None)

    all_entries = await backend.all_entries()
    assert len(all_entries) == 2
    contents = {e.content for e, _, _ in all_entries}
    assert contents == {"first", "second"}


@pytest.mark.asyncio
async def test_in_memory_backend_clear_expired_removes_past_ttl() -> None:
    """InMemoryBusBackend.clear_expired removes entries whose ttl has passed."""
    import time

    backend = InMemoryBusBackend()
    entry = _make_entry(content="expiring soon")
    # Store with expire_at already in the past
    await backend.store(entry, agent_id="a1", ttl=None)

    # Directly inject expired timestamp
    async with backend._lock:  # type: ignore[attr-defined]
        # Force expire_at to be in the past
        backend._entries[0] = (backend._entries[0][0], backend._entries[0][1], time.time() - 1)  # type: ignore[index]

    expired_ids = await backend.clear_expired()
    assert entry.id in expired_ids
    results = await backend.query(query="expiring", agent_id="a2")
    assert results == []


@pytest.mark.asyncio
async def test_in_memory_backend_data_lost_after_clear() -> None:
    """InMemoryBusBackend loses data when a new instance is created."""
    backend1 = InMemoryBusBackend()
    entry = _make_entry(content="temporary knowledge")
    await backend1.store(entry, agent_id="a1", ttl=None)

    backend2 = InMemoryBusBackend()
    results = await backend2.query(query="temporary", agent_id="a2")
    assert results == [], "New instance should have no data"


# ---------------------------------------------------------------------------
# P3-T2-2: SqliteBusBackend — data persists across instances
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_sqlite_backend_store_and_query() -> None:
    """SqliteBusBackend stores and retrieves entries."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    try:
        backend = SqliteBusBackend(path=db_path)
        entry = _make_entry(content="persistent knowledge")
        await backend.store(entry, agent_id="a1", ttl=None)

        results = await backend.query(query="persistent", agent_id="a2")
        assert len(results) == 1
        assert results[0].content == "persistent knowledge"
    finally:
        os.unlink(db_path)


@pytest.mark.asyncio
async def test_sqlite_backend_persists_across_instances() -> None:
    """Two SqliteBusBackend instances pointing to the same file share state."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    try:
        backend1 = SqliteBusBackend(path=db_path)
        entry = _make_entry(content="durable fact")
        await backend1.store(entry, agent_id="a1", ttl=None)

        backend2 = SqliteBusBackend(path=db_path)
        results = await backend2.query(query="durable", agent_id="a2")

        assert len(results) == 1
        assert results[0].content == "durable fact"
    finally:
        os.unlink(db_path)


@pytest.mark.asyncio
async def test_sqlite_backend_clear_expired() -> None:
    """SqliteBusBackend.clear_expired removes entries past their TTL."""

    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    try:
        backend = SqliteBusBackend(path=db_path)
        entry = _make_entry(content="short-lived")
        # Store with a past expiry
        await backend.store(entry, agent_id="a1", ttl=-1.0)  # already expired

        expired_ids = await backend.clear_expired()
        assert entry.id in expired_ids

        results = await backend.query(query="short-lived", agent_id="a2")
        assert results == []
    finally:
        os.unlink(db_path)


# ---------------------------------------------------------------------------
# P3-T2-3: MemoryBusBackend Protocol — isinstance check
# ---------------------------------------------------------------------------


def test_memory_bus_backend_protocol_isinstance() -> None:
    """A class implementing MemoryBusBackend Protocol passes isinstance() check."""
    # InMemoryBusBackend should satisfy the Protocol
    backend = InMemoryBusBackend()
    assert isinstance(backend, MemoryBusBackend)


def test_memory_bus_backend_protocol_sqlite_isinstance() -> None:
    """SqliteBusBackend satisfies the MemoryBusBackend Protocol."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    try:
        backend = SqliteBusBackend(path=db_path)
        assert isinstance(backend, MemoryBusBackend)
    finally:
        os.unlink(db_path)


# ---------------------------------------------------------------------------
# P3-T2-4: RedisBusBackend — skip if redis not available
# ---------------------------------------------------------------------------


@pytest.mark.redis
@pytest.mark.asyncio
async def test_redis_backend_store_and_query() -> None:
    """RedisBusBackend stores and retrieves entries (requires running Redis)."""
    pytest.importorskip("redis")

    # Check if Redis is actually reachable before running the test
    import socket

    try:
        s = socket.create_connection(("localhost", 6379), timeout=0.5)
        s.close()
    except OSError:
        pytest.skip("Redis not reachable on localhost:6379")

    from syrin.swarm.backends._redis import RedisBusBackend

    backend = RedisBusBackend(url="redis://localhost:6379")
    entry = _make_entry(content="redis knowledge")

    try:
        await backend.store(entry, agent_id="a1", ttl=None)
        results = await backend.query(query="redis", agent_id="a2")
        assert any(e.content == "redis knowledge" for e in results)
    finally:
        # Clean up
        with contextlib.suppress(Exception):
            await backend.clear_expired()
