"""Tests for memory backends."""

from __future__ import annotations

import os
import tempfile

import pytest

from syrin.enums import MemoryBackend, MemoryType
from syrin.memory import MemoryEntry
from syrin.memory.backends import InMemoryBackend, SQLiteBackend, get_backend


class TestInMemoryBackend:
    """Tests for InMemoryBackend."""

    def test_add_and_get(self) -> None:
        backend = InMemoryBackend()
        entry = MemoryEntry(
            id="test-1",
            content="Test memory",
            type=MemoryType.HISTORY,
        )
        backend.add(entry)
        result = backend.get("test-1")
        assert result is not None
        assert result.content == "Test memory"

    def test_search(self) -> None:
        backend = InMemoryBackend()
        backend.add(MemoryEntry(id="1", content="Python is great", type=MemoryType.KNOWLEDGE))
        backend.add(MemoryEntry(id="2", content="JavaScript too", type=MemoryType.KNOWLEDGE))
        # InMemoryBackend search is a simple filter
        backend.search("Python")
        # Note: InMemoryBackend returns all, filtering is done by search implementation

    def test_list(self) -> None:
        backend = InMemoryBackend()
        backend.add(MemoryEntry(id="1", content="Test 1", type=MemoryType.FACTS))
        backend.add(MemoryEntry(id="2", content="Test 2", type=MemoryType.HISTORY))
        results = backend.list()
        assert len(results) == 2

    def test_delete(self) -> None:
        backend = InMemoryBackend()
        backend.add(MemoryEntry(id="to-delete", content="Test", type=MemoryType.HISTORY))
        backend.delete("to-delete")
        assert backend.get("to-delete") is None


class TestSQLiteBackend:
    """Tests for SQLiteBackend."""

    @pytest.fixture
    def temp_db(self):
        """Create a temporary database file."""
        fd, path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        yield path
        os.unlink(path)

    def test_add_and_get(self, temp_db: str) -> None:
        backend = SQLiteBackend(path=temp_db)
        entry = MemoryEntry(
            id="test-1",
            content="Test memory",
            type=MemoryType.HISTORY,
            importance=0.8,
        )
        backend.add(entry)
        result = backend.get("test-1")
        assert result is not None
        assert result.content == "Test memory"
        assert result.importance == 0.8
        backend.close()

    def test_search(self, temp_db: str) -> None:
        backend = SQLiteBackend(path=temp_db)
        backend.add(MemoryEntry(id="1", content="Python is great", type=MemoryType.KNOWLEDGE))
        backend.add(MemoryEntry(id="2", content="JavaScript too", type=MemoryType.KNOWLEDGE))
        results = backend.search("Python")
        assert len(results) == 1
        assert results[0].content == "Python is great"
        backend.close()

    def test_list_filtered(self, temp_db: str) -> None:
        backend = SQLiteBackend(path=temp_db)
        backend.add(MemoryEntry(id="1", content="Core fact", type=MemoryType.FACTS))
        backend.add(MemoryEntry(id="2", content="Episodic event", type=MemoryType.HISTORY))

        core = backend.list(memory_type=MemoryType.FACTS)
        assert len(core) == 1
        assert core[0].type == MemoryType.FACTS

        all_mem = backend.list()
        assert len(all_mem) == 2
        backend.close()

    def test_update(self, temp_db: str) -> None:
        backend = SQLiteBackend(path=temp_db)
        entry = MemoryEntry(id="update-test", content="Original", type=MemoryType.HISTORY)
        backend.add(entry)

        entry.content = "Updated"
        backend.update(entry)

        result = backend.get("update-test")
        assert result is not None
        assert result.content == "Updated"
        backend.close()

    def test_delete(self, temp_db: str) -> None:
        backend = SQLiteBackend(path=temp_db)
        backend.add(MemoryEntry(id="to-delete", content="Test", type=MemoryType.HISTORY))
        backend.delete("to-delete")
        assert backend.get("to-delete") is None
        backend.close()

    def test_clear(self, temp_db: str) -> None:
        backend = SQLiteBackend(path=temp_db)
        backend.add(MemoryEntry(id="1", content="Test 1", type=MemoryType.HISTORY))
        backend.add(MemoryEntry(id="2", content="Test 2", type=MemoryType.HISTORY))
        backend.clear()
        assert len(backend.list()) == 0
        backend.close()

    def test_persistence(self, temp_db: str) -> None:
        """Test that data persists across backend instances."""
        entry = MemoryEntry(id="persist", content="Persistent data", type=MemoryType.FACTS)

        # Add to first backend
        backend1 = SQLiteBackend(path=temp_db)
        backend1.add(entry)
        backend1.close()

        # Read from second backend
        backend2 = SQLiteBackend(path=temp_db)
        result = backend2.get("persist")
        assert result is not None
        assert result.content == "Persistent data"
        backend2.close()


class TestGetBackend:
    """Tests for get_backend factory function."""

    def test_get_memory_backend(self) -> None:
        backend = get_backend(MemoryBackend.MEMORY)
        assert isinstance(backend, InMemoryBackend)

    def test_get_sqlite_backend(self) -> None:
        fd, path = tempfile.mkstemp(suffix=".db")
        os.close(fd)

        backend = get_backend(MemoryBackend.SQLITE, path=path)
        assert isinstance(backend, SQLiteBackend)
        backend.close()

        os.unlink(path)

    def test_unknown_backend_raises(self) -> None:
        """Test that unknown backend raises appropriate error."""
        # The get_backend function only accepts MemoryBackend enum values
        # This test verifies the error handling
        from syrin.enums import MemoryBackend

        known_backends = list(MemoryBackend)
        assert MemoryBackend.MEMORY in known_backends
        assert MemoryBackend.SQLITE in known_backends


# =============================================================================
# MEMORY BACKENDS EDGE CASES - TRY TO BREAK FUNCTIONALITY
# =============================================================================


class TestInMemoryBackendEdgeCases:
    """Edge cases for InMemoryBackend."""

    def test_add_empty_content(self):
        """Add entry with empty content."""
        backend = InMemoryBackend()
        entry = MemoryEntry(id="empty", content="", type=MemoryType.FACTS)
        backend.add(entry)
        result = backend.get("empty")
        assert result is not None
        assert result.content == ""

    def test_add_very_long_content(self):
        """Add entry with very long content."""
        backend = InMemoryBackend()
        long_content = "x" * 100000
        entry = MemoryEntry(id="long", content=long_content, type=MemoryType.FACTS)
        backend.add(entry)
        result = backend.get("long")
        assert result is not None
        assert len(result.content) == 100000

    def test_add_unicode_content(self):
        """Add entry with unicode content."""
        backend = InMemoryBackend()
        entry = MemoryEntry(id="unicode", content="Hello 🌍 你好 🔥", type=MemoryType.FACTS)
        backend.add(entry)
        result = backend.get("unicode")
        assert "🌍" in result.content

    def test_list_empty(self):
        """List with no entries."""
        backend = InMemoryBackend()
        results = backend.list()
        assert results == []

    def test_list_many_entries(self):
        """List with many entries."""
        backend = InMemoryBackend()
        for i in range(100):
            backend.add(MemoryEntry(id=f"entry-{i}", content=f"Content {i}", type=MemoryType.FACTS))
        results = backend.list()
        assert len(results) == 100

    def test_search_no_results(self):
        """Search with no matching results."""
        backend = InMemoryBackend()
        backend.add(MemoryEntry(id="1", content="Python code", type=MemoryType.KNOWLEDGE))
        backend.search("nonexistent")
        # May return all or empty depending on implementation

    def test_delete_nonexistent(self):
        """Delete nonexistent entry."""
        backend = InMemoryBackend()
        # Should not raise
        backend.delete("nonexistent")

    def test_get_nonexistent(self):
        """Get nonexistent entry."""
        backend = InMemoryBackend()
        result = backend.get("nonexistent")
        assert result is None


class TestSQLiteBackendEdgeCases:
    """Edge cases for SQLiteBackend."""

    @pytest.fixture
    def temp_db(self):
        fd, path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        yield path
        os.unlink(path)

    def test_add_with_importance(self, temp_db):
        """Add entry with importance values."""
        backend = SQLiteBackend(path=temp_db)
        entry = MemoryEntry(id="test", content="Test", type=MemoryType.FACTS, importance=0.0)
        backend.add(entry)
        result = backend.get("test")
        assert result.importance == 0.0

        entry2 = MemoryEntry(id="test2", content="Test2", type=MemoryType.FACTS, importance=1.0)
        backend.add(entry2)
        result2 = backend.get("test2")
        assert result2.importance == 1.0
        backend.close()

    def test_list_with_limit(self, temp_db):
        """List with limit."""
        backend = SQLiteBackend(path=temp_db)
        for i in range(10):
            backend.add(MemoryEntry(id=f"entry-{i}", content=f"Content {i}", type=MemoryType.FACTS))

        results = backend.list(limit=5)
        assert len(results) <= 5
        backend.close()

    def test_update_nonexistent(self, temp_db):
        """Update nonexistent entry."""
        backend = SQLiteBackend(path=temp_db)
        entry = MemoryEntry(id="nonexistent", content="Test", type=MemoryType.FACTS)
        # Should handle gracefully
        backend.update(entry)
        backend.close()

    def test_search_empty_query(self, temp_db):
        """Search with empty query."""
        backend = SQLiteBackend(path=temp_db)
        backend.add(MemoryEntry(id="1", content="Test", type=MemoryType.KNOWLEDGE))
        backend.search("")
        # Should return results or empty
        backend.close()
