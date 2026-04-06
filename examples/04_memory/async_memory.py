"""WriteMode.ASYNC — fire-and-forget memory writes.

Demonstrates:
- SYNC: Blocks until complete; use for tests or when you need immediate persistence.
- ASYNC: Fire-and-forget; returns immediately; never blocks response.

Run: python -m examples.04_memory.async_memory
"""

from __future__ import annotations

import tempfile

from syrin.enums import MemoryBackend, MemoryType, WriteMode
from syrin.memory import Memory


def main() -> None:
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        path = f.name

    try:
        # ASYNC (default): returns immediately; background thread does the write
        memory_async = Memory(
            backend=MemoryBackend.SQLITE,
            path=path,
            write_mode=WriteMode.ASYNC,
        )
        memory_async.remember("Async write", memory_type=MemoryType.HISTORY)
        # Returns immediately; write happens in background

        # SYNC: blocks until complete; use when you need immediate persistence
        memory_sync = Memory(
            backend=MemoryBackend.SQLITE,
            path=path,
            write_mode=WriteMode.SYNC,
        )
        memory_sync.remember("Sync write", memory_type=MemoryType.FACTS)
        results = memory_sync.recall(query="Sync", limit=5)
        print(f"Recalled {len(results)} memories after sync write")
    finally:
        import os

        if os.path.exists(path):
            os.unlink(path)


if __name__ == "__main__":
    main()
