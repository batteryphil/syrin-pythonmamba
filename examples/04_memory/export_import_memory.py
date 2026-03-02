"""Memory export/import — backup, GDPR export, migration.

Demonstrates:
- memory.export() → MemorySnapshot (JSON-serializable)
- memory.import_from(snapshot) → append memories
- MemorySnapshot.from_dict / to_dict for roundtrip

Run: python -m examples.04_memory.export_import_memory
"""

from __future__ import annotations

import json
import tempfile

from syrin.enums import MemoryBackend, MemoryType, WriteMode
from syrin.memory import Memory, MemorySnapshot


def main() -> None:
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        path = f.name

    try:
        # Use SYNC so writes complete before export
        memory = Memory(
            backend=MemoryBackend.SQLITE,
            path=path,
            write_mode=WriteMode.SYNC,
        )
        memory.remember("User prefers Python", memory_type=MemoryType.CORE)
        memory.remember("Last session: discussed async patterns", memory_type=MemoryType.EPISODIC)

        # Export to snapshot
        snapshot = memory.export()
        print(f"Exported {len(snapshot.memories)} memories")
        for m in snapshot.memories:
            print(f"  - [{m.type}] {m.content[:50]}...")

        # Serialize to JSON (e.g. for GDPR export)
        js = snapshot.to_json()
        data = json.loads(js)

        # Roundtrip: load from dict
        snapshot2 = MemorySnapshot.from_dict(data)
        print(f"\nRoundtrip: {len(snapshot2.memories)} memories restored")

        # Import into a new memory instance (append mode)
        memory2 = Memory(
            backend=MemoryBackend.SQLITE,
            path=path,
            write_mode=WriteMode.SYNC,
        )
        count = memory2.import_from(snapshot2)
        print(f"Imported {count} memories")
    finally:
        import os

        if os.path.exists(path):
            os.unlink(path)


if __name__ == "__main__":
    main()
