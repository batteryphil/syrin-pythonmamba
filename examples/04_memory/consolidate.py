"""Consolidate Example — Memory deduplication.

Demonstrates:
- Storing duplicate memories
- Running Memory.consolidate(deduplicate=True)
- Verifying that duplicates are removed (highest importance kept)

Run:
    python examples/04_memory/consolidate.py
"""

from __future__ import annotations

from syrin import Memory, Model
from syrin.enums import MemoryType


def main() -> None:
    print("=" * 60)
    print("Memory Consolidation Demo")
    print("=" * 60)

    # Create a Memory instance directly (no agent needed for this demo)
    memory = Memory()

    # Store some memories, including intentional duplicates
    entries = [
        ("User prefers dark mode", MemoryType.CORE, 0.9),
        ("User prefers dark mode", MemoryType.CORE, 0.7),  # duplicate, lower importance
        ("User prefers dark mode", MemoryType.CORE, 0.5),  # duplicate, even lower
        ("Python is a programming language", MemoryType.SEMANTIC, 0.8),
        ("Python is a programming language", MemoryType.SEMANTIC, 0.6),  # duplicate
        ("Yesterday had a team standup", MemoryType.EPISODIC, 0.7),
    ]

    print("\nStoring memories (with duplicates):")
    for content, mem_type, importance in entries:
        memory.remember(content, memory_type=mem_type, importance=importance)
        print(f"  [{mem_type.value:12s}] importance={importance:.1f} | {content}")

    before = memory.recall(count=100)
    print(f"\nTotal memories before consolidation: {len(before)}")

    # Run consolidation — deduplicates by content, keeps highest importance
    removed = memory.consolidate(deduplicate=True)
    print(f"Duplicates removed: {removed}")

    after = memory.recall(count=100)
    print(f"Total memories after consolidation: {len(after)}")

    print("\nRemaining memories:")
    for m in after:
        print(f"  [{m.type.value:12s}] importance={m.importance:.4f} | {m.content}")

    print("\nDone!")


if __name__ == "__main__":
    main()

    # Optional: serve the agent in the playground
    # from syrin import Agent
    # agent = Agent(
    #     model=Model.Almock(),
    #     system_prompt="You are a helpful assistant. Use memory to remember user preferences.",
    #     memory=Memory(),
    # )
    # agent.serve(port=8000, enable_playground=True, debug=True)
