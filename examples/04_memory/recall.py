"""Recall Example — Storing and retrieving memories by query.

Demonstrates:
- Creating an agent with persistent memory
- Storing multiple memories with agent.remember()
- Retrieving relevant memories with agent.recall(query=...)
- Listing all memories with agent.recall()

Run:
    python examples/04_memory/recall.py
"""

from __future__ import annotations

from syrin import Agent, Memory, Model
from syrin.enums import MemoryType


def main() -> None:
    print("=" * 60)
    print("Memory Recall Demo")
    print("=" * 60)

    # Create an agent with in-memory persistence (no API key needed)
    assistant = Agent(
        model=Model.mock(),
        system_prompt="You are a helpful assistant.",
        memory=Memory(),
    )

    # Store some memories
    facts = [
        ("User likes Python programming", MemoryType.FACTS),
        ("User works at a startup", MemoryType.FACTS),
        ("User prefers afternoon meetings", MemoryType.HISTORY),
        ("Python was created by Guido van Rossum", MemoryType.KNOWLEDGE),
        ("How to deploy: push to main, CI runs, auto-deploy", MemoryType.INSTRUCTIONS),
    ]

    print("\nStoring memories:")
    for content, mem_type in facts:
        assistant.remember(content, memory_type=mem_type)
        print(f"  [{mem_type.value:12s}] {content}")

    # Recall by query — returns memories whose content matches
    print("\n" + "-" * 60)
    print("Recall results for query='programming work':")
    results = assistant.recall(query="programming work")
    if results:
        for r in results:
            print(f"  - [{r.type.value:12s}] {r.content}")
    else:
        print("  (no matches)")

    print("\nRecall results for query='Python':")
    results = assistant.recall(query="Python")
    for r in results:
        print(f"  - [{r.type.value:12s}] {r.content}")

    # Recall by type — list all core memories
    print("\nAll CORE memories:")
    core = assistant.recall(memory_type=MemoryType.FACTS)
    for r in core:
        print(f"  - {r.content}")

    # Recall everything
    print(f"\nTotal memories stored: {len(assistant.recall())}")

    print("\nDone!")


if __name__ == "__main__":
    main()

    # Optional: serve the agent in the playground
    # assistant = Agent(
    #     model=Model.mock(),
    #     system_prompt="You are a helpful assistant.",
    #     memory=Memory(),
    # )
    # assistant.serve(port=8000, enable_playground=True, debug=True)
