"""Memory Types and Decay — Comprehensive memory features demo.

Demonstrates:
- 4 memory types: Facts, History, Knowledge, Instructions
- Memory type classes (FactsMemory, HistoryMemory, etc.)
- Factory function create_memory()
- MemoryStore: add, recall, forget
- Decay strategies (exponential, linear)
- Budget-aware memory operations (budget_extraction flat field)
- Storage backends (in-memory)
- Agent with persistent memory (all 4 types)

Run:
    python examples/04_memory/memory_types_and_decay.py
"""

from __future__ import annotations

from datetime import datetime, timedelta

from syrin import Agent, Decay, Memory, MemoryEntry, Model
from syrin.enums import DecayStrategy, MemoryBackend, MemoryType
from syrin.memory import (
    FactsMemory,
    HistoryMemory,
    InstructionsMemory,
    KnowledgeMemory,
    MemoryStore,
    create_memory,
    get_backend,
)


def main() -> None:
    # ── 1. Memory type classes ────────────────────────────────────────
    print("=" * 60)
    print("1. Memory Type Classes")
    print("=" * 60)

    core = FactsMemory(id="core-1", content="My name is John Smith", importance=0.95)
    episodic = HistoryMemory(id="ep-1", content="Yesterday I visited the Eiffel Tower")
    semantic = KnowledgeMemory(id="sem-1", content="Python is a programming language")
    procedural = InstructionsMemory(
        id="proc-1", content="How to make coffee: boil water, add grounds"
    )
    factory = create_memory(MemoryType.FACTS, "factory-1", "Created via factory")

    for label, mem in [
        ("Facts", core),
        ("History", episodic),
        ("Knowledge", semantic),
        ("Instructions", procedural),
        ("Factory", factory),
    ]:
        print(f"  {label:12s} | type={mem.type.value:12s} | {mem.content[:40]}")

    # ── 2. MemoryStore — add, recall, forget ──────────────────────────
    print("\n" + "=" * 60)
    print("2. MemoryStore — add, recall, forget")
    print("=" * 60)

    store = MemoryStore()
    store.add(content="User prefers dark mode", memory_type=MemoryType.FACTS)
    store.add(content="Yesterday's meeting was at 3pm", memory_type=MemoryType.HISTORY)
    store.add(content="Paris is the capital of France", memory_type=MemoryType.KNOWLEDGE)
    store.add(
        content="How to reset password: click forgot password", memory_type=MemoryType.INSTRUCTIONS
    )

    core_memories = store.recall(memory_type=MemoryType.FACTS)
    print(f"  Core memories ({len(core_memories)}):")
    for m in core_memories:
        print(f"    - {m.content}")

    related = store.recall("password")
    print(f"  Recall 'password' ({len(related)}):")
    for m in related:
        print(f"    - [{m.type.value}] {m.content}")

    # ── 3. Decay curves ───────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("3. Decay Curves")
    print("=" * 60)

    decay = Decay(
        strategy=DecayStrategy.EXPONENTIAL,
        rate=0.95,
        reinforce_on_access=True,
        min_importance=0.1,
    )

    old_memory = MemoryEntry(
        id="old-1",
        content="Old information",
        type=MemoryType.HISTORY,
        importance=1.0,
        created_at=datetime.now() - timedelta(hours=24),
    )

    print(f"  Before decay: importance={old_memory.importance:.4f}")
    decay.apply(old_memory)
    print(f"  After decay (24h old, rate=0.95): importance={old_memory.importance:.4f}")
    decay.on_access(old_memory)
    print(f"  After reinforcement on access: importance={old_memory.importance:.4f}")

    # ── 4. Budget-aware memory (flat fields) ──────────────────────────
    print("\n" + "=" * 60)
    print("4. Budget-Aware Memory")
    print("=" * 60)

    # budget_extraction and budget_on_exceeded are now flat fields on Memory/MemoryStore
    warnings_caught: list[object] = []

    def warn_handler(ctx: object) -> None:
        warnings_caught.append(ctx)
        # return without raising → warn-only; store still proceeds

    budgeted_store = MemoryStore(budget_extraction=0.001, budget_on_exceeded=warn_handler)
    added = budgeted_store.add(content="Short fact", memory_type=MemoryType.KNOWLEDGE)
    print(f"  Added with budget constraint: {added}")

    # ── 5. Agent with persistent memory (all 4 types) ─────────────────
    print("\n" + "=" * 60)
    print("5. Agent with Persistent Memory")
    print("=" * 60)

    agent = Agent(
        model=Model.mock(),
        memory=Memory(
            # restrict_to replaces the old types= parameter; omit for all 4 types (default)
            top_k=5,
        ),
    )

    agent.remember("My name is John", memory_type=MemoryType.FACTS)
    agent.remember("I live in San Francisco", memory_type=MemoryType.FACTS)
    agent.remember("Yesterday I had pizza", memory_type=MemoryType.HISTORY)
    agent.remember("Python uses indentation", memory_type=MemoryType.KNOWLEDGE)
    agent.remember("How to make tea: boil water, steep", memory_type=MemoryType.INSTRUCTIONS)

    agent_core = agent.recall(memory_type=MemoryType.FACTS)
    print(f"  Core memories ({len(agent_core)}):")
    for m in agent_core:
        print(f"    - {m.content}")

    agent_related = agent.recall("name")
    print(f"  Recall 'name' ({len(agent_related)}):")
    for m in agent_related:
        print(f"    - [{m.type.value}] {m.content}")

    if agent_core:
        forgotten = agent.forget(memory_id=agent_core[0].id)
        print(f"  Forgot '{agent_core[0].content}' -> deleted {forgotten}")

    # ── 6. Storage backends (in-memory) ───────────────────────────────
    print("\n" + "=" * 60)
    print("6. Storage Backends")
    print("=" * 60)

    mem_backend = get_backend(MemoryBackend.MEMORY)
    mem_backend.add(MemoryEntry(id="mem-1", content="In-memory is fast", type=MemoryType.KNOWLEDGE))
    results = mem_backend.search("fast")
    print(f"  In-memory search 'fast': {results[0].content if results else 'none'}")

    print("\nDone!")


if __name__ == "__main__":
    main()

    # Optional: serve the agent in the playground
    # agent = Agent(
    #     model=Model.mock(),
    #     memory=Memory(
    #         types=[MemoryType.FACTS, MemoryType.HISTORY, MemoryType.KNOWLEDGE, MemoryType.INSTRUCTIONS],
    #         top_k=5,
    #     ),
    # )
    # agent.serve(port=8000, enable_playground=True, debug=True)
