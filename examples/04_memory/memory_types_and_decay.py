"""Memory Types and Decay — Comprehensive memory features demo.

Demonstrates:
- 4 memory types: Core, Episodic, Semantic, Procedural
- Memory type classes (CoreMemory, EpisodicMemory, etc.)
- Factory function create_memory()
- MemoryStore: add, recall, forget
- Decay strategies (exponential, linear)
- MemoryBudget for cost-aware memory operations
- Storage backends (in-memory)
- Agent with persistent memory (all 4 types)

Run:
    python examples/04_memory/memory_types_and_decay.py
"""

from __future__ import annotations

from datetime import datetime, timedelta

from syrin import Agent, Decay, Memory, MemoryBudget, MemoryEntry, Model
from syrin.budget import warn_on_exceeded
from syrin.enums import DecayStrategy, MemoryBackend, MemoryType
from syrin.memory import (
    CoreMemory,
    EpisodicMemory,
    MemoryStore,
    ProceduralMemory,
    SemanticMemory,
    create_memory,
    get_backend,
)


def main() -> None:
    # ── 1. Memory type classes ────────────────────────────────────────
    print("=" * 60)
    print("1. Memory Type Classes")
    print("=" * 60)

    core = CoreMemory(id="core-1", content="My name is John Smith", importance=0.95)
    episodic = EpisodicMemory(id="ep-1", content="Yesterday I visited the Eiffel Tower")
    semantic = SemanticMemory(id="sem-1", content="Python is a programming language")
    procedural = ProceduralMemory(id="proc-1", content="How to make coffee: boil water, add grounds")
    factory = create_memory(MemoryType.CORE, "factory-1", "Created via factory")

    for label, mem in [
        ("Core", core),
        ("Episodic", episodic),
        ("Semantic", semantic),
        ("Procedural", procedural),
        ("Factory", factory),
    ]:
        print(f"  {label:12s} | type={mem.type.value:12s} | {mem.content[:40]}")

    # ── 2. MemoryStore — add, recall, forget ──────────────────────────
    print("\n" + "=" * 60)
    print("2. MemoryStore — add, recall, forget")
    print("=" * 60)

    store = MemoryStore()
    store.add(content="User prefers dark mode", memory_type=MemoryType.CORE)
    store.add(content="Yesterday's meeting was at 3pm", memory_type=MemoryType.EPISODIC)
    store.add(content="Paris is the capital of France", memory_type=MemoryType.SEMANTIC)
    store.add(content="How to reset password: click forgot password", memory_type=MemoryType.PROCEDURAL)

    core_memories = store.recall(memory_type=MemoryType.CORE)
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
        type=MemoryType.EPISODIC,
        importance=1.0,
        created_at=datetime.now() - timedelta(hours=24),
    )

    print(f"  Before decay: importance={old_memory.importance:.4f}")
    decay.apply(old_memory)
    print(f"  After decay (24h old, rate=0.95): importance={old_memory.importance:.4f}")
    decay.on_access(old_memory)
    print(f"  After reinforcement on access: importance={old_memory.importance:.4f}")

    # ── 4. MemoryBudget ───────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("4. MemoryBudget")
    print("=" * 60)

    budget = MemoryBudget(extraction_budget=0.001, on_exceeded=warn_on_exceeded)
    budgeted_store = MemoryStore(budget=budget)
    added = budgeted_store.add(content="Short fact", memory_type=MemoryType.SEMANTIC)
    print(f"  Added with budget constraint: {added}")

    # ── 5. Agent with persistent memory (all 4 types) ─────────────────
    print("\n" + "=" * 60)
    print("5. Agent with Persistent Memory")
    print("=" * 60)

    agent = Agent(
        model=Model.Almock(),
        memory=Memory(
            types=[MemoryType.CORE, MemoryType.EPISODIC, MemoryType.SEMANTIC, MemoryType.PROCEDURAL],
            top_k=5,
        ),
    )

    agent.remember("My name is John", memory_type=MemoryType.CORE)
    agent.remember("I live in San Francisco", memory_type=MemoryType.CORE)
    agent.remember("Yesterday I had pizza", memory_type=MemoryType.EPISODIC)
    agent.remember("Python uses indentation", memory_type=MemoryType.SEMANTIC)
    agent.remember("How to make tea: boil water, steep", memory_type=MemoryType.PROCEDURAL)

    agent_core = agent.recall(memory_type=MemoryType.CORE)
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
    mem_backend.add(MemoryEntry(id="mem-1", content="In-memory is fast", type=MemoryType.SEMANTIC))
    results = mem_backend.search("fast")
    print(f"  In-memory search 'fast': {results[0].content if results else 'none'}")

    print("\nDone!")


if __name__ == "__main__":
    main()

    # Optional: serve the agent in the playground
    # agent = Agent(
    #     model=Model.Almock(),
    #     memory=Memory(
    #         types=[MemoryType.CORE, MemoryType.EPISODIC, MemoryType.SEMANTIC, MemoryType.PROCEDURAL],
    #         top_k=5,
    #     ),
    # )
    # agent.serve(port=8000, enable_playground=True, debug=True)
