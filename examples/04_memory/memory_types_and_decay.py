"""Memory Types and Decay Example.

Demonstrates:
- 4 memory types: Core, Episodic, Semantic, Procedural
- Memory type classes (CoreMemory, EpisodicMemory, etc.)
- Decay strategies (exponential, linear)
- half_life_hours, reinforce_on_access, min_importance
- MemoryBudget for cost-aware memory operations
- Storage backends (memory, sqlite)

Run: python -m examples.04_memory.memory_types_and_decay
Visit: http://localhost:8000/playground
Requires: uv pip install syrin[serve]
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

from dotenv import load_dotenv

from examples.models.models import almock
from syrin import Agent
from syrin.enums import DecayStrategy, MemoryBackend, MemoryType
from syrin.memory import (
    CoreMemory,
    Decay,
    EpisodicMemory,
    Memory,
    MemoryBudget,
    MemoryEntry,
    MemoryStore,
    ProceduralMemory,
    SemanticMemory,
    create_memory,
    get_backend,
)

load_dotenv(Path(__file__).resolve().parent.parent / ".env")


# 1. Memory type classes
core = CoreMemory(id="core-1", content="My name is John Smith", importance=0.95)
episodic = EpisodicMemory(id="ep-1", content="Yesterday I visited the Eiffel Tower")
semantic = SemanticMemory(id="sem-1", content="Python is a programming language")
procedural = ProceduralMemory(id="proc-1", content="How to make coffee: boil water, add grounds")
mem = create_memory(MemoryType.CORE, "factory-1", "Created via factory")

# 2. MemoryStore — add, recall, forget
store = MemoryStore()
store.add(content="User prefers dark mode", memory_type=MemoryType.CORE)
store.add(content="Yesterday's meeting was at 3pm", memory_type=MemoryType.EPISODIC)
store.add(content="Paris is the capital of France", memory_type=MemoryType.SEMANTIC)
store.add(content="How to reset password: click forgot password", memory_type=MemoryType.PROCEDURAL)
core_memories = store.recall(memory_type=MemoryType.CORE)
related = store.recall("password")

# 3. Decay curves
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
decay.apply(old_memory)
decay.on_access(old_memory)

# 4. MemoryBudget
from syrin.budget import warn_on_exceeded

budget = MemoryBudget(extraction_budget=0.001, on_exceeded=warn_on_exceeded)
store = MemoryStore(budget=budget)
store.add(content="Short fact", memory_type=MemoryType.SEMANTIC)

# 5. Agent with persistent memory (all 4 types)
agent = Agent(
    model=almock,
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
core_memories = agent.recall(memory_type=MemoryType.CORE)
related = agent.recall("name")
if core_memories:
    agent.forget(memory_id=core_memories[0].id)

# 6. Storage backends
mem_backend = get_backend(MemoryBackend.MEMORY)
mem_backend.add(MemoryEntry(id="mem-1", content="In-memory is fast", type=MemoryType.SEMANTIC))
results = mem_backend.search("fast")
print(f"In-memory: {results[0].content if results else 'none'}")


# 7. Agent class for serving
class MemoryDemoAgent(Agent):
    _agent_name = "memory-demo"
    _agent_description = (
        "Agent with 4 memory types (Core, Episodic, Semantic, Procedural) and decay"
    )
    model = almock
    system_prompt = "You are a helpful assistant with persistent memory."
    memory = Memory(
        types=[MemoryType.CORE, MemoryType.EPISODIC, MemoryType.SEMANTIC, MemoryType.PROCEDURAL],
        top_k=5,
    )


if __name__ == "__main__":
    agent = MemoryDemoAgent()
    print("Serving at http://localhost:8000/playground")
    agent.serve(port=8000, enable_playground=True, debug=True)
