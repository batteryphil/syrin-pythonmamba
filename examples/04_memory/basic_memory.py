"""Basic Memory — Make your agent remember things.

Shows the 4 memory types: Core, Episodic, Semantic, Procedural.
Memory persists across conversations and decays over time.

Run:
    python examples/04_memory/basic_memory.py
"""

from syrin import Agent, Memory, MemoryType, Model

model = Model.mock()

# Create an agent with memory
agent = Agent(
    model=model,
    system_prompt="You are a helpful assistant that remembers user preferences.",
    memory=Memory(),
)

# ============================================================
# 1. Store memories (4 types)
# ============================================================
# CORE: permanent facts (never decays) — user identity, preferences
agent.remember("The user's name is Alice.", memory_type=MemoryType.FACTS, importance=1.0)

# EPISODIC: past events (decays over time) — what happened in conversations
agent.remember("User asked about machine learning yesterday.", memory_type=MemoryType.HISTORY)

# SEMANTIC: knowledge and facts — learned information
agent.remember("Alice prefers Python over JavaScript.", memory_type=MemoryType.KNOWLEDGE)

# PROCEDURAL: how-to knowledge — processes and workflows
agent.remember(
    "When Alice asks for code, use type hints and docstrings.",
    memory_type=MemoryType.INSTRUCTIONS,
)

print("Stored 4 memories (one of each type)")
print()

# ============================================================
# 2. Recall memories by query
# ============================================================
results = agent.recall("What do I know about Alice?")
print(f"=== Recall 'Alice' ({len(results)} results) ===")
for entry in results:
    print(f"  [{entry.type}] {entry.content}")
print()

# Recall by type
core_memories = agent.recall(memory_type=MemoryType.FACTS)
print(f"=== Core memories ({len(core_memories)}) ===")
for entry in core_memories:
    print(f"  {entry.content}")
print()

# ============================================================
# 3. Forget memories
# ============================================================
deleted = agent.forget(query="machine learning")
print(f"Forgot {deleted} memory(ies) matching 'machine learning'")

# Verify it's gone
remaining = agent.recall("machine learning")
print(f"Remaining matches: {len(remaining)}")
print()

# ============================================================
# 4. Use memory in conversation
# ============================================================
response = agent.run("What's my name?")
print(f"Agent says: {response.content}")

# --- Serve (uncomment to try playground) ---
# agent.serve(port=8000, enable_playground=True)
