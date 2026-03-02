# Memory System

Syrin provides a first-class persistent memory system for AI agents. Memories are stored separately from the conversation context, allowing agents to remember information across sessions and retrieve relevant knowledge when needed.

> **Agent integration:** For the agent API (`remember()`, `recall()`, `forget()`, constructor `memory=` param), see [Agent: Memory](agent/memory.md).

## Installation

No additional dependencies required. The memory system is built into Syrin.

## Quick Start

```python
from Syrin import Agent, Model
from Syrin.memory import Memory
from Syrin.enums import MemoryType

# Agent now has persistent memory ENABLED BY DEFAULT
agent = Agent(
    # model=Model("openai/gpt-4o-mini"),
    model=Model.Almock(),  # No API Key needed
    # memory is automatically enabled with sensible defaults!
)

# Store memories
agent.remember("My name is John", memory_type=MemoryType.CORE)
agent.remember("I visited Paris yesterday", memory_type=MemoryType.EPISODIC)

# Memories are automatically injected into context!
result = agent.response("What do you know about me?")
# The agent will automatically see relevant memories in context

# Manual recall also works
memories = agent.recall("name")
print([m.content for m in memories])
# Output: ['My name is John']
```

### Disable Memory

```python
# To disable memory entirely
agent = Agent(
    # model=Model("openai/gpt-4o-mini"),
    model=Model.Almock(),  # No API Key needed
    memory=False,  # Disable memory
)
```

### Memory Configuration

```python
from Syrin.memory import Memory, Decay
from Syrin.enums import MemoryType, InjectionStrategy

agent = Agent(
    # model=Model("openai/gpt-4o-mini"),
    model=Model.Almock(),  # No API Key needed
    memory=Memory(
        types=[MemoryType.CORE, MemoryType.EPISODIC],  # Types to store
        top_k=10,  # Max memories to retrieve
        auto_store=True,  # Auto-store conversations
        decay=Decay(strategy="exponential", rate=0.99),
        injection_strategy=InjectionStrategy.ATTENTION_OPTIMIZED,
    ),
)
```

---

## Memory Types

Syrin supports four types of memory, each with different characteristics:

### Core Memory

**Purpose:** Persistent facts about the agent or user.

**Characteristics:**
- High importance (default: 0.9)
- Should rarely decay
- Stores identity, preferences, key relationships

```python
from Syrin.memory import CoreMemory

core = CoreMemory(
    id="user-name",
    content="My name is John Smith",
    importance=0.95,
)
```

**Best for:** User's name, preferences, identity, important relationships.

### Episodic Memory

**Purpose:** Specific events and experiences.

**Characteristics:**
- Medium importance (default: 0.7)
- Decays over time unless reinforced
- Stores conversations, activities, experiences

```python
from Syrin.memory import EpisodicMemory

episodic = EpisodicMemory(
    id="event-1",
    content="Yesterday I visited the Eiffel Tower",
    importance=0.6,
)
```

**Best for:** Past conversations, events, activities.

### Semantic Memory

**Purpose:** Facts and knowledge.

**Characteristics:**
- High importance (default: 0.8)
- Stores factual information
- Should decay slowly

```python
from Syrin.memory import SemanticMemory

semantic = SemanticMemory(
    id="fact-1",
    content="Paris is the capital of France",
    importance=0.85,
)
```

**Best for:** Factual knowledge, learned information, definitions.

### Procedural Memory

**Purpose:** How-to knowledge and skills.

**Characteristics:**
- High importance (default: 0.85)
- Should decay slowly (skills are remembered longer)
- Stores instructions, procedures

```python
from Syrin.memory import ProceduralMemory

procedural = ProceduralMemory(
    id="skill-1",
    content="How to make coffee: boil water, add coffee, pour hot water",
    importance=0.9,
)
```

**Best for:** Instructions, procedures, learned skills.

### Factory Function

Use `create_memory()` for flexible memory creation:

```python
from Syrin.memory import create_memory
from Syrin.enums import MemoryType

# Create memory by type - uses type-specific defaults
mem = create_memory(
    memory_type=MemoryType.CORE,
    id="my-id",
    content="My content",
)
# Returns CoreMemory with default importance 0.9
```

---

## MemoryStore

`MemoryStore` provides a complete API for managing memories independently of agents:

```python
from Syrin.memory import MemoryStore
from Syrin.enums import MemoryType

# Create a store
store = MemoryStore()

# Add memories
store.add(content="User prefers dark mode", memory_type=MemoryType.CORE)
store.add(content="Yesterday had a meeting", memory_type=MemoryType.EPISODIC)

# Add with full control
from Syrin.memory import MemoryEntry
entry = MemoryEntry(
    id="my-id",
    content="Custom memory",
    type=MemoryType.SEMANTIC,
    importance=0.8,
)
store.add(entry)

# Recall memories
results = store.recall("dark mode")  # Search by query
core_only = store.recall(memory_type=MemoryType.CORE)  # Filter by type

# List all memories
all_memories = store.list()

# Forget memories
store.forget(memory_id="my-id")  # Delete by ID
store.forget(memory_type=MemoryType.EPISODIC)  # Delete all of type

# Get statistics
stats = store.get_stats()
# {'total': 5, 'by_type': {'core': 2, 'episodic': 3}, 'by_scope': {'user': 5}}
```

---

## Decay Curves

Memories naturally lose importance over time using Ebbinghaus-inspired forgetting curves:

```python
from Syrin.memory import Decay, MemoryEntry
from Syrin.enums import DecayStrategy
from datetime import datetime, timedelta

# Configure decay
decay = Decay(
    strategy=DecayStrategy.EXPONENTIAL,  # decay type
    rate=0.95,  # how fast (higher = slower decay)
    reinforce_on_access=True,  # boost importance when accessed
    min_importance=0.1,  # floor for importance
)

# Apply decay to a memory
entry = MemoryEntry(
    id="old-memory",
    content="Old info",
    type=MemoryType.EPISODIC,
    importance=1.0,
    created_at=datetime.now() - timedelta(hours=24),
)

decay.apply(entry)
print(entry.importance)  # ~0.29 after 24 hours

# Access reinforces the memory
decay.on_access(entry)
print(entry.importance)  # ~0.30 after access
print(entry.access_count)  # 1
```

### Decay Strategies

| Strategy | Description | Use Case |
|----------|-------------|----------|
| `EXPONENTIAL` | Fast initial decay, then slower | Typical forgetfulness |
| `LINEAR` | Constant rate | Gradual aging |
| `LOGARITHMIC` | Slow initial, faster later | Long-term retention |
| `STEP` | Decay at intervals | Scheduled forgetting |
| `NONE` | No decay | Permanent memories |

### Using Decay with MemoryStore

```python
from Syrin.memory import MemoryStore, Decay
from Syrin.enums import DecayStrategy

store = MemoryStore(
    decay=Decay(
        strategy=DecayStrategy.EXPONENTIAL,
        rate=0.99,  # Slow decay
    )
)

# Add memories
store.add(content="Recent info", memory_type=MemoryType.EPISODIC)

# Recall applies decay automatically
results = store.recall("recent info", apply_decay=True)
```

---

## Budget Awareness

Memory operations can respect budget constraints:

```python
from Syrin.memory import MemoryStore, MemoryBudget
from syrin.budget import warn_on_exceeded

# Create budget constraints
budget = MemoryBudget(
    extraction_budget=0.01,  # $0.01 max per operation
    on_exceeded=warn_on_exceeded,  # warn but allow
)

store = MemoryStore(budget=budget)

# Operations check budget
store.add(content="Short fact")  # Under budget - succeeds

# Budget exceeded behavior:
# - WARN: Allow but log warning
# - ERROR: Raise exception
# - STOP: Allow but don't process
```

---

## Agent Integration

The simplest way to use memory is with the Agent:

```python
from Syrin import Agent, Model
from Syrin.memory import Memory
from Syrin.enums import MemoryType

# Create agent with persistent memory
agent = Agent(
    # model=Model("openai/gpt-4o-mini"),
    model=Model.Almock(),  # No API Key needed
    memory=Memory(  # Accepts Memory or ConversationMemory
        types=[MemoryType.CORE, MemoryType.EPISODIC],
        top_k=10,
    ),
)

# Store memories
agent.remember("My name is John", memory_type=MemoryType.CORE)
agent.remember("I work at Acme Corp", memory_type=MemoryType.CORE)

# Recall memories
memories = agent.recall()  # All memories
core_memories = agent.recall(memory_type=MemoryType.CORE)  # Filtered
search_results = agent.recall("Acme")  # Search

# Forget specific memory
agent.forget(memory_id="memory-id-here")

# Or forget all of a type
agent.forget(memory_type=MemoryType.EPISODIC)
```

### Configuration Options

```python
from Syrin.memory import Memory
from Syrin.enums import MemoryType, MemoryBackend, InjectionStrategy, WriteMode

memory = Memory(
    # Which memory types to store
    types=[
        MemoryType.CORE,
        MemoryType.EPISODIC,
        MemoryType.SEMANTIC,
        MemoryType.PROCEDURAL,
    ],
    
    # Storage backend (see below)
    backend=MemoryBackend.SQLITE,
    path="./memory.db",  # For SQLite backend
    
    # Retrieval settings
    top_k=10,  # Max memories to retrieve
    relevance_threshold=0.7,  # Minimum relevance score
    
    # How to inject into context
    injection_strategy=InjectionStrategy.ATTENTION_OPTIMIZED,
    
    # Auto-store conversations
    auto_store=False,  # Automatically store user/agent messages
    
    # Decay settings
    decay=Decay(strategy=DecayStrategy.EXPONENTIAL, rate=0.99),
    
    # Budget
    memory_budget=MemoryBudget(extraction_budget=0.01),
    
    # Scope (session, agent, user, global)
    scope=MemoryScope.USER,
    
    # Write mode: SYNC blocks until complete; ASYNC fire-and-forget (default)
    write_mode=WriteMode.ASYNC,
)
```

---

## Write Mode

Memory writes (`remember`, `forget`) can run synchronously or asynchronously:

```python
from syrin.enums import WriteMode

# ASYNC (default): Fire-and-forget. Returns immediately; never blocks response.
memory = Memory(write_mode=WriteMode.ASYNC)

# SYNC: Blocks until complete. Use when you need the actual count or immediate persistence.
memory = Memory(write_mode=WriteMode.SYNC)
```

| Mode | remember() | forget() | Use Case |
|------|------------|----------|----------|
| ASYNC | Returns True immediately | Returns 0/1 placeholder | Latency-sensitive, real-time agents |
| SYNC | Blocks until stored | Returns actual count | Export/import, tests, batch ops |

---

## Export / Import

Export memories as a snapshot for backup, GDPR export, or migration:

```python
from syrin.memory import Memory, MemorySnapshot

memory = Memory(backend=MemoryBackend.SQLITE, path="./memory.db")

# Export to snapshot (JSON-serializable)
snapshot = memory.export()
print(snapshot.version)  # 1
print(len(snapshot.memories))

# Serialize to JSON
import json
js = snapshot.to_json()
data = json.loads(js)

# Import from snapshot (append mode; does not clear existing)
snapshot = MemorySnapshot.from_dict(data)
count = memory.import_from(snapshot)
print(f"Imported {count} memories")
```

---

## Storage Backends

Syrin supports multiple storage backends:

### In-Memory (Default)

```python
from Syrin.enums import MemoryBackend

memory = Memory(
    backend=MemoryBackend.MEMORY,  # Default
)
```
- **Pros:** Fast, no setup required
- **Cons:** Data lost when agent restarts
- **Use case:** Testing, ephemeral agents

### SQLite (Persistent)

```python
from Syrin.enums import MemoryBackend

memory = Memory(
    backend=MemoryBackend.SQLITE,
    path="./memory.db",  # Optional: defaults to ~/.syrin/memory.db
)
```
- **Pros:** Persistent, single file, no external dependencies
- **Cons:** Single machine, basic search
- **Use case:** Single-user applications, persistent memory

### Qdrant (Vector/Semantic Search)

```bash
pip install syrin[qdrant]  # or pip install qdrant-client
```

```python
from syrin.enums import MemoryBackend
from syrin.memory import Memory, QdrantConfig

# Local embedded Qdrant
memory = Memory(
    backend=MemoryBackend.QDRANT,
    qdrant=QdrantConfig(
        path="./qdrant_data",       # Local path for embedded
        collection="syrin_memories",
        namespace="tenant_123",     # Per-tenant isolation
    ),
)

# Qdrant Cloud or remote server
memory = Memory(
    backend=MemoryBackend.QDRANT,
    qdrant=QdrantConfig(
        url="https://xxx.qdrant.tech",
        api_key="your-api-key",    # For Qdrant Cloud
        collection="syrin_memories",
        namespace="user_alice",
    ),
)

# Legacy: path-only (embedded, no config)
memory = Memory(backend=MemoryBackend.QDRANT, path="./qdrant_data")
```
- **Pros:** Semantic search with embeddings, persistent, namespace isolation
- **Cons:** Requires Qdrant (embedded or server)
- **Use case:** Production with semantic search, multi-tenant apps

### Chroma (Lightweight Vector DB)

```bash
pip install syrin[chroma]  # or pip install chromadb
```

```python
from syrin.enums import MemoryBackend
from syrin.memory import Memory, ChromaConfig

memory = Memory(
    backend=MemoryBackend.CHROMA,
    chroma=ChromaConfig(
        path="./chroma_db",         # Local persistent; None = ephemeral
        collection="syrin_memories",
    ),
)
```
- **Pros:** Lightweight, embedded, zero-config local
- **Cons:** Fewer features than Qdrant
- **Use case:** Prototyping, local development

### Redis (Fast Cache)

```bash
pip install syrin[redis]  # or pip install redis
```

```python
from syrin.enums import MemoryBackend
from syrin.memory import Memory, RedisConfig

memory = Memory(
    backend=MemoryBackend.REDIS,
    redis=RedisConfig(
        host="localhost",
        port=6379,
        db=0,
        prefix="syrin:memory:",  # Key prefix for isolation
        ttl=86400,  # Optional: TTL in seconds (1 day)
    ),
)
```
- **Pros:** Ultra-fast, distributed, TTL support
- **Cons:** Requires Redis server; no vector search (substring match only)
- **Use case:** High-performance applications, distributed systems
- **Example:** `examples/04_memory/redis_memory.py`

### PostgreSQL (Production)

```bash
pip install syrin[postgres]  # or pip install psycopg2-binary
```

```python
from syrin.enums import MemoryBackend
from syrin.memory import Memory, PostgresConfig

memory = Memory(
    backend=MemoryBackend.POSTGRES,
    postgres=PostgresConfig(
        host="localhost",
        port=5432,
        database="syrin",
        user="postgres",
        password="",
        table="memories",
    ),
)
```
- **Pros:** Enterprise-grade, SQL support, persistent
- **Cons:** Requires PostgreSQL server; vector search needs pgvector
- **Use case:** Production applications, team usage
- **Example:** `examples/04_memory/postgres_memory.py`

**Note:** For vector search with PostgreSQL, install pgvector and set `vector_size`:

```bash
pip install pgvector
```

```python
memory = Memory(
    backend=MemoryBackend.POSTGRES,
    postgres=PostgresConfig(
        host="localhost",
        database="syrin",
        vector_size=384,  # Enable pgvector
    ),
)
```

---

## Comparison

| Backend | Type | Semantic Search | Setup | Use Case |
|---------|------|---------------|-------|----------|
| MEMORY | In-memory | ❌ | None | Testing |
| SQLITE | File | ❌ | None | Simple apps |
| QDRANT | Vector | ✅ | Server | Semantic search |
| CHROMA | Vector | ✅ | Embedded | Prototyping |
| REDIS | Cache | ❌ | Server | Fast, distributed |
| POSTGRES | Database | Optional | Server | Production |
```

---

## Conversation Memory

For session-based memory (not persistent):

```python
from Syrin import Agent, Model
from Syrin.memory import BufferMemory, WindowMemory

# BufferMemory - keeps all messages
agent = Agent(
    # model=Model("openai/gpt-4o-mini"),
    model=Model.Almock(),  # No API Key needed
    memory=BufferMemory(),  # Session memory
)

# WindowMemory - keeps last K message pairs
agent = Agent(
    # model=Model("openai/gpt-4o-mini"),
    model=Model.Almock(),  # No API Key needed
    memory=WindowMemory(k=10),  # Last 10 pairs
)
```

---

## Observability

Memory operations are automatically traced:

```python
from Syrin.observability import set_debug, get_tracer

# Enable debug mode for console output
set_debug(True)

agent = Agent(
    # model=Model("openai/gpt-4o-mini"),
    model=Model.Almock(),  # No API Key needed
    memory=Memory(),
)

agent.remember("Test", memory_type=MemoryType.CORE)
agent.recall("Test")

# Check console for span output
set_debug(False)
```

Each operation creates spans:
- `memory.store` - When storing a memory
- `memory.recall` - When retrieving memories
- `memory.forget` - When deleting memories

---

## Event Hooks

Connect memory operations to your event system:

```python
from Syrin.memory import MemoryStore
from Syrin.events import Events, EventContext

events = Events(lambda e, ctx: print(f"Event: {e}"))

store = MemoryStore(events=events)

# Register handlers
events.on("memory.store", lambda ctx: print(f"Stored: {ctx.get('memory_id')}"))
events.on("memory.recall", lambda ctx: print(f"Recalled: {ctx.get('results_count')}"))

store.add(content="Test")
store.recall("test")
```

---

## Memory Entry Reference

```python
from Syrin.memory import MemoryEntry
from Syrin.enums import MemoryType, MemoryScope
from datetime import datetime

entry = MemoryEntry(
    id="unique-id",
    content="The memory content",
    type=MemoryType.EPISODIC,
    importance=0.8,  # 0.0 - 1.0
    
    # Metadata
    scope=MemoryScope.USER,
    source="user_input",  # Where it came from
    
    # Temporal
    created_at=datetime.now(),
    valid_from=None,  # When it becomes valid
    valid_until=None,  # When it expires
    
    # Organization
    keywords=["python", "programming"],
    related_ids=["other-memory-id"],
    supersedes=None,  # ID of memory this replaces
    
    # Extra data
    metadata={"custom": "data"},
)
```

---

## Import Reference

```python
# Core classes
from Syrin.memory import (
    Memory,           # Agent memory configuration
    MemoryStore,      # Standalone storage
    MemoryEntry,      # Individual memory
    Decay,            # Decay configuration
    MemoryBudget,     # Budget constraints
)

# Memory types
from Syrin.memory import (
    CoreMemory,
    EpisodicMemory,
    SemanticMemory,
    ProceduralMemory,
    create_memory,    # Factory function
)

# Conversation memory
from Syrin.memory import (
    ConversationMemory,  # Base class
    BufferMemory,        # Keep all
    WindowMemory,       # Keep last K
)

# Backends
from syrin.memory import (
    ChromaConfig,        # Chroma config (path, collection)
    QdrantConfig,        # Qdrant config (url, path, namespace)
    InMemoryBackend,     # In-memory (default)
    SQLiteBackend,       # File-based SQLite
    QdrantBackend,       # Vector database (semantic search)
    ChromaBackend,       # Lightweight vector DB
    RedisBackend,        # Fast cache
    PostgresBackend,     # PostgreSQL
    get_backend,         # Factory function
    BACKENDS,            # Registry of all backends
)

# Snapshot (export/import)
from syrin.memory import MemorySnapshot, MemorySnapshotEntry

# Enums
from Syrin.enums import (
    MemoryType,        # CORE, EPISODIC, SEMANTIC, PROCEDURAL
    MemoryScope,       # SESSION, AGENT, USER, GLOBAL
    DecayStrategy,     # EXPONENTIAL, LINEAR, LOGARITHMIC, STEP, NONE
    MemoryBackend,    # MEMORY, SQLITE, QDRANT, etc.
    WriteMode,        # SYNC, ASYNC
    InjectionStrategy,# CHRONOLOGICAL, ATTENTION_OPTIMIZED, etc.
)
```

---

## Examples

### Example 1: Basic Agent Memory

```python
from Syrin import Agent, Model
from Syrin.memory import Memory
from Syrin.enums import MemoryType

class Assistant(Agent):
    model = Model("openai/gpt-4o-mini")
    system_prompt = "You are a helpful assistant."

# Create with memory
assistant = Assistant(memory=Memory())

# Store preferences
assistant.remember("User prefers concise answers", memory_type=MemoryType.CORE)
assistant.remember("User likes Python", memory_type=MemoryType.SEMANTIC)

# Later session - agent recalls preferences
result = assistant.response("What's Python?")
print(result.content)
```

### Example 2: Multi-Type Memory

```python
from Syrin import Agent, Model
from Syrin.memory import Memory
from Syrin.enums import MemoryType

agent = Agent(
    # model=Model("openai/gpt-4o-mini"),
    model=Model.Almock(),  # No API Key needed
    memory=Memory(types=[
        MemoryType.CORE,       # User identity
        MemoryType.EPISODIC,   # Past conversations
        MemoryType.SEMANTIC,   # Learned facts
        MemoryType.PROCEDURAL, # Skills/instructions
    ])
)

# Store different types
agent.remember("I am Alice", memory_type=MemoryType.CORE)
agent.remember("Yesterday: Had lunch at 12pm", memory_type=MemoryType.EPISODIC)
agent.remember("Python: High-level programming language", memory_type=MemoryType.SEMANTIC)
agent.remember("Git: git commit -m 'message'", memory_type=MemoryType.PROCEDURAL)

# Recall specific type
facts = agent.recall(memory_type=MemoryType.SEMANTIC)
```

### Example 3: Standalone MemoryStore

```python
from Syrin.memory import MemoryStore
from Syrin.enums import MemoryType

# Create standalone store
store = MemoryStore()

# Add many memories
for i in range(100):
    store.add(
        content=f"Memory {i}",
        memory_type=MemoryType.EPISODIC,
    )

# Search
results = store.recall("Memory 5")
print(f"Found: {len(results)}")

# Get stats
print(store.get_stats())
# {'total': 100, 'by_type': {'episodic': 100}, 'by_scope': {'user': 100}}

# Cleanup
store.clear()
```

### Example 4: Custom Decay

```python
from Syrin.memory import MemoryStore, Decay
from Syrin.enums import DecayStrategy

# Fast decay for ephemeral info
store = MemoryStore(
    decay=Decay(
        strategy=DecayStrategy.EXPONENTIAL,
        rate=0.90,  # Fast decay
        min_importance=0.05,
    )
)

# No decay for permanent info
perm_store = MemoryStore(
    decay=Decay(strategy=DecayStrategy.NONE)
)
```

### Example 5: Handoff with Memory Transfer

```python
from Syrin import Agent, Model
from Syrin.memory import Memory
from Syrin.enums import MemoryType

class Analyzer(Agent):
    model = Model("openai/gpt-4o-mini")
    system_prompt = "You analyze data."

class Presenter(Agent):
    model = Model("openai/gpt-4o-mini")
    system_prompt = "You present findings."

# Analyzer stores findings
analyzer = Analyzer(memory=Memory())
analyzer.remember("Key finding: 40% growth", memory_type=MemoryType.SEMANTIC)

# Handoff transfers memories
result = analyzer.handoff(Presenter, "Present the findings", transfer_context=True)
```

---

## Summary

| Feature | Description |
|---------|-------------|
| **4 Memory Types** | Core, Episodic, Semantic, Procedural |
| **Decay Curves** | Exponential, Linear, Logarithmic, Step, None |
| **Budget Aware** | Control costs per operation |
| **Observability** | Built-in tracing with spans |
| **Event Hooks** | Connect to your event system |
| **Multiple Backends** | Memory, SQLite, Qdrant, Chroma, Redis, Postgres |
| **Agent Integration** | Seamless with `Agent.remember/recall/forget` |
| **Search** | Query-based and type-filtered retrieval |
