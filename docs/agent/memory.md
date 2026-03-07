# Memory

> **Full config & backends:** For Memory, backends (Redis, Chroma, etc.), memory types, decay, and standalone usage, see [Memory System](../memory.md).

Agents have two kinds of memory: **persistent memory** and **conversation memory**.

## Persistent Memory

Stores long-term facts via `remember()`, `recall()`, and `forget()`. Retrieved content is injected into the context before each request.

### remember()

```python
memory_id = agent.remember(
    "User prefers dark mode",
    memory_type=MemoryType.EPISODIC,
    importance=0.8,
    user_id="user_123",
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `content` | `str` | — | Text to store |
| `memory_type` | `MemoryType` | `EPISODIC` | Memory type |
| `importance` | `float` | `1.0` | 0.0–1.0 |
| `**metadata` | Any | — | Extra fields |

**Returns:** `str` (memory ID)

### recall()

```python
memories = agent.recall(
    query="user preferences",
    memory_type=MemoryType.EPISODIC,
    limit=10,
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `query` | `str \| None` | `None` | Search query |
| `memory_type` | `MemoryType \| None` | `None` | Filter by type |
| `limit` | `int` | `10` | Max results |

**Returns:** `list[MemoryEntry]`

### forget()

```python
deleted = agent.forget(memory_id="abc-123")
deleted = agent.forget(query="obsolete")
deleted = agent.forget(memory_type=MemoryType.EPISODIC)
```

**Returns:** `int` (number of deleted memories)

---

## Memory Types

| Type | Use |
|------|-----|
| `CORE` | Identity, stable preferences |
| `EPISODIC` | Past events and interactions |
| `SEMANTIC` | Facts and concepts |
| `PROCEDURAL` | Learned patterns and behaviors |

---

## Memory

Control behavior via `memory=Memory(...)`:

```python
from syrin.memory.config import Memory as Memory
from syrin.enums import MemoryType, MemoryBackend

agent = Agent(
    model=model,
    memory=Memory(
        backend=MemoryBackend.MEMORY,
        types=[MemoryType.CORE, MemoryType.EPISODIC],
        top_k=10,
        relevance_threshold=0.7,
        auto_store=False,
    ),
)
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `backend` | `MEMORY` | Storage backend |
| `types` | `[CORE, EPISODIC, SEMANTIC, PROCEDURAL]` | Enabled types |
| `top_k` | `10` | Max recalled memories per request |
| `relevance_threshold` | `0.7` | Relevance cutoff |
| `auto_store` | `False` | Auto-store turns as episodic memories |
| `path` | `None` | Path for file-based backends |

---

## Auto-Injection

On each request, the agent:

1. Calls `recall(user_input)` with `top_k` and `relevance_threshold`
2. Formats results as "## Relevant Memories:"
3. Adds them to the prompt

---

## Memory

Single memory type for conversation and optional persistent recall:

```python
from syrin.memory import Memory

agent = Agent(
    model=model,
    memory=Memory(),  # Default when omitted.
)
```

- Default: `memory=Memory()` when omitted (multi-turn enabled).
- `memory=None` or `memory=False` — Disable memory (no history, no recall).

---

## Properties

- `agent.memory` — Current memory config (persistent or conversation).
- `agent.conversation_memory` — Conversation memory, if set (read-only).
- `agent.persistent_memory` — Persistent memory config, if set.

---

## See Also

- [Memory System](../memory.md) — Full config, backends, types, standalone usage
- [Use Case 3: Agent with Memory](../agent-with-memory.md)
