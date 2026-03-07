# Checkpointing

> **Config & backends:** For CheckpointConfig, storage backends (sqlite, filesystem), and CheckpointTrigger details, see [Checkpoints](../checkpoint.md).

Checkpoints save and restore agent state so you can resume runs or recover from failures.

## Configuration

```python
from syrin import Agent
from syrin.checkpoint import CheckpointConfig
from syrin.enums import CheckpointTrigger

agent = Agent(
    model=model,
    checkpoint=CheckpointConfig(
        enabled=True,
        storage="sqlite",
        path="/tmp/agent_checkpoints.db",
        trigger=CheckpointTrigger.STEP,
        max_checkpoints=10,
    ),
)
```

## CheckpointTrigger

| Trigger | When |
|---------|------|
| `MANUAL` | Only when `save_checkpoint()` is called |
| `STEP` | After each response step |
| `TOOL` | After each tool call |
| `ERROR` | On exception |
| `BUDGET` | On budget exceeded |

## Manual Checkpoints

### save_checkpoint()

```python
checkpoint_id = agent.save_checkpoint(
    name="my_agent",
    reason="before_expensive_step",
)
```

**Returns:** `str | None` (checkpoint ID or `None` if disabled)

### load_checkpoint()

```python
success = agent.load_checkpoint(checkpoint_id)
```

**Returns:** `bool`

### list_checkpoints()

```python
ids = agent.list_checkpoints(name="my_agent")
```

**Returns:** `list[str]`

## get_checkpoint_report()

```python
report = agent.get_checkpoint_report()
# report.checkpoints.saves
# report.checkpoints.loads
```

## Checkpoint State

State includes:

- `iteration` — Loop iteration count (restored on load)
- `messages` — Conversation history from `agent.messages` (restored to conversation memory)
- `memory_data`
- `budget_state`
- `checkpoint_reason`
- `context_snapshot` — Point-in-time context view (breakdown, utilization, provenance; for debug/viz)

On **load_checkpoint**, the agent restores: `messages` → conversation memory via `load_messages()`, `iteration` → `_last_iteration`, and `budget_state` (when budget is configured).

## Long-running sessions

For agents that run across multiple sessions (restarts, long runs):

1. **Checkpoint** — Save and restore conversation + iteration
2. **Memory** — Use `BufferMemory` or `WindowMemory` for session history (restored from checkpoint)
3. **auto_compact_at** — Proactive compaction (e.g. 60%) to reduce context rot

Example: `examples/12_checkpoints/long_running_agent.py`

```python
from syrin import Agent, CheckpointConfig, CheckpointTrigger, Context
from syrin.memory.conversation import BufferMemory

mem = BufferMemory()
agent = Agent(
    model=model,
    memory=mem,
    context=Context(auto_compact_at=0.6),
    checkpoint=CheckpointConfig(storage="sqlite", path="/tmp/agent.db", trigger=CheckpointTrigger.STEP),
)
# ... run conversation ...
cid = agent.save_checkpoint()
# On restart:
agent2 = Agent(..., memory=BufferMemory(), checkpoint=...)
agent2.load_checkpoint(cid)  # Restores messages + iteration
```

For multi-user or multi-conversation setups, use `save_checkpoint(name=f"{agent_name}_{conversation_id}")` to scope checkpoints per conversation.

## Automatic Checkpoints

With `trigger=STEP` or `TOOL`, checkpoints are saved automatically at the configured points. With `ERROR` or `BUDGET`, they are saved when those events occur.

## See Also

- [Checkpoints](../checkpoint.md) — Full config, storage backends, triggers
