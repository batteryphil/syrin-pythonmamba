# Handoff & Spawn

Transfer control to another agent or create sub-agents for subtasks.

## handoff()

Transfer control to another agent. Optional context and budget transfer.

```python
from syrin import Agent

class Researcher(Agent):
    model = model
    system_prompt = "You research topics."

class Writer(Agent):
    model = model
    system_prompt = "You write articles."

researcher = Researcher()
response = researcher.handoff(
    Writer,
    "Write an article based on your research",
    transfer_context=True,
    transfer_budget=False,
)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `target_agent` | `type[Agent]` | — | Target agent class |
| `task` | `str` | — | Task for target |
| `transfer_context` | `bool` | `True` | Copy persistent memories |
| `transfer_budget` | `bool` | `False` | Share remaining budget |

**Returns:** `Response[str]`

Handoff interception (e.g. `syrin.on(Hook.HANDOFF_START, fn)`) is supported; see [Events & Hooks](events-hooks.md). Step 4 extends this with richer handoff context.

---

## spawn()

Create a sub-agent to run a task. Budget can be inherited or “pocket money.”

```python
child_response = parent.spawn(
    ChildAgent,
    task="Research topic X",
    budget=Budget(run=0.10),
    max_children=5,
)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `agent_class` | `type[Agent]` | — | Sub-agent class |
| `task` | `str \| None` | `None` | Task to run immediately |
| `budget` | `Budget \| None` | `None` | Pocket money or shared |
| `max_children` | `int \| None` | `None` | Max concurrent children |

**Returns:** `Agent` if `task` is `None`, else `Response[str]`

### Budget Inheritance

- **Shared budget** (`budget.shared=True`): Child uses parent budget.
- **Pocket money**: Child gets own budget; must be ≤ parent’s remaining.

```python
parent = ParentAgent(budget=Budget(run=1.0, shared=True))
response = parent.spawn(ChildAgent, task="...")  # Child uses parent budget
```

---

## spawn_parallel()

Run multiple sub-agents in parallel.

```python
results = parent.spawn_parallel([
    (Researcher, "Research A"),
    (Analyst, "Analyze B"),
    (Writer, "Write C"),
])
# results: list[Response[str]]
```
