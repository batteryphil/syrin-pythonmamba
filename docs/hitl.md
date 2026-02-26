# Human-in-the-Loop (HITL)

HITL gates tool execution behind human approval. Use for safety-critical or high-risk operations.

## Per-Tool Approval

Mark specific tools with `requires_approval=True` and pass an `ApprovalGate`:

```python
from syrin import Agent, ApprovalGate, tool

@tool(requires_approval=True, description="Delete a record")
def delete_record(id: str) -> str:
    return f"Deleted {id}"

def approve(msg: str, timeout: int, ctx: dict) -> bool:
    return input(f"Approve? [y/n]: ").lower() == "y"

gate = ApprovalGate(callback=approve)
agent = Agent(
    model=model,
    tools=[delete_record],
    approval_gate=gate,
    hitl_timeout=300,
)
agent.response("Delete record abc123")
```

## ApprovalGate

`ApprovalGate(callback=fn)` — callback receives `(message, timeout, context)` and returns `bool`. Sync or async.

```python
async def slack_approve(msg: str, timeout: int, ctx: dict) -> bool:
    # Post to Slack, wait for reaction
    return await wait_for_slack_reaction(msg, timeout=timeout)

gate = ApprovalGate(callback=slack_approve)
```

## All-Tools Approval: HumanInTheLoop

For safety-critical agents where every tool needs approval, use the HumanInTheLoop loop:

```python
from syrin import Agent, HumanInTheLoop

async def approve(tool_name: str, args: dict) -> bool:
    return input(f"Approve {tool_name}? [y/n]: ").lower() == "y"

agent = Agent(
    model=model,
    tools=[...],
    loop=HumanInTheLoop(approve=approve, timeout=300),
)
```

Or with ApprovalGate:

```python
gate = ApprovalGate(callback=lambda msg, t, ctx: ...)
agent = Agent(..., loop=HumanInTheLoop(approval_gate=gate, timeout=300))
```

## Timeout

`hitl_timeout` (Agent) or `timeout` (HumanInTheLoop): seconds to wait. On timeout, treat as rejection.

## Hooks

| Hook | When |
|------|------|
| `Hook.HITL_PENDING` | Before waiting for approval |
| `Hook.HITL_APPROVED` | Approval granted |
| `Hook.HITL_REJECTED` | Rejected or timeout |

```python
agent.events.on(Hook.HITL_PENDING, lambda ctx: log_pending(ctx.name, ctx.arguments))
agent.events.on(Hook.HITL_REJECTED, lambda ctx: alert_rejection(ctx.name))
```
