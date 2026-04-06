---
title: Budget Delegation
description: How agents carve budget slices for child agents — self.spawn(), self.spawn_many(), SpawnResult, and BudgetAllocationError
weight: 74
---

## Budget Flows Downward

When an agent spawns a child, it carves a slice of its remaining budget for that child. The child cannot spend more than its allocation. Any unused budget flows back to the parent's pool when the child finishes.

This creates a hierarchy of budget control — an orchestrator can spawn ten analysts and guarantee total cost never exceeds its budget, regardless of what each analyst does.

## Spawning a Single Child

Use `self.spawn()` inside any agent method:

```python
from syrin import Agent, Budget, Model
from syrin.response import Response
from syrin.swarm._spawn import SpawnResult

class OrchestratorAgent(Agent):
    model = Model.mock()
    system_prompt = "Decompose tasks and delegate to specialists."

    async def arun(self, input_text: str) -> Response[str]:
        result: SpawnResult = await self.spawn(
            AnalysisAgent,
            task="Analyse the market data",
            budget=1.00,
        )
        print(result.content)
        print(f"Spent: ${result.cost:.4f}")
        print(f"Pool remaining: ${result.budget_remaining:.4f}")
        return Response(content=result.content, cost=result.cost)
```

When you pass `budget=1.00`, Syrin allocates $1.00 from the parent's remaining pool, runs the child with that allocation, and returns any unused portion to the pool when the child finishes.

If `budget` exceeds the parent's remaining balance, `ValueError` is raised immediately — no LLM call is made.

When `budget` is omitted and the parent has a budget, the child borrows the full remaining balance and returns unused funds automatically.

`SpawnResult` has five attributes:

`content` — The child agent's output text.

`cost` — Actual cost incurred by the child in USD.

`budget_remaining` — Remaining pool balance after the child completes, in USD.

`stop_reason` — Why the child terminated (`END_TURN`, `BUDGET`, etc.).

`child_agent_id` — Unique ID in `"parent::child::uuid"` format.

## Spawning Multiple Children Concurrently

Use `self.spawn_many()` with a list of `SpawnSpec` objects to run children in parallel:

```python
from syrin.swarm._spawn import SpawnResult, SpawnSpec

class OrchestratorAgent(Agent):
    model = Model.mock()

    async def arun(self, input_text: str) -> Response[str]:
        specs = [
            SpawnSpec(agent=ResearchAgent, task="Find papers on topic A", budget=0.50),
            SpawnSpec(agent=ResearchAgent, task="Find papers on topic B", budget=0.50),
            SpawnSpec(agent=SummaryAgent,  task="Summarise all findings",  budget=0.25),
        ]
        results: list[SpawnResult] = await self.spawn_many(specs)

        combined = "\n\n".join(r.content for r in results)
        total_cost = sum(r.cost for r in results)
        return Response(content=combined, cost=total_cost)
```

Budget allocations happen concurrently from the pool. If any allocation fails (not enough budget), the others still proceed. Partial failure is the default behavior.

`SpawnSpec` has four fields: `agent` (the agent class to instantiate, required), `task` (the task string, required), `budget` (the allocation in USD, required), and `timeout` (optional per-child timeout in seconds).

## Budget Pool in a Swarm

When you pass a `Budget` to a `Swarm`, budget sharing is automatic — every agent in the swarm draws from the same pool. No extra flags needed:

```python
from syrin import Budget
from syrin.swarm import Swarm

budget = Budget(max_cost=10.00)

swarm = Swarm(
    agents=[OrchestratorAgent()],
    goal="...",
    budget=budget,
)
```

If a child agent's requested allocation exceeds the pool's remaining balance, `BudgetAllocationError` is raised:

```python
from syrin.budget._pool import BudgetAllocationError

try:
    result = await self.spawn(HeavyAgent, task="...", budget=5.00)
except BudgetAllocationError as e:
    print(f"Cannot allocate ${e.requested:.2f}; pool has ${e.available:.2f}")
```

## How Budget Flows

Here's the flow for a two-child spawn with a $10 pool:

```
Pool: $10.00
  └── OrchestratorAgent borrows from pool:
        ├── ChildA allocated $2.00 → spends $1.50 → returns $0.50 to pool
        └── ChildB allocated $1.50 → spends $1.50 → returns $0.00 to pool
  Pool remaining: $10.00 - $1.50 - $1.50 = $7.00
```

Unused budget from each child returns to the pool automatically after the child completes.

## Hooks

```python
from syrin.enums import Hook

agent.events.on(Hook.SPAWN_START, lambda ctx: print(
    f"Spawning {ctx['child_agent']} with budget ${ctx['child_budget']:.2f}"
))

agent.events.on(Hook.SPAWN_END, lambda ctx: print(
    f"Child done: cost=${ctx['cost']:.4f}, duration={ctx['duration']:.1f}s"
))
```

`Hook.SPAWN_START` context includes `source_agent`, `child_agent`, `child_task`, and `child_budget`.

`Hook.SPAWN_END` context includes `source_agent`, `child_agent`, `child_task`, `cost`, and `duration`.

## Runtime Budget Reallocation

After agents are spawned and running, a supervisor can adjust their allocations without stopping or restarting them.  Two operations are available on `SwarmController`:

| Method | What it does |
|--------|-------------|
| `topup_budget(agent, amount)` | Adds `amount` on top of the agent's current allocation |
| `reallocate_budget(agent, new_amount)` | Replaces the agent's allocation with `new_amount` entirely |

Both are atomic (backed by `asyncio.Lock`), require `CONTROL` permission, and are recorded in the audit log as `ControlAction.TOPUP_BUDGET` / `ControlAction.REALLOCATE_BUDGET`.

### Wiring it up

Pass a `BudgetPool` to `SwarmController` at construction time.  The same pool must be shared with the agents whose allocations you want to adjust.

```python
import asyncio
from syrin.budget._pool import BudgetPool
from syrin.enums import AgentRole
from syrin.swarm._authority import SwarmAuthorityGuard
from syrin.swarm._control import AgentStateSnapshot, AgentStatus, SwarmController

async def main() -> None:
    pool = BudgetPool(total=10.00)
    await pool.allocate("cto", 2.00)
    await pool.allocate("marketing", 3.00)

    guard = SwarmAuthorityGuard(
        roles={
            "ceo": AgentRole.ORCHESTRATOR,
            "cto": AgentRole.WORKER,
            "marketing": AgentRole.WORKER,
        },
        teams={"ceo": ["cto", "marketing"]},
    )

    ctrl = SwarmController(
        actor_id="ceo",
        guard=guard,
        state_registry={...},   # your AgentStateSnapshot dict
        task_registry={},
        budget_pool=pool,        # ← wire the pool here
    )

    # CTO is performing well — give it $1 more
    await ctrl.topup_budget("cto", 1.00)
    # Marketing underspent — trim its cap, return funds to pool
    await ctrl.reallocate_budget("marketing", 1.50)

asyncio.run(main())
```

### `topup_budget(target, additional)`

Draws `additional` from the pool's free balance and adds it to the agent's allocation.  The agent keeps running — no restart required.

```python
# CTO started with $2; add $1.50 mid-run
await ctrl.topup_budget("cto", 1.50)
# pool.snapshot()["cto"]["allocated"] → 3.50
# pool.remaining reduced by 1.50
```

Raises `BudgetAllocationError` if:
- The agent has no active allocation (not yet spawned or already finished)
- The pool has insufficient remaining balance
- The new total would exceed `per_agent_max`

### `reallocate_budget(target, new_amount)`

Replaces the agent's allocation with `new_amount`.

- **Increasing** (`new_amount > current`): draws the difference from the pool
- **Decreasing** (`new_amount < current`): returns the difference to the pool

```python
# Raise CTO's cap from $2 → $5 (draws $3 from pool)
await ctrl.reallocate_budget("cto", 5.00)

# Trim marketing from $3 → $1.50 (returns $1.50 to pool)
await ctrl.reallocate_budget("marketing", 1.50)
```

Raises `BudgetAllocationError` if:
- The agent has no active allocation
- `new_amount` is less than what the agent has already spent
- Pool has insufficient balance (when increasing)
- `new_amount` would exceed `per_agent_max`

### CEO rebalancing departments mid-run

The pattern works naturally for the Virtual Office use case: CEO watches
mid-run cost snapshots and rebalances across departments without pausing anyone.

```python
snap = pool.snapshot()

# Engineering is close to its limit — top it up
if snap["engineering"]["spent"] / snap["engineering"]["allocated"] > 0.80:
    await ctrl.topup_budget("engineering", 2.00)

# Marketing underspent — reclaim unused allocation
if snap["marketing"]["spent"] < snap["marketing"]["allocated"] * 0.30:
    await ctrl.reallocate_budget(
        "marketing",
        max(snap["marketing"]["spent"], snap["marketing"]["allocated"] * 0.50),
    )
```

### Audit trail

Every `topup_budget` and `reallocate_budget` call is recorded in the authority guard's audit log:

```python
for entry in ctrl._guard.audit_log():
    print(f"{entry.action:<24} {entry.actor_id} → {entry.target_id}")

# reallocate_budget        ceo → marketing
# topup_budget             ceo → engineering
```

## Cross-Swarm Budget

Budget delegation is scoped to the owning swarm's `BudgetPool`. Agents from different swarms don't share pools. If you need cross-swarm budgeting, create a single swarm with a shared budget and structure your agents as a hierarchy within it.

## See Also

- [Swarm](/multi-agent/swarm) — Swarm topologies
- [Hierarchy](/multi-agent/hierarchy) — Multi-level agent hierarchies, Virtual Office pattern
- [Budget](/core/budget) — Budget configuration
