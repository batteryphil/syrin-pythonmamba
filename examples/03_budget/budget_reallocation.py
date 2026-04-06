"""Runtime Budget Reallocation — topup and reallocate agent budgets mid-run.

Once agents are running inside a swarm, the orchestrator (CEO, supervisor) can
adjust their budgets without stopping or restarting them.

Two operations:

  topup_budget(agent, amount)     — add more budget on top of what is already
                                    allocated.  Use when an agent is producing
                                    good results and you want to let it do more.

  reallocate_budget(agent, amt)   — replace the agent's allocation entirely.
                                    Use when you want to raise a cap, trim a
                                    department, or rebalance across workers.

Both operations are async-safe (backed by asyncio.Lock), permission-checked
(CONTROL permission required), and recorded in the SwarmController audit log.

Run:
    python examples/03_budget/budget_reallocation.py
"""

from __future__ import annotations

import asyncio

from syrin.budget._pool import BudgetPool
from syrin.budget.exceptions import BudgetAllocationError
from syrin.enums import AgentRole, AgentStatus, ControlAction
from syrin.swarm._authority import SwarmAuthorityGuard
from syrin.swarm._control import AgentStateSnapshot, SwarmController

# ── helpers ──────────────────────────────────────────────────────────────────


def _make_snapshot(agent_id: str, supervisor_id: str | None = None) -> AgentStateSnapshot:
    return AgentStateSnapshot(
        agent_id=agent_id,
        status=AgentStatus.RUNNING,
        role=AgentRole.WORKER,
        last_output_summary="",
        cost_spent=0.0,
        task="working",
        context_override=None,
        supervisor_id=supervisor_id,
    )


def _print_pool(pool: BudgetPool, label: str = "") -> None:
    snap = pool.snapshot()
    if label:
        print(f"\n  [{label}]")
    for agent_id, entry in snap.items():
        print(
            f"    {agent_id:<20} allocated=${entry['allocated']:.2f}  spent=${entry['spent']:.2f}"
        )
    print(f"    {'pool.remaining':<20} ${pool.remaining:.2f}")


# ─────────────────────────────────────────────────────────────────────────────
# Example 1 — topup: CEO gives CTO more budget mid-run
# ─────────────────────────────────────────────────────────────────────────────


async def example_topup() -> None:
    print("\n── Example 1: topup_budget — add more budget to a running agent ──")

    # $10 pool: CTO gets $2, engineer gets $1
    pool = BudgetPool(total=10.00)
    await pool.allocate("cto", 2.00)
    await pool.allocate("engineer", 1.00)

    guard = SwarmAuthorityGuard(
        roles={
            "ceo": AgentRole.ORCHESTRATOR,
            "cto": AgentRole.SUPERVISOR,
            "engineer": AgentRole.WORKER,
        },
        teams={"ceo": ["cto", "engineer"], "cto": ["engineer"]},
    )
    state = {
        "cto": _make_snapshot("cto", supervisor_id="ceo"),
        "engineer": _make_snapshot("engineer", supervisor_id="cto"),
    }
    ctrl = SwarmController(
        actor_id="ceo",
        guard=guard,
        state_registry=state,
        task_registry={},
        budget_pool=pool,
    )

    _print_pool(pool, "before topup")

    # CTO is doing well — CEO tops up by $1.50
    await ctrl.topup_budget("cto", 1.50)
    print("\n  CEO topped up CTO by $1.50")
    _print_pool(pool, "after topup")

    # Audit log confirms it
    entry = ctrl._guard.audit_log()[-1]
    print(f"\n  Audit: {entry.actor_id} → {entry.target_id}  action={entry.action}")
    assert entry.action == ControlAction.TOPUP_BUDGET


# ─────────────────────────────────────────────────────────────────────────────
# Example 2 — reallocate up: raise a department's cap
# ─────────────────────────────────────────────────────────────────────────────


async def example_reallocate_up() -> None:
    print("\n── Example 2: reallocate_budget up — raise an agent's cap ────────")

    pool = BudgetPool(total=10.00)
    await pool.allocate("cto", 2.00)

    guard = SwarmAuthorityGuard(
        roles={"ceo": AgentRole.ORCHESTRATOR, "cto": AgentRole.SUPERVISOR},
        teams={"ceo": ["cto"]},
    )
    ctrl = SwarmController(
        actor_id="ceo",
        guard=guard,
        state_registry={"cto": _make_snapshot("cto")},
        task_registry={},
        budget_pool=pool,
    )

    _print_pool(pool, "before reallocate")

    # CTO started with $2; CEO raises cap to $5
    await ctrl.reallocate_budget("cto", 5.00)
    print("\n  CEO reallocated CTO from $2.00 → $5.00")
    _print_pool(pool, "after reallocate up")

    entry = ctrl._guard.audit_log()[-1]
    print(f"\n  Audit: {entry.actor_id} → {entry.target_id}  action={entry.action}")
    assert entry.action == ControlAction.REALLOCATE_BUDGET


# ─────────────────────────────────────────────────────────────────────────────
# Example 3 — reallocate down: trim a department that spent less than expected
# ─────────────────────────────────────────────────────────────────────────────


async def example_reallocate_down() -> None:
    print("\n── Example 3: reallocate_budget down — trim unused allocation ─────")

    pool = BudgetPool(total=10.00)
    await pool.allocate("marketing", 5.00)
    # Marketing only spent $1.20 so far
    await pool.spend("marketing", 1.20)

    guard = SwarmAuthorityGuard(
        roles={"ceo": AgentRole.ORCHESTRATOR, "marketing": AgentRole.WORKER},
        teams={"ceo": ["marketing"]},
    )
    ctrl = SwarmController(
        actor_id="ceo",
        guard=guard,
        state_registry={"marketing": _make_snapshot("marketing")},
        task_registry={},
        budget_pool=pool,
    )

    _print_pool(pool, "before trim")

    # CEO reallocates marketing down to $2 (must be ≥ already spent $1.20)
    await ctrl.reallocate_budget("marketing", 2.00)
    print("\n  CEO trimmed marketing from $5.00 → $2.00")
    print("  ($3 returned to pool for reuse)")
    _print_pool(pool, "after trim")


# ─────────────────────────────────────────────────────────────────────────────
# Example 4 — guard rails: errors you will encounter
# ─────────────────────────────────────────────────────────────────────────────


async def example_guard_rails() -> None:
    print("\n── Example 4: guard rails — errors that protect the pool ─────────")

    pool = BudgetPool(total=5.00)
    await pool.allocate("agent", 3.00)
    await pool.spend("agent", 2.00)

    guard = SwarmAuthorityGuard(
        roles={"ceo": AgentRole.ORCHESTRATOR, "agent": AgentRole.WORKER},
        teams={"ceo": ["agent"]},
    )
    ctrl = SwarmController(
        actor_id="ceo",
        guard=guard,
        state_registry={"agent": _make_snapshot("agent")},
        task_registry={},
        budget_pool=pool,
    )

    # 1. Cannot shrink below what is already spent
    print("\n  [1] Reallocate below spent ($2.00) → should raise")
    try:
        await ctrl.reallocate_budget("agent", 1.00)
    except BudgetAllocationError as e:
        print(f"      BudgetAllocationError: {e}")

    # 2. Cannot topup beyond pool.remaining ($2.00)
    print("\n  [2] Topup by $3.00 when pool only has $2.00 remaining → should raise")
    try:
        await ctrl.topup_budget("agent", 3.00)
    except BudgetAllocationError as e:
        print(f"      BudgetAllocationError: {e}")

    # 3. No budget_pool set → RuntimeError
    print("\n  [3] Controller with no budget_pool → should raise RuntimeError")
    ctrl_no_pool = SwarmController(
        actor_id="ceo",
        guard=guard,
        state_registry={"agent": _make_snapshot("agent")},
        task_registry={},
        budget_pool=None,
    )
    try:
        await ctrl_no_pool.topup_budget("agent", 1.00)
    except RuntimeError as e:
        print(f"      RuntimeError: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Example 5 — CEO rebalances across departments after seeing mid-run results
# ─────────────────────────────────────────────────────────────────────────────


async def example_ceo_rebalance() -> None:
    print("\n── Example 5: CEO rebalances budgets mid-run based on results ─────")

    pool = BudgetPool(total=10.00)
    await pool.allocate("engineering", 4.00)
    await pool.allocate("marketing", 4.00)
    # Pool remaining: $2

    # Simulate mid-run: engineering is 70% through its budget, marketing barely started
    await pool.spend("engineering", 2.80)
    await pool.spend("marketing", 0.40)

    guard = SwarmAuthorityGuard(
        roles={
            "ceo": AgentRole.ORCHESTRATOR,
            "engineering": AgentRole.WORKER,
            "marketing": AgentRole.WORKER,
        },
        teams={"ceo": ["engineering", "marketing"]},
    )
    state = {
        "engineering": _make_snapshot("engineering"),
        "marketing": _make_snapshot("marketing"),
    }
    ctrl = SwarmController(
        actor_id="ceo",
        guard=guard,
        state_registry=state,
        task_registry={},
        budget_pool=pool,
    )

    print("\n  Mid-run snapshot (before rebalance):")
    _print_pool(pool)

    # Engineering is almost out — CEO tops it up by $1
    # Marketing has plenty — CEO trims it back to $2 (releasing $1.60 to pool)
    await ctrl.reallocate_budget("marketing", 2.00)  # returns $2 to pool
    await ctrl.topup_budget("engineering", 2.00)  # draws $2 from pool

    print("\n  After CEO rebalance (marketing trimmed, engineering topped up):")
    _print_pool(pool)

    log = ctrl._guard.audit_log()
    print(f"\n  Audit log ({len(log)} entries):")
    for entry in log:
        print(f"    {entry.action:<22} {entry.actor_id} → {entry.target_id}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────


async def main() -> None:
    await example_topup()
    await example_reallocate_up()
    await example_reallocate_down()
    await example_guard_rails()
    await example_ceo_rebalance()
    print("\nAll budget reallocation examples completed.")


if __name__ == "__main__":
    asyncio.run(main())
