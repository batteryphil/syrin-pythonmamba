"""TDD: runtime budget reallocation — BudgetPool.topup/reallocate + SwarmController."""

from __future__ import annotations

import asyncio

import pytest

from syrin.budget._pool import BudgetPool
from syrin.budget.exceptions import BudgetAllocationError
from syrin.enums import AgentRole, AgentStatus, ControlAction
from syrin.swarm._authority import AgentPermissionError, SwarmAuthorityGuard
from syrin.swarm._control import AgentStateSnapshot, SwarmController

# ── helpers ──────────────────────────────────────────────────────────────────


def _make_guard(actor: str = "sup", target: str = "w1") -> SwarmAuthorityGuard:
    return SwarmAuthorityGuard(
        roles={actor: AgentRole.SUPERVISOR, target: AgentRole.WORKER},
        teams={actor: [target]},
    )


def _make_snapshot(agent_id: str = "w1") -> AgentStateSnapshot:
    return AgentStateSnapshot(
        agent_id=agent_id,
        status=AgentStatus.RUNNING,
        role=AgentRole.WORKER,
        last_output_summary="working",
        cost_spent=0.0,
        task="analyse",
        context_override=None,
        supervisor_id="sup",
    )


async def _setup(
    pool_total: float = 10.00,
    initial_alloc: float = 2.00,
    initial_spent: float = 0.00,
) -> tuple[SwarmController, BudgetPool]:
    pool = BudgetPool(total=pool_total)
    await pool.allocate("w1", initial_alloc)
    if initial_spent > 0:
        await pool.spend("w1", initial_spent)
    guard = _make_guard()
    state = {"w1": _make_snapshot("w1")}
    ctrl = SwarmController(
        actor_id="sup",
        guard=guard,
        state_registry=state,
        task_registry={},
        budget_pool=pool,
    )
    return ctrl, pool


# ─────────────────────────────────────────────────────────────────────────────
# BudgetPool.topup
# ─────────────────────────────────────────────────────────────────────────────


class TestBudgetPoolTopup:
    """BudgetPool.topup() adds to an agent's active allocation from the pool."""

    def test_topup_increases_allocated_in_snapshot(self) -> None:
        """topup(agent, 1.00) raises snapshot allocated from 2.00 → 3.00."""
        pool = BudgetPool(total=10.00)
        asyncio.run(pool.allocate("a1", 2.00))
        asyncio.run(pool.topup("a1", 1.00))
        assert pool.snapshot()["a1"]["allocated"] == pytest.approx(3.00)

    def test_topup_reduces_pool_remaining(self) -> None:
        """topup draws the additional amount from the pool's free balance."""
        pool = BudgetPool(total=10.00)
        asyncio.run(pool.allocate("a1", 2.00))
        asyncio.run(pool.topup("a1", 1.00))
        assert pool.remaining == pytest.approx(7.00)

    def test_topup_zero_is_noop(self) -> None:
        """topup(0.00) leaves allocation and remaining unchanged."""
        pool = BudgetPool(total=10.00)
        asyncio.run(pool.allocate("a1", 2.00))
        asyncio.run(pool.topup("a1", 0.00))
        assert pool.snapshot()["a1"]["allocated"] == pytest.approx(2.00)
        assert pool.remaining == pytest.approx(8.00)

    def test_topup_unallocated_agent_raises(self) -> None:
        """topup on an agent with no active allocation raises BudgetAllocationError."""
        pool = BudgetPool(total=10.00)
        with pytest.raises(BudgetAllocationError):
            asyncio.run(pool.topup("ghost", 1.00))

    def test_topup_insufficient_pool_raises(self) -> None:
        """topup raises when additional exceeds pool.remaining."""
        pool = BudgetPool(total=3.00)
        asyncio.run(pool.allocate("a1", 2.00))
        # remaining = 1.00, requesting 2.00
        with pytest.raises(BudgetAllocationError):
            asyncio.run(pool.topup("a1", 2.00))

    def test_topup_respects_per_agent_max(self) -> None:
        """topup raises when it would push allocated above per_agent_max."""
        pool = BudgetPool(total=10.00, per_agent_max=3.00)
        asyncio.run(pool.allocate("a1", 2.00))
        # 2.00 + 2.00 = 4.00 > per_agent_max=3.00
        with pytest.raises(BudgetAllocationError):
            asyncio.run(pool.topup("a1", 2.00))

    def test_topup_exactly_to_per_agent_max_succeeds(self) -> None:
        """topup to exactly per_agent_max is allowed."""
        pool = BudgetPool(total=10.00, per_agent_max=3.00)
        asyncio.run(pool.allocate("a1", 2.00))
        asyncio.run(pool.topup("a1", 1.00))  # 2.00 + 1.00 == 3.00 == max
        assert pool.snapshot()["a1"]["allocated"] == pytest.approx(3.00)

    def test_topup_does_not_affect_other_agents(self) -> None:
        """topup on a1 does not change a2's allocation or spent."""
        pool = BudgetPool(total=10.00)
        asyncio.run(pool.allocate("a1", 2.00))
        asyncio.run(pool.allocate("a2", 2.00))
        asyncio.run(pool.topup("a1", 1.00))
        assert pool.snapshot()["a2"]["allocated"] == pytest.approx(2.00)

    @pytest.mark.asyncio
    async def test_topup_is_atomic_under_concurrency(self) -> None:
        """Concurrent topups never exceed pool total."""
        pool = BudgetPool(total=4.00)
        await pool.allocate("a1", 1.00)
        # remaining = 3.00; fire 5 concurrent topups of 1.00 each
        errors: list[BudgetAllocationError] = []

        async def try_topup() -> None:
            try:
                await pool.topup("a1", 1.00)
            except BudgetAllocationError as e:
                errors.append(e)

        await asyncio.gather(*[try_topup() for _ in range(5)])
        # Only 3 should succeed (pool.remaining was 3.00)
        assert pool.remaining >= 0.0
        assert pool.snapshot()["a1"]["allocated"] <= 4.00


# ─────────────────────────────────────────────────────────────────────────────
# BudgetPool.reallocate
# ─────────────────────────────────────────────────────────────────────────────


class TestBudgetPoolReallocate:
    """BudgetPool.reallocate() replaces an agent's allocation in-place."""

    def test_reallocate_up_updates_snapshot(self) -> None:
        """reallocate to a higher amount reflects in snapshot."""
        pool = BudgetPool(total=10.00)
        asyncio.run(pool.allocate("a1", 2.00))
        asyncio.run(pool.reallocate("a1", 5.00))
        assert pool.snapshot()["a1"]["allocated"] == pytest.approx(5.00)

    def test_reallocate_up_reduces_remaining(self) -> None:
        """Increasing allocation draws the difference from pool.remaining."""
        pool = BudgetPool(total=10.00)
        asyncio.run(pool.allocate("a1", 2.00))
        asyncio.run(pool.reallocate("a1", 5.00))
        # remaining was 8.00, delta = 3.00 drawn → 5.00
        assert pool.remaining == pytest.approx(5.00)

    def test_reallocate_down_updates_snapshot(self) -> None:
        """reallocate to a lower amount reflects in snapshot."""
        pool = BudgetPool(total=10.00)
        asyncio.run(pool.allocate("a1", 5.00))
        asyncio.run(pool.reallocate("a1", 3.00))
        assert pool.snapshot()["a1"]["allocated"] == pytest.approx(3.00)

    def test_reallocate_down_returns_difference_to_pool(self) -> None:
        """Decreasing allocation returns the difference to pool.remaining."""
        pool = BudgetPool(total=10.00)
        asyncio.run(pool.allocate("a1", 5.00))
        asyncio.run(pool.reallocate("a1", 3.00))
        # remaining was 5.00, delta = -2.00 returned → 7.00
        assert pool.remaining == pytest.approx(7.00)

    def test_reallocate_to_same_amount_is_noop(self) -> None:
        """reallocate to current allocation changes nothing."""
        pool = BudgetPool(total=10.00)
        asyncio.run(pool.allocate("a1", 3.00))
        asyncio.run(pool.reallocate("a1", 3.00))
        assert pool.snapshot()["a1"]["allocated"] == pytest.approx(3.00)
        assert pool.remaining == pytest.approx(7.00)

    def test_reallocate_below_spent_raises(self) -> None:
        """Cannot reallocate to less than what the agent has already spent."""
        pool = BudgetPool(total=10.00)
        asyncio.run(pool.allocate("a1", 5.00))
        asyncio.run(pool.spend("a1", 2.00))
        with pytest.raises(BudgetAllocationError):
            asyncio.run(pool.reallocate("a1", 1.00))  # 1.00 < spent 2.00

    def test_reallocate_exactly_to_spent_succeeds(self) -> None:
        """Reallocating to exactly the spent amount is the minimum valid value."""
        pool = BudgetPool(total=10.00)
        asyncio.run(pool.allocate("a1", 5.00))
        asyncio.run(pool.spend("a1", 2.00))
        asyncio.run(pool.reallocate("a1", 2.00))  # exactly at spent
        assert pool.snapshot()["a1"]["allocated"] == pytest.approx(2.00)

    def test_reallocate_unallocated_agent_raises(self) -> None:
        """reallocate on an agent with no active allocation raises."""
        pool = BudgetPool(total=10.00)
        with pytest.raises(BudgetAllocationError):
            asyncio.run(pool.reallocate("ghost", 5.00))

    def test_reallocate_up_insufficient_pool_raises(self) -> None:
        """reallocate up raises when pool has insufficient remaining balance."""
        pool = BudgetPool(total=5.00)
        asyncio.run(pool.allocate("a1", 2.00))
        # remaining = 3.00, trying to increase by 4.00 (2→6)
        with pytest.raises(BudgetAllocationError):
            asyncio.run(pool.reallocate("a1", 6.00))

    def test_reallocate_respects_per_agent_max(self) -> None:
        """reallocate raises when new_amount exceeds per_agent_max."""
        pool = BudgetPool(total=10.00, per_agent_max=3.00)
        asyncio.run(pool.allocate("a1", 2.00))
        with pytest.raises(BudgetAllocationError):
            asyncio.run(pool.reallocate("a1", 4.00))

    def test_reallocate_does_not_affect_other_agents(self) -> None:
        """reallocate on a1 does not touch a2's snapshot entry."""
        pool = BudgetPool(total=10.00)
        asyncio.run(pool.allocate("a1", 2.00))
        asyncio.run(pool.allocate("a2", 2.00))
        asyncio.run(pool.reallocate("a1", 4.00))
        assert pool.snapshot()["a2"]["allocated"] == pytest.approx(2.00)


# ─────────────────────────────────────────────────────────────────────────────
# SwarmController.topup_budget
# ─────────────────────────────────────────────────────────────────────────────


class TestSwarmControllerTopupBudget:
    """SwarmController.topup_budget() is a permissioned topup on the pool."""

    @pytest.mark.asyncio
    async def test_topup_increases_pool_allocation(self) -> None:
        """topup_budget delegates to BudgetPool.topup and increases allocation."""
        ctrl, pool = await _setup()
        await ctrl.topup_budget("w1", 1.00)
        assert pool.snapshot()["w1"]["allocated"] == pytest.approx(3.00)

    @pytest.mark.asyncio
    async def test_topup_reduces_pool_remaining(self) -> None:
        """topup_budget draws from pool.remaining."""
        ctrl, pool = await _setup()
        await ctrl.topup_budget("w1", 1.00)
        assert pool.remaining == pytest.approx(7.00)

    @pytest.mark.asyncio
    async def test_topup_requires_control_permission(self) -> None:
        """Worker cannot topup another worker — AgentPermissionError raised."""
        pool = BudgetPool(total=10.00)
        await pool.allocate("w1", 2.00)
        guard = SwarmAuthorityGuard(
            roles={"w2": AgentRole.WORKER, "w1": AgentRole.WORKER},
            teams={},
        )
        ctrl = SwarmController(
            actor_id="w2",
            guard=guard,
            state_registry={"w1": _make_snapshot("w1")},
            task_registry={},
            budget_pool=pool,
        )
        with pytest.raises(AgentPermissionError):
            await ctrl.topup_budget("w1", 1.00)

    @pytest.mark.asyncio
    async def test_topup_records_topup_budget_in_audit_log(self) -> None:
        """topup_budget records ControlAction.TOPUP_BUDGET in the audit log."""
        ctrl, _ = await _setup()
        await ctrl.topup_budget("w1", 1.00)
        actions = [e.action for e in ctrl._guard.audit_log()]
        assert ControlAction.TOPUP_BUDGET in actions

    @pytest.mark.asyncio
    async def test_topup_without_pool_raises_runtime_error(self) -> None:
        """topup_budget with no budget_pool raises RuntimeError."""
        guard = _make_guard()
        ctrl = SwarmController(
            actor_id="sup",
            guard=guard,
            state_registry={"w1": _make_snapshot()},
            task_registry={},
            budget_pool=None,
        )
        with pytest.raises(RuntimeError, match="budget_pool"):
            await ctrl.topup_budget("w1", 1.00)

    @pytest.mark.asyncio
    async def test_topup_insufficient_pool_propagates_error(self) -> None:
        """BudgetAllocationError from the pool propagates out of topup_budget."""
        ctrl, _ = await _setup(pool_total=2.50, initial_alloc=2.00)
        # remaining = 0.50, requesting 1.00
        with pytest.raises(BudgetAllocationError):
            await ctrl.topup_budget("w1", 1.00)


# ─────────────────────────────────────────────────────────────────────────────
# SwarmController.reallocate_budget
# ─────────────────────────────────────────────────────────────────────────────


class TestSwarmControllerReallocateBudget:
    """SwarmController.reallocate_budget() is a permissioned reallocate on the pool."""

    @pytest.mark.asyncio
    async def test_reallocate_up_updates_pool(self) -> None:
        """reallocate_budget delegates to BudgetPool.reallocate."""
        ctrl, pool = await _setup()
        await ctrl.reallocate_budget("w1", 5.00)
        assert pool.snapshot()["w1"]["allocated"] == pytest.approx(5.00)

    @pytest.mark.asyncio
    async def test_reallocate_down_restores_pool_remaining(self) -> None:
        """reallocate_budget down returns difference to pool."""
        ctrl, pool = await _setup(initial_alloc=5.00)
        await ctrl.reallocate_budget("w1", 2.00)
        assert pool.remaining == pytest.approx(8.00)  # 5 + 3 returned = 8

    @pytest.mark.asyncio
    async def test_reallocate_requires_control_permission(self) -> None:
        """Worker cannot reallocate another worker — AgentPermissionError raised."""
        pool = BudgetPool(total=10.00)
        await pool.allocate("w1", 2.00)
        guard = SwarmAuthorityGuard(
            roles={"w2": AgentRole.WORKER, "w1": AgentRole.WORKER},
            teams={},
        )
        ctrl = SwarmController(
            actor_id="w2",
            guard=guard,
            state_registry={"w1": _make_snapshot("w1")},
            task_registry={},
            budget_pool=pool,
        )
        with pytest.raises(AgentPermissionError):
            await ctrl.reallocate_budget("w1", 5.00)

    @pytest.mark.asyncio
    async def test_reallocate_records_reallocate_budget_in_audit_log(self) -> None:
        """reallocate_budget records ControlAction.REALLOCATE_BUDGET in audit log."""
        ctrl, _ = await _setup()
        await ctrl.reallocate_budget("w1", 4.00)
        actions = [e.action for e in ctrl._guard.audit_log()]
        assert ControlAction.REALLOCATE_BUDGET in actions

    @pytest.mark.asyncio
    async def test_reallocate_without_pool_raises_runtime_error(self) -> None:
        """reallocate_budget with no budget_pool raises RuntimeError."""
        guard = _make_guard()
        ctrl = SwarmController(
            actor_id="sup",
            guard=guard,
            state_registry={"w1": _make_snapshot()},
            task_registry={},
            budget_pool=None,
        )
        with pytest.raises(RuntimeError, match="budget_pool"):
            await ctrl.reallocate_budget("w1", 3.00)

    @pytest.mark.asyncio
    async def test_reallocate_below_spent_propagates_error(self) -> None:
        """BudgetAllocationError from pool propagates when new_amount < spent."""
        ctrl, pool = await _setup(initial_alloc=5.00, initial_spent=2.00)
        with pytest.raises(BudgetAllocationError):
            await ctrl.reallocate_budget("w1", 1.00)  # 1.00 < spent 2.00

    @pytest.mark.asyncio
    async def test_reallocate_insufficient_pool_propagates_error(self) -> None:
        """BudgetAllocationError propagates when pool has insufficient balance."""
        ctrl, _ = await _setup(pool_total=3.00, initial_alloc=2.00)
        # remaining = 1.00, trying to increase to 5.00 (delta = 3.00)
        with pytest.raises(BudgetAllocationError):
            await ctrl.reallocate_budget("w1", 5.00)
