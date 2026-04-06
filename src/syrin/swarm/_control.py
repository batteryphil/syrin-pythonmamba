"""SwarmController — agent control actions for a swarm."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

from syrin.budget._pool import BudgetPool
from syrin.enums import AgentPermission, AgentRole, AgentStatus, ControlAction, PauseMode
from syrin.swarm._agent_ref import AgentRef, _aid
from syrin.swarm._authority import SwarmAuthorityGuard

_MAX_SUMMARY_LEN: int = 500


# ---------------------------------------------------------------------------
# AgentStateSnapshot
# ---------------------------------------------------------------------------


@dataclass
class AgentStateSnapshot:
    """State snapshot for a single agent.

    Attributes:
        agent_id: Unique identifier of the agent.
        status: Current :class:`~syrin.enums.AgentStatus`.
        role: :class:`~syrin.enums.AgentRole` assigned to this agent.
        last_output_summary: Truncated (≤ 500 chars) summary of the last output.
        cost_spent: Total cost spent by this agent so far.
        task: Description of the current task.
        context_override: Injected context string, or ``None`` if unset.
        supervisor_id: ID of the supervisor agent, or ``None`` if this agent
            has no supervisor.
    """

    agent_id: str
    status: AgentStatus
    role: AgentRole
    last_output_summary: str
    cost_spent: float
    task: str
    context_override: str | None
    supervisor_id: str | None

    def __post_init__(self) -> None:
        """Truncate last_output_summary to ≤ 500 characters."""
        if len(self.last_output_summary) > _MAX_SUMMARY_LEN:
            self.last_output_summary = self.last_output_summary[:_MAX_SUMMARY_LEN]


# ---------------------------------------------------------------------------
# SwarmController
# ---------------------------------------------------------------------------


class SwarmController:
    """Agent control actions for a swarm.

    All actions pass through a :class:`~syrin.swarm._authority.SwarmAuthorityGuard`
    before executing.  Successful actions are recorded via
    :meth:`~syrin.swarm._authority.SwarmAuthorityGuard.record_action`.

    Example::

        guard = SwarmAuthorityGuard(
            roles={"sup": AgentRole.SUPERVISOR, "w1": AgentRole.WORKER},
            teams={"sup": ["w1"]},
        )
        ctrl = SwarmController(
            actor_id="sup",
            guard=guard,
            state_registry=state,
            task_registry=tasks,
        )
        await ctrl.pause_agent("w1")
    """

    def __init__(
        self,
        actor_id: AgentRef | str,
        guard: SwarmAuthorityGuard,
        state_registry: dict[str, AgentStateSnapshot],
        task_registry: dict[str, asyncio.Task[object]],
        budget_pool: BudgetPool | None = None,
    ) -> None:
        """Initialise SwarmController.

        Args:
            actor_id: Agent instance or agent ID string initiating control actions.
            guard: Authority guard for permission checks.
            state_registry: Mapping of agent_id → :class:`AgentStateSnapshot`.
            task_registry: Mapping of agent_id → running :class:`asyncio.Task`.
            budget_pool: Optional shared :class:`~syrin.budget._pool.BudgetPool`.
                Required for :meth:`topup_budget` and :meth:`reallocate_budget`.
        """
        self._actor: AgentRef | str = actor_id
        self._actor_id: str = _aid(actor_id)
        self._guard = guard
        self._state: dict[str, AgentStateSnapshot] = state_registry
        self._tasks: dict[str, asyncio.Task[object]] = task_registry
        self._pool: BudgetPool | None = budget_pool

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _require(self, target: AgentRef | str, permission: AgentPermission) -> None:
        """Check permission via guard; raise AgentPermissionError if denied."""
        self._guard.require(self._actor, permission, target)

    def _cancel_task(self, target_id: str) -> None:
        """Cancel the asyncio task for *target_id* if one exists."""
        task = self._tasks.get(target_id)
        if task is not None and not task.done():
            task.cancel()

    def _set_status(self, target_id: str, status: AgentStatus) -> None:
        """Update the status in the state registry for *target_id*."""
        snap = self._state.get(target_id)
        if snap is not None:
            snap.status = status

    # ------------------------------------------------------------------
    # Control actions
    # ------------------------------------------------------------------

    async def pause_agent(self, target: AgentRef, *, mode: PauseMode = PauseMode.IMMEDIATE) -> None:
        """Pause the target agent.

        Args:
            target: Agent instance to pause.
            mode: When to apply the pause.
                :attr:`~syrin.enums.PauseMode.IMMEDIATE` pauses right away.
                :attr:`~syrin.enums.PauseMode.DRAIN` waits for the current
                step to complete before pausing.

        Raises:
            AgentPermissionError: If the actor lacks CONTROL permission.
        """
        target_id = _aid(target)
        self._require(target, AgentPermission.CONTROL)
        if mode == PauseMode.DRAIN:
            self._set_status(target_id, AgentStatus.DRAINING)
        else:
            self._set_status(target_id, AgentStatus.PAUSED)
        self._guard.record_action(self._actor_id, target_id, ControlAction.PAUSE)

    async def resume_agent(self, target: AgentRef) -> None:
        """Resume a paused agent.

        Args:
            target: Agent instance to resume.

        Raises:
            AgentPermissionError: If the actor lacks CONTROL permission.
        """
        target_id = _aid(target)
        self._require(target, AgentPermission.CONTROL)
        self._set_status(target_id, AgentStatus.RUNNING)
        self._guard.record_action(self._actor_id, target_id, ControlAction.RESUME)

    async def skip_agent(self, target: AgentRef) -> None:
        """Skip the target agent's current task and set status to IDLE.

        Args:
            target: Agent instance to skip.

        Raises:
            AgentPermissionError: If the actor lacks CONTROL permission.
        """
        target_id = _aid(target)
        self._require(target, AgentPermission.CONTROL)
        self._cancel_task(target_id)
        self._set_status(target_id, AgentStatus.IDLE)
        self._guard.record_action(self._actor_id, target_id, ControlAction.SKIP)

    async def change_context(self, target: AgentRef, new_context: str) -> None:
        """Inject a new context override for the target agent.

        Args:
            target: Agent instance whose context to change.
            new_context: The new context string to inject.

        Raises:
            AgentPermissionError: If the actor lacks CONTROL permission.
        """
        target_id = _aid(target)
        self._require(target, AgentPermission.CONTROL)
        snap = self._state.get(target_id)
        if snap is not None:
            snap.context_override = new_context
        self._guard.record_action(self._actor_id, target_id, ControlAction.CHANGE_CONTEXT)

    async def kill_agent(self, target: AgentRef) -> None:
        """Forcibly terminate the target agent.

        Args:
            target: Agent instance to terminate.

        Raises:
            AgentPermissionError: If the actor lacks CONTROL permission.
        """
        target_id = _aid(target)
        self._require(target, AgentPermission.CONTROL)
        self._cancel_task(target_id)
        self._set_status(target_id, AgentStatus.KILLED)
        self._guard.record_action(self._actor_id, target_id, ControlAction.KILL)

    async def topup_budget(self, target: AgentRef | str, additional: float) -> None:
        """Add *additional* USD to an agent's active budget allocation at runtime.

        The actor must have ``CONTROL`` permission over the target.  The extra
        amount is drawn from the shared :class:`~syrin.budget._pool.BudgetPool`
        passed at construction time.

        Args:
            target: Agent instance or agent ID string to top up.
            additional: Extra USD to add.  ``0.0`` is a no-op.

        Raises:
            RuntimeError: If no ``budget_pool`` was provided at construction.
            AgentPermissionError: If the actor lacks ``CONTROL`` permission.
            BudgetAllocationError: If the pool has insufficient balance or the
                agent has no active allocation.

        Example::

            ctrl = SwarmController(actor_id=ceo, ..., budget_pool=pool)
            await ctrl.topup_budget(cto, 2.00)   # CTO gets $2 more mid-run
        """
        if self._pool is None:
            raise RuntimeError(
                "topup_budget requires a budget_pool. "
                "Pass budget_pool=pool when constructing SwarmController."
            )
        target_id = _aid(target)
        self._require(target, AgentPermission.CONTROL)
        await self._pool.topup(target_id, additional)
        self._guard.record_action(self._actor_id, target_id, ControlAction.TOPUP_BUDGET)

    async def reallocate_budget(self, target: AgentRef | str, new_amount: float) -> None:
        """Replace an agent's budget allocation with *new_amount* USD at runtime.

        If *new_amount* is greater than the current allocation the difference is
        drawn from the pool; if smaller, the difference is returned.  The new
        amount must be at least as large as what the agent has already spent.

        Args:
            target: Agent instance or agent ID string to reallocate.
            new_amount: New total allocation in USD.

        Raises:
            RuntimeError: If no ``budget_pool`` was provided at construction.
            AgentPermissionError: If the actor lacks ``CONTROL`` permission.
            BudgetAllocationError: If the new amount is below the agent's
                already-spent total, the pool has insufficient balance, or
                *new_amount* would exceed ``per_agent_max``.

        Example::

            ctrl = SwarmController(actor_id=ceo, ..., budget_pool=pool)
            await ctrl.reallocate_budget(cto, 5.00)   # raise CTO cap to $5
            await ctrl.reallocate_budget(cto, 1.00)   # trim back to $1
        """
        if self._pool is None:
            raise RuntimeError(
                "reallocate_budget requires a budget_pool. "
                "Pass budget_pool=pool when constructing SwarmController."
            )
        target_id = _aid(target)
        self._require(target, AgentPermission.CONTROL)
        await self._pool.reallocate(target_id, new_amount)
        self._guard.record_action(self._actor_id, target_id, ControlAction.REALLOCATE_BUDGET)

    async def read_agent_state(self, target: AgentRef) -> AgentStateSnapshot:
        """Return the current :class:`AgentStateSnapshot` for *target*.

        Args:
            target: Agent instance to read.

        Returns:
            :class:`AgentStateSnapshot` for the target agent.

        Raises:
            AgentPermissionError: If the actor lacks READ permission.
            KeyError: If the agent is not registered in the state registry.
        """
        target_id = _aid(target)
        self._require(target, AgentPermission.READ)
        return self._state[target_id]
