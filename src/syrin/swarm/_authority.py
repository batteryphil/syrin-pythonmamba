"""SwarmAuthorityGuard — permission-gated control for agent authority."""

from __future__ import annotations

import contextlib
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime

from syrin.enums import AgentPermission, AgentRole, ControlAction, DelegationScope, Hook
from syrin.swarm._agent_ref import AgentRef, _aid

# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class AgentPermissionError(Exception):
    """Raised when an agent attempts an unauthorized control action.

    Attributes:
        actor_id: ID of the agent requesting the action.
        target_id: ID of the target agent.
        attempted_action: The action that was denied.
        reason: Human-readable explanation of why the action was denied.
    """

    def __init__(
        self,
        actor_id: str,
        target_id: str,
        attempted_action: str,
        reason: str = "",
    ) -> None:
        """Initialise AgentPermissionError.

        Args:
            actor_id: Agent ID of the requester.
            target_id: Agent ID of the target.
            attempted_action: The permission or action string that was denied.
            reason: Optional human-readable reason.
        """
        self.actor_id = actor_id
        self.target_id = target_id
        self.attempted_action = attempted_action
        self.reason = reason
        super().__init__(
            f"Agent {actor_id!r} cannot {attempted_action!r} on {target_id!r}: {reason}"
        )


# ---------------------------------------------------------------------------
# Audit
# ---------------------------------------------------------------------------


@dataclass
class AuditEntry:
    """Audit record for a successful control action.

    Attributes:
        actor_id: ID of the agent that performed the action.
        target_id: ID of the agent the action was performed on.
        action: :class:`~syrin.enums.ControlAction` describing what was done.
        timestamp: UTC timestamp when the action was recorded.
    """

    actor_id: str
    target_id: str
    action: ControlAction
    timestamp: datetime


# ---------------------------------------------------------------------------
# Delegation record (internal)
# ---------------------------------------------------------------------------


@dataclass
class _DelegationRecord:
    """Internal record of a permission delegation.

    Attributes:
        delegator_id: ID of the agent that granted the delegation.
        delegate_id: ID of the agent that received the delegation.
        permissions: List of granted permissions.
        scope: Scope under which the delegation is valid.
    """

    delegator_id: str
    delegate_id: str
    permissions: list[AgentPermission]
    scope: DelegationScope


# ---------------------------------------------------------------------------
# SwarmAuthorityGuard
# ---------------------------------------------------------------------------


class SwarmAuthorityGuard:
    """Permission gate for agent control actions within a swarm.

    This is the single source of truth for all permission decisions in a
    swarm.  It enforces role-based and team-based access control, supports
    temporary delegation, and emits lifecycle hooks on every grant or denial.

    Permission rules:

    - ``ADMIN`` may perform **any** action on **any** agent.
    - ``ORCHESTRATOR`` may ``CONTROL``, ``CONTEXT``, and ``SPAWN`` on agents
      listed in *teams[actor_id]*.
    - ``SUPERVISOR`` may ``CONTROL`` (pause/resume) agents listed in
      *teams[actor_id]*.
    - ``WORKER`` may only ``SIGNAL`` (send A2A messages) — no control.
    - Actors not present in *roles* are treated as ``WORKER``.

    Example::

        guard = SwarmAuthorityGuard(
            roles={"sup": AgentRole.SUPERVISOR, "w1": AgentRole.WORKER},
            teams={"sup": ["w1"]},
        )
        guard.require("sup", AgentPermission.CONTROL, "w1")  # passes
        guard.require("w1", AgentPermission.CONTROL, "sup")  # raises AgentPermissionError
    """

    def __init__(
        self,
        roles: dict[AgentRef, AgentRole],
        teams: dict[AgentRef, list[AgentRef]],
        fire_event_fn: Callable[[Hook, dict[str, object]], None] | None = None,
    ) -> None:
        """Initialise SwarmAuthorityGuard.

        Args:
            roles: Mapping of agent instance → :class:`~syrin.enums.AgentRole`.
            teams: Mapping of supervisor/orchestrator agent instance → list of
                worker agent instances they manage.
            fire_event_fn: Optional callable invoked with
                ``(Hook, dict)`` on permission decisions. Defaults to no-op.
        """
        self._roles: dict[str, AgentRole] = {_aid(a): r for a, r in roles.items()}
        self._teams: dict[str, list[str]] = {
            _aid(a): [_aid(m) for m in ms] for a, ms in teams.items()
        }
        self._fire: Callable[[Hook, dict[str, object]], None] = fire_event_fn or (
            lambda _h, _d: None
        )
        self._audit: list[AuditEntry] = []
        self._delegations: list[_DelegationRecord] = []

    # ------------------------------------------------------------------
    # Role helpers
    # ------------------------------------------------------------------

    def _role_of(self, agent_id: str) -> AgentRole:
        """Return the role for *agent_id*, defaulting to WORKER."""
        return self._roles.get(agent_id, AgentRole.WORKER)

    def _is_in_team(self, actor_id: str, target_id: str) -> bool:
        """Return True if *target_id* is in *actor_id*'s team."""
        return target_id in self._teams.get(actor_id, [])

    # ------------------------------------------------------------------
    # Delegation helpers
    # ------------------------------------------------------------------

    def _delegated_permissions(self, agent_id: str) -> set[AgentPermission]:
        """Return the set of permissions delegated to *agent_id*."""
        result: set[AgentPermission] = set()
        for rec in self._delegations:
            if rec.delegate_id == agent_id:
                result.update(rec.permissions)
        return result

    # ------------------------------------------------------------------
    # Core permission logic
    # ------------------------------------------------------------------

    def _evaluate(
        self, actor_id: str, permission: AgentPermission, target_id: str
    ) -> tuple[bool, str]:
        """Evaluate permission and return (granted, reason).

        Args:
            actor_id: Requesting agent ID.
            permission: Permission being requested.
            target_id: Target agent ID.

        Returns:
            Tuple of (granted: bool, reason: str).
        """
        role = self._role_of(actor_id)

        # ADMIN: unconditional grant
        if role == AgentRole.ADMIN:
            return True, "admin role"

        # Check delegated permissions (any target is allowed for delegated perms)
        delegated = self._delegated_permissions(actor_id)
        if permission in delegated:
            return True, f"delegated {permission}"

        # ORCHESTRATOR: CONTROL, CONTEXT, SPAWN on team members
        if role == AgentRole.ORCHESTRATOR:
            if permission in (
                AgentPermission.CONTROL,
                AgentPermission.CONTEXT,
                AgentPermission.SPAWN,
            ):
                if self._is_in_team(actor_id, target_id):
                    return True, "orchestrator team member"
                return False, f"target {target_id!r} not in orchestrator's team"
            if permission == AgentPermission.SIGNAL:
                return True, "orchestrator may signal"
            if permission == AgentPermission.READ:
                return True, "orchestrator may read"
            return False, f"orchestrator cannot {permission}"

        # SUPERVISOR: CONTROL on team members only
        if role == AgentRole.SUPERVISOR:
            if permission == AgentPermission.CONTROL:
                if self._is_in_team(actor_id, target_id):
                    return True, "supervisor team member"
                return False, f"target {target_id!r} not in supervisor's team"
            if permission in (AgentPermission.SIGNAL, AgentPermission.READ):
                return True, f"supervisor may {permission}"
            return False, f"supervisor cannot {permission}"

        # WORKER: SIGNAL only
        if permission == AgentPermission.SIGNAL:
            return True, "worker may signal"

        return False, f"worker role cannot {permission}"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def check(
        self, actor: AgentRef | str, permission: AgentPermission, target: AgentRef | str
    ) -> bool:
        """Check permission without raising.

        Args:
            actor: Requesting agent instance.
            permission: Permission being requested.
            target: Target agent instance.

        Returns:
            ``True`` if the action is allowed, ``False`` otherwise.
        """
        granted, _ = self._evaluate(_aid(actor), permission, _aid(target))
        return granted

    def require(
        self, actor: AgentRef | str, permission: AgentPermission, target: AgentRef | str
    ) -> None:
        """Check permission, raise :class:`AgentPermissionError` if denied.

        Also fires :attr:`~syrin.enums.Hook.AGENT_PERMISSION_DENIED` on denial.

        Args:
            actor: Requesting agent instance.
            permission: Permission being requested.
            target: Target agent instance.

        Raises:
            AgentPermissionError: If the action is not permitted.
        """
        actor_id = _aid(actor)
        target_id = _aid(target)
        granted, reason = self._evaluate(actor_id, permission, target_id)
        if not granted:
            self._fire(
                Hook.AGENT_PERMISSION_DENIED,
                {
                    "actor_id": actor_id,
                    "target_id": target_id,
                    "action": str(permission),
                    "reason": reason,
                },
            )
            raise AgentPermissionError(
                actor_id=actor_id,
                target_id=target_id,
                attempted_action=str(permission),
                reason=reason,
            )

    def record_action(self, actor_id: str, target_id: str, action: ControlAction | str) -> None:
        """Record a successful action to the audit log and fire AGENT_CONTROL_ACTION.

        This is an internal method used by :class:`SwarmController`; callers
        already hold resolved string IDs.

        Args:
            actor_id: ID of the agent that performed the action.
            target_id: ID of the target agent.
            action: :class:`~syrin.enums.ControlAction` describing what was done.
        """
        # Normalise to ControlAction if a plain string was passed
        with contextlib.suppress(ValueError):
            action = ControlAction(action)
        entry = AuditEntry(
            actor_id=actor_id,
            target_id=target_id,
            action=action,  # type: ignore[arg-type]
            timestamp=datetime.utcnow(),
        )
        self._audit.append(entry)
        self._fire(
            Hook.AGENT_CONTROL_ACTION,
            {
                "actor_id": actor_id,
                "target_id": target_id,
                "action": str(action),
            },
        )

    def audit_log(self) -> list[AuditEntry]:
        """Return a copy of all recorded audit entries.

        Returns:
            List of :class:`AuditEntry` records, oldest first.
        """
        return list(self._audit)

    # ------------------------------------------------------------------
    # Delegation
    # ------------------------------------------------------------------

    def delegate(
        self,
        delegator_id: AgentRef | str,
        delegate_id: AgentRef | str,
        permissions: list[AgentPermission],
        scope: DelegationScope = DelegationScope.CURRENT_RUN,
    ) -> None:
        """Grant *delegate_id* the listed permissions on behalf of *delegator_id*.

        Args:
            delegator_id: Agent instance or ID granting the delegation.
            delegate_id: Agent instance or ID receiving the delegation.
            permissions: Permissions to delegate.
            scope: How long the delegation remains in effect.

        Raises:
            AgentPermissionError: If *delegator_id* tries to delegate
                :attr:`~syrin.enums.AgentPermission.ADMIN` without holding
                the ADMIN role, or uses PERMANENT scope without ADMIN role.
        """
        delegator_id = _aid(delegator_id)
        delegate_id = _aid(delegate_id)

        if scope == DelegationScope.PERMANENT:
            delegator_role = self._role_of(delegator_id)
            if delegator_role != AgentRole.ADMIN:
                raise AgentPermissionError(
                    actor_id=delegator_id,
                    target_id=delegate_id,
                    attempted_action="permanent delegation",
                    reason=f"DelegationScope.PERMANENT requires ADMIN role; got {delegator_role}",
                )

        if AgentPermission.ADMIN in permissions:
            delegator_role = self._role_of(delegator_id)
            if delegator_role != AgentRole.ADMIN:
                raise AgentPermissionError(
                    actor_id=delegator_id,
                    target_id=delegate_id,
                    attempted_action="delegate ADMIN permission",
                    reason=f"only ADMIN role can delegate ADMIN permission; got {delegator_role}",
                )

        record = _DelegationRecord(
            delegator_id=delegator_id,
            delegate_id=delegate_id,
            permissions=list(permissions),
            scope=scope,
        )
        self._delegations.append(record)
        self._fire(
            Hook.AGENT_DELEGATION,
            {
                "delegator_id": delegator_id,
                "delegate_id": delegate_id,
                "permissions": list(permissions),
                "scope": scope,
            },
        )

    def revoke_delegation(self, delegator_id: AgentRef | str, delegate_id: AgentRef | str) -> None:
        """Remove all delegations from *delegator_id* to *delegate_id*.

        Args:
            delegator_id: Agent instance or ID that originally granted the delegation.
            delegate_id: Agent instance or ID whose delegated permissions are revoked.
        """
        delegator_id = _aid(delegator_id)
        delegate_id = _aid(delegate_id)
        self._delegations = [
            r
            for r in self._delegations
            if not (r.delegator_id == delegator_id and r.delegate_id == delegate_id)
        ]


# ---------------------------------------------------------------------------
# Factory helper
# ---------------------------------------------------------------------------


def build_guard_from_agents(
    agents: Sequence[AgentRef],
    fire_event_fn: Callable[[Hook, dict[str, object]], None] | None = None,
) -> SwarmAuthorityGuard:
    """Build a :class:`SwarmAuthorityGuard` automatically from agent class attributes.

    Reads the ``role`` and ``team`` :data:`~typing.ClassVar` attributes declared on
    each agent class to construct the ``roles`` and ``teams`` mappings without any
    free strings.

    Args:
        agents: List of agent instances.  Each agent's class should declare
            a ``role`` ClassVar (:class:`~syrin.enums.AgentRole`) and optionally
            a ``team`` ClassVar (list of sub-agent classes it supervises).
        fire_event_fn: Optional event hook callback.

    Returns:
        A :class:`SwarmAuthorityGuard` wired from the agents' class metadata.

    Example::

        class Supervisor(Agent):
            role = AgentRole.SUPERVISOR
            team = [WorkerAgent]

        class WorkerAgent(Agent):
            role = AgentRole.WORKER

        supervisor = Supervisor()
        worker = WorkerAgent()
        guard = build_guard_from_agents([supervisor, worker])
    """
    roles: dict[AgentRef, AgentRole] = {}
    teams: dict[AgentRef, list[AgentRef]] = {}

    # Index agents by their class for team membership lookup
    class_to_agents: dict[type, list[AgentRef]] = {}
    for agent in agents:
        cls = type(agent)
        class_to_agents.setdefault(cls, []).append(agent)
        agent_role: AgentRole = getattr(cls, "role", AgentRole.WORKER)
        if not isinstance(agent_role, AgentRole):
            agent_role = AgentRole.WORKER
        roles[agent] = agent_role

    for agent in agents:
        cls = type(agent)
        # Use `team` ClassVar (existing pattern in syrin Agent)
        managed_classes: list[type] | None = getattr(cls, "team", None)
        if not managed_classes:
            continue
        managed_agents: list[AgentRef] = []
        for managed_cls in managed_classes:
            managed_agents.extend(class_to_agents.get(managed_cls, []))
        if managed_agents:
            teams[agent] = managed_agents

    return SwarmAuthorityGuard(roles=roles, teams=teams, fire_event_fn=fire_event_fn)


__all__ = [
    "AgentPermissionError",
    "AuditEntry",
    "SwarmAuthorityGuard",
    "build_guard_from_agents",
]
