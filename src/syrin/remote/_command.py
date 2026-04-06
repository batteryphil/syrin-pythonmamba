"""RemoteCommand and CommandProcessor — remote control plane command processing."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from syrin.enums import Hook


class RemoteCommand(StrEnum):
    """Commands that can be sent to a running agent via the control plane.

    Attributes:
        PAUSE: Pause the agent's execution.
        RESUME: Resume a paused agent.
        KILL: Terminate the agent permanently.
        ROLLBACK: Roll back to previous configuration version.
        UPDATE_CONFIG: Push a configuration update.
    """

    PAUSE = "pause"
    RESUME = "resume"
    KILL = "kill"
    ROLLBACK = "rollback"
    UPDATE_CONFIG = "update_config"


@dataclass
class CommandResult:
    """Result of executing a remote command.

    Attributes:
        success: True if the command was accepted and executed.
        reason: Human-readable explanation when success=False; empty string on success.
        command: The command that was attempted.
    """

    success: bool
    reason: str
    command: RemoteCommand


@dataclass
class CommandAuditEntry:
    """Audit record for a remote command execution attempt.

    Attributes:
        command: The command that was executed or rejected.
        timestamp: When the command was received, as an ISO-8601 UTC string.
        actor_id: Identifier of the entity that sent the command (or "anonymous").
        success: True if the command was executed.
        reason: Why the command was rejected; empty string if success=True.
    """

    command: RemoteCommand
    timestamp: str
    actor_id: str
    success: bool
    reason: str


class CommandProcessor:
    """Processes remote control commands for a single agent.

    Manages the PAUSE → running/paused/killed state machine, optional
    two-step kill confirmation with a configurable time window, and optional
    Ed25519 signed-command verification.

    Each ``CommandProcessor`` instance maintains completely independent state —
    there is no shared class-level state between instances.

    Example:
        >>> processor = CommandProcessor(agent_id="my-agent")
        >>> result = processor.execute(RemoteCommand.PAUSE)
        >>> result.success
        True
        >>> processor.status
        'paused'
    """

    def __init__(
        self,
        agent_id: str,
        kill_requires_confirmation: bool = False,
        confirmation_window_seconds: float = 30.0,
        require_signed_commands: bool = False,
        fire_fn: Callable[[Hook, dict[str, object]], None] | None = None,
    ) -> None:
        """Initialise a CommandProcessor for the given agent.

        Args:
            agent_id: Unique identifier for the agent this processor controls.
            kill_requires_confirmation: When True, the first KILL command arms a
                confirmation window. A second KILL within the window completes the
                kill; after the window expires the pending confirmation resets.
            confirmation_window_seconds: Duration of the kill confirmation window
                in seconds. Default is 30.0.
            require_signed_commands: When True, every command must carry a valid
                Ed25519 signature. Unsigned commands are rejected.
            fire_fn: Optional hook-firing callback ``(Hook, dict) -> None``.
        """
        self._agent_id = agent_id
        self._kill_requires_confirmation = kill_requires_confirmation
        self._confirmation_window_seconds = confirmation_window_seconds
        self._require_signed_commands = require_signed_commands
        self._fire_fn = fire_fn

        self._status: str = "running"
        self._pending_kill_at: float | None = None
        self._audit: list[CommandAuditEntry] = []

    def execute(
        self,
        command: RemoteCommand,
        actor_id: str = "anonymous",
        signature: bytes | None = None,
        public_key: bytes | None = None,
    ) -> CommandResult:
        """Execute a remote command.

        Validates the command (signature check when required), applies state
        transitions, and records an audit entry.  Fires ``Hook.COMMAND_EXECUTED``
        on success and ``Hook.COMMAND_REJECTED`` on rejection.

        Args:
            command: The :class:`RemoteCommand` to execute.
            actor_id: Identifier of the sender (used in the audit log).
            signature: Optional Ed25519 signature bytes for signed commands.
            public_key: Optional Ed25519 public key bytes for signature verification.

        Returns:
            A :class:`CommandResult` describing whether the command was accepted.
        """
        from syrin.enums import Hook

        # --- Signature verification ---
        if self._require_signed_commands:
            if signature is None or public_key is None:
                result = CommandResult(success=False, reason="unsigned", command=command)
                self._record(result, actor_id)
                if self._fire_fn is not None:
                    self._fire_fn(
                        Hook.COMMAND_REJECTED,
                        {
                            "agent_id": self._agent_id,
                            "command": command,
                            "reason": "unsigned",
                            "actor_id": actor_id,
                        },
                    )
                return result

            from syrin.security.identity import AgentIdentity

            valid = AgentIdentity.verify(
                message=command.encode(),
                signature=signature,
                public_key=public_key,
            )
            if not valid:
                result = CommandResult(success=False, reason="invalid_signature", command=command)
                self._record(result, actor_id)
                if self._fire_fn is not None:
                    self._fire_fn(
                        Hook.COMMAND_REJECTED,
                        {
                            "agent_id": self._agent_id,
                            "command": command,
                            "reason": "invalid_signature",
                            "actor_id": actor_id,
                        },
                    )
                return result

        # --- State machine ---
        result = self._apply_command(command)
        self._record(result, actor_id)

        if self._fire_fn is not None:
            hook = Hook.COMMAND_EXECUTED if result.success else Hook.COMMAND_REJECTED
            self._fire_fn(
                hook,
                {
                    "agent_id": self._agent_id,
                    "command": command,
                    "success": result.success,
                    "reason": result.reason,
                    "actor_id": actor_id,
                    "status": self._status,
                },
            )

        return result

    def _apply_command(self, command: RemoteCommand) -> CommandResult:
        """Apply command state transitions without side effects (no hooks/audit).

        Args:
            command: The command to apply.

        Returns:
            A :class:`CommandResult` for the attempted transition.
        """
        if command == RemoteCommand.PAUSE:
            self._status = "paused"
            return CommandResult(success=True, reason="", command=command)

        if command == RemoteCommand.RESUME:
            self._status = "running"
            return CommandResult(success=True, reason="", command=command)

        if command == RemoteCommand.KILL:
            return self._handle_kill(command)

        if command == RemoteCommand.ROLLBACK:
            return CommandResult(success=True, reason="", command=command)

        if command == RemoteCommand.UPDATE_CONFIG:
            return CommandResult(success=True, reason="", command=command)

        return CommandResult(success=False, reason="unknown_command", command=command)

    def _handle_kill(self, command: RemoteCommand) -> CommandResult:
        """Handle KILL with optional two-step confirmation.

        Args:
            command: The KILL command.

        Returns:
            CommandResult — success=True only when kill is fully confirmed.
        """
        if not self._kill_requires_confirmation:
            self._status = "killed"
            self._pending_kill_at = None
            return CommandResult(success=True, reason="", command=command)

        now = time.monotonic()

        if self._pending_kill_at is None:
            # First KILL: arm the confirmation window
            self._pending_kill_at = now
            return CommandResult(
                success=False,
                reason="kill_pending_confirmation",
                command=command,
            )

        elapsed = now - self._pending_kill_at
        if elapsed <= self._confirmation_window_seconds:
            # Second KILL within window: execute
            self._status = "killed"
            self._pending_kill_at = None
            return CommandResult(success=True, reason="", command=command)

        # Window expired: reset and re-arm
        self._pending_kill_at = now
        return CommandResult(
            success=False,
            reason="kill_pending_confirmation",
            command=command,
        )

    def _record(self, result: CommandResult, actor_id: str) -> None:
        """Append an audit entry for a command execution attempt.

        Args:
            result: The result to record.
            actor_id: Identifier of the command sender.
        """
        self._audit.append(
            CommandAuditEntry(
                command=result.command,
                timestamp=datetime.now(tz=UTC).isoformat(),
                actor_id=actor_id,
                success=result.success,
                reason=result.reason,
            )
        )

    @property
    def status(self) -> str:
        """Current agent lifecycle status: ``'running'``, ``'paused'``, or ``'killed'``.

        Returns:
            String status of the controlled agent.
        """
        return self._status

    def audit_log(self) -> list[CommandAuditEntry]:
        """Return all recorded command attempts for this processor.

        Returns:
            List of :class:`CommandAuditEntry` objects in chronological order.
        """
        return list(self._audit)
