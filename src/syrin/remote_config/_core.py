"""RemoteConfig, ConfigVersion, RemoteConfigSnapshot — Remote Config Control Plane core."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from syrin.remote_config._command import RemoteCommandConfig
from syrin.remote_config._validator import ConfigValidationError, RemoteConfigValidator

if TYPE_CHECKING:
    pass


class ConfigRejectedError(Exception):
    """Raised when a remote config push is rejected due to a denied field.

    Attributes:
        field: The field name that was denied.
        reason: Human-readable explanation.
    """

    def __init__(self, field: str, reason: str = "field_denied") -> None:
        """Create a ConfigRejectedError.

        Args:
            field: The field name that was denied.
            reason: Human-readable explanation.
        """
        super().__init__(f"Config change rejected for field '{field}': {reason}")
        self.field = field
        self.reason = reason


@dataclass
class ConfigVersion:
    """A single versioned record of a config change.

    Each :meth:`RemoteConfig.apply` call appends one :class:`ConfigVersion`
    to the history.

    Attributes:
        version: Monotonically increasing version number, starting at 1.
        applied_at: UTC timestamp when the change was applied.
        applied_by: Identifier of who pushed this config (e.g. a Nexus user ID).
        fields_changed: List of top-level field keys that were changed.
        previous_values: Snapshot of the values *before* this change.
        new_values: The new values that were applied.
        rollback_token: UUID4 string used to safely reference this version for rollback.
    """

    version: int
    applied_at: datetime
    applied_by: str
    fields_changed: list[str]
    previous_values: dict[str, object]
    new_values: dict[str, object]
    rollback_token: str


@dataclass(frozen=True)
class RemoteConfigSnapshot:
    """Current agent config snapshot at a point in time.

    Returned by :meth:`~syrin.agent.Agent.current_config`.

    Attributes:
        agent_id: The agent's unique identifier.
        version: The current config version number.
        values: Mapping of field name to current value.
        captured_at: UTC timestamp when the snapshot was taken.
    """

    agent_id: str
    version: int
    values: dict[str, object]
    captured_at: datetime


class RemoteConfig:
    """Remote Config Control Plane client for a single agent.

    Manages the lifecycle of remote configuration: transport settings,
    field-level access control, validation, versioning, and rollback.

    Pass an instance as ``remote_config=RemoteConfig(...)`` on
    :class:`~syrin.Agent`.

    Example:
        >>> remote = RemoteConfig(
        ...     url="https://nexus.syrin.dev/config",
        ...     agent_id="hr-agent-prod",
        ...     api_key=os.environ["NEXUS_API_KEY"],
        ...     transport=RemoteTransport.SSE,
        ...     allow=[RemoteField.MODEL, RemoteField.BUDGET],
        ...     deny=[RemoteField.IDENTITY],
        ...     validators=[RemoteConfigValidator.max_budget(10.00)],
        ... )
    """

    def __init__(
        self,
        *,
        url: str,
        agent_id: str,
        api_key: str = "",
        transport: object = None,
        poll_interval: int = 30,
        reconnect_on_failure: bool = True,
        max_reconnect_attempts: int = 10,
        allow: list[object] | None = None,
        deny: list[object] | None = None,
        validators: list[RemoteConfigValidator] | None = None,
        command_config: RemoteCommandConfig | None = None,
    ) -> None:
        """Initialise a RemoteConfig instance.

        Args:
            url: The remote config server URL.
            agent_id: Unique identifier for this agent (used as namespace).
            api_key: API key for authenticating with the remote config server.
            transport: Transport mechanism (:class:`~syrin.enums.RemoteTransport`).
                Defaults to ``RemoteTransport.POLLING``.
            poll_interval: Seconds between polls when using POLLING transport.
            reconnect_on_failure: Automatically reconnect on transport failure.
            max_reconnect_attempts: Maximum number of reconnect attempts.
            allow: Whitelist of :class:`~syrin.enums.RemoteField` values that
                can be changed remotely.  When set, only listed fields are
                accepted.  When ``None``, all fields are allowed (minus deny).
            deny: Blacklist of :class:`~syrin.enums.RemoteField` values that
                cannot be changed remotely.  Takes priority over *allow*.
            validators: List of :class:`~syrin.remote_config.RemoteConfigValidator`
                instances to run before each config push.
            command_config: Options for remote command handling.
        """
        from syrin.enums import RemoteTransport

        self.url = url
        self.agent_id = agent_id
        self.api_key = api_key
        self.transport: object = transport if transport is not None else RemoteTransport.POLLING
        self.poll_interval = poll_interval
        self.reconnect_on_failure = reconnect_on_failure
        self.max_reconnect_attempts = max_reconnect_attempts
        self._allow: frozenset[object] = frozenset(allow) if allow is not None else frozenset()
        self._deny: frozenset[object] = frozenset(deny) if deny is not None else frozenset()
        self._validators: list[RemoteConfigValidator] = validators or []
        self.command_config: RemoteCommandConfig = command_config or RemoteCommandConfig()

        # Internal state
        self._current_values: dict[str, object] = {}
        self._version: int = 0
        self._history: list[ConfigVersion] = []

        # Optional hook emitter — injected by Agent when mounting
        self._fire_fn: object = None

    def is_field_allowed(self, field: object) -> bool:
        """Check whether a field is permitted to be changed remotely.

        Deny list takes priority: if *field* is in the deny list it is always
        rejected.  Next, if an allow list is set, only listed fields are
        accepted.  When neither list is set, all fields are allowed.

        Args:
            field: A :class:`~syrin.enums.RemoteField` value to check.

        Returns:
            ``True`` if the field may be changed remotely.
        """
        if self._deny and field in self._deny:
            return False
        if self._allow:
            return field in self._allow
        return True

    async def apply(self, changes: dict[str, object], changed_by: str = "remote") -> ConfigVersion:
        """Apply a config change, validate, version, and emit hooks.

        Steps:
        1. Check every changed field is allowed (deny list first, then allow list).
        2. Run all registered validators.
        3. Record a :class:`ConfigVersion` in history.
        4. Emit ``Hook.CONFIG_APPLIED`` on success or ``Hook.CONFIG_REJECTED`` on failure.

        Args:
            changes: Mapping of field name to new value.
            changed_by: Identifier of who is pushing this config (e.g. user ID).

        Returns:
            The :class:`ConfigVersion` record for this change.

        Raises:
            ConfigRejectedError: If any changed field is denied by the access policy.
            ConfigValidationError: If any validator rejects the new config.
        """
        from syrin.enums import Hook, RemoteField

        # 1. Field access control
        for key in changes:
            try:
                remote_field = RemoteField(key)
            except ValueError:
                # Unknown field key — treat as allowed (not in deny/allow lists)
                remote_field = key  # type: ignore[assignment]
            if not self.is_field_allowed(remote_field):
                self._fire_hook(
                    Hook.CONFIG_REJECTED,
                    {
                        "agent_id": self.agent_id,
                        "field": key,
                        "reason": "field_denied",
                        "changed_by": changed_by,
                    },
                )
                raise ConfigRejectedError(field=key)

        # 2. Validators
        for validator in self._validators:
            try:
                validator(changes, self)
            except ConfigValidationError as exc:
                self._fire_hook(
                    Hook.CONFIG_REJECTED,
                    {
                        "agent_id": self.agent_id,
                        "field": exc.field,
                        "reason": str(exc),
                        "changed_by": changed_by,
                    },
                )
                raise

        # 3. Build version record
        previous = {k: self._current_values[k] for k in changes if k in self._current_values}
        self._version += 1
        self._current_values.update(changes)

        version_record = ConfigVersion(
            version=self._version,
            applied_at=datetime.now(tz=UTC),
            applied_by=changed_by,
            fields_changed=list(changes.keys()),
            previous_values=previous,
            new_values=dict(changes),
            rollback_token=str(uuid.uuid4()),
        )
        self._history.append(version_record)

        # 4. Emit CONFIG_APPLIED
        self._fire_hook(
            Hook.CONFIG_APPLIED,
            {
                "agent_id": self.agent_id,
                "version": self._version,
                "fields_changed": version_record.fields_changed,
                "changed_by": changed_by,
            },
        )

        return version_record

    async def rollback(self, version: int | None = None) -> ConfigVersion:
        """Roll back to a previous config version.

        Args:
            version: The specific version number to roll back to.  When
                ``None``, rolls back to the version immediately before the
                current one.

        Returns:
            A new :class:`ConfigVersion` record representing the rollback.

        Raises:
            ValueError: If there is no history to roll back to, or if the
                requested version does not exist.
        """
        from syrin.enums import Hook

        if not self._history:
            raise ValueError("No config history available — cannot rollback.")

        if version is None:
            # Roll back to the version before the current one
            if len(self._history) < 2:
                # We can only roll back to the empty initial state
                target_values: dict[str, object] = {}
                rolled_back_from = self._history[-1]
            else:
                rolled_back_from = self._history[-1]
                target_values = dict(self._history[-2].new_values)
        else:
            # Find the specific target version
            target_entry = next((v for v in self._history if v.version == version), None)
            if target_entry is None:
                raise ValueError(f"Config version {version} not found in history.")
            rolled_back_from = self._history[-1]
            target_values = dict(target_entry.new_values)

        # Apply the rollback as a new version entry
        previous = dict(self._current_values)
        self._version += 1
        self._current_values = target_values

        rollback_record = ConfigVersion(
            version=self._version,
            applied_at=datetime.now(tz=UTC),
            applied_by="rollback",
            fields_changed=list(rolled_back_from.fields_changed),
            previous_values=previous,
            new_values=target_values,
            rollback_token=str(uuid.uuid4()),
        )
        self._history.append(rollback_record)

        self._fire_hook(
            Hook.CONFIG_ROLLBACK,
            {
                "agent_id": self.agent_id,
                "from_version": rolled_back_from.version,
                "to_version": self._version,
            },
        )

        return rollback_record

    async def get_history(self, last_n: int = 10) -> list[ConfigVersion]:
        """Return the most recent config change history entries.

        Args:
            last_n: Maximum number of entries to return, most recent first.

        Returns:
            A list of :class:`ConfigVersion` records, newest last (chronological).
        """
        return list(self._history[-last_n:])

    @property
    def config_history(self) -> list[ConfigVersion]:
        """All applied config versions in chronological order.

        Returns:
            Full list of :class:`ConfigVersion` records.
        """
        return list(self._history)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _fire_hook(self, hook: object, data: dict[str, object]) -> None:
        """Fire a hook via the injected fire function, if available.

        Args:
            hook: Hook enum value to fire.
            data: Event context data.
        """
        if callable(self._fire_fn):
            self._fire_fn(hook, data)
