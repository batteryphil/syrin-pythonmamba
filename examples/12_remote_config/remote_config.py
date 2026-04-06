"""Remote Config Control Plane — live config push to a running agent.

Shows how to attach a RemoteConfig to an agent so its settings can be
changed at runtime without restarting.

Key concepts:
  - RemoteConfig(url=..., agent_id=..., allow=[RemoteField.MODEL, ...])
  - agent.config_schema() — export the full JSON Schema for dashboard rendering
  - config.apply(changes, changed_by="...") — push a config change
  - RemoteCommand.PAUSE — send a control command over the wire
  - config.rollback() — revert to the previous config version
  - Hook.CONFIG_APPLIED, Hook.CONFIG_ROLLBACK, Hook.CONFIG_REJECTED

Note: This example uses a fake URL — no network calls are made. The
RemoteConfig operates entirely in-process here, demonstrating the API.

Run:
    uv run python examples/remote_config.py
"""

from __future__ import annotations

import asyncio

from syrin import Agent, Model
from syrin.enums import Hook, RemoteField
from syrin.remote_config import ConfigRejectedError, RemoteConfig
from syrin.response import Response

# ── Agent with RemoteConfig ───────────────────────────────────────────────────
#
# The allow list restricts which fields can be changed remotely.
# Here we allow MODEL and BUDGET, but deny IDENTITY (security boundary).


class HRAgent(Agent):
    """HR assistant that supports live config updates."""

    model = Model.mock(latency_seconds=0.05, lorem_length=8)
    system_prompt = "You are an HR assistant."

    # RemoteConfig is attached at class level; in production you would
    # pass it to Agent(..., remote_config=RemoteConfig(...))
    async def arun(self, input_text: str) -> Response[str]:
        return Response(content=f"HR response to: {input_text}", cost=0.002)


# ── Example 1: config_schema() ────────────────────────────────────────────────
#
# Export the full JSON Schema for this agent so a dashboard can render the
# correct form fields for remote configuration.


async def example_config_schema() -> None:
    print("\n── Example 1: agent.config_schema() ────────────────────────────")

    agent = HRAgent()
    schema = agent.config_schema()

    print(f"Schema type: {schema.get('type')}")
    properties = schema.get("properties", {})
    print(f"Configurable sections ({len(properties)}):")
    for section, defn in list(properties.items())[:6]:
        print(f"  {section}: {type(defn).__name__}")


# ── Example 2: Apply a config change ─────────────────────────────────────────
#
# apply() pushes a change, validates it, versions it, and emits hooks.


async def example_apply_config() -> None:
    print("\n── Example 2: config.apply() — push a model change ─────────────")

    config = RemoteConfig(
        url="https://nexus.syrin.dev/config",  # fake URL; no network calls
        agent_id="hr-agent-prod",
        allow=[RemoteField.MODEL, RemoteField.BUDGET],
        deny=[RemoteField.IDENTITY],
    )

    # Track hooks
    applied: list[dict[str, object]] = []
    config._fire_fn = (  # type: ignore[assignment]
        lambda hook, data: (
            applied.append({"hook": hook, **data})
            if hook in (Hook.CONFIG_APPLIED, Hook.CONFIG_REJECTED)
            else None
        )
    )

    # Push a model change
    version = await config.apply(
        changes={"model": "gpt-4o-mini"},
        changed_by="ops-team",
    )

    print(f"Version number:      {version.version}")
    print(f"Applied at:          {version.applied_at.isoformat()}")
    print(f"Applied by:          {version.applied_by}")
    print(f"Fields changed:      {version.fields_changed}")
    print(f"Previous values:     {version.previous_values}")
    print(f"New values:          {version.new_values}")
    print(f"Rollback token:      {version.rollback_token[:8]}...")

    print(f"\nHooks fired: {[str(e['hook']) for e in applied]}")

    # Push a second change
    await config.apply(
        changes={"budget": {"max_cost": 5.00}},
        changed_by="ops-team",
    )
    print(f"Config history length: {len(config.config_history)}")


# ── Example 3: Rollback ───────────────────────────────────────────────────────
#
# rollback() reverts to the version before the last change and fires
# Hook.CONFIG_ROLLBACK.


async def example_rollback() -> None:
    print("\n── Example 3: config.rollback() ─────────────────────────────────")

    config = RemoteConfig(
        url="https://nexus.syrin.dev/config",
        agent_id="hr-agent-prod",
        allow=[RemoteField.MODEL, RemoteField.BUDGET],
    )

    rollback_events: list[dict[str, object]] = []

    def _track(hook: object, data: dict[str, object]) -> None:
        rollback_events.append({"hook": str(hook), **data})

    config._fire_fn = _track  # type: ignore[assignment]

    # Apply two changes
    await config.apply({"model": "claude-haiku-4-5-20251001"}, changed_by="deploy")
    v2 = await config.apply({"model": "gpt-4o"}, changed_by="ops")

    print(f"Current model after v2: {config._current_values.get('model')}")

    # Rollback to previous version (v1)
    rollback_version = await config.rollback()

    print(f"Model after rollback:   {config._current_values.get('model')}")
    print(f"Rollback version:       {rollback_version.version}")
    print(f"Rolled from version:    {v2.version}")

    rb_hooks = [e["hook"] for e in rollback_events if "rollback" in e["hook"].lower()]
    print(f"Rollback hooks:         {rb_hooks}")


# ── Example 4: Rejected field (IDENTITY denied) ───────────────────────────────
#
# Fields in the deny list raise ConfigRejectedError and fire Hook.CONFIG_REJECTED.


async def example_rejected_field() -> None:
    print("\n── Example 4: ConfigRejectedError on denied field ───────────────")

    config = RemoteConfig(
        url="https://nexus.syrin.dev/config",
        agent_id="hr-agent-prod",
        allow=[RemoteField.MODEL],
        deny=[RemoteField.IDENTITY],
    )

    rejected_events: list[str] = []
    config._fire_fn = (  # type: ignore[assignment]
        lambda hook, _data: (
            rejected_events.append(str(hook)) if "rejected" in str(hook).lower() else None
        )
    )

    try:
        await config.apply({"identity": "hacked-identity"}, changed_by="attacker")
    except ConfigRejectedError as exc:
        print(f"ConfigRejectedError caught: field={exc.field}  reason={exc.reason}")
        print(
            f"Hook.CONFIG_REJECTED fired: {Hook.CONFIG_REJECTED in rejected_events or bool(rejected_events)}"
        )


# ── Example 5: RemoteCommand simulation ──────────────────────────────────────
#
# RemoteCommand values represent control actions sent to a running agent.
# Show the available commands and how they would be dispatched.


async def example_remote_commands() -> None:
    print("\n── Example 5: RemoteCommand values ─────────────────────────────")

    from syrin.enums import RemoteCommand as RC

    commands = [
        (RC.PAUSE, "pause execution after current step"),
        (RC.RESUME, "resume a paused agent"),
        (RC.KILL, "terminate immediately"),
        (RC.ROLLBACK, "roll back to last checkpoint"),
        (RC.FLUSH_MEMORY, "clear agent memory"),
        (RC.DRAIN, "complete current run then pause"),
    ]

    print("Available RemoteCommand values:")
    for cmd, desc in commands:
        print(f"  {cmd!s:<30}  # {desc}")

    # Demonstrate how a PAUSE command would look over the wire
    print("\nExample wire message:")
    print("  {")
    print(f'    "command": "{RC.PAUSE}",')
    print('    "agent_id": "hr-agent-prod",')
    print('    "issued_by": "nexus-dashboard"')
    print("  }")


# ── Example 6: Config history ─────────────────────────────────────────────────


async def example_config_history() -> None:
    print("\n── Example 6: config.get_history() ────────────────────────────")

    config = RemoteConfig(
        url="https://nexus.syrin.dev/config",
        agent_id="hr-agent-prod",
        allow=[RemoteField.MODEL, RemoteField.BUDGET],
    )

    await config.apply({"model": "gpt-4o-mini"}, changed_by="alice")
    await config.apply({"budget": {"max_cost": 2.0}}, changed_by="bob")
    await config.apply({"model": "claude-haiku-4-5-20251001"}, changed_by="alice")

    history = await config.get_history(last_n=5)
    print(f"History entries ({len(history)}):")
    for v in history:
        print(f"  v{v.version}  by={v.applied_by}  fields={v.fields_changed}")


# ── Main ──────────────────────────────────────────────────────────────────────


async def main() -> None:
    await example_config_schema()
    await example_apply_config()
    await example_rollback()
    await example_rejected_field()
    await example_remote_commands()
    await example_config_history()
    print("\nAll remote config examples completed.")


if __name__ == "__main__":
    asyncio.run(main())
