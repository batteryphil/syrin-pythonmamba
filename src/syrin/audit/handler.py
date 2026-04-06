"""Audit hook handler: maps hooks to AuditEntry and writes to backend."""

from __future__ import annotations

from typing import TYPE_CHECKING

from syrin.audit.models import AuditEntry, AuditLog
from syrin.enums import AuditEventType, Hook

if TYPE_CHECKING:
    from syrin.events import EventContext

# Hook -> AuditEventType mapping
_HOOK_TO_AUDIT_EVENT: dict[Hook, str] = {
    Hook.AGENT_RUN_START: AuditEventType.AGENT_RUN_START,
    Hook.AGENT_RUN_END: AuditEventType.AGENT_RUN_END,
    Hook.AGENT_INIT: AuditEventType.AGENT_INIT,
    Hook.AGENT_RESET: AuditEventType.AGENT_RESET,
    Hook.SERVE_REQUEST_START: AuditEventType.SERVE_REQUEST_START,
    Hook.SERVE_REQUEST_END: AuditEventType.SERVE_REQUEST_END,
    Hook.LLM_REQUEST_START: AuditEventType.LLM_CALL,
    Hook.LLM_REQUEST_END: AuditEventType.LLM_CALL,
    Hook.LLM_RETRY: AuditEventType.LLM_RETRY,
    Hook.LLM_FALLBACK: AuditEventType.LLM_FALLBACK,
    Hook.TOOL_CALL_START: AuditEventType.TOOL_CALL,
    Hook.TOOL_CALL_END: AuditEventType.TOOL_CALL,
    Hook.TOOL_ERROR: AuditEventType.TOOL_ERROR,
    Hook.HANDOFF_START: AuditEventType.HANDOFF_START,
    Hook.HANDOFF_END: AuditEventType.HANDOFF_END,
    Hook.HANDOFF_BLOCKED: AuditEventType.HANDOFF_BLOCKED,
    Hook.SPAWN_START: AuditEventType.SPAWN_START,
    Hook.SPAWN_END: AuditEventType.SPAWN_END,
    Hook.BUDGET_CHECK: AuditEventType.BUDGET_CHECK,
    Hook.BUDGET_THRESHOLD: AuditEventType.BUDGET_THRESHOLD,
    Hook.BUDGET_EXCEEDED: AuditEventType.BUDGET_EXCEEDED,
    Hook.GUARDRAIL_INPUT: AuditEventType.GUARDRAIL_INPUT,
    Hook.GUARDRAIL_OUTPUT: AuditEventType.GUARDRAIL_OUTPUT,
    Hook.GUARDRAIL_BLOCKED: AuditEventType.GUARDRAIL_BLOCKED,
    Hook.MEMORY_STORE: AuditEventType.MEMORY_STORE,
    Hook.MEMORY_RECALL: AuditEventType.MEMORY_RECALL,
    Hook.MEMORY_FORGET: AuditEventType.MEMORY_FORGET,
    Hook.DYNAMIC_PIPELINE_START: AuditEventType.DYNAMIC_PIPELINE_START,
    Hook.DYNAMIC_PIPELINE_PLAN: AuditEventType.DYNAMIC_PIPELINE_PLAN,
    Hook.DYNAMIC_PIPELINE_EXECUTE: AuditEventType.DYNAMIC_PIPELINE_EXECUTE,
    Hook.DYNAMIC_PIPELINE_AGENT_SPAWN: AuditEventType.DYNAMIC_PIPELINE_AGENT_SPAWN,
    Hook.DYNAMIC_PIPELINE_AGENT_COMPLETE: AuditEventType.DYNAMIC_PIPELINE_AGENT_COMPLETE,
    Hook.DYNAMIC_PIPELINE_END: AuditEventType.DYNAMIC_PIPELINE_END,
    Hook.DYNAMIC_PIPELINE_ERROR: AuditEventType.DYNAMIC_PIPELINE_ERROR,
}

# Events that are LLM-related (for include_llm_calls)
_LLM_EVENTS = {AuditEventType.LLM_CALL, AuditEventType.LLM_RETRY, AuditEventType.LLM_FALLBACK}

# Events that are tool-related (for include_tool_calls)
_TOOL_EVENTS = {AuditEventType.TOOL_CALL, AuditEventType.TOOL_ERROR}

# Events for handoff/spawn (for include_handoff_spawn)
_HANDOFF_SPAWN_EVENTS = {
    AuditEventType.HANDOFF_START,
    AuditEventType.HANDOFF_END,
    AuditEventType.HANDOFF_BLOCKED,
    AuditEventType.SPAWN_START,
    AuditEventType.SPAWN_END,
}

# Events for budget (for include_budget)
_BUDGET_EVENTS = {
    AuditEventType.BUDGET_CHECK,
    AuditEventType.BUDGET_THRESHOLD,
    AuditEventType.BUDGET_EXCEEDED,
}


def _should_log(audit_config: AuditLog, event_type: str) -> bool:
    return not (
        (event_type in _LLM_EVENTS and not audit_config.include_llm_calls)
        or (event_type in _TOOL_EVENTS and not audit_config.include_tool_calls)
        or (event_type in _HANDOFF_SPAWN_EVENTS and not audit_config.include_handoff_spawn)
        or (event_type in _BUDGET_EVENTS and not audit_config.include_budget)
    )


def _ctx_val(ctx: EventContext, key: str, default: object = None) -> object:
    return ctx.get(key, default)


def _ctx_float(ctx: EventContext, key: str) -> float | None:
    v = _ctx_val(ctx, key)
    if v is None:
        return None
    try:
        return float(v)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _ctx_int(ctx: EventContext, key: str) -> int | None:
    v = _ctx_val(ctx, key)
    if v is None:
        return None
    try:
        return int(v)  # type: ignore[call-overload, no-any-return]
    except (TypeError, ValueError):
        return None


def _build_entry(
    source: str,
    event_type: str,
    hook: Hook,
    ctx: EventContext,
    audit_config: AuditLog,
) -> AuditEntry:
    """Build AuditEntry from EventContext."""
    cost = _ctx_float(ctx, "cost")
    duration = _ctx_float(ctx, "duration")
    duration_ms = duration * 1000.0 if duration is not None else None
    tokens_val = _ctx_val(ctx, "tokens")
    tokens: dict[str, int] | None = None
    if isinstance(tokens_val, dict):
        tokens = {k: int(v) for k, v in tokens_val.items() if isinstance(v, (int, float))}
    elif isinstance(tokens_val, (int, float)):
        tokens = {"total": int(tokens_val)}

    extra: dict[str, object] = {}
    if audit_config.include_user_input and "input" in ctx:
        extra["input"] = str(ctx["input"])[:500]
    if audit_config.include_model_output and "content" in ctx:
        extra["content"] = str(ctx["content"])[:1000]

    return AuditEntry(
        source=source,
        event=event_type,
        model=_ctx_val(ctx, "model") if isinstance(_ctx_val(ctx, "model"), str) else None,  # type: ignore[arg-type]
        tokens=tokens,
        cost_usd=cost,
        budget_percent=_ctx_float(ctx, "budget_percent"),
        duration_ms=duration_ms,
        trace_id=str(_ctx_val(ctx, "trace_id")) if _ctx_val(ctx, "trace_id") else None,
        run_id=str(_ctx_val(ctx, "run_id")) if _ctx_val(ctx, "run_id") else None,
        iteration=_ctx_int(ctx, "iteration"),
        tool_name=_ctx_val(ctx, "name") if isinstance(_ctx_val(ctx, "name"), str) else None,  # type: ignore[arg-type]
        tool_error=_ctx_val(ctx, "error") if isinstance(_ctx_val(ctx, "error"), str) else None,  # type: ignore[arg-type]
        stop_reason=_ctx_val(ctx, "stop_reason")  # type: ignore[arg-type]
        if isinstance(_ctx_val(ctx, "stop_reason"), str)
        else None,
        extra=extra if extra else None,
    )


class AuditHookHandler:
    """Subscribes to hooks and writes audit entries to backend."""

    def __init__(self, source: str, config: AuditLog) -> None:
        self._source = source
        self._config = config
        self._backend = config.get_backend()

    def handle(self, hook: Hook, ctx: EventContext) -> None:
        """Handle a hook event. Builds AuditEntry and writes if configured to include."""
        event_type = _HOOK_TO_AUDIT_EVENT.get(hook)
        if event_type is None:
            return
        if not _should_log(self._config, event_type):
            return
        try:
            entry = _build_entry(self._source, event_type, hook, ctx, self._config)
            self._backend.write(entry)  # type: ignore[attr-defined]
        except Exception:
            pass  # Audit must not break execution

    def __call__(self, hook: Hook, ctx: EventContext) -> None:
        """Allow use as events.on_all(handler)."""
        self.handle(hook, ctx)
