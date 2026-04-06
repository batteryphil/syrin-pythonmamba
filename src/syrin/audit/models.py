"""Audit data models: AuditEntry, AuditEvent, AuditFilters, AuditLog."""

from __future__ import annotations

from datetime import UTC, datetime

from pydantic import BaseModel, Field

from syrin.enums import AuditBackend


class AuditEntry(BaseModel):  # type: ignore[explicit-any]
    """Single audit log entry. Written to audit backend as JSONL line.

    Attributes:
        timestamp: When the event occurred.
        source: Agent class name, Pipeline, or DynamicPipeline.
        event: Audit event type (e.g. llm_call, tool_call).
        model: Model ID used.
        tokens: Input/output/total token counts.
        cost_usd: Cost in USD.
        budget_percent: Budget utilization percentage.
        duration_ms: Duration in milliseconds.
        trace_id, run_id: Observability IDs.
        iteration: Loop iteration number.
        tool_calls, tool_name, tool_error: Tool execution info.
        stop_reason: Why the run ended.
        extra: Hook-specific fields.
    """

    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    source: str = Field(description="Agent class name or AgentRouter")
    event: str = Field(description="Audit event type (e.g. llm_call, tool_call)")
    model: str | None = None
    tokens: dict[str, int] | None = Field(default=None, description="input, output, total")
    cost_usd: float | None = None
    budget_percent: float | None = None
    duration_ms: float | None = None
    trace_id: str | None = None
    run_id: str | None = None
    iteration: int | None = None
    tool_calls: list[dict[str, object]] | None = None
    tool_name: str | None = None
    tool_error: str | None = None
    stop_reason: str | None = None
    extra: dict[str, object] | None = Field(default=None, description="Hook-specific fields")

    def model_dump_json_line(self) -> str:
        """Serialize as JSON line for JSONL output."""
        return self.model_dump_json()


class AuditFilters(BaseModel):  # type: ignore[explicit-any]
    """Filters for querying audit entries (optional backend support).

    Attributes:
        agent: Filter by source agent name.
        event: Filter by event type.
        since, until: Time range.
        limit: Max entries to return (default 100).
    """

    agent: str | None = None
    event: str | None = None
    since: datetime | None = None
    until: datetime | None = None
    limit: int = 100


class AuditLog(BaseModel):  # type: ignore[explicit-any]
    """Audit configuration for Agent, Pipeline, or DynamicPipeline.

    Pass to Agent(audit=AuditLog(...)) or Pipeline(audit=...). Events are
    written to the backend (JSONL by default).

    Attributes:
        backend: AuditBackend (JSONL, FILE, OTLP).
        path: File path for JSONL backend.
        include_llm_calls, include_tool_calls, include_handoff_spawn: Event filters.
        include_budget, include_user_input, include_model_output: Content filters.
        custom_backend: Optional AuditBackendProtocol (overrides backend/path).
    """

    backend: AuditBackend | str = AuditBackend.JSONL
    path: str | None = Field(default=None, description="File path for JSONL backend")
    include_llm_calls: bool = True
    include_tool_calls: bool = True
    include_handoff_spawn: bool = True
    include_budget: bool = False
    include_user_input: bool = False
    include_model_output: bool = True
    custom_backend: object = Field(
        default=None,
        description="Optional AuditBackendProtocol instance (overrides backend/path)",
    )

    def get_backend(self, default_path: str = "./audit.jsonl") -> object:
        """Resolve backend instance. Returns custom_backend or creates JsonlAuditBackend."""
        if self.custom_backend is not None:
            return self.custom_backend
        from syrin.audit.backend import JsonlAuditBackend

        p = self.path or default_path
        return JsonlAuditBackend(path=p)
