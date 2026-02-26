"""Audit data models: AuditEntry, AuditEvent, AuditFilters, AuditLog."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field

from syrin.enums import AuditBackend


class AuditEntry(BaseModel):
    """Single audit log entry. Written to backend as JSONL line."""

    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    source: str = Field(description="Agent class name, Pipeline, or DynamicPipeline")
    event: str = Field(description="Audit event type (e.g. llm_call, tool_call)")
    model: str | None = None
    tokens: dict[str, int] | None = Field(default=None, description="input, output, total")
    cost_usd: float | None = None
    budget_percent: float | None = None
    duration_ms: float | None = None
    trace_id: str | None = None
    run_id: str | None = None
    iteration: int | None = None
    tool_calls: list[dict[str, Any]] | None = None
    tool_name: str | None = None
    tool_error: str | None = None
    stop_reason: str | None = None
    extra: dict[str, Any] | None = Field(default=None, description="Hook-specific fields")

    def model_dump_json_line(self) -> str:
        """Serialize as JSON line for JSONL output."""
        return self.model_dump_json()


class AuditFilters(BaseModel):
    """Filters for querying audit entries (optional backend support)."""

    agent: str | None = None
    event: str | None = None
    since: datetime | None = None
    until: datetime | None = None
    limit: int = 100


class AuditLog(BaseModel):
    """Audit configuration for Agent, Pipeline, or DynamicPipeline."""

    backend: AuditBackend | str = AuditBackend.JSONL
    path: str | None = Field(default=None, description="File path for JSONL backend")
    include_llm_calls: bool = True
    include_tool_calls: bool = True
    include_handoff_spawn: bool = True
    include_budget: bool = False
    include_user_input: bool = False
    include_model_output: bool = True
    custom_backend: Any = Field(
        default=None,
        description="Optional AuditBackendProtocol instance (overrides backend/path)",
    )

    def get_backend(self, default_path: str = "./audit.jsonl") -> Any:
        """Resolve backend instance. Returns custom_backend or creates JsonlAuditBackend."""
        if self.custom_backend is not None:
            return self.custom_backend
        from syrin.audit.backend import JsonlAuditBackend

        p = self.path or default_path
        return JsonlAuditBackend(path=p)
