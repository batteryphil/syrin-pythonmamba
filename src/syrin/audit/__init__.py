"""Audit logging for compliance and observability.

Agent, Pipeline, and DynamicPipeline can be configured with AuditLog to
persist lifecycle events (LLM calls, tool calls, handoffs, spawns) to
JSONL files or custom backends.
"""

from syrin.audit.backend import JsonlAuditBackend
from syrin.audit.handler import AuditHookHandler
from syrin.audit.models import (
    AuditEntry,
    AuditFilters,
    AuditLog,
)
from syrin.audit.protocol import AuditBackendProtocol

__all__ = [
    "AuditBackendProtocol",
    "AuditEntry",
    "AuditEvent",
    "AuditFilters",
    "AuditHookHandler",
    "AuditLog",
    "JsonlAuditBackend",
]
