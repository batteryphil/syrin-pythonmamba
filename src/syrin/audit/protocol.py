"""Audit backend protocol."""

from __future__ import annotations

from typing import Protocol

from syrin.audit.models import AuditEntry, AuditFilters


class AuditBackendProtocol(Protocol):
    """Protocol for audit log backends.

    write() is required. query() is optional for backends that support filtering.
    """

    def write(self, entry: AuditEntry) -> None:
        """Persist an audit entry. Must not raise for normal operation."""
        ...

    def query(self, filters: AuditFilters) -> list[AuditEntry]:
        """Query entries by filters. Optional; may raise NotImplementedError."""
        ...
