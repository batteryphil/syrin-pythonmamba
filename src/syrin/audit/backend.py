"""Built-in audit backends."""

from __future__ import annotations

import json
from pathlib import Path

from syrin.audit.models import AuditEntry, AuditFilters
from syrin.audit.protocol import AuditBackendProtocol


class JsonlAuditBackend(AuditBackendProtocol):
    """Write audit entries to a JSONL file. Supports optional query by reading file."""

    def __init__(self, path: str = "./audit.jsonl") -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, entry: AuditEntry) -> None:
        """Append entry as JSON line."""
        line = entry.model_dump_json() + "\n"
        with open(self._path, "a", encoding="utf-8") as f:
            f.write(line)

    def query(self, filters: AuditFilters) -> list[AuditEntry]:
        """Read file and filter. Simple implementation; not for large files."""
        if not self._path.exists():
            return []
        entries: list[AuditEntry] = []
        with open(self._path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    e = AuditEntry.model_validate(json.loads(line))
                except Exception:
                    continue
                if filters.agent is not None and e.source != filters.agent:
                    continue
                if filters.event is not None and e.event != filters.event:
                    continue
                if filters.since is not None and e.timestamp < filters.since:
                    continue
                if filters.until is not None and e.timestamp > filters.until:
                    continue
                entries.append(e)
                if len(entries) >= filters.limit:
                    break
        return entries[-filters.limit :] if len(entries) > filters.limit else entries
