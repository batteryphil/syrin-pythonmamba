"""Tests for JsonlAuditBackend."""

import json
import tempfile
from pathlib import Path

from syrin.audit import AuditEntry, AuditFilters, JsonlAuditBackend
from syrin.enums import AuditEventType


class TestJsonlAuditBackend:
    """Valid and invalid JsonlAuditBackend behavior."""

    def test_write_creates_file(self) -> None:
        """Write creates file and appends JSON lines."""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "audit.jsonl"
            backend = JsonlAuditBackend(path=str(path))

            e = AuditEntry(source="TestAgent", event=AuditEventType.LLM_CALL)
            backend.write(e)
            backend.write(e)

            lines = path.read_text().strip().split("\n")
            assert len(lines) == 2
            for line in lines:
                parsed = json.loads(line)
                assert parsed["source"] == "TestAgent"
                assert parsed["event"] == AuditEventType.LLM_CALL

    def test_query_empty_file_returns_empty_list(self) -> None:
        """Query on non-existent or empty file returns []."""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "nonexistent.jsonl"
            backend = JsonlAuditBackend(path=str(path))
            result = backend.query(AuditFilters())
            assert result == []

    def test_query_filters_by_agent(self) -> None:
        """Query filters by agent."""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "audit.jsonl"
            backend = JsonlAuditBackend(path=str(path))

            backend.write(AuditEntry(source="AgentA", event=AuditEventType.LLM_CALL))
            backend.write(AuditEntry(source="AgentB", event=AuditEventType.LLM_CALL))
            backend.write(AuditEntry(source="AgentA", event=AuditEventType.TOOL_CALL))

            result = backend.query(AuditFilters(agent="AgentA"))
            assert len(result) == 2
            assert all(e.source == "AgentA" for e in result)

    def test_query_respects_limit(self) -> None:
        """Query respects limit."""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "audit.jsonl"
            backend = JsonlAuditBackend(path=str(path))
            for _ in range(10):
                backend.write(AuditEntry(source="A", event=AuditEventType.LLM_CALL))
            result = backend.query(AuditFilters(limit=3))
            assert len(result) <= 3
