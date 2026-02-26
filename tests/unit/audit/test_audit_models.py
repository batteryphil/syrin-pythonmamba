"""Tests for audit models: AuditEntry, AuditLog, AuditFilters."""

from datetime import datetime, timezone

from syrin.audit import AuditEntry, AuditFilters, AuditLog
from syrin.enums import AuditEventType


class TestAuditEntry:
    """Valid and invalid AuditEntry creation."""

    def test_minimal_valid_entry(self) -> None:
        """Minimal valid entry has required fields."""
        e = AuditEntry(source="TestAgent", event=AuditEventType.LLM_CALL)
        assert e.source == "TestAgent"
        assert e.event == AuditEventType.LLM_CALL
        assert e.timestamp is not None
        assert e.model is None
        assert e.cost_usd is None

    def test_full_entry(self) -> None:
        """Full entry with all optional fields."""
        e = AuditEntry(
            source="ComplianceAgent",
            event=AuditEventType.TOOL_CALL,
            model="gpt-4o",
            tokens={"input": 100, "output": 50, "total": 150},
            cost_usd=0.002,
            duration_ms=500.0,
            iteration=2,
            tool_name="search",
        )
        assert e.tokens["input"] == 100
        assert e.cost_usd == 0.002
        assert e.tool_name == "search"

    def test_model_dump_json_line(self) -> None:
        """JSON line serialization is valid JSON."""
        e = AuditEntry(source="A", event=AuditEventType.AGENT_RUN_END)
        line = e.model_dump_json_line()
        import json

        parsed = json.loads(line)
        assert parsed["source"] == "A"
        assert parsed["event"] == AuditEventType.AGENT_RUN_END


class TestAuditLog:
    """Valid and invalid AuditLog config."""

    def test_default_config(self) -> None:
        """Default config has sensible includes."""
        config = AuditLog()
        assert config.include_llm_calls is True
        assert config.include_tool_calls is True
        assert config.include_handoff_spawn is True
        assert config.include_budget is False
        assert config.path is None

    def test_custom_path(self) -> None:
        """Custom path is preserved."""
        config = AuditLog(path="/tmp/audit.jsonl")
        backend = config.get_backend()
        assert backend._path.name == "audit.jsonl"
        assert str(backend._path).endswith("audit.jsonl")

    def test_get_backend_uses_default_path(self) -> None:
        """Without path, default path is used."""
        config = AuditLog()
        backend = config.get_backend()
        assert "audit" in str(backend._path).lower()

    def test_custom_backend_overrides_path(self) -> None:
        """custom_backend takes precedence."""

        class FakeBackend:
            def write(self, entry):  # noqa: ARG002
                pass

        config = AuditLog(custom_backend=FakeBackend())
        backend = config.get_backend()
        assert isinstance(backend, FakeBackend)


class TestAuditFilters:
    """AuditFilters validation."""

    def test_default_filters(self) -> None:
        """Default limit is 100."""
        f = AuditFilters()
        assert f.limit == 100

    def test_filters_with_all_fields(self) -> None:
        """All filter fields accepted."""
        since = datetime.now(timezone.utc)
        f = AuditFilters(
            agent="TestAgent",
            event=AuditEventType.LLM_CALL,
            since=since,
            limit=50,
        )
        assert f.agent == "TestAgent"
        assert f.event == AuditEventType.LLM_CALL
        assert f.since == since
        assert f.limit == 50
