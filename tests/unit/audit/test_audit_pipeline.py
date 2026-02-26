"""Pipeline and DynamicPipeline audit integration tests."""

import json
import tempfile
from pathlib import Path

import pytest

from syrin import Agent, AuditLog, Model, Pipeline
from syrin.agent.multi_agent import DynamicPipeline
from syrin.enums import AuditEventType


class QuickAgent(Agent):
    """Minimal agent for pipeline tests."""

    model = Model.Almock()
    system_prompt = "You are helpful."


class TestPipelineAudit:
    """Pipeline with audit logs pipeline-level events."""

    def test_pipeline_with_audit_writes_entries(self) -> None:
        """Pipeline with audit writes PIPELINE_START, AGENT_*, PIPELINE_END."""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "pipeline_audit.jsonl"
            audit = AuditLog(path=str(path))
            pipeline = Pipeline(audit=audit)

            result = pipeline.run([(QuickAgent, "Hello")]).content
            assert result

            lines = [line for line in path.read_text().strip().split("\n") if line]
            assert len(lines) >= 2
            events = [json.loads(line)["event"] for line in lines]
            assert AuditEventType.PIPELINE_START in events
            assert AuditEventType.PIPELINE_END in events

    def test_pipeline_audit_invalid_type_raises(self) -> None:
        """Pipeline with non-AuditLog audit raises TypeError."""
        with pytest.raises(TypeError) as exc_info:
            Pipeline(audit="invalid")  # type: ignore[arg-type]
        assert "audit must be AuditLog" in str(exc_info.value)


class TestDynamicPipelineAudit:
    """DynamicPipeline with audit."""

    def test_dynamic_pipeline_with_audit_writes_entries(self) -> None:
        """DynamicPipeline with audit writes DYNAMIC_PIPELINE_* events."""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "dynamic_audit.jsonl"
            audit = AuditLog(path=str(path))
            pipeline = DynamicPipeline(
                agents=[QuickAgent],
                model=Model.Almock(),
                audit=audit,
            )

            result = pipeline.run("Hello")
            assert result.content

            lines = [line for line in path.read_text().strip().split("\n") if line]
            assert len(lines) >= 2
            events = [json.loads(line)["event"] for line in lines]
            assert AuditEventType.DYNAMIC_PIPELINE_START in events
            assert AuditEventType.DYNAMIC_PIPELINE_END in events

    def test_dynamic_pipeline_audit_invalid_type_raises(self) -> None:
        """DynamicPipeline with non-AuditLog audit raises TypeError."""
        with pytest.raises(TypeError) as exc_info:
            DynamicPipeline(
                model=Model.Almock(),
                audit="invalid",  # type: ignore[arg-type]
            )
        assert "audit must be AuditLog" in str(exc_info.value)
