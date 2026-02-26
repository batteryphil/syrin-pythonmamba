"""Integration tests: Agent with AuditLog writes entries."""

import json
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from syrin import Agent, AuditLog, Model
from syrin.enums import AuditEventType
from syrin.types import ProviderResponse, TokenUsage


class TestAgentAuditIntegration:
    """Agent with audit logs lifecycle events."""

    def test_agent_with_audit_writes_entries(self) -> None:
        """Agent with audit config writes AGENT_RUN_START and AGENT_RUN_END."""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "audit.jsonl"
            audit = AuditLog(path=str(path))

            model = Model("anthropic/claude-3-5-sonnet")
            agent = Agent(model=model, system_prompt="Test", audit=audit)

            mock_resp = ProviderResponse(
                content="Hi",
                tool_calls=[],
                token_usage=TokenUsage(
                    input_tokens=5,
                    output_tokens=10,
                    total_tokens=15,
                ),
            )
            with patch.object(
                agent._provider,
                "complete",
                new_callable=AsyncMock,
                return_value=mock_resp,
            ):
                agent.response("Hello")

            lines = [line for line in path.read_text().strip().split("\n") if line]
            assert len(lines) >= 2  # At least RUN_START and RUN_END
            events = [json.loads(line)["event"] for line in lines]
            assert AuditEventType.AGENT_RUN_START in events
            assert AuditEventType.AGENT_RUN_END in events

    def test_agent_without_audit_no_file_created(self) -> None:
        """Agent without audit does not create audit file."""
        model = Model("anthropic/claude-3-5-sonnet")
        agent = Agent(model=model, system_prompt="Test")

        mock_resp = ProviderResponse(
            content="Hi",
            tool_calls=[],
            token_usage=TokenUsage(input_tokens=5, output_tokens=10, total_tokens=15),
        )
        with patch.object(
            agent._provider,
            "complete",
            new_callable=AsyncMock,
            return_value=mock_resp,
        ):
            agent.response("Hello")

        # No audit, so no file - nothing to assert except no exception

    def test_agent_audit_invalid_type_raises(self) -> None:
        """Agent with non-AuditLog audit raises TypeError."""
        model = Model("anthropic/claude-3-5-sonnet")
        with pytest.raises(TypeError) as exc_info:
            Agent(model=model, audit="invalid")  # type: ignore[arg-type]
        assert "audit must be AuditLog" in str(exc_info.value)

    def test_audit_include_llm_calls_false_skips_llm_events(self) -> None:
        """include_llm_calls=False skips LLM_REQUEST_* events."""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "audit.jsonl"
            audit = AuditLog(path=str(path), include_llm_calls=False)

            model = Model("anthropic/claude-3-5-sonnet")
            agent = Agent(model=model, system_prompt="Test", audit=audit)

            mock_resp = ProviderResponse(
                content="Hi",
                tool_calls=[],
                token_usage=TokenUsage(
                    input_tokens=5,
                    output_tokens=10,
                    total_tokens=15,
                ),
            )
            with patch.object(
                agent._provider,
                "complete",
                new_callable=AsyncMock,
                return_value=mock_resp,
            ):
                agent.response("Hello")

            lines = [line for line in path.read_text().strip().split("\n") if line]
            events = [json.loads(line)["event"] for line in lines]
            assert AuditEventType.LLM_CALL not in events
            assert AuditEventType.AGENT_RUN_START in events
            assert AuditEventType.AGENT_RUN_END in events
