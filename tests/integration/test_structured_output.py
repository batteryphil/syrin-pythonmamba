"""Structured output: when output_type/Output(type=...) is set, validation runs and retries applied."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from pydantic import BaseModel

from syrin import Agent
from syrin.exceptions import OutputValidationError
from syrin.model import Model
from syrin.output import Output
from syrin.types import ProviderResponse, TokenUsage


class SimpleOut(BaseModel):
    """Minimal Pydantic model for structured output tests."""

    name: str
    value: int


def _mock_provider_response(content: str) -> ProviderResponse:
    return ProviderResponse(
        content=content,
        tool_calls=[],
        token_usage=TokenUsage(input_tokens=1, output_tokens=1, total_tokens=2),
    )


class TestStructuredOutputValidation:
    """When Output(type=...) is set, validation runs and response.structured is populated."""

    def test_valid_json_populates_structured_parsed(self) -> None:
        """Valid JSON matching schema produces response.structured.parsed."""
        model = Model("anthropic/claude-3-5-sonnet")
        agent = Agent(model=model, system_prompt="Return JSON.", output=Output(SimpleOut))
        mock_resp = _mock_provider_response(content='{"name": "Alice", "value": 42}')
        with patch.object(
            agent._provider,
            "complete",
            new_callable=AsyncMock,
            return_value=mock_resp,
        ):
            r = agent.run("Return name and value.")
        assert r.structured is not None
        assert r.structured.is_valid
        assert r.structured.parsed is not None
        assert r.structured.parsed.name == "Alice"
        assert r.structured.parsed.value == 42

    def test_response_data_property_when_structured_set(self) -> None:
        """Response.data returns dict when structured output is valid."""
        model = Model("anthropic/claude-3-5-sonnet")
        agent = Agent(model=model, system_prompt="Return JSON.", output=Output(SimpleOut))
        mock_resp = _mock_provider_response(content='{"name": "Bob", "value": 10}')
        with patch.object(
            agent._provider,
            "complete",
            new_callable=AsyncMock,
            return_value=mock_resp,
        ):
            r = agent.run("Return JSON.")
        assert r.output is not None
        assert r.output.name == "Bob"  # type: ignore[union-attr]
        assert r.output.value == 10  # type: ignore[union-attr]

    def test_invalid_json_raises_output_validation_error(self) -> None:
        """When LLM returns invalid JSON/schema and retries exhausted, OutputValidationError is raised."""
        model = Model("anthropic/claude-3-5-sonnet")
        agent = Agent(
            model=model,
            system_prompt="Return JSON.",
            output=Output(SimpleOut, validation_retries=0),
        )
        mock_resp = _mock_provider_response(content="Not JSON at all")
        with (
            patch.object(
                agent._provider,
                "complete",
                new_callable=AsyncMock,
                return_value=mock_resp,
            ),
            pytest.raises(OutputValidationError),
        ):
            agent.run("Return JSON.")
