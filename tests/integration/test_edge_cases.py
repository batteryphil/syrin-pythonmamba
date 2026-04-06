"""Edge cases: empty tools, no budget, no memory, missing provider — no crashes; errors are typed."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from syrin import Agent, Model
from syrin.exceptions import SyrinError
from syrin.types import ProviderResponse, TokenUsage


def _mock_provider_response(content: str = "Ok") -> ProviderResponse:
    return ProviderResponse(
        content=content,
        tool_calls=[],
        token_usage=TokenUsage(input_tokens=5, output_tokens=10, total_tokens=15),
    )


class TestEmptyTools:
    """Agent with empty tools list runs without crashing."""

    def test_agent_with_empty_tools_list_responds(self) -> None:
        """tools=[] is valid; response() completes."""
        model = Model("anthropic/claude-3-5-sonnet")
        agent = Agent(model=model, system_prompt="Test.", tools=[])
        with patch.object(
            agent._provider,
            "complete",
            new_callable=AsyncMock,
            return_value=_mock_provider_response(content="Hi"),
        ):
            r = agent.run("Hello")
        assert r.content == "Hi"
        assert r.tool_calls == []


class TestNoBudget:
    """Agent without budget runs without crashing."""

    def test_agent_without_budget_responds(self) -> None:
        """No budget set: response() completes and cost is tracked in response."""
        model = Model("anthropic/claude-3-5-sonnet")
        agent = Agent(model=model, system_prompt="Test.")
        with patch.object(
            agent._provider,
            "complete",
            new_callable=AsyncMock,
            return_value=_mock_provider_response(),
        ):
            r = agent.run("Hi")
        assert r.cost >= 0
        assert agent._budget is None or r.budget_used is not None


class TestNoPersistentMemory:
    """Agent with memory=None or only conversation memory: recall/remember handled."""

    def test_agent_memory_none_recall_raises_runtime_error(self) -> None:
        """memory=None: recall() raises RuntimeError (not generic Exception)."""
        model = Model("anthropic/claude-3-5-sonnet")
        agent = Agent(model=model, system_prompt="Test.", memory=None)
        with pytest.raises(RuntimeError, match="persistent memory"):
            agent.recall("x")

    def test_agent_memory_none_remember_raises_runtime_error(self) -> None:
        """memory=None: remember() raises RuntimeError."""
        model = Model("anthropic/claude-3-5-sonnet")
        agent = Agent(model=model, system_prompt="Test.", memory=None)
        with pytest.raises(RuntimeError, match="persistent memory"):
            agent.remember("x")


class TestMissingProvider:
    """Missing or invalid provider raises typed error."""

    def test_registry_unknown_provider_strict_raises_provider_not_found(self) -> None:
        """get_provider(unknown, strict=True) raises ProviderNotFoundError (SyrinError)."""
        from syrin.exceptions import ProviderNotFoundError
        from syrin.providers.registry import get_provider

        with pytest.raises(ProviderNotFoundError) as exc_info:
            get_provider("nonexistent-provider-xyz", strict=True)
        assert issubclass(type(exc_info.value), SyrinError)


class TestNoMemoryBackendConfigured:
    """Agent with default memory (no explicit backend) works."""

    def test_agent_default_memory_remember_recall_work(self) -> None:
        """Default persistent memory: remember and recall work."""
        from syrin.enums import MemoryType
        from syrin.memory import Memory

        model = Model("anthropic/claude-3-5-sonnet")
        agent = Agent(model=model, system_prompt="Test.", memory=Memory())
        agent.remember("Fact", memory_type=MemoryType.HISTORY)
        entries = agent.recall(limit=5)
        assert isinstance(entries, list)
