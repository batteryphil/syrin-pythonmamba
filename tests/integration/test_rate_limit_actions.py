"""Rate limit: exceeded raises; threshold action callback runs (user implements STOP/WAIT/etc.)."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from syrin import Agent, Model
from syrin.enums import ThresholdMetric
from syrin.ratelimit import APIRateLimit, create_rate_limit_manager
from syrin.threshold import RateLimitThreshold
from syrin.types import ProviderResponse, TokenUsage


def _mock_provider_response(content: str = "Ok") -> ProviderResponse:
    return ProviderResponse(
        content=content,
        tool_calls=[],
        token_usage=TokenUsage(input_tokens=5, output_tokens=10, total_tokens=15),
    )


class TestRateLimitExceededRaises:
    """When rate limit is exceeded (check returns False), agent raises RuntimeError."""

    def test_rate_limit_exceeded_raises_runtime_error(self) -> None:
        """Agent with rpm=1: first call ok, second call in same minute raises."""
        model = Model("anthropic/claude-3-5-sonnet")
        config = APIRateLimit(rpm=1)

        agent = Agent(
            model=model,
            system_prompt="Test.",
            rate_limit=config,
        )
        with patch.object(
            agent._provider,
            "complete",
            new_callable=AsyncMock,
            return_value=_mock_provider_response(content="First"),
        ):
            r1 = agent.run("Hi")
        assert r1.content == "First"

        with (
            patch.object(
                agent._provider,
                "complete",
                new_callable=AsyncMock,
                return_value=_mock_provider_response(content="Second"),
            ),
            pytest.raises(RuntimeError, match="Rate limit exceeded"),
        ):
            agent.run("Hi again")


class TestRateLimitThresholdActionCallback:
    """Threshold action callback is executed; user can raise, wait, switch model, etc."""

    def test_threshold_action_callback_raises_when_triggered(self) -> None:
        """User action callback that raises: when threshold triggers, RuntimeError is raised."""

        def stop_at_50(ctx: object) -> None:
            pct = getattr(ctx, "percentage", 0)
            if pct >= 50:
                raise RuntimeError("Rate limit threshold reached")

        model = Model("anthropic/claude-3-5-sonnet")
        config = APIRateLimit(
            rpm=2,
            thresholds=[
                RateLimitThreshold(at=50, action=stop_at_50, metric=ThresholdMetric.RPM),
            ],
        )

        agent = Agent(
            model=model,
            system_prompt="Test.",
            rate_limit=config,
        )
        with patch.object(
            agent._provider,
            "complete",
            new_callable=AsyncMock,
            return_value=_mock_provider_response(),
        ):
            agent.run("First")
        with (
            patch.object(
                agent._provider,
                "complete",
                new_callable=AsyncMock,
                return_value=_mock_provider_response(),
            ),
            pytest.raises(RuntimeError, match="Rate limit threshold reached"),
        ):
            agent.run("Second")

    def test_get_triggered_action_returns_none(self) -> None:
        """get_triggered_action() returns None; threshold behavior is via action callback only."""
        config = APIRateLimit(rpm=100)
        manager = create_rate_limit_manager(config)
        assert manager.get_triggered_action() is None

        config_with_threshold = APIRateLimit(
            rpm=10,
            thresholds=[
                RateLimitThreshold(at=50, action=lambda _: None, metric=ThresholdMetric.RPM),
            ],
        )
        manager2 = create_rate_limit_manager(config_with_threshold)
        for _ in range(6):
            manager2.record()
        assert manager2.get_triggered_action() is None
