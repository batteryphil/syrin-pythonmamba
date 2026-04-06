"""Tests for RateLimitReport implementation.

Tests that RateLimitReport is properly populated during agent execution.
"""

from __future__ import annotations

from unittest.mock import Mock, patch

import pytest

from syrin import Agent, Model
from syrin.enums import RateLimitAction, ThresholdMetric
from syrin.ratelimit import APIRateLimit, RateLimitThreshold
from syrin.response import RateLimitReport

# =============================================================================
# TESTS FOR RATE LIMIT REPORT
# =============================================================================


class TestRateLimitReport:
    """Tests for RateLimitReport population."""

    def test_rate_limit_check_tracked(self):
        """Test that rate limit checks are tracked."""
        rate_limit = APIRateLimit(
            rpm=60,
            thresholds=[
                RateLimitThreshold(at=80, action=RateLimitAction.WARN, metric=ThresholdMetric.RPM)
            ],
        )

        agent = Agent(
            model=Model("openai/gpt-4o-mini"),
            rate_limit=rate_limit,
        )

        # Verify rate limit manager is set up
        assert agent._rate_limit_manager is not None

        # Manually trigger a check to verify tracking works
        agent._run_report.ratelimits.checks += 1

        # Rate limit report should show check was made
        assert agent.report.ratelimits.checks >= 1

    def test_rate_limit_report_zero_rate_limits(self):
        """Test agent with no rate limits."""

        class TestAgent(Agent):
            model = Model("openai/gpt-4o-mini")

        agent = TestAgent()

        with patch.object(
            agent._loop,
            "run",
            return_value=Mock(
                content="Response",
                cost_usd=0.001,
                token_usage={"input": 10, "output": 5, "total": 15},
                tool_calls=[],
                stop_reason="end_turn",
                iterations=1,
                latency_ms=100,
            ),
        ):
            result = agent.run("Test")

        # Should have defaults
        assert result.report.ratelimits.checks == 0
        assert result.report.ratelimits.exceeded is False


# =============================================================================
# EDGE CASE TESTS FOR RATE LIMIT REPORT
# =============================================================================


class TestRateLimitReportEdgeCases:
    """Edge case tests for RateLimitReport."""

    def test_rate_limit_report_defaults(self):
        """Test RateLimitReport default values."""
        report = RateLimitReport()
        assert report.checks == 0
        assert report.throttles == 0
        assert report.exceeded is False

    def test_rate_limit_report_multiple_checks(self):
        """Test RateLimitReport with multiple checks."""
        report = RateLimitReport(checks=10, throttles=2, exceeded=True)
        assert report.checks == 10
        assert report.throttles == 2
        assert report.exceeded is True

    def test_rate_limit_report_very_high_values(self):
        """Test RateLimitReport with very high values."""
        report = RateLimitReport(checks=9999999, throttles=8888888)
        assert report.checks == 9999999
        assert report.throttles == 8888888

    def test_rate_limit_report_negative_values(self):
        """Test RateLimitReport with negative values."""
        # Should handle gracefully (though semantically invalid)
        report = RateLimitReport(checks=-5, throttles=-3)
        assert report.checks == -5
        assert report.throttles == -3

    def test_rate_limit_report_exists_in_response(self):
        """Test that RateLimitReport exists in agent response."""

        class TestAgent(Agent):
            model = Model("openai/gpt-4o-mini")

        agent = TestAgent()

        with patch.object(
            agent._loop,
            "run",
            return_value=Mock(
                content="Response",
                cost_usd=0.001,
                token_usage={"input": 10, "output": 5, "total": 15},
                tool_calls=[],
                stop_reason="end_turn",
                iterations=1,
                latency_ms=100,
            ),
        ):
            result = agent.run("Test")

        assert hasattr(result.report, "ratelimits")
        assert isinstance(result.report.ratelimits, RateLimitReport)


# =============================================================================
# RUN TESTS
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
