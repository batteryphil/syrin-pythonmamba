"""Tests for MemoryReport implementation.

Tests that MemoryReport is properly populated during agent execution.
"""

from __future__ import annotations

import contextlib
from unittest.mock import Mock, patch

import pytest

from syrin import Agent, Model
from syrin.enums import MemoryType
from syrin.response import MemoryReport

# =============================================================================
# TESTS FOR MEMORY REPORT POPULATION
# =============================================================================


class TestMemoryReport:
    """Tests for MemoryReport population."""

    def test_memory_store_tracked_in_report(self):
        """Test that remember() increments memory.stores."""

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
            agent.run("Test")

        # Initial store count should be 0
        initial_stores = agent.report.memory.stores

        # Mock memory backend
        mock_backend = Mock()
        mock_backend.add = Mock()
        agent._memory_backend = mock_backend

        # Store a memory
        agent.remember("Test memory", memory_type=MemoryType.HISTORY)

        # Report should be updated
        assert agent.report.memory.stores == initial_stores + 1

    def test_memory_recall_tracked_in_report(self):
        """Test that recall() increments memory.recalls."""

        class TestAgent(Agent):
            model = Model("openai/gpt-4o-mini")

        agent = TestAgent()

        # Mock memory backend
        mock_backend = Mock()
        mock_backend.search = Mock(return_value=[])
        agent._memory_backend = mock_backend

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
            agent.run("Test")

        # Initial recall count should be 0
        initial_recalls = agent.report.memory.recalls

        # Recall memories
        agent.recall("test query")

        # Report should be updated
        assert agent.report.memory.recalls == initial_recalls + 1

    def test_memory_forget_tracked_in_report(self):
        """Test that forget() increments memory.forgets."""

        class TestAgent(Agent):
            model = Model("openai/gpt-4o-mini")

        agent = TestAgent()

        # Mock memory backend
        mock_backend = Mock()
        mock_backend.delete = Mock()
        agent._memory_backend = mock_backend

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
            agent.run("Test")

        # Initial forget count should be 0
        initial_forgets = agent.report.memory.forgets

        # Forget a memory
        agent.forget(memory_id="test-id")

        # Report should be updated
        assert agent.report.memory.forgets == initial_forgets + 1

    def test_memory_operations_multiple(self):
        """Test multiple memory operations are all tracked."""

        class TestAgent(Agent):
            model = Model("openai/gpt-4o-mini")

        agent = TestAgent()

        # Mock memory backend
        mock_backend = Mock()
        mock_backend.add = Mock()
        mock_backend.search = Mock(return_value=[])
        mock_backend.delete = Mock()
        agent._memory_backend = mock_backend

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
            agent.run("Test")

        # Perform multiple operations
        agent.remember("Memory 1")
        agent.remember("Memory 2")
        agent.recall("query")
        agent.forget(memory_id="id1")

        # Check report
        assert agent.report.memory.stores == 2
        assert agent.report.memory.recalls == 1
        assert agent.report.memory.forgets == 1


# =============================================================================
# EDGE CASE TESTS FOR MEMORY REPORT
# =============================================================================


class TestMemoryReportEdgeCases:
    """Edge case tests for MemoryReport."""

    def test_memory_report_zero_operations(self):
        """Test MemoryReport with zero operations."""
        report = MemoryReport()
        assert report.recalls == 0
        assert report.stores == 0
        assert report.forgets == 0
        assert report.consolidated == 0

    def test_memory_report_negative_values_invalid(self):
        """Test that negative values are handled gracefully."""
        # The dataclass allows negative values, but they're semantically invalid
        report = MemoryReport(recalls=-5, stores=-3, forgets=-1)
        assert report.recalls == -5
        assert report.stores == -3
        assert report.forgets == -1

    def test_memory_report_large_values(self):
        """Test MemoryReport with very large values."""
        report = MemoryReport(recalls=999999, stores=888888, forgets=777777)
        assert report.recalls == 999999
        assert report.stores == 888888
        assert report.forgets == 777777

    def test_memory_report_operations_cumulative(self):
        """Test that memory operations accumulate correctly across multiple calls."""

        class TestAgent(Agent):
            model = Model("openai/gpt-4o-mini")

        agent = TestAgent()

        # Mock memory backend
        mock_backend = Mock()
        mock_backend.add = Mock()
        mock_backend.search = Mock(return_value=[])
        agent._memory_backend = mock_backend

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
            agent.run("Test")

        # Perform operations multiple times
        for i in range(5):
            agent.remember(f"Memory {i}")

        for i in range(3):
            agent.recall(f"query {i}")

        # Check cumulative counts
        assert agent.report.memory.stores == 5
        assert agent.report.memory.recalls == 3

    def test_memory_report_with_exception(self):
        """Test memory operations with exceptions don't corrupt report."""

        class TestAgent(Agent):
            model = Model("openai/gpt-4o-mini")

        agent = TestAgent()

        # Mock backend that raises exception
        mock_backend = Mock()
        mock_backend.add = Mock(side_effect=RuntimeError("DB Error"))
        agent._memory_backend = mock_backend

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
            agent.run("Test")

        initial_stores = agent.report.memory.stores

        # This should raise but report should still be valid
        with contextlib.suppress(RuntimeError):
            agent.remember("Test")

        # Report should not have been updated due to exception
        assert agent.report.memory.stores == initial_stores


# =============================================================================
# RUN TESTS
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
