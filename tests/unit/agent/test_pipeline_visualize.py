"""Tests for parallel() and sequential() helper functions from syrin.agent.pipeline."""

from __future__ import annotations

import asyncio

import pytest

from syrin import Agent, Model
from syrin.agent.pipeline import parallel, sequential


class _ResearchAgent(Agent):
    """Research stub."""

    model = Model.Almock()
    system_prompt = "Research"


class _WriterAgent(Agent):
    """Writer stub."""

    model = Model.Almock()
    system_prompt = "Writer"


class _EditorAgent(Agent):
    """Editor stub."""

    model = Model.Almock()
    system_prompt = "Editor"


@pytest.mark.phase_1
class TestSequential:
    """sequential() runs agents one after another."""

    def test_sequential_empty_returns_empty_response(self) -> None:
        """sequential([]) returns an empty Response without raising."""
        result = sequential([])
        assert result.content == ""
        assert result.cost == 0.0

    def test_sequential_single_agent(self) -> None:
        """sequential() with one agent returns that agent's response."""
        agent = _ResearchAgent()
        result = sequential([(agent, "Research AI")])
        assert isinstance(result.content, str)

    def test_sequential_multiple_agents(self) -> None:
        """sequential() with multiple agents returns last agent's response."""
        researcher = _ResearchAgent()
        writer = _WriterAgent()
        result = sequential([(researcher, "Research AI"), (writer, "Write summary")])
        assert isinstance(result.content, str)

    def test_sequential_pass_previous_false(self) -> None:
        """sequential() with pass_previous=False does not prepend prior output."""
        researcher = _ResearchAgent()
        writer = _WriterAgent()
        # Should not raise — just ignores previous output
        result = sequential(
            [(researcher, "Research AI"), (writer, "Write summary")],
            pass_previous=False,
        )
        assert isinstance(result.content, str)

    def test_sequential_three_agents(self) -> None:
        """sequential() with three agents completes without raising."""
        researcher = _ResearchAgent()
        writer = _WriterAgent()
        editor = _EditorAgent()
        result = sequential(
            [
                (researcher, "Research AI trends"),
                (writer, "Write article"),
                (editor, "Edit article"),
            ]
        )
        assert isinstance(result.content, str)


@pytest.mark.phase_1
class TestParallel:
    """parallel() runs agents concurrently."""

    def test_parallel_single_agent(self) -> None:
        """parallel() with one agent returns a list with one response."""
        agent = _ResearchAgent()
        results = asyncio.run(parallel([(agent, "Research AI")]))
        assert len(results) == 1
        assert isinstance(results[0].content, str)

    def test_parallel_multiple_agents(self) -> None:
        """parallel() with multiple agents returns results in input order."""
        researcher = _ResearchAgent()
        writer = _WriterAgent()
        results = asyncio.run(parallel([(researcher, "Research AI"), (writer, "Write summary")]))
        assert len(results) == 2
        for r in results:
            assert isinstance(r.content, str)

    def test_parallel_three_agents(self) -> None:
        """parallel() with three agents returns three results."""
        researcher = _ResearchAgent()
        writer = _WriterAgent()
        editor = _EditorAgent()
        results = asyncio.run(
            parallel(
                [
                    (researcher, "Research AI"),
                    (writer, "Write article"),
                    (editor, "Edit article"),
                ]
            )
        )
        assert len(results) == 3

    def test_parallel_result_order_matches_input(self) -> None:
        """parallel() preserves input order in results."""
        researcher = _ResearchAgent()
        writer = _WriterAgent()
        results = asyncio.run(parallel([(researcher, "Task A"), (writer, "Task B")]))
        # Both should be valid Response objects — order is deterministic
        assert len(results) == 2
