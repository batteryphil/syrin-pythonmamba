"""
Integration tests for Agent variants: budget, memory, tools.

Uses Model.Almock(latency_seconds=0.01) for fast tests.
"""

from __future__ import annotations

from syrin import Agent, Budget, Memory, MemoryType, Model
from syrin.memory import Memory
from syrin.tool import tool


def _almock() -> Model:
    return Model.Almock(latency_seconds=0.01, lorem_length=50)


# -----------------------------------------------------------------------------
# Agent with budget vs without budget
# -----------------------------------------------------------------------------


class TestAgentWithWithoutBudget:
    def test_agent_with_budget_almock_real_flow(self) -> None:
        """Agent with budget (Almock) — real flow, cost tracked."""
        model = _almock()
        budget = Budget(run=10.0)
        agent = Agent(model=model, system_prompt="You are helpful.", budget=budget)

        response = agent.response("Hello")

        assert response.content is not None
        assert response.cost >= 0
        if agent.budget_state is not None:
            assert agent.budget_state.spent >= 0

    def test_agent_without_budget_almock(self) -> None:
        """Agent without budget — runs successfully, no budget tracking."""
        model = _almock()
        agent = Agent(model=model, system_prompt="You are helpful.")

        response = agent.response("Hello")

        assert response.content is not None
        assert response.cost >= 0


# -----------------------------------------------------------------------------
# Agent with memory vs without memory
# -----------------------------------------------------------------------------


class TestAgentWithWithoutMemory:
    def test_agent_with_buffer_memory(self) -> None:
        """Agent with Memory — conversation history included in context."""

        model = _almock()
        mem = Memory()
        mem.add_conversation_segment("My name is Alice.", role="user")
        mem.add_conversation_segment("Hello Alice!", role="assistant")
        agent = Agent(model=model, system_prompt="You are helpful.", memory=mem)

        response = agent.response("What is my name?")

        assert response.content is not None
        assert len(mem.get_conversation_messages()) >= 2

    def test_agent_with_memory_4type(self) -> None:
        """Agent with Memory (4-type) — remember/recall works."""
        model = _almock()
        mem = Memory()
        agent = Agent(model=model, system_prompt="You are helpful.", memory=mem)

        agent.remember("User prefers Python over Java", memory_type=MemoryType.CORE)
        response = agent.response("What do I prefer?")

        assert response.content is not None

    def test_agent_without_memory_memory_false(self) -> None:
        """Agent with memory=False — no conversation persistence."""
        model = _almock()
        agent = Agent(model=model, system_prompt="You are helpful.", memory=False)

        response = agent.response("Hello")

        assert response.content is not None


# -----------------------------------------------------------------------------
# Agent with tools vs without tools
# -----------------------------------------------------------------------------


class TestAgentWithWithoutTools:
    @staticmethod
    def _search_tool():
        @tool
        def search(query: str) -> str:
            return f"Results for: {query}"

        return search

    def test_agent_with_tools(self) -> None:
        """Agent with tools — tool execution flow."""
        model = _almock()
        search = self._search_tool()
        agent = Agent(
            model=model,
            system_prompt="You are helpful. Use search when needed.",
            tools=[search],
        )

        response = agent.response("Search for AI trends")

        assert response.content is not None

    def test_agent_without_tools(self) -> None:
        """Agent without tools — single LLM call."""
        model = _almock()
        agent = Agent(model=model, system_prompt="You are helpful.")

        response = agent.response("What is 2+2?")

        assert response.content is not None
