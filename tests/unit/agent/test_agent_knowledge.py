"""Tests for Agent + Knowledge integration (Step 5)."""

from __future__ import annotations

from syrin import Agent
from syrin.enums import KnowledgeBackend
from syrin.knowledge import Knowledge
from syrin.model import Model


def _make_fake_embedding(dim: int = 4):
    """Minimal EmbeddingProvider for tests."""

    class FakeEmbedding:
        @property
        def dimensions(self) -> int:
            return dim

        @property
        def model_id(self) -> str:
            return "fake"

        async def embed(
            self,
            texts: list[str],
            budget_tracker: object | None = None,
        ) -> list[list[float]]:
            return [[0.1] * dim for _ in texts]

    return FakeEmbedding()


class TestAgentKnowledgeIntegration:
    """Agent with knowledge adds search_knowledge tool."""

    def test_agent_with_knowledge_has_search_tool(self) -> None:
        """Agent with knowledge= gets search_knowledge tool."""
        embedding = _make_fake_embedding()
        knowledge = Knowledge(
            sources=[Knowledge.Text("Fact about Python.")],
            embedding=embedding,
            backend=KnowledgeBackend.MEMORY,
        )
        agent = Agent(
            model=Model.Almock(latency_min=0, latency_max=0),
            system_prompt="You are helpful.",
            knowledge=knowledge,
        )
        tool_names = [t.name for t in agent.tools]
        assert "search_knowledge" in tool_names

    def test_agent_without_knowledge_no_search_tool(self) -> None:
        """Agent without knowledge has no search_knowledge tool."""
        agent = Agent(
            model=Model.Almock(latency_min=0, latency_max=0),
            system_prompt="You are helpful.",
        )
        tool_names = [t.name for t in agent.tools]
        assert "search_knowledge" not in tool_names

    def test_agent_with_agentic_knowledge_has_deep_and_verify_tools(self) -> None:
        """Agent with knowledge= and agentic=True gets search_knowledge_deep and verify_knowledge."""
        embedding = _make_fake_embedding()
        knowledge = Knowledge(
            sources=[Knowledge.Text("Fact about Python.")],
            embedding=embedding,
            backend=KnowledgeBackend.MEMORY,
            agentic=True,
        )
        agent = Agent(
            model=Model.Almock(latency_min=0, latency_max=0),
            system_prompt="You are helpful.",
            knowledge=knowledge,
        )
        tool_names = [t.name for t in agent.tools]
        assert "search_knowledge" in tool_names
        assert "search_knowledge_deep" in tool_names
        assert "verify_knowledge" in tool_names

    def test_agent_without_agentic_no_deep_or_verify_tools(self) -> None:
        """Agent with knowledge= and agentic=False does not get agentic tools."""
        embedding = _make_fake_embedding()
        knowledge = Knowledge(
            sources=[Knowledge.Text("Fact.")],
            embedding=embedding,
            backend=KnowledgeBackend.MEMORY,
            agentic=False,
        )
        agent = Agent(
            model=Model.Almock(latency_min=0, latency_max=0),
            system_prompt="You are helpful.",
            knowledge=knowledge,
        )
        tool_names = [t.name for t in agent.tools]
        assert "search_knowledge" in tool_names
        assert "search_knowledge_deep" not in tool_names
        assert "verify_knowledge" not in tool_names
