"""Agentic RAG: multi-step retrieval with search_knowledge_deep and verify_knowledge.

When agentic=True, the agent gets three tools:
- search_knowledge: Single search (same as non-agentic)
- search_knowledge_deep: Decompose complex queries, grade results, refine if needed
- verify_knowledge: Check if a claim is SUPPORTED, CONTRADICTED, or NOT_FOUND

Run:
    uv run python examples/19_knowledge/agentic_rag.py

Requires: syrin[knowledge]. Set OPENAI_API_KEY in examples/.env.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from syrin import Agent
from syrin.embedding import Embedding
from syrin.enums import KnowledgeBackend
from syrin.knowledge import AgenticRAGConfig, Knowledge
from syrin.model import Model


async def main() -> None:
    # 1. Create Knowledge with agentic=True
    knowledge = Knowledge(
        sources=[
            Knowledge.Text(
                "Python is a high-level programming language created by Guido van Rossum."
            ),
            Knowledge.Text("Syrin is a library for building AI agents with budget management."),
            Knowledge.Text("RAG means Retrieval-Augmented Generation."),
        ],
        embedding=Embedding.OpenAI("text-embedding-3-small", api_key=os.getenv("OPENAI_API_KEY")),
        backend=KnowledgeBackend.MEMORY,
        agentic=True,
        agentic_config=AgenticRAGConfig(
            max_search_iterations=3,
            decompose_complex=True,
            grade_results=True,
            relevance_threshold=0.5,
        ),
    )

    # 2. Attach to Agent — gets search_knowledge, search_knowledge_deep, verify_knowledge
    agent = Agent(
        model=Model.OpenAI("gpt-4o-mini"),
        system_prompt=(
            "You are a helpful assistant. "
            "Use search_knowledge for simple questions. "
            "Use search_knowledge_deep for complex or multi-part questions. "
            "Use verify_knowledge to fact-check before answering."
        ),
        knowledge=knowledge,
    )

    # 3. Verify agent has all three tools
    tool_names = [t.name for t in agent.tools]
    assert "search_knowledge" in tool_names
    assert "search_knowledge_deep" in tool_names
    assert "verify_knowledge" in tool_names
    print("Tools:", tool_names)

    # 4. Direct search (triggers lazy ingest)
    results = await knowledge.search("What is Syrin?")
    print("Search results:", [r.chunk.content[:50] for r in results[:2]])


if __name__ == "__main__":
    asyncio.run(main())
