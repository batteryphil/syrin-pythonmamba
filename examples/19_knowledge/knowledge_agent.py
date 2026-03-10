"""Knowledge + Agent: RAG with search_knowledge tool.

Shows the full pipeline: load → chunk → embed → store, plus Agent
integration (auto search_knowledge tool).
For agentic retrieval (search_knowledge_deep, verify_knowledge), see agentic_rag.py.

Run:
    uv run python examples/19_knowledge/knowledge_agent.py

Requires: syrin[knowledge] (chonkie). Set OPENAI_API_KEY in examples/.env.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from syrin import Agent
from syrin.embedding import Embedding
from syrin.enums import KnowledgeBackend
from syrin.knowledge import Knowledge
from syrin.model import Model


async def main() -> None:
    # 1. Create Knowledge with sources and embedding
    knowledge = Knowledge(
        sources=[
            Knowledge.Text("Python is a high-level programming language."),
            Knowledge.Text("Syrin is a library for building AI agents with budget management."),
            Knowledge.Text("RAG means Retrieval-Augmented Generation."),
        ],
        embedding=Embedding.OpenAI("text-embedding-3-small"),
        backend=KnowledgeBackend.MEMORY,
    )

    # 2. Attach to Agent — search_knowledge tool is auto-added
    _ = Agent(
        model=Model.OpenAI("gpt-4o-mini"),
        system_prompt="You are a helpful assistant. Use search_knowledge when asked about Python, Syrin, or RAG.",
        knowledge=knowledge,
    )

    # 3. Search directly (triggers lazy ingest)
    results = await knowledge.search("What is Syrin?")
    print("Search results:", [r.chunk.content[:50] for r in results[:2]])

    # 4. Agent can call search_knowledge via tools
    # r = agent.response("What is RAG?")
    # print(r.content)


if __name__ == "__main__":
    asyncio.run(main())
