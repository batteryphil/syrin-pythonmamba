"""Knowledge Agent -- Retrieval-Augmented Generation (RAG) with Syrin.

Demonstrates:
- Knowledge() with text sources for a simple knowledge base
- Embedding.OpenAI for vector embeddings
- In-memory backend for quick prototyping
- Attaching knowledge to an Agent (auto-adds search_knowledge tool)
- Direct knowledge.search() for manual retrieval

Run:
    python examples/19_knowledge/knowledge_agent.py

Requires:
    syrin[knowledge] (chonkie) and OPENAI_API_KEY env var for embeddings.
    For a fully offline example, see the Almock-based examples instead.
"""

from __future__ import annotations

import asyncio
import os
import sys

from syrin import Agent, Knowledge, KnowledgeBackend, Model
from syrin.embedding import Embedding

# ---------------------------------------------------------------------------
# Note: This example requires an OpenAI API key for embeddings.
# Set OPENAI_API_KEY in your environment before running.
# ---------------------------------------------------------------------------
if not os.getenv("OPENAI_API_KEY"):
    print("Skipped: set OPENAI_API_KEY to run this example.")
    sys.exit(0)


async def main() -> None:
    # ------------------------------------------------------------------
    # 1. Create a knowledge base from text sources
    # ------------------------------------------------------------------
    print("-- 1. Create knowledge base --")

    knowledge = Knowledge(
        sources=[
            Knowledge.Text("Python is a high-level programming language."),
            Knowledge.Text("Syrin is a library for building AI agents with budget management."),
            Knowledge.Text("RAG means Retrieval-Augmented Generation."),
        ],
        embedding=Embedding.OpenAI("text-embedding-3-small"),
        backend=KnowledgeBackend.MEMORY,
    )
    print("  3 text sources loaded (in-memory backend)")

    # ------------------------------------------------------------------
    # 2. Search the knowledge base directly
    # ------------------------------------------------------------------
    print("\n-- 2. Direct search (triggers lazy ingest) --")

    results = await knowledge.search("What is Syrin?")
    for r in results[:2]:
        print(f"  Found: {r.chunk.content[:60]}...")

    # ------------------------------------------------------------------
    # 3. Attach knowledge to an Agent
    # ------------------------------------------------------------------
    print("\n-- 3. Agent with knowledge --")

    agent = Agent(
        model=Model.Almock(),
        system_prompt=(
            "You are a helpful assistant. "
            "Use search_knowledge when asked about Python, Syrin, or RAG."
        ),
        knowledge=knowledge,
    )

    r = agent.response("Tell me about Syrin.")
    print(f"  Agent response: {r.content[:80]}...")


if __name__ == "__main__":
    asyncio.run(main())

# ---------------------------------------------------------------------------
# Optional: serve with playground UI (requires syrin[serve])
# ---------------------------------------------------------------------------
# agent = Agent(
#     model=Model.Almock(),
#     system_prompt="You are a helpful assistant.",
#     knowledge=knowledge,
# )
# agent.serve(port=8000, enable_playground=True, debug=True)
