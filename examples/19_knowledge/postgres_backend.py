"""Knowledge with PostgreSQL (pgvector) backend.

Shows the full RAG pipeline using KnowledgeBackend.POSTGRES for production storage.
Requires: syrin[knowledge,knowledge-postgres]. PostgreSQL with pgvector extension.

Run:
    uv run python examples/19_knowledge/postgres_backend.py

Setup:
    1. Install: uv pip install "syrin[knowledge,knowledge-postgres]"
    2. Start PostgreSQL with pgvector:
       - Docker: docker run -d -p 5432:5432 -e POSTGRES_PASSWORD=postgres pgvector/pgvector:pg16
       - Local: ensure pgvector extension is installed
    3. Set in examples/.env:
       - OPENAI_API_KEY=sk-...
       - DATABASE_URL=postgresql://postgres:postgres@localhost:5432/postgres
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
from syrin.knowledge import Knowledge
from syrin.model import Model


async def main() -> None:
    api_key = os.getenv("OPENAI_API_KEY")
    connection_url = os.getenv("DATABASE_URL")

    if not api_key:
        print("Set OPENAI_API_KEY in examples/.env")
        return
    if not connection_url:
        print(
            "Set DATABASE_URL in examples/.env (e.g. postgresql://postgres:postgres@localhost:5432/postgres)\n"
            "Start Postgres with pgvector: docker run -d -p 5432:5432 -e POSTGRES_PASSWORD=postgres pgvector/pgvector:pg16"
        )
        return

    # Knowledge with Postgres backend
    knowledge = Knowledge(
        sources=[
            Knowledge.Text(
                "Python is a high-level programming language created by Guido van Rossum."
            ),
            Knowledge.Text("Syrin is a library for building AI agents with budget management."),
            Knowledge.Text("RAG means Retrieval-Augmented Generation."),
            Knowledge.Text("PostgreSQL with pgvector enables vector similarity search in SQL."),
        ],
        embedding=Embedding.OpenAI("text-embedding-3-small", api_key=api_key),
        backend=KnowledgeBackend.POSTGRES,
        connection_url=connection_url,
    )

    # Ingest and search
    await knowledge.ingest()
    stats = await knowledge.stats()
    print(f"Ingested: {stats['chunk_count']} chunks, {stats['source_count']} sources")

    results = await knowledge.search("What is Syrin?", top_k=3)
    print(f"Search 'What is Syrin?': {len(results)} results")
    for r in results:
        print(f"  [{r.rank}] score={r.score:.2f} | {r.chunk.content[:60]}...")

    # Agent with Postgres-backed knowledge
    agent = Agent(
        model=Model.OpenAI("gpt-4o-mini", api_key=api_key),
        system_prompt="Use search_knowledge to answer questions about Python, Syrin, RAG, or PostgreSQL.",
        knowledge=knowledge,
    )
    print("\nAgent tools:", [t.name for t in agent.tools])

    # Cleanup (optional; keeps data in Postgres for inspection)
    # await knowledge.clear()
    print("\nOK: Postgres backend example complete.")


if __name__ == "__main__":
    asyncio.run(main())
