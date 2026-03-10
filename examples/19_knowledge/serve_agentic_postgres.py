"""Serve Agentic RAG with PostgreSQL backend over HTTP.

Starts an HTTP server; use POST /chat to verify agentic RAG (search_knowledge,
search_knowledge_deep, verify_knowledge tools) backed by Postgres.

Run:
    # 1. Start Postgres (if not already running)
    docker run -d --name syrin-pg -p 5432:5432 -e POSTGRES_PASSWORD=postgres pgvector/pgvector:pg16

    # 2. Set env (or use examples/.env)
    export DATABASE_URL=postgresql://postgres:postgres@localhost:5432/postgres
    export OPENAI_API_KEY=sk-...

    # 3. Serve
    uv run python examples/19_knowledge/serve_agentic_postgres.py

    # 4. Verify
    curl -X POST http://localhost:8000/chat -H "Content-Type: application/json" \
      -d '{"message": "What is Syrin and what does RAG mean? Use the knowledge base."}'

Requires: syrin[knowledge,knowledge-postgres,serve]
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from syrin import Agent
from syrin.embedding import Embedding
from syrin.enums import KnowledgeBackend
from syrin.knowledge import AgenticRAGConfig, Knowledge
from syrin.model import Model


def main() -> None:
    api_key = os.getenv("OPENAI_API_KEY")
    connection_url = os.getenv("DATABASE_URL")

    if not api_key:
        raise SystemExit("Set OPENAI_API_KEY in examples/.env")
    if not connection_url:
        raise SystemExit(
            "Set DATABASE_URL (e.g. postgresql://postgres:postgres@localhost:5432/postgres). "
            "Start Postgres: docker run -d --name syrin-pg -p 5432:5432 -e POSTGRES_PASSWORD=postgres pgvector/pgvector:pg16"
        )

    knowledge = Knowledge(
        sources=[
            Knowledge.Text(
                "Python is a high-level programming language created by Guido van Rossum."
            ),
            Knowledge.Text("Syrin is a library for building AI agents with budget management."),
            Knowledge.Text("RAG means Retrieval-Augmented Generation."),
            Knowledge.Text(
                "Agentic RAG uses query decomposition, result grading, and verification."
            ),
            Knowledge.Text("PostgreSQL with pgvector enables vector similarity search in SQL."),
        ],
        embedding=Embedding.OpenAI("text-embedding-3-small", api_key=api_key),
        backend=KnowledgeBackend.POSTGRES,
        connection_url=connection_url,
        agentic=True,
        agentic_config=AgenticRAGConfig(
            max_search_iterations=3,
            decompose_complex=True,
            grade_results=True,
            relevance_threshold=0.4,
        ),
    )

    agent = Agent(
        model=Model.OpenAI("gpt-4o-mini", api_key=api_key),
        system_prompt=(
            "Use search_knowledge for simple questions. "
            "Use search_knowledge_deep for complex or multi-part questions. "
            "Use verify_knowledge to fact-check. Answer using the knowledge base."
        ),
        knowledge=knowledge,
    )

    print("Agentic RAG + Postgres | POST /chat, /stream | port 8000")
    agent.serve(port=8000, enable_playground=True, debug=True)


if __name__ == "__main__":
    main()
