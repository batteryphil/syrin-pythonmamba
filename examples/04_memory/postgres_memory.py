"""PostgreSQL Memory Example.

Demonstrates:
- Memory with PostgresConfig for persistent, SQL-backed storage
- Production-ready persistence
- Text search (LIKE); for vector/semantic search enable pgvector

Requires: pip install syrin[postgres]  # or pip install psycopg2-binary

Prerequisites: PostgreSQL running at localhost:5432
  docker run -d -p 5432:5432 -e POSTGRES_PASSWORD= -e POSTGRES_DB=syrin postgres:alpine

Run: python -m examples.04_memory.postgres_memory
"""

from __future__ import annotations

from examples.models.models import almock
from syrin import Agent, Memory, MemoryType
from syrin.enums import MemoryBackend
from syrin.memory import PostgresConfig


def main() -> None:
    memory = Memory(
        backend=MemoryBackend.POSTGRES,
        postgres=PostgresConfig(
            host="localhost",
            port=5432,
            database="syrin",
            user="postgres",
            password="",
            table="memories",
        ),
    )

    agent = Agent(
        model=almock,
        system_prompt="You are a helpful assistant with Postgres-backed memory.",
        memory=memory,
    )

    agent.remember("User prefers Python over JavaScript", memory_type=MemoryType.CORE)
    agent.remember("Last discussed async/await patterns", memory_type=MemoryType.EPISODIC)
    agent.remember("Likes functional programming style", memory_type=MemoryType.SEMANTIC)

    # Recall by query (SQL LIKE; for vector search enable pgvector with vector_size>0)
    entries = agent.recall(query="programming", limit=5)
    print(f"Recalled {len(entries)} memories:")
    for e in entries:
        print(f"  - {e.content[:70]}...")

    # Production example (custom host, credentials via env):
    # import os
    # memory = Memory(
    #     backend=MemoryBackend.POSTGRES,
    #     postgres=PostgresConfig(
    #         host=os.getenv("PGHOST", "localhost"),
    #         port=int(os.getenv("PGPORT", "5432")),
    #         database=os.getenv("PGDATABASE", "syrin"),
    #         user=os.getenv("PGUSER", "postgres"),
    #         password=os.getenv("PGPASSWORD", ""),
    #         table="agent_memories",
    #     ),
    # )


if __name__ == "__main__":
    main()
