"""Qdrant Vector Memory Example.

Demonstrates:
- Memory with QdrantConfig for vector/semantic search
- Local embedded Qdrant (path) or remote (url)
- Namespace isolation for multi-tenant

Requires: pip install syrin[qdrant]  # or pip install qdrant-client

Run: python -m examples.04_memory.qdrant_memory
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from syrin import Agent, Memory, MemoryType, Model
from syrin.enums import MemoryBackend
from syrin.memory import QdrantConfig


def main() -> None:
    # Use temp dir for demo; in production use persistent path or Qdrant Cloud
    with tempfile.TemporaryDirectory() as tmp:
        qdrant_path = Path(tmp) / "qdrant_data"

        memory = Memory(
            backend=MemoryBackend.QDRANT,
            qdrant=QdrantConfig(
                path=str(qdrant_path),
                collection="syrin_memories",
                namespace="demo",  # Per-tenant isolation
            ),
        )

        agent = Agent(
            model=Model.mock(),
            system_prompt="You are a helpful assistant with memory.",
            memory=memory,
        )

        agent.remember("User prefers Python over JavaScript", memory_type=MemoryType.FACTS)
        agent.remember("Last discussed async/await patterns", memory_type=MemoryType.HISTORY)

        # Semantic recall - vector search (MD5 pseudo-embeddings by default;
        # use EmbeddingConfig for real semantic similarity)
        entries = agent.recall(query="programming language", limit=5)
        print(f"Recalled {len(entries)} memories:")
        for e in entries:
            print(f"  - {e.content[:60]}...")

        # Qdrant Cloud example (uncomment with your URL and API key):
        # memory_cloud = Memory(
        #     backend=MemoryBackend.QDRANT,
        #     qdrant=QdrantConfig(
        #         url="https://xxx.qdrant.tech",
        #         api_key="your-api-key",
        #         collection="syrin_memories",
        #         namespace="tenant_123",
        #     ),
        # )


if __name__ == "__main__":
    main()
