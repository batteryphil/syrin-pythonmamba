"""Knowledge Store (vector backends) — upsert chunks, semantic search, delete.

Run: uv run python examples/19_knowledge/vector_store.py

Uses InMemoryKnowledgeStore (no extra deps). For Postgres/Qdrant/Chroma/SQLite,
install syrin[knowledge-postgres], syrin[qdrant], syrin[chroma], or syrin[knowledge-sqlite].
"""

from __future__ import annotations

import asyncio
import math


def _simple_embed(text: str, dim: int = 4) -> list[float]:
    """Deterministic pseudo-embedding for demo (not semantic)."""
    h = 0
    for c in text.encode():
        h = (h * 31 + c) & 0xFFFF_FFFF
    norm = max(1.0, math.sqrt(h % 1000))
    return [(float((h >> (i * 8)) % 256) / 255 - 0.5) / norm for i in range(dim)]


async def main() -> None:
    from syrin.knowledge import Chunk
    from syrin.knowledge.stores import InMemoryKnowledgeStore

    # In-memory store (no deps)
    store = InMemoryKnowledgeStore(embedding_dimensions=4)

    # Create chunks
    chunks = [
        Chunk(
            content="Python is a high-level programming language.",
            metadata={"source_type": "text"},
            document_id="doc1",
            chunk_index=0,
            token_count=8,
        ),
        Chunk(
            content="Rust provides memory safety without garbage collection.",
            metadata={"source_type": "text"},
            document_id="doc1",
            chunk_index=1,
            token_count=8,
        ),
        Chunk(
            content="JavaScript runs in browsers and Node.js.",
            metadata={"source_type": "text"},
            document_id="doc2",
            chunk_index=0,
            token_count=6,
        ),
    ]
    embeddings = [_simple_embed(c.content) for c in chunks]

    # Upsert
    await store.upsert(chunks, embeddings)
    print(f"Upserted {await store.count()} chunks")

    # Search
    query_emb = _simple_embed("Python programming")
    results = await store.search(query_emb, top_k=2)
    for r in results:
        print(f"  rank {r.rank}: score={r.score:.3f} | {r.chunk.content[:50]}...")

    # Search with filter
    results_pdf = await store.search(query_emb, top_k=5, filter={"source_type": "text"})
    print(f"  Filtered: {len(results_pdf)} results")

    # Delete by document
    n = await store.delete(document_id="doc1")
    print(f"Deleted {n} chunks for doc1. Remaining: {await store.count()}")


if __name__ == "__main__":
    asyncio.run(main())
