"""Full RAG + Agentic RAG lifecycle — every feature in one example.

Covers the complete pipeline: loaders → document model → chunking → store → search,
plus agentic retrieval (decompose, grade, refine, verify) and agent integration.

Features demonstrated:
- Document model (content, source, source_type, metadata)
- Loaders: Knowledge.Text, Knowledge.Texts (also: .Markdown, .PDF, .Python, .YAML,
  .JSON, .Directory, .URL, .GitHub, .TextFile)
- Chunking: ChunkConfig, ChunkStrategy (RECURSIVE, AUTO, PAGE, MARKDOWN, CODE,
  SENTENCE, TOKEN, SEMANTIC)
- Knowledge: sources, embedding, backend (MEMORY|SQLITE|POSTGRES|QDRANT|CHROMA),
  chunk_config, top_k, score_threshold, inject_system_prompt
- Pipeline: ingest(), search(), add_source(), stats(), clear()
- Search: top_k, filter, score_threshold
- Agent: knowledge= adds search_knowledge tool
- Agentic: agentic=True adds search_knowledge_deep, verify_knowledge
- AgenticRAGConfig: max_search_iterations, decompose_complex, grade_results,
  relevance_threshold, web_fallback

Run:
    uv run python examples/19_knowledge/full_rag_lifecycle.py

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
from syrin.knowledge import (
    AgenticRAGConfig,
    ChunkConfig,
    ChunkStrategy,
    Document,
    Knowledge,
    get_chunker,
)
from syrin.model import Model


async def main() -> None:
    print("=" * 60)
    print("1. DOCUMENT MODEL & LOADERS")
    print("=" * 60)

    # Document: immutable, required fields content, source, source_type
    doc = Document(
        content="Syrin is a Python library for building AI agents with budget management.",
        source="intro",
        source_type="text",
        metadata={"section": "overview"},
    )
    print(f"  Document: {doc.content[:50]}... (source={doc.source})")

    # Knowledge.* factory methods create loaders
    raw = Knowledge.Text("Python is a high-level programming language created by Guido van Rossum.")
    raw_multi = Knowledge.Texts(
        [
            "RAG means Retrieval-Augmented Generation.",
            "Agentic RAG adds query decomposition, grading, and verification.",
        ]
    )
    print(f"  Knowledge.Text -> 1 doc, Knowledge.Texts -> {len(raw_multi.load())} docs")

    print("\n" + "=" * 60)
    print("2. CHUNKING (ChunkConfig, ChunkStrategy, get_chunker)")
    print("=" * 60)

    docs_for_chunking = raw.load() + raw_multi.load()
    config = ChunkConfig(
        strategy=ChunkStrategy.RECURSIVE,
        chunk_size=64,
        min_chunk_size=0,
    )
    chunker = get_chunker(config)
    chunks = chunker.chunk(docs_for_chunking)
    print(f"  ChunkStrategy.RECURSIVE: {len(chunks)} chunks from {len(docs_for_chunking)} docs")
    for c in chunks[:2]:
        print(f"    [{c.chunk_index}] {c.content[:40]}... (doc={c.document_id})")

    # Auto strategy selects by source_type
    config_auto = ChunkConfig(strategy=ChunkStrategy.AUTO)
    chunker_auto = get_chunker(config_auto)
    chunks_auto = chunker_auto.chunk(docs_for_chunking)
    print(f"  ChunkStrategy.AUTO: {len(chunks_auto)} chunks")

    print("\n" + "=" * 60)
    print("3. KNOWLEDGE ORCHESTRATOR (load → chunk → embed → store)")
    print("=" * 60)

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("  SKIP: OPENAI_API_KEY not set. Set it in examples/.env to run embeddings.")
        return

    embedding = Embedding.OpenAI("text-embedding-3-small", api_key=api_key)
    knowledge = Knowledge(
        sources=[
            Knowledge.Text(
                "Python is a high-level programming language created by Guido van Rossum."
            ),
            Knowledge.Text("Syrin is a library for building AI agents with budget management."),
            Knowledge.Text("RAG means Retrieval-Augmented Generation."),
            Knowledge.Text(
                "Agentic RAG uses query decomposition, result grading, and iterative refinement."
            ),
            Knowledge.Text(
                "Budget control is first-class in Syrin: per-agent limits and cost tracking."
            ),
        ],
        embedding=embedding,
        backend=KnowledgeBackend.MEMORY,
        chunk_config=ChunkConfig(strategy=ChunkStrategy.RECURSIVE, chunk_size=64, min_chunk_size=0),
        chunk_strategy=None,  # chunk_config takes precedence
        top_k=5,
        score_threshold=0.2,
        inject_system_prompt=True,
    )

    # Explicit ingest (or lazy on first search)
    await knowledge.ingest()
    stats = await knowledge.stats()
    print(f"  Ingested: {stats['chunk_count']} chunks from {stats['source_count']} sources")

    print("\n" + "=" * 60)
    print("4. SEARCH (semantic retrieval)")
    print("=" * 60)

    results = await knowledge.search("What is Syrin?")
    print(f"  Query 'What is Syrin?': {len(results)} results")
    for r in results[:2]:
        print(f"    rank {r.rank} score={r.score:.2f}: {r.chunk.content[:50]}...")

    # Search with filter (by source)
    results_filtered = await knowledge.search(
        "programming language",
        top_k=2,
        filter={"source": "user_provided"},
    )
    print(f"  Filtered search: {len(results_filtered)} results")

    print("\n" + "=" * 60)
    print("5. SOURCE MANAGEMENT (add_source, stats, remove_source)")
    print("=" * 60)

    extra = Knowledge.Text("LiteLLM provides a unified interface to many LLM providers.")
    knowledge.add_source(extra)
    print(f"  add_source: {await knowledge.stats()}")

    # Re-ingest to include new source (Knowledge doesn't auto re-ingest)
    await knowledge.ingest()
    stats_after = await knowledge.stats()
    print(f"  After ingest: {stats_after}")

    print("\n" + "=" * 60)
    print("6. AGENT + RAG (search_knowledge tool)")
    print("=" * 60)

    agent = Agent(
        model=Model.OpenAI("gpt-4o-mini", api_key=api_key),
        system_prompt="Use search_knowledge when asked about Python, Syrin, RAG, or agents.",
        knowledge=knowledge,
    )
    tool_names = [t.name for t in agent.tools]
    print(f"  Agent tools: {tool_names}")
    assert "search_knowledge" in tool_names

    print("\n" + "=" * 60)
    print("7. AGENTIC RAG (search_knowledge_deep, verify_knowledge)")
    print("=" * 60)

    # Create fresh Knowledge with agentic enabled
    knowledge_agentic = Knowledge(
        sources=[
            Knowledge.Text("Python is a high-level programming language."),
            Knowledge.Text("Syrin is a library for building AI agents with budget management."),
            Knowledge.Text("RAG means Retrieval-Augmented Generation."),
            Knowledge.Text("Agentic RAG adds decomposition, grading, and verification."),
        ],
        embedding=embedding,
        backend=KnowledgeBackend.MEMORY,
        agentic=True,
        agentic_config=AgenticRAGConfig(
            max_search_iterations=3,
            decompose_complex=True,
            grade_results=True,
            relevance_threshold=0.4,
            web_fallback=False,
        ),
    )

    agent_agentic = Agent(
        model=Model.OpenAI("gpt-4o-mini", api_key=api_key),
        system_prompt=(
            "Use search_knowledge for simple questions. "
            "Use search_knowledge_deep for complex or multi-part questions. "
            "Use verify_knowledge to fact-check before answering."
        ),
        knowledge=knowledge_agentic,
    )
    agentic_tools = [t.name for t in agent_agentic.tools]
    print(f"  Agentic tools: {agentic_tools}")
    assert "search_knowledge" in agentic_tools
    assert "search_knowledge_deep" in agentic_tools
    assert "verify_knowledge" in agentic_tools

    # Direct semantic search
    simple_results = await knowledge_agentic.search("What is Syrin?")
    print(f"  search('What is Syrin?'): {len(simple_results)} results")

    # Agent run: agent uses tools (arun in async context; use response() in sync)
    response = await agent_agentic.arun(
        "What is Syrin and what does RAG mean? Use the knowledge base."
    )
    print(f"  Agent response (excerpt): {response.content[:120]}...")

    print("\n" + "=" * 60)
    print("8. CLEANUP (clear, stats)")
    print("=" * 60)

    await knowledge.clear()
    await knowledge_agentic.clear()
    print(f"  Cleared. Final stats (first): {await knowledge.stats()}")
    print(f"  Cleared. Final stats (agentic): {await knowledge_agentic.stats()}")

    print("\n" + "=" * 60)
    print("OK: Full RAG + Agentic RAG lifecycle complete.")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
