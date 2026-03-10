"""Knowledge chunking: ChunkConfig, ChunkStrategy, get_chunker.

Shows how to split documents into retrieval-optimized chunks.
Requires: uv pip install syrin[knowledge] (chonkie).

Run:
    uv run python examples/19_knowledge/chunking.py
"""

from __future__ import annotations

from syrin.knowledge import (
    ChunkConfig,
    ChunkStrategy,
    Document,
    get_chunker,
)


def main() -> None:
    # Sample documents
    docs = [
        Document(
            content="First paragraph. Second paragraph. Third. " * 30,
            source="doc1.txt",
            source_type="text",
        ),
        Document(
            content="# Section A\n\nContent under A.\n\n## Subsection\n\nMore content.",
            source="doc2.md",
            source_type="markdown",
        ),
    ]

    # Recursive chunker (general text)
    config = ChunkConfig(
        strategy=ChunkStrategy.RECURSIVE,
        chunk_size=80,
        min_chunk_size=0,
    )
    chunker = get_chunker(config)
    chunks = chunker.chunk(docs)
    print(f"Recursive: {len(chunks)} chunks from {len(docs)} docs")
    for c in chunks[:3]:
        print(
            f"  {c.document_id} idx={c.chunk_index} tokens={c.token_count} strategy={c.metadata.get('chunk_strategy')}"
        )

    # Auto: selects strategy per document (markdown -> MARKDOWN, text -> RECURSIVE)
    config_auto = ChunkConfig(strategy=ChunkStrategy.AUTO, min_chunk_size=0)
    chunker_auto = get_chunker(config_auto)
    chunks_auto = chunker_auto.chunk(docs)
    print(f"\nAuto: {len(chunks_auto)} chunks")
    for c in chunks_auto:
        print(f"  {c.document_id} strategy={c.metadata.get('chunk_strategy')}")

    # Page chunker: one chunk per document (e.g. PDF pages)
    page_doc = Document(
        content="Page content here.",
        source="file.pdf",
        source_type="pdf",
        metadata={"page": 1, "has_pages": True},
    )
    config_page = ChunkConfig(strategy=ChunkStrategy.PAGE, min_chunk_size=0)
    page_chunker = get_chunker(config_page)
    page_chunks = page_chunker.chunk([page_doc])
    assert len(page_chunks) == 1
    assert page_chunks[0].metadata.get("chunk_strategy") == "page"
    print("\nPage: 1 chunk per document (e.g. PDF).")

    print("\nOK: Chunking works.")


if __name__ == "__main__":
    main()
