"""Chunking protocol, Chunk model, and configuration for Knowledge module."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Protocol, TypeAlias, runtime_checkable

from syrin.knowledge._document import Document, DocumentMetadata

if TYPE_CHECKING:
    from syrin.embedding._protocol import EmbeddingProvider

# Chunk metadata: same value types as Document (primitives, list[str] for heading_hierarchy, etc.).
ChunkMetadata: TypeAlias = DocumentMetadata


class ChunkStrategy(StrEnum):
    """Chunking strategy selection.

    Attributes:
        RECURSIVE: Split at paragraphs, then sentences, then words. Best for general text.
        SEMANTIC: Split at similarity drops (requires embeddings). Best for dense text.
        MARKDOWN: Header-aware, preserves section hierarchy and tables/code blocks.
        CODE: AST-aware via tree-sitter. Best for source code.
        PAGE: One chunk per page. Best for PDFs.
        SENTENCE: Sentence-level grouping.
        TOKEN: Fixed token-count windows.
        AUTO: Select strategy from document source_type (code → CODE, markdown → MARKDOWN, pdf → PAGE, else RECURSIVE).
    """

    RECURSIVE = "recursive"
    SEMANTIC = "semantic"
    MARKDOWN = "markdown"
    CODE = "code"
    PAGE = "page"
    SENTENCE = "sentence"
    TOKEN = "token"
    AUTO = "auto"


@dataclass(frozen=True)
class Chunk:
    """A chunk of a document, ready for embedding and storage.

    Attributes:
        content: The chunk text.
        metadata: Inherited from Document plus chunk-specific (e.g. chunk_strategy, heading_hierarchy).
        document_id: Reference to source Document (same as Document.source).
        chunk_index: Zero-based position of this chunk within the document.
        token_count: Estimated token count for the chunk.
    """

    content: str
    metadata: ChunkMetadata
    document_id: str
    chunk_index: int
    token_count: int

    def __post_init__(self) -> None:
        if not self.document_id:
            raise ValueError("Chunk.document_id must be non-empty")
        if self.chunk_index < 0:
            raise ValueError("Chunk.chunk_index must be >= 0")
        if self.token_count < 0:
            raise ValueError("Chunk.token_count must be >= 0")


@dataclass
class ChunkConfig:
    """Chunking configuration.

    Attributes:
        strategy: Which chunking strategy to use. Default AUTO selects by source_type.
        chunk_size: Target tokens per chunk.
        chunk_overlap: Overlap in tokens between consecutive chunks (0 = no overlap).
        min_chunk_size: Chunks with fewer tokens are dropped or merged.
        preserve_tables: Never split markdown/HTML tables (MarkdownChunker).
        preserve_code_blocks: Never split fenced code blocks (MarkdownChunker).
        preserve_headers: Keep heading hierarchy in chunk metadata (MarkdownChunker).
        similarity_threshold: For SEMANTIC strategy.
        embedding: Required for SEMANTIC strategy; used to compute sentence similarity.
        language: For CODE strategy (e.g. "python", "javascript").
    """

    strategy: ChunkStrategy = ChunkStrategy.AUTO
    chunk_size: int = 512
    chunk_overlap: int = 0
    min_chunk_size: int = 50
    preserve_tables: bool = True
    preserve_code_blocks: bool = True
    preserve_headers: bool = True
    similarity_threshold: float = 0.5
    embedding: EmbeddingProvider | None = None
    language: str | None = None

    def __post_init__(self) -> None:
        if self.chunk_size < 1:
            raise ValueError("chunk_size must be >= 1")
        if self.chunk_overlap < 0:
            raise ValueError("chunk_overlap must be >= 0")
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be < chunk_size")
        if self.min_chunk_size < 0:
            raise ValueError("min_chunk_size must be >= 0")
        if not 0.0 <= self.similarity_threshold <= 1.0:
            raise ValueError("similarity_threshold must be between 0 and 1")


@runtime_checkable
class Chunker(Protocol):
    """Protocol for document chunking strategies.

    Implementations split documents into retrieval-optimized chunks.
    Use chunk() for sync and achunk() for async (e.g. when strategy needs embeddings).
    """

    def chunk(self, documents: list[Document]) -> list[Chunk]:
        """Split documents into chunks synchronously.

        Returns:
            List of Chunks, order preserved per document.
        """
        ...

    async def achunk(self, documents: list[Document]) -> list[Chunk]:
        """Split documents into chunks asynchronously.

        Default implementation delegates to chunk(). Override for strategies
        that require async work (e.g. SemanticChunker with EmbeddingProvider).

        Returns:
            List of Chunks, order preserved per document.
        """
        ...


def _estimate_tokens(text: str) -> int:
    """Estimate token count (same heuristic as router: ~4 chars per token)."""
    return max(1, len(text) // 4)
