"""Tests for Chunk, ChunkConfig, Chunker protocol, and chunking strategies."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from syrin.knowledge import Document
from syrin.knowledge._chunker import Chunk, ChunkConfig, ChunkStrategy

try:
    import chonkie  # noqa: F401

    HAS_CHONKIE = True
except ImportError:
    HAS_CHONKIE = False

requires_chonkie = pytest.mark.skipif(
    not HAS_CHONKIE, reason="chonkie not installed (pip install syrin[knowledge])"
)


class TestChunk:
    """Tests for Chunk dataclass."""

    def test_create_chunk(self) -> None:
        """Chunk can be created with required fields."""
        c = Chunk(
            content="Hello world",
            metadata={"chunk_strategy": "recursive"},
            document_id="doc.txt",
            chunk_index=0,
            token_count=3,
        )
        assert c.content == "Hello world"
        assert c.document_id == "doc.txt"
        assert c.chunk_index == 0
        assert c.token_count == 3
        assert c.metadata["chunk_strategy"] == "recursive"

    def test_chunk_is_immutable(self) -> None:
        """Chunk is frozen."""
        c = Chunk(
            content="x",
            metadata={},
            document_id="d",
            chunk_index=0,
            token_count=1,
        )
        with pytest.raises(AttributeError):
            c.content = "y"

    def test_chunk_empty_document_id_raises(self) -> None:
        """Chunk raises if document_id is empty."""
        with pytest.raises(ValueError, match="document_id must be non-empty"):
            Chunk(
                content="x",
                metadata={},
                document_id="",
                chunk_index=0,
                token_count=1,
            )

    def test_chunk_negative_chunk_index_raises(self) -> None:
        """Chunk raises if chunk_index is negative."""
        with pytest.raises(ValueError, match="chunk_index must be >= 0"):
            Chunk(
                content="x",
                metadata={},
                document_id="d",
                chunk_index=-1,
                token_count=1,
            )

    def test_chunk_negative_token_count_raises(self) -> None:
        """Chunk raises if token_count is negative."""
        with pytest.raises(ValueError, match="token_count must be >= 0"):
            Chunk(
                content="x",
                metadata={},
                document_id="d",
                chunk_index=0,
                token_count=-1,
            )


class TestChunkConfig:
    """Tests for ChunkConfig."""

    def test_defaults(self) -> None:
        """ChunkConfig has expected defaults."""
        config = ChunkConfig()
        assert config.strategy == ChunkStrategy.AUTO
        assert config.chunk_size == 512
        assert config.chunk_overlap == 0
        assert config.min_chunk_size == 50
        assert config.preserve_headers is True
        assert config.embedding is None

    def test_valid_strategy(self) -> None:
        """ChunkConfig accepts all ChunkStrategy values."""
        for strategy in ChunkStrategy:
            config = ChunkConfig(strategy=strategy, min_chunk_size=0)
            assert config.strategy == strategy

    def test_chunk_size_zero_raises(self) -> None:
        """ChunkConfig raises if chunk_size < 1."""
        with pytest.raises(ValueError, match="chunk_size must be >= 1"):
            ChunkConfig(chunk_size=0)

    def test_chunk_overlap_negative_raises(self) -> None:
        """ChunkConfig raises if chunk_overlap < 0."""
        with pytest.raises(ValueError, match="chunk_overlap must be >= 0"):
            ChunkConfig(chunk_overlap=-1)

    def test_chunk_overlap_ge_chunk_size_raises(self) -> None:
        """ChunkConfig raises if chunk_overlap >= chunk_size."""
        with pytest.raises(ValueError, match="chunk_overlap must be < chunk_size"):
            ChunkConfig(chunk_size=100, chunk_overlap=100)
        with pytest.raises(ValueError, match="chunk_overlap must be < chunk_size"):
            ChunkConfig(chunk_size=100, chunk_overlap=150)

    def test_chunk_overlap_zero_allowed(self) -> None:
        """ChunkConfig allows chunk_overlap=0 (default)."""
        config = ChunkConfig(chunk_overlap=0, chunk_size=512)
        assert config.chunk_overlap == 0

    def test_min_chunk_size_negative_raises(self) -> None:
        """ChunkConfig raises if min_chunk_size < 0."""
        with pytest.raises(ValueError, match="min_chunk_size must be >= 0"):
            ChunkConfig(min_chunk_size=-1)

    def test_similarity_threshold_out_of_range_raises(self) -> None:
        """ChunkConfig raises if similarity_threshold not in [0, 1]."""
        with pytest.raises(ValueError, match="similarity_threshold"):
            ChunkConfig(similarity_threshold=-0.1)
        with pytest.raises(ValueError, match="similarity_threshold"):
            ChunkConfig(similarity_threshold=1.5)


class TestGetChunker:
    """Tests for get_chunker factory."""

    @requires_chonkie
    def test_returns_recursive_chunker(self) -> None:
        """get_chunker returns RecursiveChunker for RECURSIVE strategy."""
        from syrin.knowledge import get_chunker
        from syrin.knowledge.chunkers import RecursiveChunker

        config = ChunkConfig(strategy=ChunkStrategy.RECURSIVE)
        chunker = get_chunker(config)
        assert isinstance(chunker, RecursiveChunker)

    @requires_chonkie
    def test_returns_auto_chunker(self) -> None:
        """get_chunker returns AutoChunker for AUTO strategy."""
        from syrin.knowledge import get_chunker
        from syrin.knowledge.chunkers import AutoChunker

        config = ChunkConfig(strategy=ChunkStrategy.AUTO, min_chunk_size=0)
        chunker = get_chunker(config)
        assert isinstance(chunker, AutoChunker)

    def test_returns_page_chunker(self) -> None:
        """get_chunker returns PageChunker for PAGE strategy."""
        from syrin.knowledge import get_chunker
        from syrin.knowledge.chunkers import PageChunker

        config = ChunkConfig(strategy=ChunkStrategy.PAGE, min_chunk_size=0)
        chunker = get_chunker(config)
        assert isinstance(chunker, PageChunker)

    @requires_chonkie
    def test_returns_markdown_chunker(self) -> None:
        """get_chunker returns MarkdownChunker for MARKDOWN strategy."""
        from syrin.knowledge import get_chunker
        from syrin.knowledge.chunkers import MarkdownChunker

        config = ChunkConfig(strategy=ChunkStrategy.MARKDOWN, min_chunk_size=0)
        chunker = get_chunker(config)
        assert isinstance(chunker, MarkdownChunker)


@requires_chonkie
class TestRecursiveChunker:
    """Tests for RecursiveChunker (requires chonkie)."""

    def test_empty_documents_returns_empty_list(self) -> None:
        """RecursiveChunker returns [] for empty document list."""
        from syrin.knowledge import get_chunker

        config = ChunkConfig(strategy=ChunkStrategy.RECURSIVE, min_chunk_size=0)
        chunker = get_chunker(config)
        assert chunker.chunk([]) == []

    def test_empty_content_document_produces_no_chunks(self) -> None:
        """Document with empty or whitespace-only content produces no chunks."""
        from syrin.knowledge import get_chunker

        config = ChunkConfig(strategy=ChunkStrategy.RECURSIVE, min_chunk_size=0)
        chunker = get_chunker(config)
        doc = Document(content="   \n\n  ", source="empty.txt", source_type="text")
        chunks = chunker.chunk([doc])
        assert chunks == []

    def test_single_document_produces_chunks_with_document_id(self) -> None:
        """Chunks have document_id equal to Document.source."""
        from syrin.knowledge import get_chunker

        config = ChunkConfig(
            strategy=ChunkStrategy.RECURSIVE,
            chunk_size=50,
            min_chunk_size=0,
        )
        chunker = get_chunker(config)
        doc = Document(
            content="First sentence. Second sentence. Third. " * 10,
            source="test.txt",
            source_type="text",
        )
        chunks = chunker.chunk([doc])
        assert len(chunks) >= 1
        for c in chunks:
            assert c.document_id == "test.txt"
            assert c.metadata.get("chunk_strategy") == "recursive"
            assert c.chunk_index >= 0
            assert c.token_count >= 0

    def test_chunk_indices_sequential(self) -> None:
        """Chunk indices are 0, 1, 2, ... per document."""
        from syrin.knowledge import get_chunker

        config = ChunkConfig(
            strategy=ChunkStrategy.RECURSIVE,
            chunk_size=30,
            min_chunk_size=0,
        )
        chunker = get_chunker(config)
        doc = Document(
            content="A. B. C. D. E. F. G. H. I. J. " * 5,
            source="seq.txt",
            source_type="text",
        )
        chunks = chunker.chunk([doc])
        for i, c in enumerate(chunks):
            assert c.chunk_index == i

    def test_min_chunk_size_filters_small_chunks(self) -> None:
        """Chunks below min_chunk_size are excluded."""
        from syrin.knowledge import get_chunker

        config = ChunkConfig(
            strategy=ChunkStrategy.RECURSIVE,
            chunk_size=500,
            min_chunk_size=100,
        )
        chunker = get_chunker(config)
        doc = Document(
            content="Short. " * 5,
            source="short.txt",
            source_type="text",
        )
        chunks = chunker.chunk([doc])
        for c in chunks:
            assert c.token_count >= 100
        if chunks:
            assert all(c.token_count >= 100 for c in chunks)


class TestPageChunker:
    """Tests for PageChunker."""

    def test_one_chunk_per_document(self) -> None:
        """PageChunker produces one chunk per document."""
        from syrin.knowledge import get_chunker

        config = ChunkConfig(strategy=ChunkStrategy.PAGE, min_chunk_size=0)
        chunker = get_chunker(config)
        docs = [
            Document(content="Page 1.", source="a.pdf", source_type="pdf", metadata={"page": 1}),
            Document(content="Page 2.", source="a.pdf", source_type="pdf", metadata={"page": 2}),
        ]
        chunks = chunker.chunk(docs)
        assert len(chunks) == 2
        assert chunks[0].document_id == "a.pdf"
        assert chunks[0].chunk_index == 0
        assert chunks[1].chunk_index == 1
        assert chunks[0].metadata.get("chunk_strategy") == "page"

    def test_empty_content_skipped(self) -> None:
        """Documents with empty content produce no chunks."""
        from syrin.knowledge import get_chunker

        config = ChunkConfig(strategy=ChunkStrategy.PAGE, min_chunk_size=0)
        chunker = get_chunker(config)
        doc = Document(
            content="  \n  ", source="blank.pdf", source_type="pdf", metadata={"page": 1}
        )
        assert chunker.chunk([doc]) == []


@requires_chonkie
class TestAutoChunker:
    """Tests for AutoChunker strategy selection."""

    def test_pdf_with_has_pages_uses_page_strategy(self) -> None:
        """Document with source_type pdf and has_pages uses PAGE chunker."""
        from syrin.knowledge import get_chunker

        config = ChunkConfig(strategy=ChunkStrategy.AUTO, min_chunk_size=0)
        chunker = get_chunker(config)
        doc = Document(
            content="PDF page content. " * 20,
            source="x.pdf",
            source_type="pdf",
            metadata={"page": 1, "has_pages": True},
        )
        chunks = chunker.chunk([doc])
        assert len(chunks) >= 1
        assert chunks[0].metadata.get("chunk_strategy") == "page"

    def test_markdown_uses_markdown_strategy(self) -> None:
        """Document with source_type markdown uses MARKDOWN chunker."""
        from syrin.knowledge import get_chunker

        config = ChunkConfig(strategy=ChunkStrategy.AUTO, min_chunk_size=0)
        chunker = get_chunker(config)
        doc = Document(
            content="# Title\n\nBody. " * 30,
            source="x.md",
            source_type="markdown",
        )
        chunks = chunker.chunk([doc])
        assert len(chunks) >= 1
        assert chunks[0].metadata.get("chunk_strategy") == "markdown"

    def test_text_uses_recursive_strategy(self) -> None:
        """Document with source_type text uses RECURSIVE chunker."""
        from syrin.knowledge import get_chunker

        config = ChunkConfig(strategy=ChunkStrategy.AUTO, min_chunk_size=0)
        chunker = get_chunker(config)
        doc = Document(
            content="Plain text. " * 50,
            source="x.txt",
            source_type="text",
        )
        chunks = chunker.chunk([doc])
        assert len(chunks) >= 1
        assert chunks[0].metadata.get("chunk_strategy") == "recursive"


class TestSemanticChunker:
    """Tests for SemanticChunker (requires embedding provider)."""

    def test_requires_embedding_raises(self) -> None:
        """SemanticChunker raises if config.embedding is None."""
        from syrin.knowledge.chunkers import SemanticChunker

        config = ChunkConfig(strategy=ChunkStrategy.SEMANTIC, min_chunk_size=0)
        with pytest.raises(ValueError, match="requires config.embedding"):
            SemanticChunker(config)

    @pytest.mark.asyncio
    async def test_achunk_returns_chunks_with_mock_embedding(self) -> None:
        """SemanticChunker.achunk returns chunks when embedding is provided."""
        from syrin.knowledge.chunkers import SemanticChunker

        mock_embedding = AsyncMock()
        mock_embedding.embed.return_value = [[0.1] * 8, [0.2] * 8, [0.1] * 8]
        mock_embedding.dimensions = 8
        mock_embedding.model_id = "test"

        config = ChunkConfig(
            strategy=ChunkStrategy.SEMANTIC,
            min_chunk_size=0,
            similarity_threshold=0.5,
            embedding=mock_embedding,
        )
        chunker = SemanticChunker(config)
        doc = Document(
            content="First. Second. Third.",
            source="s.txt",
            source_type="text",
        )
        chunks = await chunker.achunk([doc])
        assert len(chunks) >= 1
        assert chunks[0].document_id == "s.txt"
        mock_embedding.embed.assert_called_once()


@requires_chonkie
class TestChunkerProtocol:
    """Tests for Chunker protocol (achunk delegates to chunk for sync chunkers)."""

    @pytest.mark.asyncio
    async def test_recursive_achunk_same_as_chunk(self) -> None:
        """RecursiveChunker.achunk returns same result as chunk."""
        from syrin.knowledge import get_chunker

        config = ChunkConfig(strategy=ChunkStrategy.RECURSIVE, chunk_size=50, min_chunk_size=0)
        chunker = get_chunker(config)
        doc = Document(content="Hello. World. " * 20, source="t.txt", source_type="text")
        sync_chunks = chunker.chunk([doc])
        async_chunks = await chunker.achunk([doc])
        assert len(async_chunks) == len(sync_chunks)
        for a, s in zip(async_chunks, sync_chunks, strict=True):
            assert a.content == s.content
            assert a.document_id == s.document_id
            assert a.chunk_index == s.chunk_index
