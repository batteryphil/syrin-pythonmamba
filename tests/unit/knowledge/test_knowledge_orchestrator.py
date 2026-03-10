"""Tests for Knowledge orchestrator (Step 5).

TDD: Valid, invalid, and edge cases for the Knowledge class.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import pytest

from syrin.enums import KnowledgeBackend
from syrin.knowledge import Knowledge
from syrin.knowledge._chunker import ChunkConfig, ChunkStrategy
from syrin.knowledge.loaders import RawTextLoader

if TYPE_CHECKING:
    from syrin.embedding._protocol import EmbeddingProvider


def _make_fake_embedding(dim: int = 4) -> type[EmbeddingProvider]:
    """Create a minimal EmbeddingProvider implementation for tests."""

    class FakeEmbedding:
        @property
        def dimensions(self) -> int:
            return dim

        @property
        def model_id(self) -> str:
            return "fake-embedding"

        async def embed(
            self,
            texts: list[str],
            budget_tracker: object | None = None,
        ) -> list[list[float]]:
            return [[0.1] * dim for _ in texts]

    return FakeEmbedding  # type: ignore[return-value]


class TestKnowledgeConstructor:
    """Constructor validation and config."""

    def test_requires_embedding(self) -> None:
        """Knowledge requires embedding provider."""
        with pytest.raises(ValueError, match="embedding is required"):
            Knowledge(
                sources=[Knowledge.Text("x")],
                embedding=None,
            )

    def test_requires_sources(self) -> None:
        """Knowledge requires at least one source."""
        FakeEmb = _make_fake_embedding()
        with pytest.raises(ValueError, match="sources.*non-empty"):
            Knowledge(
                sources=[],
                embedding=FakeEmb(),
            )

    def test_valid_minimal_memory_backend(self) -> None:
        """Minimal config with MEMORY backend succeeds."""
        FakeEmb = _make_fake_embedding()
        k = Knowledge(
            sources=[Knowledge.Text("fact")],
            embedding=FakeEmb(),
            backend=KnowledgeBackend.MEMORY,
        )
        assert k.embedding is not None
        assert k._store is not None
        assert len(k._sources) == 1

    def test_sqlite_requires_path_or_defaults(self) -> None:
        """SQLITE backend uses default path when path is None."""
        FakeEmb = _make_fake_embedding()
        try:
            k = Knowledge(
                sources=[Knowledge.Text("x")],
                embedding=FakeEmb(),
                backend=KnowledgeBackend.SQLITE,
                path=None,
            )
            assert k._path is not None
            assert "knowledge" in k._path
        except ImportError:
            pytest.skip("sqlite-vec not installed")

    def test_sqlite_with_explicit_path(self) -> None:
        """SQLITE backend accepts explicit path."""
        FakeEmb = _make_fake_embedding()
        try:
            k = Knowledge(
                sources=[Knowledge.Text("x")],
                embedding=FakeEmb(),
                backend=KnowledgeBackend.SQLITE,
                path="/tmp/kb.db",
            )
            assert k._path == "/tmp/kb.db"
        except ImportError:
            pytest.skip("sqlite-vec not installed")

    def test_chunk_config_from_shorthand(self) -> None:
        """chunk_strategy shorthand builds ChunkConfig."""
        FakeEmb = _make_fake_embedding()
        k = Knowledge(
            sources=[Knowledge.Text("x")],
            embedding=FakeEmb(),
            backend=KnowledgeBackend.MEMORY,
            chunk_strategy=ChunkStrategy.RECURSIVE,
            chunk_size=256,
        )
        assert k._chunk_config is not None
        assert k._chunk_config.strategy == ChunkStrategy.RECURSIVE
        assert k._chunk_config.chunk_size == 256

    def test_chunk_config_explicit_overrides_shorthand(self) -> None:
        """Explicit chunk_config takes precedence."""
        FakeEmb = _make_fake_embedding()
        cfg = ChunkConfig(strategy=ChunkStrategy.PAGE, chunk_size=100)
        k = Knowledge(
            sources=[Knowledge.Text("x")],
            embedding=FakeEmb(),
            backend=KnowledgeBackend.MEMORY,
            chunk_config=cfg,
            chunk_strategy=ChunkStrategy.RECURSIVE,
        )
        assert k._chunk_config.strategy == ChunkStrategy.PAGE
        assert k._chunk_config.chunk_size == 100

    def test_agentic_true_creates_config(self) -> None:
        """agentic=True creates AgenticRAGConfig with defaults."""
        FakeEmb = _make_fake_embedding()
        k = Knowledge(
            sources=[Knowledge.Text("x")],
            embedding=FakeEmb(),
            backend=KnowledgeBackend.MEMORY,
            agentic=True,
        )
        assert k._agentic is True
        assert k._agentic_config is not None
        assert k._agentic_config.max_search_iterations == 3
        assert k._agentic_config.decompose_complex is True
        assert k._agentic_config.grade_results is True


class TestKnowledgeIngest:
    """Ingest pipeline: load -> chunk -> embed -> store."""

    @pytest.mark.asyncio
    async def test_ingest_loads_chunks_embeds_stores(self) -> None:
        """ingest() runs full pipeline and stores chunks."""
        FakeEmb = _make_fake_embedding()
        # Use longer text so chunker produces at least one chunk (min_chunk_size=50 tokens)
        long_text = "Python is a programming language. " * 20
        k = Knowledge(
            sources=[Knowledge.Text(long_text)],
            embedding=FakeEmb(),
            backend=KnowledgeBackend.MEMORY,
        )
        await k.ingest()
        count = await k._store.count()
        assert count >= 1

    @pytest.mark.asyncio
    async def test_ingest_idempotent(self) -> None:
        """Multiple ingest() calls do not duplicate chunks."""
        FakeEmb = _make_fake_embedding()
        k = Knowledge(
            sources=[Knowledge.Text("Unique content for idempotence.")],
            embedding=FakeEmb(),
            backend=KnowledgeBackend.MEMORY,
        )
        await k.ingest()
        c1 = await k._store.count()
        await k.ingest()
        c2 = await k._store.count()
        assert c1 == c2

    @pytest.mark.asyncio
    async def test_ingest_emits_hooks(self) -> None:
        """ingest emits INGEST_START and INGEST_END hooks."""
        FakeEmb = _make_fake_embedding()
        emit = MagicMock()
        long_text = "Hook test content. " * 20
        k = Knowledge(
            sources=[Knowledge.Text(long_text)],
            embedding=FakeEmb(),
            backend=KnowledgeBackend.MEMORY,
            emit=emit,
        )
        await k.ingest()
        assert emit.call_count >= 2
        hook_values = [c[0][0] for c in emit.call_args_list]
        assert "knowledge.ingest.start" in hook_values
        assert "knowledge.ingest.end" in hook_values


class TestKnowledgeSearch:
    """Search method."""

    @pytest.mark.asyncio
    async def test_search_triggers_lazy_ingest(self) -> None:
        """First search() triggers ingest if not yet done."""
        FakeEmb = _make_fake_embedding()
        long_text = "Searchable fact about dogs. " * 20
        k = Knowledge(
            sources=[Knowledge.Text(long_text)],
            embedding=FakeEmb(),
            backend=KnowledgeBackend.MEMORY,
        )
        results = await k.search("dogs")
        assert isinstance(results, list)
        # May or may not have results depending on similarity
        assert all(
            hasattr(r, "chunk") and hasattr(r, "score") and hasattr(r, "rank") for r in results
        )

    @pytest.mark.asyncio
    async def test_search_respects_top_k(self) -> None:
        """search() returns at most top_k results."""
        FakeEmb = _make_fake_embedding()
        long_a = "Content A. " * 20
        long_b = "Content B. " * 20
        long_c = "Content C. " * 20
        k = Knowledge(
            sources=[
                Knowledge.Text(long_a),
                Knowledge.Text(long_b),
                Knowledge.Text(long_c),
            ],
            embedding=FakeEmb(),
            backend=KnowledgeBackend.MEMORY,
        )
        await k.ingest()
        results = await k.search("query", top_k=2)
        assert len(results) <= 2

    @pytest.mark.asyncio
    async def test_search_with_filter(self) -> None:
        """search() accepts metadata filter."""
        FakeEmb = _make_fake_embedding()
        long_text = "Filtered content. " * 20
        k = Knowledge(
            sources=[Knowledge.Text(long_text)],
            embedding=FakeEmb(),
            backend=KnowledgeBackend.MEMORY,
        )
        await k.ingest()
        results = await k.search("content", filter={"source_type": "text"})
        assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_search_empty_query_returns_list(self) -> None:
        """search returns list even with no matching results."""
        FakeEmb = _make_fake_embedding()
        long_text = "Some content. " * 20
        k = Knowledge(
            sources=[Knowledge.Text(long_text)],
            embedding=FakeEmb(),
            backend=KnowledgeBackend.MEMORY,
        )
        await k.ingest()
        results = await k.search("xyznonexistent")
        assert isinstance(results, list)


class TestKnowledgeLifecycle:
    """add_source, remove_source, clear, stats."""

    @pytest.mark.asyncio
    async def test_add_source(self) -> None:
        """add_source() adds a new loader."""
        FakeEmb = _make_fake_embedding()
        k = Knowledge(
            sources=[Knowledge.Text("Initial")],
            embedding=FakeEmb(),
            backend=KnowledgeBackend.MEMORY,
        )
        k.add_source(Knowledge.Text("Added"))
        assert len(k._sources) == 2

    @pytest.mark.asyncio
    async def test_remove_source(self) -> None:
        """remove_source() removes loader and deletes its chunks."""
        FakeEmb = _make_fake_embedding()
        loader = Knowledge.Text("To remove")
        k = Knowledge(
            sources=[Knowledge.Text("Keep"), loader],
            embedding=FakeEmb(),
            backend=KnowledgeBackend.MEMORY,
        )
        await k.ingest()
        await k.remove_source(loader)
        assert len(k._sources) == 1

    @pytest.mark.asyncio
    async def test_clear_deletes_all_chunks(self) -> None:
        """clear() deletes all chunks."""
        FakeEmb = _make_fake_embedding()
        long_text = "Clear me. " * 20
        k = Knowledge(
            sources=[Knowledge.Text(long_text)],
            embedding=FakeEmb(),
            backend=KnowledgeBackend.MEMORY,
        )
        await k.ingest()
        assert await k._store.count() >= 1
        await k.clear()
        assert await k._store.count() == 0

    @pytest.mark.asyncio
    async def test_stats_returns_dict(self) -> None:
        """stats() returns chunk count and source count."""
        FakeEmb = _make_fake_embedding()
        long_text = "Stats test. " * 20
        k = Knowledge(
            sources=[Knowledge.Text(long_text)],
            embedding=FakeEmb(),
            backend=KnowledgeBackend.MEMORY,
        )
        await k.ingest()
        s = await k.stats()
        assert "chunk_count" in s
        assert "source_count" in s
        assert s["source_count"] == 1
        assert s["chunk_count"] >= 0


class TestKnowledgeSourceConstructors:
    """Static source constructors remain unchanged."""

    def test_text_returns_raw_text_loader(self) -> None:
        """Knowledge.Text returns RawTextLoader."""
        loader = Knowledge.Text("hello")
        assert isinstance(loader, RawTextLoader)
        docs = loader.load()
        assert len(docs) == 1
        assert docs[0].content == "hello"

    def test_texts_returns_raw_text_loader(self) -> None:
        """Knowledge.Texts returns RawTextLoader with multiple texts."""
        loader = Knowledge.Texts(["a", "b"])
        docs = loader.load()
        assert len(docs) == 2
