"""Tests for KnowledgeStore protocol and implementations.

TDD: Tests define the contract. Run against InMemory first; parametrize for other backends.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import pytest

from syrin.knowledge._chunker import Chunk
from syrin.knowledge._store import MetadataFilter, SearchResult, chunk_id

if TYPE_CHECKING:
    from syrin.knowledge.stores._memory import InMemoryKnowledgeStore


def _make_chunk(
    content: str = "test content",
    document_id: str = "doc1",
    chunk_index: int = 0,
    token_count: int = 2,
    metadata: MetadataFilter | None = None,
) -> Chunk:
    return Chunk(
        content=content,
        metadata=metadata or {},
        document_id=document_id,
        chunk_index=chunk_index,
        token_count=token_count,
    )


def _embed(text: str, dim: int = 4) -> list[float]:
    """Simple deterministic embedding for tests (normalized)."""
    import hashlib

    h = hashlib.sha256(text.encode()).digest()
    vec = [float((b - 128) / 128) for b in h[:dim]]
    norm = math.sqrt(sum(x * x for x in vec)) or 1.0
    return [x / norm for x in vec]


class TestChunkId:
    """Tests for chunk_id helper."""

    def test_chunk_id_format(self) -> None:
        """chunk_id returns document_id::chunk_index."""
        c = _make_chunk(document_id="resume.pdf", chunk_index=3)
        assert chunk_id(c) == "resume.pdf::3"

    def test_chunk_id_stable(self) -> None:
        """Same chunk produces same id."""
        c = _make_chunk(document_id="x", chunk_index=0)
        assert chunk_id(c) == chunk_id(c)


class TestSearchResult:
    """Tests for SearchResult dataclass."""

    def test_valid_search_result(self) -> None:
        """SearchResult accepts valid score and rank."""
        c = _make_chunk()
        r = SearchResult(chunk=c, score=0.9, rank=1)
        assert r.score == 0.9
        assert r.rank == 1

    def test_score_out_of_range_raises(self) -> None:
        """SearchResult rejects score outside [0, 1]."""
        c = _make_chunk()
        with pytest.raises(ValueError, match="score must be in"):
            SearchResult(chunk=c, score=1.5, rank=1)
        with pytest.raises(ValueError, match="score must be in"):
            SearchResult(chunk=c, score=-0.1, rank=1)

    def test_rank_negative_raises(self) -> None:
        """SearchResult rejects rank < 1."""
        c = _make_chunk()
        with pytest.raises(ValueError, match="rank must be >= 1"):
            SearchResult(chunk=c, score=0.5, rank=0)


@pytest.fixture
def in_memory_store() -> InMemoryKnowledgeStore:
    """Create InMemoryKnowledgeStore with 4-dim vectors."""
    from syrin.knowledge.stores._memory import InMemoryKnowledgeStore

    return InMemoryKnowledgeStore(embedding_dimensions=4)


@pytest.fixture
async def populated_store(in_memory_store: InMemoryKnowledgeStore) -> InMemoryKnowledgeStore:
    """Store with 3 chunks from 2 documents."""
    chunks = [
        _make_chunk("first chunk", "doc1", 0, metadata={"source_type": "pdf"}),
        _make_chunk("second chunk", "doc1", 1, metadata={"source_type": "pdf"}),
        _make_chunk("other doc chunk", "doc2", 0, metadata={"source_type": "markdown"}),
    ]
    embeddings = [_embed(c.content) for c in chunks]
    await in_memory_store.upsert(chunks, embeddings)
    return in_memory_store


class TestKnowledgeStoreProtocol:
    """Contract tests for KnowledgeStore. Parametrize over backends when available."""

    @pytest.mark.asyncio
    async def test_upsert_then_count(self, in_memory_store: InMemoryKnowledgeStore) -> None:
        """After upsert, count returns chunk count."""
        chunks = [_make_chunk("a", "d1", 0), _make_chunk("b", "d1", 1)]
        embeddings = [_embed("a"), _embed("b")]
        await in_memory_store.upsert(chunks, embeddings)
        assert await in_memory_store.count() == 2

    @pytest.mark.asyncio
    async def test_upsert_empty_list(self, in_memory_store: InMemoryKnowledgeStore) -> None:
        """Upsert empty list is valid; count stays 0."""
        await in_memory_store.upsert([], [])
        assert await in_memory_store.count() == 0

    @pytest.mark.asyncio
    async def test_upsert_mismatched_lengths_raises(
        self, in_memory_store: InMemoryKnowledgeStore
    ) -> None:
        """Upsert raises when chunks and embeddings lengths differ."""
        chunks = [_make_chunk("a", "d1", 0)]
        embeddings = [_embed("a"), _embed("b")]
        with pytest.raises(ValueError, match="length"):
            await in_memory_store.upsert(chunks, embeddings)

    @pytest.mark.asyncio
    async def test_upsert_idempotent(self, in_memory_store: InMemoryKnowledgeStore) -> None:
        """Upsert same chunk twice updates (no duplicate)."""
        c = _make_chunk("same", "d1", 0)
        emb = _embed("same")
        await in_memory_store.upsert([c], [emb])
        await in_memory_store.upsert([c], [emb])
        assert await in_memory_store.count() == 1

    @pytest.mark.asyncio
    async def test_search_returns_results(self, populated_store: InMemoryKnowledgeStore) -> None:
        """Search returns list of SearchResult with scores and ranks."""
        q_emb = _embed("first chunk")
        results = await populated_store.search(q_emb, top_k=3)
        assert len(results) <= 3
        for i, r in enumerate(results):
            assert isinstance(r, SearchResult)
            assert 0 <= r.score <= 1
            assert r.rank == i + 1
            assert isinstance(r.chunk, Chunk)

    @pytest.mark.asyncio
    async def test_search_top_k_zero(self, populated_store: InMemoryKnowledgeStore) -> None:
        """Search with top_k=0 returns empty list."""
        results = await populated_store.search(_embed("x"), top_k=0)
        assert results == []

    @pytest.mark.asyncio
    async def test_search_respects_score_threshold(
        self, populated_store: InMemoryKnowledgeStore
    ) -> None:
        """Results below score_threshold are excluded."""
        q_emb = _embed("totally unrelated gibberish xyz")
        results = await populated_store.search(q_emb, top_k=5, score_threshold=0.99)
        assert all(r.score >= 0.99 for r in results)

    @pytest.mark.asyncio
    async def test_search_with_filter(self, populated_store: InMemoryKnowledgeStore) -> None:
        """Filter restricts results by metadata."""
        q_emb = _embed("chunk")
        results = await populated_store.search(q_emb, top_k=5, filter={"source_type": "markdown"})
        assert len(results) <= 2
        for r in results:
            assert r.chunk.metadata.get("source_type") == "markdown"

    @pytest.mark.asyncio
    async def test_delete_by_document_id(self, populated_store: InMemoryKnowledgeStore) -> None:
        """Delete by document_id removes matching chunks and returns count."""
        n = await populated_store.delete(document_id="doc1")
        assert n == 2
        assert await populated_store.count() == 1

    @pytest.mark.asyncio
    async def test_delete_by_source(self, populated_store: InMemoryKnowledgeStore) -> None:
        """Delete by source removes matching chunks (source = document_id)."""
        n = await populated_store.delete(source="doc2")
        assert n == 1
        assert await populated_store.count() == 2

    @pytest.mark.asyncio
    async def test_delete_nonexistent_returns_zero(
        self, populated_store: InMemoryKnowledgeStore
    ) -> None:
        """Delete with no matching chunks returns 0."""
        n = await populated_store.delete(document_id="nonexistent")
        assert n == 0
        assert await populated_store.count() == 3

    @pytest.mark.asyncio
    async def test_delete_requires_source_or_document_id(
        self, in_memory_store: InMemoryKnowledgeStore
    ) -> None:
        """Delete raises when both source and document_id are None."""
        with pytest.raises(ValueError, match="source or document_id"):
            await in_memory_store.delete()

    @pytest.mark.asyncio
    async def test_count_empty_store(self, in_memory_store: InMemoryKnowledgeStore) -> None:
        """Empty store returns count 0."""
        assert await in_memory_store.count() == 0


class TestChromaKnowledgeStore:
    """Tests for ChromaKnowledgeStore when chromadb is available."""

    @pytest.fixture
    def chroma_store(self):
        try:
            from syrin.knowledge.stores._chroma import ChromaKnowledgeStore

            return ChromaKnowledgeStore(
                embedding_dimensions=4,
                collection_name="syrin_test_" + str(id(self)),
                path=None,
            )
        except ImportError:
            pytest.skip("chromadb not installed")

    @pytest.mark.asyncio
    async def test_chroma_upsert_search_delete(self, chroma_store) -> None:
        """Chroma store supports full upsert/search/delete/count cycle."""
        chunks = [
            _make_chunk("alpha", "d1", 0, metadata={"source_type": "txt"}),
            _make_chunk("beta", "d1", 1, metadata={"source_type": "txt"}),
        ]
        embs = [_embed("alpha"), _embed("beta")]
        await chroma_store.upsert(chunks, embs)
        assert await chroma_store.count() == 2
        results = await chroma_store.search(_embed("alpha"), top_k=2)
        assert len(results) >= 1
        assert results[0].chunk.content == "alpha"
        n = await chroma_store.delete(document_id="d1")
        assert n == 2
        assert await chroma_store.count() == 0
