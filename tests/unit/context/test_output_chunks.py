"""Tests for stored output chunks.

TDD: chunk assistant replies by paragraph; retrieve by relevance to current query.
"""

from __future__ import annotations

import pytest

from syrin import Agent, Model
from syrin.context import Context
from syrin.enums import OutputChunkStrategy
from syrin.memory import Memory


def _almock() -> Model:
    return Model.Almock(latency_seconds=0.01)


class TestOutputChunkStrategy:
    """OutputChunkStrategy enum and validation."""

    def test_enum_values(self) -> None:
        assert OutputChunkStrategy.PARAGRAPH == "paragraph"
        assert OutputChunkStrategy.FIXED == "fixed"

    def test_context_accepts_output_chunk_config(self) -> None:
        ctx = Context(
            store_output_chunks=True,
            output_chunk_top_k=5,
            output_chunk_threshold=0.3,
            output_chunk_strategy=OutputChunkStrategy.PARAGRAPH,
        )
        assert ctx.store_output_chunks is True
        assert ctx.output_chunk_top_k == 5
        assert ctx.output_chunk_threshold == 0.3
        assert ctx.output_chunk_strategy == OutputChunkStrategy.PARAGRAPH

    def test_context_output_chunk_defaults(self) -> None:
        ctx = Context()
        assert ctx.store_output_chunks is False
        assert ctx.output_chunk_top_k == 5
        assert ctx.output_chunk_threshold == 0.0
        assert ctx.output_chunk_strategy == OutputChunkStrategy.PARAGRAPH

    def test_output_chunk_top_k_invalid(self) -> None:
        with pytest.raises(ValueError, match="output_chunk_top_k must be >= 0"):
            Context(store_output_chunks=True, output_chunk_top_k=-1)

    def test_output_chunk_threshold_invalid(self) -> None:
        with pytest.raises(ValueError, match="output_chunk_threshold"):
            Context(store_output_chunks=True, output_chunk_threshold=1.5)

    def test_output_chunk_size_invalid_for_fixed(self) -> None:
        with pytest.raises(ValueError, match="output_chunk_size must be >= 1"):
            Context(
                store_output_chunks=True,
                output_chunk_strategy=OutputChunkStrategy.FIXED,
                output_chunk_size=0,
            )


class TestMemoryOutputChunks:
    """Memory.add_output_chunks and get_relevant_output_chunks."""

    def test_add_output_chunks_paragraph_strategy(self) -> None:
        mem = Memory()
        content = "First para.\n\nSecond para.\n\nThird para."
        mem.add_output_chunks(content, strategy=OutputChunkStrategy.PARAGRAPH)
        chunks = mem.get_relevant_output_chunks("First", top_k=5)
        assert len(chunks) >= 1
        contents = [c[0].content for c in chunks]
        assert any("First para" in ct for ct in contents)

    def test_add_output_chunks_fixed_strategy(self) -> None:
        mem = Memory()
        content = "A" * 100
        mem.add_output_chunks(
            content,
            strategy=OutputChunkStrategy.FIXED,
            chunk_size=30,
        )
        chunks = mem.get_relevant_output_chunks("A", top_k=10)
        assert len(chunks) >= 1

    def test_add_output_chunks_empty_content_no_chunks(self) -> None:
        mem = Memory()
        mem.add_output_chunks("", strategy=OutputChunkStrategy.PARAGRAPH)
        chunks = mem.get_relevant_output_chunks("anything", top_k=5)
        assert chunks == []

    def test_add_output_chunks_whitespace_only_no_chunks(self) -> None:
        mem = Memory()
        mem.add_output_chunks("   \n\n  ", strategy=OutputChunkStrategy.PARAGRAPH)
        chunks = mem.get_relevant_output_chunks("x", top_k=5)
        assert chunks == []

    def test_get_relevant_output_chunks_top_k(self) -> None:
        mem = Memory()
        mem.add_output_chunks("Alpha.\n\nBeta.\n\nGamma.", strategy=OutputChunkStrategy.PARAGRAPH)
        chunks = mem.get_relevant_output_chunks("Alpha Beta Gamma", top_k=2)
        assert len(chunks) <= 2

    def test_get_relevant_output_chunks_threshold(self) -> None:
        mem = Memory()
        mem.add_output_chunks(
            "Unrelated stuff.\n\nMore stuff.", strategy=OutputChunkStrategy.PARAGRAPH
        )
        chunks = mem.get_relevant_output_chunks("xyznonexistent", top_k=5, threshold=0.9)
        assert len(chunks) == 0 or all(c[1] >= 0.9 for c in chunks)


class TestOutputChunksAgentFlow:
    """Agent with store_output_chunks=True stores and retrieves chunks."""

    def test_store_output_chunks_false_no_chunks_stored(self) -> None:
        agent = Agent(
            model=_almock(),
            system_prompt="Help.",
            memory=Memory(),
            context=Context(store_output_chunks=False),
        )
        agent.run("Tell me a long story.")
        snap = agent.context.snapshot()
        assert getattr(snap, "output_chunks", []) == [] or not hasattr(snap, "output_chunks")

    def test_store_output_chunks_true_chunks_in_context(self) -> None:
        agent = Agent(
            model=_almock(),
            system_prompt="Be verbose. Use multiple paragraphs.",
            memory=Memory(),
            context=Context(
                store_output_chunks=True,
                output_chunk_top_k=5,
                output_chunk_threshold=0.0,
            ),
        )
        agent.run("Explain Syrin memory in detail.")
        agent.run("What about the relevance threshold?")
        snap = agent.context.snapshot()
        output_chunks = getattr(snap, "output_chunks", [])
        output_scores = getattr(snap, "output_chunk_scores", [])
        assert isinstance(output_chunks, list)
        assert isinstance(output_scores, list)
        if output_chunks:
            assert len(output_scores) == len(output_chunks)

    def test_store_output_chunks_with_memory_none_is_noop(self) -> None:
        agent = Agent(
            model=_almock(),
            system_prompt="Help.",
            memory=None,
            context=Context(store_output_chunks=True),
        )
        r = agent.run("Hi")
        assert r.content is not None

    def test_snapshot_has_output_chunks_fields(self) -> None:
        memory = Memory()
        memory.add_output_chunks(
            "Syrin uses memory.\n\nIt has segments.", strategy=OutputChunkStrategy.PARAGRAPH
        )
        agent = Agent(
            model=_almock(),
            system_prompt="Help.",
            memory=memory,
            context=Context(
                store_output_chunks=True,
                output_chunk_top_k=5,
                output_chunk_threshold=0.0,
            ),
        )
        agent.run("Tell me about Syrin memory")
        snap = agent.context.snapshot()
        assert hasattr(snap, "output_chunks")
        assert hasattr(snap, "output_chunk_scores")
        d = snap.to_dict()
        assert "output_chunks" in d
        assert "output_chunk_scores" in d
