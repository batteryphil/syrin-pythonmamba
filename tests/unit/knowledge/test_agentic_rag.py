"""Tests for Agentic RAG (Step 6): AgenticRAGConfig, decompose, grade, refine, verify."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from syrin.knowledge._agentic import (
    AgenticRAGConfig,
    _parse_grade,
    _parse_sub_queries,
    _parse_verdict,
    grade_results,
    refine_query,
    verify_claim,
)
from syrin.knowledge._chunker import Chunk
from syrin.knowledge._store import SearchResult


def _make_fake_chunk(content: str, doc_id: str = "test", idx: int = 0) -> Chunk:
    return Chunk(
        content=content,
        metadata={},
        document_id=doc_id,
        chunk_index=idx,
        token_count=max(1, len(content) // 4),
    )


def _make_fake_search_result(chunk: Chunk, score: float = 0.8, rank: int = 1) -> SearchResult:
    return SearchResult(chunk=chunk, score=score, rank=rank)


class TestAgenticRAGConfig:
    """AgenticRAGConfig validation."""

    def test_defaults(self) -> None:
        cfg = AgenticRAGConfig()
        assert cfg.max_search_iterations == 3
        assert cfg.decompose_complex is True
        assert cfg.grade_results is True
        assert cfg.relevance_threshold == 0.5
        assert cfg.web_fallback is False
        assert cfg.search_model is None

    def test_max_search_iterations_min(self) -> None:
        with pytest.raises(ValueError, match="max_search_iterations"):
            AgenticRAGConfig(max_search_iterations=0)

    def test_max_search_iterations_negative(self) -> None:
        with pytest.raises(ValueError, match="max_search_iterations"):
            AgenticRAGConfig(max_search_iterations=-1)

    def test_relevance_threshold_too_low(self) -> None:
        with pytest.raises(ValueError, match="relevance_threshold"):
            AgenticRAGConfig(relevance_threshold=-0.1)

    def test_relevance_threshold_too_high(self) -> None:
        with pytest.raises(ValueError, match="relevance_threshold"):
            AgenticRAGConfig(relevance_threshold=1.5)

    def test_relevance_threshold_boundary_zero(self) -> None:
        cfg = AgenticRAGConfig(relevance_threshold=0.0)
        assert cfg.relevance_threshold == 0.0

    def test_relevance_threshold_boundary_one(self) -> None:
        cfg = AgenticRAGConfig(relevance_threshold=1.0)
        assert cfg.relevance_threshold == 1.0


class TestParseSubQueries:
    """_parse_sub_queries edge cases."""

    def test_empty_string(self) -> None:
        assert _parse_sub_queries("") == []
        assert _parse_sub_queries("   \n  ") == []

    def test_single_line(self) -> None:
        assert _parse_sub_queries("What is Python?") == ["What is Python?"]

    def test_multiple_lines(self) -> None:
        text = "What is Python?\nHow does it work?\nWhere is it used?"
        assert _parse_sub_queries(text) == [
            "What is Python?",
            "How does it work?",
            "Where is it used?",
        ]

    def test_numbered_lines(self) -> None:
        text = "1. What is Python?\n2. How does it work?"
        assert _parse_sub_queries(text) == ["What is Python?", "How does it work?"]

    def test_bullet_prefixes(self) -> None:
        text = "- Q1\n* Q2\n• Q3"
        result = _parse_sub_queries(text)
        assert len(result) == 3
        assert "Q1" in result[0]
        assert "Q2" in result[1]
        assert "Q3" in result[2]

    def test_caps_at_five(self) -> None:
        text = "Q1\nQ2\nQ3\nQ4\nQ5\nQ6"
        result = _parse_sub_queries(text)
        assert len(result) == 5
        assert "Q6" not in result


class TestParseGrade:
    """_parse_grade edge cases."""

    def test_empty(self) -> None:
        assert _parse_grade("") == 0.0

    def test_simple_float(self) -> None:
        assert _parse_grade("0.75") == 0.75
        assert _parse_grade("0.5") == 0.5

    def test_with_text(self) -> None:
        assert _parse_grade("The relevance is 0.82") == 0.82

    def test_integer(self) -> None:
        assert _parse_grade("1") == 1.0

    def test_out_of_range_clamped(self) -> None:
        assert _parse_grade("1.5") == 1.0
        assert _parse_grade("2.0") == 1.0

    def test_no_number_returns_zero(self) -> None:
        assert _parse_grade("irrelevant") == 0.0


class TestParseVerdict:
    """_parse_verdict edge cases."""

    def test_empty(self) -> None:
        assert _parse_verdict("") == "NOT_FOUND"

    def test_supported(self) -> None:
        assert _parse_verdict("SUPPORTED") == "SUPPORTED"
        assert _parse_verdict("The evidence SUPPORTED the claim") == "SUPPORTED"

    def test_contradicted(self) -> None:
        assert _parse_verdict("CONTRADICTED") == "CONTRADICTED"
        assert _parse_verdict("It CONTRADICTED the claim") == "CONTRADICTED"

    def test_not_found(self) -> None:
        assert _parse_verdict("NOT_FOUND") == "NOT_FOUND"
        assert _parse_verdict("No information") == "NOT_FOUND"


class TestGradeResults:
    """grade_results with mocked model."""

    @pytest.mark.asyncio
    async def test_empty_results_returns_zero(self) -> None:
        model = MagicMock()
        model.acomplete = AsyncMock()
        score = await grade_results("query", [], model)
        assert score == 0.0
        model.acomplete.assert_not_called()

    @pytest.mark.asyncio
    async def test_calls_model_and_parses(self) -> None:
        model = MagicMock()
        model.acomplete = AsyncMock(return_value=MagicMock(content="0.85"))
        chunk = _make_fake_chunk("content")
        results = [_make_fake_search_result(chunk)]
        score = await grade_results("query", results, model)
        assert score == 0.85
        model.acomplete.assert_called_once()


class TestRefineQuery:
    """refine_query with mocked model."""

    @pytest.mark.asyncio
    async def test_returns_refined_query(self) -> None:
        model = MagicMock()
        model.acomplete = AsyncMock(return_value=MagicMock(content="Improved query"))
        chunk = _make_fake_chunk("content")
        results = [_make_fake_search_result(chunk)]
        refined = await refine_query("original", results, model)
        assert refined == "Improved query"

    @pytest.mark.asyncio
    async def test_empty_response_fallback_to_original(self) -> None:
        model = MagicMock()
        model.acomplete = AsyncMock(return_value=MagicMock(content="   "))
        chunk = _make_fake_chunk("content")
        results = [_make_fake_search_result(chunk)]
        refined = await refine_query("original query", results, model)
        assert refined == "original query"


class TestVerifyClaim:
    """verify_claim with mocked model."""

    @pytest.mark.asyncio
    async def test_empty_results_returns_not_found(self) -> None:
        model = MagicMock()
        model.acomplete = AsyncMock()
        verdict = await verify_claim("claim", [], model)
        assert verdict == "NOT_FOUND"
        model.acomplete.assert_not_called()

    @pytest.mark.asyncio
    async def test_returns_supported(self) -> None:
        model = MagicMock()
        model.acomplete = AsyncMock(return_value=MagicMock(content="SUPPORTED"))
        chunk = _make_fake_chunk("evidence")
        results = [_make_fake_search_result(chunk)]
        verdict = await verify_claim("claim", results, model)
        assert verdict == "SUPPORTED"
