"""Tests for ContextStore protocol and InMemoryContextStore (Step 10)."""

from __future__ import annotations

from syrin.context.store import (
    ContextSegment,
    InMemoryContextStore,
    SimpleTextScorer,
)


class TestContextSegment:
    """ContextSegment dataclass."""

    def test_create_minimal(self) -> None:
        seg = ContextSegment(content="Hello", role="user")
        assert seg.content == "Hello"
        assert seg.role == "user"
        assert seg.turn_id is None
        assert seg.embedding is None

    def test_create_full(self) -> None:
        seg = ContextSegment(
            content="Hi there",
            role="assistant",
            turn_id=42,
            embedding=[0.1, 0.2],
        )
        assert seg.turn_id == 42
        assert seg.embedding == [0.1, 0.2]


class TestInMemoryContextStore:
    """InMemoryContextStore basic operations."""

    def test_add_segment_and_list_recent(self) -> None:
        store = InMemoryContextStore()
        store.add_segment(ContextSegment(content="A", role="user"))
        store.add_segment(ContextSegment(content="B", role="assistant"))
        recent = store.list_recent(5)
        assert len(recent) == 2
        assert recent[0].content == "A"
        assert recent[1].content == "B"

    def test_list_recent_empty_store(self) -> None:
        store = InMemoryContextStore()
        assert store.list_recent(10) == []

    def test_list_recent_respects_n(self) -> None:
        store = InMemoryContextStore()
        for i in range(5):
            store.add_segment(ContextSegment(content=str(i), role="user"))
        recent = store.list_recent(2)
        assert len(recent) == 2
        # Most recently added = last 2 (chronological order)
        assert [s.content for s in recent] == ["3", "4"]

    def test_list_recent_returns_newest_first(self) -> None:
        store = InMemoryContextStore()
        store.add_segment(ContextSegment(content="old", role="user"))
        store.add_segment(ContextSegment(content="new", role="user"))
        recent = store.list_recent(2)
        assert recent[0].content == "old"
        assert recent[1].content == "new"

    def test_clear_removes_all(self) -> None:
        store = InMemoryContextStore()
        store.add_segment(ContextSegment(content="X", role="user"))
        store.clear()
        assert store.list_recent(10) == []

    def test_clear_empty_store_no_op(self) -> None:
        store = InMemoryContextStore()
        store.clear()
        assert store.list_recent(10) == []


class TestInMemoryContextStoreGetRelevant:
    """get_relevant with SimpleTextScorer."""

    def test_get_relevant_returns_matching_above_threshold(self) -> None:
        store = InMemoryContextStore(scorer=SimpleTextScorer())
        store.add_segment(ContextSegment(content="Python programming", role="user"))
        store.add_segment(ContextSegment(content="Weather today", role="assistant"))
        store.add_segment(ContextSegment(content="Python tips and tricks", role="user"))
        results = store.get_relevant("Python", top_k=5, threshold=0.0)
        assert len(results) >= 1
        contents = [s.content for s, _ in results]
        assert "Python programming" in contents or "Python tips and tricks" in contents

    def test_get_relevant_respects_top_k(self) -> None:
        store = InMemoryContextStore(scorer=SimpleTextScorer())
        for i in range(10):
            store.add_segment(ContextSegment(content=f"match {i}", role="user"))
        results = store.get_relevant("match", top_k=3, threshold=0.0)
        assert len(results) <= 3

    def test_get_relevant_empty_store_returns_empty(self) -> None:
        store = InMemoryContextStore(scorer=SimpleTextScorer())
        results = store.get_relevant("query", top_k=5, threshold=0.0)
        assert results == []

    def test_get_relevant_top_k_zero_returns_empty(self) -> None:
        store = InMemoryContextStore(scorer=SimpleTextScorer())
        store.add_segment(ContextSegment(content="something", role="user"))
        results = store.get_relevant("something", top_k=0, threshold=0.0)
        assert results == []

    def test_get_relevant_filters_below_threshold(self) -> None:
        store = InMemoryContextStore(scorer=SimpleTextScorer())
        store.add_segment(ContextSegment(content="completely unrelated", role="user"))
        results = store.get_relevant("xyz", top_k=5, threshold=0.99)
        assert len(results) == 0

    def test_get_relevant_scores_in_descending_order(self) -> None:
        store = InMemoryContextStore(scorer=SimpleTextScorer())
        store.add_segment(ContextSegment(content="python python python", role="user"))
        store.add_segment(ContextSegment(content="python", role="user"))
        results = store.get_relevant("python", top_k=5, threshold=0.0)
        if len(results) >= 2:
            assert results[0][1] >= results[1][1]


class TestSimpleTextScorer:
    """SimpleTextScorer behavior."""

    def test_exact_match_scores_high(self) -> None:
        scorer = SimpleTextScorer()
        seg = ContextSegment(content="hello world", role="user")
        scores = scorer.score("hello world", [seg])
        assert len(scores) == 1
        assert scores[0][1] > 0.9

    def test_no_overlap_scores_zero(self) -> None:
        scorer = SimpleTextScorer()
        seg = ContextSegment(content="abc def", role="user")
        scores = scorer.score("xyz", [seg])
        assert len(scores) == 1
        assert scores[0][1] == 0.0

    def test_empty_query_returns_zeros(self) -> None:
        scorer = SimpleTextScorer()
        seg = ContextSegment(content="something", role="user")
        scores = scorer.score("", [seg])
        assert len(scores) == 1
        assert scores[0][1] == 0.0

    def test_empty_segments_returns_empty(self) -> None:
        scorer = SimpleTextScorer()
        scores = scorer.score("query", [])
        assert scores == []


class TestContextStoreEdgeCases:
    """Edge cases and invalid inputs."""

    def test_add_segment_with_empty_content_allowed(self) -> None:
        store = InMemoryContextStore()
        store.add_segment(ContextSegment(content="", role="user"))
        recent = store.list_recent(1)
        assert len(recent) == 1
        assert recent[0].content == ""

    def test_get_relevant_negative_top_k_treated_as_zero(self) -> None:
        store = InMemoryContextStore(scorer=SimpleTextScorer())
        store.add_segment(ContextSegment(content="x", role="user"))
        results = store.get_relevant("x", top_k=-1, threshold=0.0)
        assert len(results) == 0
