"""Context store for pull-based context formation (Step 10) and output chunks (Step 11).

Store conversation segments; retrieve by relevance to the current query.
Memory uses InMemoryContextStore internally for formation_mode=PULL and output chunks.
ContextStore/InMemoryContextStore remain available for custom implementations.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass
class ContextSegment:
    """A single conversation segment stored for pull retrieval.

    Attributes:
        content: Message content text.
        role: Message role (user, assistant, etc.).
        turn_id: Optional turn index for ordering.
        embedding: Optional precomputed embedding vector.
    """

    content: str
    role: str = "user"
    turn_id: int | None = None
    embedding: list[float] | None = None


@runtime_checkable
class RelevanceScorer(Protocol):
    """Protocol for scoring segment relevance to a query.

    Implement for custom scoring (e.g. embedding-based).
    Default: SimpleTextScorer (token overlap, no deps).
    """

    def score(
        self,
        query: str,
        segments: Sequence[ContextSegment],
    ) -> list[tuple[ContextSegment, float]]:
        """Score segments by relevance to query. Higher = more relevant."""
        ...


class SimpleTextScorer:
    """Token-overlap scorer. No embeddings; works without extra deps.

    Uses Jaccard-like similarity on word tokens.
    """

    def score(
        self,
        query: str,
        segments: Sequence[ContextSegment],
    ) -> list[tuple[ContextSegment, float]]:
        """Score each segment by token overlap with query."""
        if not segments:
            return []
        q_tokens = set(_tokenize(query.lower()))
        if not q_tokens:
            return [(s, 0.0) for s in segments]

        result: list[tuple[ContextSegment, float]] = []
        for seg in segments:
            s_tokens = set(_tokenize(seg.content.lower()))
            if not s_tokens:
                result.append((seg, 0.0))
                continue
            overlap = len(q_tokens & s_tokens) / len(q_tokens | s_tokens)
            result.append((seg, float(overlap)))
        return result


def _tokenize(text: str) -> list[str]:
    """Split text into word tokens."""
    return re.findall(r"\b\w+\b", text)


def chunk_assistant_content(
    content: str,
    strategy: str = "paragraph",
    chunk_size: int = 300,
) -> list[str]:
    """Split assistant content into chunks for relevance retrieval.

    Args:
        content: Full assistant reply text.
        strategy: "paragraph" (split on \\n\\n) or "fixed" (by chunk_size chars).
        chunk_size: Character size per chunk when strategy="fixed". Ignored for paragraph.

    Returns:
        List of non-empty chunk strings. Empty list if content is empty or whitespace-only.
    """
    text = (content or "").strip()
    if not text:
        return []

    if strategy == "fixed":
        chunks: list[str] = []
        for i in range(0, len(text), chunk_size):
            piece = text[i : i + chunk_size]
            if piece.strip():
                chunks.append(piece)
        return chunks

    # paragraph (default)
    raw = re.split(r"\n\s*\n", text)
    return [p.strip() for p in raw if p.strip()]


@runtime_checkable
class ContextStore(Protocol):
    """Protocol for pull-based context storage.

    Store conversation segments; retrieve by relevance.
    """

    def add_segment(self, segment: ContextSegment) -> None:
        """Add a segment to the store."""
        ...

    def get_relevant(
        self,
        query: str,
        top_k: int = 10,
        threshold: float = 0.0,
    ) -> list[tuple[ContextSegment, float]]:
        """Return segments most relevant to query, with scores.

        Args:
            query: Current user prompt (or query text).
            top_k: Maximum number of segments to return.
            threshold: Minimum score (0.0-1.0). Segments below are excluded.

        Returns:
            List of (segment, score) sorted by score descending.
        """
        ...

    def list_recent(self, n: int = 10) -> list[ContextSegment]:
        """Return the n most recently added segments (oldest first for ordering)."""
        ...

    def clear(self) -> None:
        """Remove all segments. Optional."""
        ...


class InMemoryContextStore:
    """In-memory context store with pluggable relevance scorer.

    Use for pull-based context formation. Segments are stored in order;
    get_relevant uses the scorer to rank by relevance to the query.
    """

    def __init__(self, scorer: RelevanceScorer | None = None) -> None:
        """Create store. Uses SimpleTextScorer if scorer not provided."""
        self._segments: list[ContextSegment] = []
        self._scorer = scorer if scorer is not None else SimpleTextScorer()

    def add_segment(self, segment: ContextSegment) -> None:
        """Add a segment to the store."""
        self._segments.append(segment)

    def get_relevant(
        self,
        query: str,
        top_k: int = 10,
        threshold: float = 0.0,
    ) -> list[tuple[ContextSegment, float]]:
        """Return segments most relevant to query, above threshold, up to top_k."""
        if not self._segments:
            return []
        top_k = max(0, top_k)
        if top_k == 0:
            return []

        scored = self._scorer.score(query, self._segments)
        filtered = [(s, sc) for s, sc in scored if sc >= threshold]
        filtered.sort(key=lambda x: x[1], reverse=True)
        return filtered[:top_k]

    def list_recent(self, n: int = 10) -> list[ContextSegment]:
        """Return the n most recently added segments (chronological order)."""
        return self._segments[-n:] if n > 0 else []

    def clear(self) -> None:
        """Remove all segments."""
        self._segments.clear()


__all__ = [
    "ContextSegment",
    "ContextStore",
    "InMemoryContextStore",
    "RelevanceScorer",
    "SimpleTextScorer",
    "chunk_assistant_content",
]
