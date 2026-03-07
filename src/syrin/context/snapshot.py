"""Context snapshot types: full view of context window for visibility and export."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Literal


class ContextSegmentSource(StrEnum):
    """Source of a context segment. Used for provenance and message preview."""

    SYSTEM = "system"
    MEMORY = "memory"
    CONVERSATION = "conversation"
    TOOLS = "tools"
    CURRENT_PROMPT = "current_prompt"
    INJECTED = "injected"
    PULLED = "pulled"


ContextRotRisk = Literal["low", "medium", "high"]

# Default utilization thresholds for context rot risk (research-backed).
_CONTEXT_ROT_LOW_PCT = 60.0
_CONTEXT_ROT_HIGH_PCT = 70.0


def _context_rot_risk_from_utilization(utilization_pct: float) -> ContextRotRisk:
    """Derive context rot risk from utilization percentage."""
    if utilization_pct < _CONTEXT_ROT_LOW_PCT:
        return "low"
    if utilization_pct < _CONTEXT_ROT_HIGH_PCT:
        return "medium"
    return "high"


@dataclass
class ContextBreakdown:
    """Token counts by component (system, tools, memory, conversation messages, injected)."""

    system_tokens: int = 0
    """Tokens in system prompt."""
    tools_tokens: int = 0
    """Tokens in tool definitions."""
    memory_tokens: int = 0
    """Tokens in recalled memory block."""
    messages_tokens: int = 0
    """Tokens in conversation + current user message."""
    injected_tokens: int = 0
    """Tokens in runtime-injected context (RAG, dynamic blocks)."""

    @property
    def total_tokens(self) -> int:
        """Total tokens across all components."""
        return (
            self.system_tokens
            + self.tools_tokens
            + self.memory_tokens
            + self.messages_tokens
            + self.injected_tokens
        )


@dataclass
class MessagePreview:
    """Single message preview: role, snippet, token count, and source."""

    role: str
    content_snippet: str
    token_count: int
    source: ContextSegmentSource


@dataclass
class ContextSegmentProvenance:
    """Provenance for one segment: where it came from and optional detail."""

    segment_id: str
    source: ContextSegmentSource
    source_detail: str | None = None


@dataclass
class ContextSnapshot:
    """Point-in-time view of context sent to the LLM.

    Full view: what is passed, why each part is there, where it came from,
    capacity, and context rot risk. Use for debugging, visualization, and export.
    """

    timestamp: float = field(default_factory=time.time)
    total_tokens: int = 0
    max_tokens: int = 0
    tokens_available: int = 0
    utilization_pct: float = 0.0
    breakdown: ContextBreakdown = field(default_factory=ContextBreakdown)
    compacted: bool = False
    compact_method: str | None = None
    messages_count: int = 0

    message_preview: list[MessagePreview] = field(default_factory=list)
    raw_messages: list[dict[str, Any]] | None = None

    provenance: list[ContextSegmentProvenance] = field(default_factory=list)
    why_included: list[str] = field(default_factory=list)
    context_rot_risk: ContextRotRisk = "low"
    context_mode: str = "full"
    """Context mode used: full, focused, or intelligent."""
    context_mode_dropped_count: int = 0
    """Number of conversation messages dropped by context_mode (focused/intelligent)."""
    pulled_segments: list[dict[str, Any]] = field(default_factory=list)
    """When formation_mode=pull: segments retrieved from context store (content, role, score)."""
    pull_scores: list[float] = field(default_factory=list)
    """Relevance scores for pulled_segments."""
    output_chunks: list[dict[str, Any]] = field(default_factory=list)
    """When store_output_chunks=True: assistant chunks retrieved by relevance (content, role, score)."""
    output_chunk_scores: list[float] = field(default_factory=list)
    """Relevance scores for output_chunks."""

    def to_dict(self, include_raw_messages: bool = False) -> dict[str, Any]:
        """Export snapshot for visualization or logging. JSON-serializable."""
        out: dict[str, Any] = {
            "timestamp": self.timestamp,
            "total_tokens": self.total_tokens,
            "max_tokens": self.max_tokens,
            "tokens_available": self.tokens_available,
            "utilization_pct": self.utilization_pct,
            "breakdown": {
                "system_tokens": self.breakdown.system_tokens,
                "tools_tokens": self.breakdown.tools_tokens,
                "memory_tokens": self.breakdown.memory_tokens,
                "messages_tokens": self.breakdown.messages_tokens,
                "injected_tokens": self.breakdown.injected_tokens,
                "total_tokens": self.breakdown.total_tokens,
            },
            "compacted": self.compacted,
            "compact_method": self.compact_method,
            "messages_count": self.messages_count,
            "message_preview": [
                {
                    "role": p.role,
                    "content_snippet": p.content_snippet,
                    "token_count": p.token_count,
                    "source": p.source.value,
                }
                for p in self.message_preview
            ],
            "provenance": [
                {
                    "segment_id": p.segment_id,
                    "source": p.source.value,
                    "source_detail": p.source_detail,
                }
                for p in self.provenance
            ],
            "why_included": self.why_included,
            "context_rot_risk": self.context_rot_risk,
            "context_mode": self.context_mode,
            "context_mode_dropped_count": self.context_mode_dropped_count,
            "pulled_segments": self.pulled_segments,
            "pull_scores": self.pull_scores,
            "output_chunks": self.output_chunks,
            "output_chunk_scores": self.output_chunk_scores,
        }
        if include_raw_messages and self.raw_messages is not None:
            out["raw_messages"] = self.raw_messages
        return out


__all__ = [
    "ContextBreakdown",
    "ContextRotRisk",
    "ContextSegmentProvenance",
    "ContextSegmentSource",
    "ContextSnapshot",
    "MessagePreview",
    "_context_rot_risk_from_utilization",
]
