"""Persistent context map (Step 12).

Durable index/summary (topics, decisions, segment pointers) that survives
context resets. Drives retrieval; optional file or Protocol backend.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable


@dataclass
class ContextMap:
    """Durable index for context formation. Survives resets; drives retrieval.

    Use for session summaries, topic tracking, and segment pointers.
    Update via update_map(); inject summary at prepare when inject_map_summary=True.

    Attributes:
        topics: Topic labels (e.g. ["Syrin memory", "budget limits"]).
        decisions: Key decisions or facts (e.g. ["User prefers Python"]).
        segment_ids: Opaque IDs pointing to stored segments (for custom backends).
        summary: Session or conversation summary string.
        last_updated: Unix timestamp of last update.
    """

    topics: list[str] = field(default_factory=list)
    decisions: list[str] = field(default_factory=list)
    segment_ids: list[str] = field(default_factory=list)
    summary: str = ""
    last_updated: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable dict for persistence."""
        return {
            "topics": list(self.topics),
            "decisions": list(self.decisions),
            "segment_ids": list(self.segment_ids),
            "summary": self.summary,
            "last_updated": self.last_updated,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> ContextMap:
        """Load from dict (e.g. JSON). Returns empty map if data is None or invalid."""
        if not data or not isinstance(data, dict):
            return ContextMap()
        return cls(
            topics=list(data.get("topics", [])) if isinstance(data.get("topics"), list) else [],
            decisions=(
                list(data.get("decisions", [])) if isinstance(data.get("decisions"), list) else []
            ),
            segment_ids=(
                list(data.get("segment_ids", []))
                if isinstance(data.get("segment_ids"), list)
                else []
            ),
            summary=str(data.get("summary", "")),
            last_updated=float(data.get("last_updated", 0)),
        )


@runtime_checkable
class ContextMapBackend(Protocol):
    """Protocol for persisting ContextMap. Implement for custom backends."""

    def load(self) -> ContextMap:
        """Load map from storage. Returns empty map if not found."""
        ...

    def save(self, m: ContextMap) -> None:
        """Persist map."""
        ...


class FileContextMapBackend:
    """File-based persistence for ContextMap. Uses JSON."""

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)

    def load(self) -> ContextMap:
        """Load from file. Returns empty map if file missing or invalid."""
        if not self._path.exists():
            return ContextMap()
        try:
            data = json.loads(self._path.read_text())
            return ContextMap.from_dict(data)
        except (json.JSONDecodeError, OSError):
            return ContextMap()

    def save(self, m: ContextMap) -> None:
        """Write to file. Creates parent dirs if needed."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(json.dumps(m.to_dict(), indent=2))


__all__ = [
    "ContextMap",
    "ContextMapBackend",
    "FileContextMapBackend",
]
