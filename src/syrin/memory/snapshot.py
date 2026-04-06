"""Memory snapshot for export/import."""

from __future__ import annotations

import json
from dataclasses import dataclass, field


@dataclass
class MemorySnapshotEntry:
    """Minimal representation of a memory for export."""

    id: str
    content: str
    type: str
    importance: float = 1.0
    scope: str = "user"
    created_at: str | None = None
    metadata: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "content": self.content,
            "type": self.type,
            "importance": self.importance,
            "scope": self.scope,
            "created_at": self.created_at,
            "metadata": self.metadata,
        }


@dataclass
class MemorySnapshot:
    """Immutable snapshot of memories for export/import."""

    version: int = 1
    memories: list[MemorySnapshotEntry] = field(default_factory=list)
    metadata: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "version": self.version,
            "memories": [m.to_dict() for m in self.memories],
            "metadata": self.metadata,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), default=str)

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> MemorySnapshot:
        memories = []
        for m in data.get("memories", []):  # type: ignore[attr-defined]
            memories.append(
                MemorySnapshotEntry(
                    id=m.get("id", ""),
                    content=m.get("content", ""),
                    type=m.get("type", "history"),
                    importance=m.get("importance", 1.0),
                    scope=m.get("scope", "user"),
                    created_at=m.get("created_at"),
                    metadata=m.get("metadata", {}),
                )
            )
        try:
            ver = int(data.get("version", 1))  # type: ignore[call-overload]
        except (TypeError, ValueError):
            ver = 1
        return cls(
            version=ver,
            memories=memories,
            metadata=data.get("metadata", {}),  # type: ignore[arg-type]
        )
