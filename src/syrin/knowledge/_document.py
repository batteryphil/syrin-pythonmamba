"""Document model for Knowledge module."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TypeAlias

# Metadata values used by loaders: primitives and list of str (e.g. for tags).
# Avoids Any while supporting common keys: title, page, section_index, repo, etc.
DocumentMetadata: TypeAlias = dict[
    str,
    str | int | float | bool | None | list[str],
]


@dataclass(frozen=True)
class Document:
    """Immutable loaded document with content, source identity, and optional metadata.

    Documents are the fundamental unit of content in the Knowledge module.
    They are created by DocumentLoaders and later chunked for embedding.

    Attributes:
        content: The text content of the document.
        source: Source identifier (e.g. file path, "user_provided", "github/org/repo").
        source_type: Type of source (e.g. "pdf", "markdown", "github", "url", "text").
        metadata: Optional extra metadata (page, section, title, etc.). Does not
            duplicate source or source_type.

    Example:
        doc = Document(
            content="This is the document text.",
            source="resume.pdf",
            source_type="pdf",
            metadata={"page": 1, "total_pages": 3},
        )
    """

    content: str
    source: str
    source_type: str
    metadata: DocumentMetadata = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate required fields."""
        if not self.source:
            raise ValueError("Document.source must be non-empty")
        if not self.source_type:
            raise ValueError("Document.source_type must be non-empty")
