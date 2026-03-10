"""Markdown loader with header-aware section parsing."""

from __future__ import annotations

import re
from pathlib import Path

from syrin.knowledge._document import Document, DocumentMetadata


class MarkdownLoader:
    """Load Markdown files with header-aware section parsing.

    Parses Markdown files into sections based on headers (# ## ###).
    Each section becomes a separate Document.

    Example:
        loader = MarkdownLoader("path/to/file.md")
        docs = loader.load()  # List of Documents, one per section
    """

    def __init__(self, path: str | Path) -> None:
        """Initialize MarkdownLoader.

        Args:
            path: Path to the Markdown file.
        """
        self._path = Path(path)

    @property
    def path(self) -> Path:
        """Path to the Markdown file."""
        return self._path

    def load(self) -> list[Document]:
        """Load the Markdown file, splitting by headers.

        Returns:
            List of Documents, one per section.
        """
        if not self._path.exists():
            raise FileNotFoundError(f"Markdown file not found: {self._path}")

        content = self._path.read_text(encoding="utf-8")
        sections = self._parse_sections(content)

        docs: list[Document] = []
        for i, (header, section_content) in enumerate(sections):
            full_content = f"{header}\n{section_content}" if section_content else header
            metadata: DocumentMetadata = {
                "section_index": i,
                "section_title": header,
            }
            docs.append(
                Document(
                    content=full_content,
                    source=str(self._path),
                    source_type="markdown",
                    metadata=metadata,
                )
            )

        return docs

    def _parse_sections(self, content: str) -> list[tuple[str, str]]:
        """Parse Markdown content into sections.

        Args:
            content: Markdown text content.

        Returns:
            List of (header, section_content) tuples.
        """
        lines = content.split("\n")
        sections: list[tuple[str, str]] = []
        current_header = "Introduction"
        current_content: list[str] = []
        header_pattern = re.compile(r"^(#{1,6})\s+(.+)$")

        for line in lines:
            match = header_pattern.match(line)
            if match:
                if current_content:
                    content_str = "\n".join(current_content).strip()
                    if content_str:
                        sections.append((current_header, content_str))
                current_header = match.group(2).strip()
                current_content = []
            else:
                current_content.append(line)

        if current_content:
            content_str = "\n".join(current_content).strip()
            if content_str:
                sections.append((current_header, content_str))

        if not sections:
            sections.append(("Content", content.strip()))

        return sections

    async def aload(self) -> list[Document]:
        """Load the Markdown file asynchronously.

        Returns:
            List of Documents, one per section.
        """
        return self.load()


__all__ = ["MarkdownLoader"]
