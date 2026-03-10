"""Text and RawText loaders."""

from __future__ import annotations

import threading
from pathlib import Path

from syrin.knowledge._document import Document, DocumentMetadata

_text_source_counter = 0
_text_source_lock = threading.Lock()


def _next_text_source() -> str:
    """Return unique source for inline text (allows multiple Knowledge.Text() without collisions)."""
    global _text_source_counter
    with _text_source_lock:
        _text_source_counter += 1
        return f"user_provided_{_text_source_counter}"


class TextLoader:
    """Load plain text files.

    Example:
        loader = TextLoader("path/to/file.txt")
        docs = loader.load()
    """

    def __init__(self, path: str | Path) -> None:
        """Initialize TextLoader.

        Args:
            path: Path to the text file.
        """
        self._path = Path(path)

    @property
    def path(self) -> Path:
        """Path to the text file."""
        return self._path

    def load(self) -> list[Document]:
        """Load the text file.

        Returns:
            List containing one Document with the file contents.
        """
        if not self._path.exists():
            raise FileNotFoundError(f"Text file not found: {self._path}")

        content = self._path.read_text(encoding="utf-8")
        return [
            Document(
                content=content,
                source=str(self._path),
                source_type="text",
            )
        ]

    async def aload(self) -> list[Document]:
        """Load the text file asynchronously.

        Returns:
            List containing one Document with the file contents.
        """
        return self.load()


class RawTextLoader:
    """Load raw text provided directly by the user.

    Useful for quick facts, structured data, or content without a file.

    Example:
        loader = RawTextLoader("I have 5 years of Python experience.")
        docs = loader.load()

        # Multiple texts
        loader = RawTextLoader(["fact 1", "fact 2"])
    """

    def __init__(
        self,
        content: str | list[str],
        metadata: DocumentMetadata | None = None,
        source: str | None = None,
    ) -> None:
        """Initialize RawTextLoader.

        Args:
            content: Single text string or list of text strings.
            metadata: Optional metadata to attach to all documents.
            source: Optional document source. If None, uses unique id per loader
                (user_provided_N for single item, user_provided_0/1/... for list).
        """
        self._content = content if isinstance(content, list) else [content]
        self._metadata = metadata or {}
        self._source = source

    def load(self) -> list[Document]:
        """Load the raw text(s).

        Returns:
            List of Documents, one per text string.
        """
        docs: list[Document] = []
        for i, text in enumerate(self._content):
            meta: DocumentMetadata = {
                **self._metadata,
            }
            if len(self._content) > 1:
                meta["index"] = i
            if self._source is not None:
                src = f"{self._source}_{i}" if len(self._content) > 1 else self._source
            elif len(self._content) > 1:
                src = f"user_provided_{i}"
            else:
                src = _next_text_source()
            docs.append(
                Document(
                    content=text,
                    source=src,
                    source_type="text",
                    metadata=meta,
                )
            )
        return docs

    async def aload(self) -> list[Document]:
        """Load the raw text(s) asynchronously.

        Returns:
            List of Documents, one per text string.
        """
        return self.load()


__all__ = ["TextLoader", "RawTextLoader"]
