"""Document loader protocol for Knowledge module."""

from __future__ import annotations

from typing import Protocol


class DocumentLoader(Protocol):
    """Protocol for document loaders.

    DocumentLoaders convert various sources (files, URLs, etc.) into
    normalized Document objects that can be chunked and embedded.

    Example:
        class MyLoader(DocumentLoader):
            def load(self) -> list[Document]:
                # Load documents synchronously
                ...

            async def aload(self) -> list[Document]:
                # Load documents asynchronously
                ...
    """

    def load(self) -> list[Document]:
        """Load and return documents synchronously.

        Returns:
            List of loaded Documents.
        """
        ...

    async def aload(self) -> list[Document]:
        """Load and return documents asynchronously.

        Returns:
            List of loaded Documents.
        """
        ...


# Import Document for type hints (late import to avoid circular dependency)
from syrin.knowledge._document import Document

__all__ = ["Document", "DocumentLoader"]
