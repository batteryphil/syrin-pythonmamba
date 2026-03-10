"""PDF loader using pypdf."""

from __future__ import annotations

from pathlib import Path

from syrin.knowledge._document import Document, DocumentMetadata


class PDFLoader:
    """Load PDF files using pypdf.

    Extracts text from each page. Each page becomes a separate Document.

    Example:
        loader = PDFLoader("path/to/file.pdf")
        docs = loader.load()  # List of Documents, one per page
    """

    def __init__(self, path: str | Path) -> None:
        """Initialize PDFLoader.

        Args:
            path: Path to the PDF file.
        """
        self._path = Path(path)

    def _check_dependency(self) -> None:
        """Check if pypdf is installed."""
        try:
            import pypdf  # type: ignore[import-not-found]  # noqa: F401
        except ImportError as err:
            raise ImportError(
                "pypdf is required for PDF loading. Install with: uv pip install pypdf"
            ) from err

    @property
    def path(self) -> Path:
        """Path to the PDF file."""
        return self._path

    def load(self) -> list[Document]:
        """Load the PDF file, one page per Document.

        Returns:
            List of Documents, one per page.

        Raises:
            ImportError: If pypdf is not installed.
            FileNotFoundError: If the PDF file does not exist.
        """
        try:
            import pypdf
        except ImportError as err:
            raise ImportError(
                "pypdf is required for PDF loading. Install with: uv pip install pypdf"
            ) from err

        if not self._path.exists():
            raise FileNotFoundError(f"PDF file not found: {self._path}")

        reader = pypdf.PdfReader(str(self._path))
        docs: list[Document] = []

        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            if text.strip():
                metadata: DocumentMetadata = {
                    "page": i + 1,
                    "total_pages": len(reader.pages),
                    "has_pages": True,
                }
                if reader.metadata and "/Title" in reader.metadata:
                    title = reader.metadata["/Title"]
                    if isinstance(title, str):
                        metadata["title"] = title
                docs.append(
                    Document(
                        content=text,
                        source=str(self._path),
                        source_type="pdf",
                        metadata=metadata,
                    )
                )

        return docs

    async def aload(self) -> list[Document]:
        """Load the PDF file asynchronously.

        Returns:
            List of Documents, one per page.
        """
        return self.load()


__all__ = ["PDFLoader"]
