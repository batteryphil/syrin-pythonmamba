"""Directory loader with glob and regex support."""

from __future__ import annotations

import re
import warnings
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from syrin.knowledge._document import Document

# Lazy imports for loaders
_LOADER_MAP: dict[str, str] | None = None


def _get_loader_map() -> dict[str, str]:
    """Get lazy-loaded loader map."""
    global _LOADER_MAP
    if _LOADER_MAP is None:
        _LOADER_MAP = {
            ".txt": "TextLoader",
            ".md": "MarkdownLoader",
            ".markdown": "MarkdownLoader",
            ".pdf": "PDFLoader",
            ".json": "JSONLoader",
            ".yaml": "YAMLLoader",
            ".yml": "YAMLLoader",
            ".py": "PythonLoader",
        }
    return _LOADER_MAP


class DirectoryLoader:
    r"""Load documents from a directory using glob or regex patterns.

    Supports both glob patterns and regex patterns for file matching.
    Automatically delegates to appropriate loader based on file extension.

    Example:
        # Glob pattern
        loader = DirectoryLoader("./docs/", glob="**/*.md")

        # Regex pattern
        loader = DirectoryLoader("./src/", pattern=r".*\.(py|md)$")

        # Load all files
        docs = loader.load()

        # Or use generator for memory efficiency
        for doc in loader.load_yield():
            process(doc)
    """

    def __init__(
        self,
        path: str | Path,
        glob: str = "**/*",
        pattern: str | None = None,
        recursive: bool = True,
    ) -> None:
        """Initialize DirectoryLoader.

        Args:
            path: Path to the directory.
            glob: Glob pattern for file matching (default: all files).
            pattern: Regex pattern as alternative to glob.
            recursive: Whether to search recursively (default: True).
        """
        self._path = Path(path)
        self._glob = glob
        self._pattern = pattern
        self._recursive = recursive

    @property
    def path(self) -> Path:
        """Path to the directory."""
        return self._path

    def load(self) -> list[Document]:
        """Load all matching files from the directory.

        Returns:
            List of Documents from all matching files.
        """
        files = self._get_matching_files()
        docs: list[Document] = []

        for file_path in files:
            try:
                loader = self._get_loader_for_file(file_path)
                if loader:
                    docs.extend(loader.load())
            except Exception as e:
                warnings.warn(f"Failed to load {file_path}: {e}", stacklevel=2)

        return docs

    def load_yield(self) -> Iterator[Document]:
        """Load files yielding Documents one at a time.

        Memory-efficient alternative to load() for large directories.

        Yields:
            Document objects one at a time.
        """
        files = self._get_matching_files()

        for file_path in files:
            try:
                loader = self._get_loader_for_file(file_path)
                if loader:
                    yield from loader.load()
            except Exception as e:
                warnings.warn(f"Failed to load {file_path}: {e}", stacklevel=2)

    def _get_matching_files(self) -> list[Path]:
        """Get list of files matching the pattern."""
        if not self._path.exists():
            raise FileNotFoundError(f"Directory not found: {self._path}")

        if not self._path.is_dir():
            raise NotADirectoryError(f"Not a directory: {self._path}")

        if self._pattern:
            return self._get_files_by_regex()
        return self._get_files_by_glob()

    def _get_files_by_glob(self) -> list[Path]:
        """Get files matching glob pattern."""
        if self._recursive:
            return sorted(self._path.glob(self._glob))
        return sorted(self._path.glob(self._glob.replace("**/", "")))

    def _get_files_by_regex(self) -> list[Path]:
        """Get files matching regex pattern."""
        assert self._pattern is not None  # Only called when pattern is set
        regex = re.compile(self._pattern)
        files: list[Path] = []

        if self._recursive:
            for path in self._path.rglob("*"):
                if path.is_file() and regex.match(path.name):
                    files.append(path)
        else:
            for path in self._path.glob("*"):
                if path.is_file() and regex.match(path.name):
                    files.append(path)

        return sorted(files)

    def _get_loader_for_file(self, file_path: Path) -> Any:
        """Get appropriate loader for a file based on extension."""
        ext = file_path.suffix.lower()
        loader_name = _get_loader_map().get(ext)

        if not loader_name:
            return None

        return self._load_loader(loader_name, file_path)

    def _load_loader(self, loader_name: str, file_path: Path) -> Any:
        """Lazy load and instantiate a loader."""
        if loader_name == "TextLoader":
            from syrin.knowledge.loaders._text import TextLoader

            return TextLoader(file_path)
        if loader_name == "MarkdownLoader":
            from syrin.knowledge.loaders._markdown import MarkdownLoader

            return MarkdownLoader(file_path)
        if loader_name == "PDFLoader":
            from syrin.knowledge.loaders._pdf import PDFLoader

            return PDFLoader(file_path)
        if loader_name == "YAMLLoader":
            from syrin.knowledge.loaders._yaml import YAMLLoader

            return YAMLLoader(file_path)
        if loader_name == "JSONLoader":
            from syrin.knowledge.loaders._json import JSONLoader

            return JSONLoader(file_path)
        if loader_name == "PythonLoader":
            from syrin.knowledge.loaders._python import PythonLoader

            return PythonLoader(file_path)
        return None

    async def aload(self) -> list[Document]:
        """Load all documents asynchronously.

        Returns:
            List of Documents from all matching files.
        """
        return self.load()


__all__ = ["DirectoryLoader"]
