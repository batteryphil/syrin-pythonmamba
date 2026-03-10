"""JSON loader with simple dot notation path support."""

from __future__ import annotations

import json
from pathlib import Path

from syrin.knowledge._document import Document


class JSONLoader:
    """Load JSON files and convert to text.

    Supports simple dot notation for extracting specific paths.
    Example paths: "data.items", "data.items.0.name", "results"

    Example:
        loader = JSONLoader("path/to/file.json")
        docs = loader.load()

        # Extract specific path
        loader = JSONLoader("path/to/file.json", jq_path="data.items")
    """

    def __init__(
        self,
        path: str | Path,
        jq_path: str | None = None,
    ) -> None:
        """Initialize JSONLoader.

        Args:
            path: Path to the JSON file.
            jq_path: Optional dot notation path to extract (e.g., "data.items.0.name").
        """
        self._path = Path(path)
        self._jq_path = jq_path

    @property
    def path(self) -> Path:
        """Path to the JSON file."""
        return self._path

    def load(self) -> list[Document]:
        """Load the JSON file and convert to text.

        Returns:
            List containing one Document with the JSON as text.
        """
        if not self._path.exists():
            raise FileNotFoundError(f"JSON file not found: {self._path}")

        content = self._path.read_text(encoding="utf-8")
        data = json.loads(content)

        if self._jq_path:
            data = self._extract_path(data, self._jq_path)

        text_representation = self._json_to_text(data)
        metadata: dict[str, str | int | float | bool | None | list[str]] = {}
        if self._jq_path is not None:
            metadata["path"] = self._jq_path
        return [
            Document(
                content=text_representation,
                source=str(self._path),
                source_type="json",
                metadata=metadata,
            )
        ]

    def _extract_path(
        self,
        data: object,
        path: str,
    ) -> object:
        """Extract a value from JSON data using dot notation.

        Args:
            data: Parsed JSON data.
            path: Dot notation path (e.g., "data.items.0.name").

        Returns:
            Extracted data.
        """
        parts = path.split(".")
        current: object = data

        for part in parts:
            if isinstance(current, dict):
                current = current.get(part)
            elif isinstance(current, list):
                try:
                    index = int(part)
                    current = current[index]
                except (ValueError, IndexError):
                    return None
            else:
                return None

            if current is None:
                return None

        return current

    def _json_to_text(self, data: object, indent: int = 0) -> str:
        """Convert JSON data to readable text.

        Args:
            data: Parsed JSON data.
            indent: Current indentation level.

        Returns:
            Text representation of the JSON data.
        """
        if isinstance(data, dict):
            lines = []
            for key, value in data.items():
                if isinstance(value, (dict, list)):
                    lines.append(f"{'  ' * indent}{key}:")
                    lines.append(self._json_to_text(value, indent + 1))
                else:
                    lines.append(f"{'  ' * indent}{key}: {json.dumps(value)}")
            return "\n".join(lines)
        elif isinstance(data, list):
            lines = []
            for i, item in enumerate(data):
                if isinstance(item, (dict, list)):
                    lines.append(f"{'  ' * indent}[{i}]:")
                    lines.append(self._json_to_text(item, indent + 1))
                else:
                    lines.append(f"{'  ' * indent}[{i}]: {json.dumps(item)}")
            return "\n".join(lines)
        else:
            return json.dumps(data)

    async def aload(self) -> list[Document]:
        """Load the JSON file asynchronously.

        Returns:
            List containing one Document with the JSON as text.
        """
        return self.load()


__all__ = ["JSONLoader"]
