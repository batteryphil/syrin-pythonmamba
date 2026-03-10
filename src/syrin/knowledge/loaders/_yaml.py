"""YAML loader."""

# mypy: disable-error-code="import-untyped"

from __future__ import annotations

from pathlib import Path

from syrin.knowledge._document import Document


class YAMLLoader:
    """Load YAML files and convert to text.

    Converts YAML data to a readable text representation.

    Example:
        loader = YAMLLoader("path/to/file.yaml")
        docs = loader.load()
    """

    def __init__(self, path: str | Path) -> None:
        """Initialize YAMLLoader.

        Args:
            path: Path to the YAML file.

        Raises:
            ImportError: If pyyaml is not installed.
        """
        self._path = Path(path)
        self._check_dependency()

    def _check_dependency(self) -> None:
        """Check if pyyaml is installed."""
        try:
            import yaml  # noqa: F401
        except ImportError as err:
            raise ImportError(
                "pyyaml is required for YAML loading. Install with: uv pip install pyyaml"
            ) from err

    @property
    def path(self) -> Path:
        """Path to the YAML file."""
        return self._path

    def load(self) -> list[Document]:
        """Load the YAML file and convert to text.

        Returns:
            List containing one Document with the YAML as text.
        """
        if not self._path.exists():
            raise FileNotFoundError(f"YAML file not found: {self._path}")

        try:
            import yaml
        except ImportError as err:
            raise ImportError(
                "pyyaml is required for YAML loading. Install with: uv pip install pyyaml"
            ) from err

        content = self._path.read_text(encoding="utf-8")
        data = yaml.safe_load(content)

        text_representation = self._yaml_to_text(data)
        return [
            Document(
                content=text_representation,
                source=str(self._path),
                source_type="yaml",
            )
        ]

    def _yaml_to_text(
        self,
        data: str | int | float | bool | list[object] | dict[str, object] | None,
        indent: int = 0,
    ) -> str:
        """Convert YAML data to readable text.

        Args:
            data: Parsed YAML data.
            indent: Current indentation level.

        Returns:
            Text representation of the YAML data.
        """
        if isinstance(data, dict):
            lines = []
            for key, value in data.items():
                if isinstance(value, (dict, list)):
                    lines.append(f"{'  ' * indent}{key}:")
                    lines.append(self._yaml_to_text(value, indent + 1))
                else:
                    lines.append(f"{'  ' * indent}{key}: {value}")
            return "\n".join(lines)
        elif isinstance(data, list):
            lines = []
            for item in data:
                if isinstance(item, (dict, list)):
                    lines.append(self._yaml_to_text(item, indent))
                else:
                    lines.append(f"{'  ' * indent}- {item}")
            return "\n".join(lines)
        else:
            return str(data)

    async def aload(self) -> list[Document]:
        """Load the YAML file asynchronously.

        Returns:
            List containing one Document with the YAML as text.
        """
        return self.load()


__all__ = ["YAMLLoader"]
