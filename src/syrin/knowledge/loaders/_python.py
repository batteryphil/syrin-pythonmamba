"""Python source code loader with AST-aware extraction."""

from __future__ import annotations

import ast
from pathlib import Path

from syrin.knowledge._document import Document


class PythonLoader:
    """Load Python source files with structure-aware extraction.

    Parses Python files using AST to extract classes, functions, and docstrings.

    Example:
        loader = PythonLoader("path/to/file.py")
        docs = loader.load()  # One doc per top-level class/function
    """

    def __init__(self, path: str | Path) -> None:
        """Initialize PythonLoader.

        Args:
            path: Path to the Python file.
        """
        self._path = Path(path)

    @property
    def path(self) -> Path:
        """Path to the Python file."""
        return self._path

    def load(self) -> list[Document]:
        """Load the Python file with AST parsing.

        Returns:
            List of Documents, one per top-level class/function.
        """
        if not self._path.exists():
            raise FileNotFoundError(f"Python file not found: {self._path}")

        source = self._path.read_text(encoding="utf-8")

        try:
            tree = ast.parse(source)
        except SyntaxError as err:
            raise ValueError(f"Failed to parse Python file: {err}") from err

        docs: list[Document] = []

        # Extract module docstring
        if ast.get_docstring(tree):
            docs.append(
                Document(
                    content=ast.get_docstring(tree) or "",
                    source=str(self._path),
                    source_type="python",
                    metadata={"type": "module", "name": self._path.stem},
                )
            )

        # Extract top-level classes and functions
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.ClassDef):
                doc = self._extract_class(node)
                if doc:
                    docs.append(doc)
            elif isinstance(node, ast.FunctionDef):
                doc = self._extract_function(node)
                if doc:
                    docs.append(doc)

        if not docs:
            # Fallback: return entire source as one document
            docs.append(
                Document(
                    content=source,
                    source=str(self._path),
                    source_type="python",
                    metadata={"type": "module", "name": self._path.stem},
                )
            )

        return docs

    def _extract_class(self, node: ast.ClassDef) -> Document | None:
        """Extract class information.

        Args:
            node: AST ClassDef node.

        Returns:
            Document with class content.
        """
        lines = [f"class {node.name}"]

        # Bases
        if node.bases:
            bases = [self._get_name(b) for b in node.bases]
            lines.append(f"({', '.join(bases)})")

        lines.append(":")

        # Docstring
        if ast.get_docstring(node):
            lines.append("")
            lines.append(ast.get_docstring(node) or "")

        # Methods
        for item in node.body:
            if isinstance(item, ast.FunctionDef):
                method_doc = self._extract_function(item, indent=4)
                if method_doc:
                    lines.append("")
                    lines.append(method_doc.content)

        content = "\n".join(lines)

        return Document(
            content=content,
            source=str(self._path),
            source_type="python",
            metadata={"type": "class", "name": node.name},
        )

    def _extract_function(
        self,
        node: ast.FunctionDef,
        indent: int = 0,
    ) -> Document | None:
        """Extract function/method information.

        Args:
            node: AST FunctionDef node.
            indent: Indentation level for methods.

        Returns:
            Document with function content.
        """
        prefix = " " * indent
        lines = [f"{prefix}def {node.name}("]

        # Parameters
        args = node.args
        params: list[str] = []
        for arg in args.args:
            params.append(arg.arg)
        for arg in args.posonlyargs:
            params.append(arg.arg)
        if args.vararg:
            params.append(f"*{args.vararg.arg}")
        if args.kwarg:
            params.append(f"**{args.kwarg.arg}")

        lines[0] += ", ".join(params) + "):"

        # Return type
        if node.returns:
            lines[0] += f" -> {self._get_name(node.returns)}"

        # Docstring
        if ast.get_docstring(node):
            lines.append("")
            doc_lines = (ast.get_docstring(node) or "").split("\n")
            for doc_line in doc_lines:
                lines.append(f"{prefix}    {doc_line}")

        content = "\n".join(lines)

        return Document(
            content=content,
            source=str(self._path),
            source_type="python",
            metadata={"type": "function", "name": node.name},
        )

    def _get_name(self, node: ast.AST) -> str:
        """Get name from AST node.

        Args:
            node: AST node.

        Returns:
            Name string.
        """
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return f"{self._get_name(node.value)}.{node.attr}"
        elif isinstance(node, ast.Subscript):
            return f"{self._get_name(node.value)}[{self._get_name(node.slice)}]"
        elif isinstance(node, ast.Constant):
            return repr(node.value)
        return ""

    async def aload(self) -> list[Document]:
        """Load the Python file asynchronously.

        Returns:
            List of Documents, one per top-level class/function.
        """
        return self.load()


__all__ = ["PythonLoader"]
