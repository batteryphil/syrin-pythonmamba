"""Auto chunker: select strategy per document from source_type."""

from __future__ import annotations

from dataclasses import replace

from syrin.knowledge._chunker import Chunk, ChunkConfig, ChunkStrategy
from syrin.knowledge._document import Document
from syrin.knowledge.chunkers._code import CodeChunker
from syrin.knowledge.chunkers._markdown import MarkdownChunker
from syrin.knowledge.chunkers._page import PageChunker
from syrin.knowledge.chunkers._recursive import RecursiveChunker


def _select_strategy(doc: Document) -> ChunkStrategy:
    """Choose chunking strategy from document source_type and metadata."""
    source_type = doc.source_type.lower()
    code_types = ("python", "javascript", "typescript", "go", "rust", "java", "c", "cpp")
    if source_type in code_types:
        return ChunkStrategy.CODE
    if source_type == "markdown":
        return ChunkStrategy.MARKDOWN
    if source_type == "pdf" and doc.metadata.get("has_pages"):
        return ChunkStrategy.PAGE
    return ChunkStrategy.RECURSIVE


class AutoChunker:
    """Select chunking strategy per document from source_type (CODE, MARKDOWN, PAGE, else RECURSIVE)."""

    def __init__(self, config: ChunkConfig) -> None:
        if config.strategy != ChunkStrategy.AUTO:
            raise ValueError("AutoChunker requires strategy=ChunkStrategy.AUTO")
        self._config = config
        self._recursive = RecursiveChunker(replace(config, strategy=ChunkStrategy.RECURSIVE))
        self._markdown = MarkdownChunker(replace(config, strategy=ChunkStrategy.MARKDOWN))
        self._page = PageChunker(replace(config, strategy=ChunkStrategy.PAGE))
        try:
            self._code: CodeChunker | None = CodeChunker(
                replace(config, strategy=ChunkStrategy.CODE)
            )
        except ImportError:
            self._code = None

    def chunk(self, documents: list[Document]) -> list[Chunk]:
        result: list[Chunk] = []
        for doc in documents:
            strategy = _select_strategy(doc)
            if strategy == ChunkStrategy.CODE and self._code is not None:
                result.extend(self._code.chunk([doc]))
            elif strategy == ChunkStrategy.MARKDOWN:
                result.extend(self._markdown.chunk([doc]))
            elif strategy == ChunkStrategy.PAGE:
                result.extend(self._page.chunk([doc]))
            else:
                result.extend(self._recursive.chunk([doc]))
        return result

    async def achunk(self, documents: list[Document]) -> list[Chunk]:
        return self.chunk(documents)


__all__ = ["AutoChunker", "_select_strategy"]
