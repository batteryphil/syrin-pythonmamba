"""Markdown chunker: header-aware, preserves structure. Uses chonkie markdown recipe when available."""

from __future__ import annotations

import re
from typing import Protocol, cast

from syrin.knowledge._chunker import Chunk, ChunkConfig, ChunkStrategy
from syrin.knowledge._document import Document
from syrin.knowledge.chunkers._base import (
    _CHONKIE_IMPORT_ERROR,
    _build_chunk_metadata,
    _filter_by_min_size,
)

_HEADER_PATTERN = re.compile(r"^(#{1,6})\s+(.+)$")


def _heading_hierarchy_from_content(content: str) -> list[str]:
    """Extract heading hierarchy from chunk content (lines starting with #)."""
    hierarchy: list[str] = []
    for line in content.split("\n"):
        m = _HEADER_PATTERN.match(line)
        if m:
            level = len(m.group(1))
            title = m.group(2).strip()
            hierarchy = hierarchy[: level - 1] + [title]
    return hierarchy


class _ChonkieChunkLike(Protocol):
    """Protocol for chonkie chunk ( .text, .token_count )."""

    @property
    def text(self) -> str: ...
    @property
    def token_count(self) -> int: ...


class _MarkdownChunkerLike(Protocol):
    """Protocol for chunker that accepts text and returns list of chunk-like objects."""

    def __call__(self, text: str) -> list[_ChonkieChunkLike]: ...


def _get_markdown_chunker(config: ChunkConfig) -> _MarkdownChunkerLike:
    """Build chonkie chunker with markdown recipe or fallback to default recursive."""
    try:
        from chonkie import RecursiveChunker
    except ImportError as e:
        raise ImportError(_CHONKIE_IMPORT_ERROR) from e
    min_chars = max(1, config.min_chunk_size * 4)
    try:
        chunker = RecursiveChunker.from_recipe(
            "markdown",
            chunk_size=config.chunk_size,
            min_characters_per_chunk=min_chars,
        )
    except Exception:
        chunker = RecursiveChunker(
            tokenizer="character",
            chunk_size=config.chunk_size,
            min_characters_per_chunk=min_chars,
        )
    return cast(_MarkdownChunkerLike, chunker)


class MarkdownChunker:
    """Chunk Markdown by headers first, then paragraphs. Preserves header hierarchy in metadata."""

    def __init__(self, config: ChunkConfig) -> None:
        if config.strategy != ChunkStrategy.MARKDOWN:
            raise ValueError("MarkdownChunker requires strategy=ChunkStrategy.MARKDOWN")
        self._config = config
        self._chunker = _get_markdown_chunker(config)

    def chunk(self, documents: list[Document]) -> list[Chunk]:
        result: list[Chunk] = []
        for doc in documents:
            if not doc.content.strip():
                continue
            raw = self._chunker(doc.content)
            for i, rc in enumerate(raw):
                meta = _build_chunk_metadata(doc, ChunkStrategy.MARKDOWN.value)
                if self._config.preserve_headers:
                    hierarchy = _heading_hierarchy_from_content(rc.text)
                    if hierarchy:
                        meta = {**meta, "heading_hierarchy": hierarchy}
                result.append(
                    Chunk(
                        content=rc.text,
                        metadata=meta,
                        document_id=doc.source,
                        chunk_index=i,
                        token_count=rc.token_count,
                    )
                )
        return _filter_by_min_size(result, self._config.min_chunk_size)

    async def achunk(self, documents: list[Document]) -> list[Chunk]:
        return self.chunk(documents)


__all__ = ["MarkdownChunker"]
