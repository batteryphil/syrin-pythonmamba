"""Chunking strategies for Knowledge module."""

from __future__ import annotations

from typing import cast

from syrin.knowledge._chunker import ChunkConfig, Chunker, ChunkStrategy
from syrin.knowledge.chunkers._auto import AutoChunker
from syrin.knowledge.chunkers._code import CodeChunker
from syrin.knowledge.chunkers._markdown import MarkdownChunker
from syrin.knowledge.chunkers._page import PageChunker
from syrin.knowledge.chunkers._recursive import RecursiveChunker
from syrin.knowledge.chunkers._semantic import SemanticChunker
from syrin.knowledge.chunkers._sentence import SentenceChunker
from syrin.knowledge.chunkers._token import TokenChunker

__all__ = [
    "AutoChunker",
    "CodeChunker",
    "MarkdownChunker",
    "PageChunker",
    "RecursiveChunker",
    "SemanticChunker",
    "SentenceChunker",
    "TokenChunker",
    "get_chunker",
]

_STRATEGY_MAP: dict[ChunkStrategy, type] = {
    ChunkStrategy.AUTO: AutoChunker,
    ChunkStrategy.RECURSIVE: RecursiveChunker,
    ChunkStrategy.MARKDOWN: MarkdownChunker,
    ChunkStrategy.PAGE: PageChunker,
    ChunkStrategy.CODE: CodeChunker,
    ChunkStrategy.SENTENCE: SentenceChunker,
    ChunkStrategy.TOKEN: TokenChunker,
    ChunkStrategy.SEMANTIC: SemanticChunker,
}


def get_chunker(config: ChunkConfig) -> Chunker:
    """Return a Chunker implementation for the given config.

    Args:
        config: ChunkConfig with strategy and options.

    Returns:
        Chunker instance (RecursiveChunker, MarkdownChunker, AutoChunker, etc.).

    Example:
        config = ChunkConfig(strategy=ChunkStrategy.RECURSIVE, chunk_size=512)
        chunker = get_chunker(config)
        chunks = chunker.chunk(documents)
    """
    cls = _STRATEGY_MAP.get(config.strategy)
    if cls is None:
        raise ValueError(f"Unknown chunk strategy: {config.strategy}")
    return cast(Chunker, cls(config))
