"""Recursive chunker using chonkie backend."""

from __future__ import annotations

from typing import cast

from syrin.knowledge._chunker import Chunk, ChunkConfig, ChunkStrategy
from syrin.knowledge._document import Document
from syrin.knowledge.chunkers._base import (
    _CHONKIE_IMPORT_ERROR,
    _build_chunk_metadata,
    _filter_by_min_size,
)


def _get_recursive_chunker() -> type:
    """Lazy import chonkie.RecursiveChunker."""
    try:
        from chonkie import RecursiveChunker

        return cast(type, RecursiveChunker)
    except ImportError as e:
        raise ImportError(_CHONKIE_IMPORT_ERROR) from e


class RecursiveChunker:
    """Chunk documents by recursively splitting at paragraphs, sentences, then words.

    Uses chonkie RecursiveChunker. Best for general text (articles, docs).
    """

    def __init__(self, config: ChunkConfig) -> None:
        if config.strategy != ChunkStrategy.RECURSIVE:
            raise ValueError("RecursiveChunker requires strategy=ChunkStrategy.RECURSIVE")
        self._config = config
        RecursiveChunkerClass = _get_recursive_chunker()
        # chonkie: tokenizer, chunk_size, rules, min_characters_per_chunk (no overlap in base)
        self._chunker = RecursiveChunkerClass(
            tokenizer="character",
            chunk_size=config.chunk_size,
            min_characters_per_chunk=max(1, config.min_chunk_size * 4),  # ~4 chars per token
        )

    def chunk(self, documents: list[Document]) -> list[Chunk]:
        result: list[Chunk] = []
        for doc in documents:
            if not doc.content.strip():
                continue
            raw = self._chunker(doc.content)
            for i, rc in enumerate(raw):
                meta = _build_chunk_metadata(doc, ChunkStrategy.RECURSIVE.value)
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


__all__ = ["RecursiveChunker"]
