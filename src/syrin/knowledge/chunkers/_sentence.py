"""Sentence-level chunker using chonkie."""

from __future__ import annotations

from typing import cast

from syrin.knowledge._chunker import Chunk, ChunkConfig, ChunkStrategy
from syrin.knowledge._document import Document
from syrin.knowledge.chunkers._base import (
    _CHONKIE_IMPORT_ERROR,
    _build_chunk_metadata,
    _filter_by_min_size,
)


def _get_sentence_chunker() -> type:
    """Lazy import chonkie.SentenceChunker."""
    try:
        from chonkie import SentenceChunker as ChonkieSentenceChunker

        return cast(type, ChonkieSentenceChunker)
    except ImportError as e:
        raise ImportError(_CHONKIE_IMPORT_ERROR) from e


class SentenceChunker:
    """Chunk by sentence boundaries. Uses chonkie SentenceChunker."""

    def __init__(self, config: ChunkConfig) -> None:
        if config.strategy != ChunkStrategy.SENTENCE:
            raise ValueError("SentenceChunker requires strategy=ChunkStrategy.SENTENCE")
        self._config = config
        Klass = _get_sentence_chunker()
        self._chunker = Klass(
            tokenizer="character",
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
        )

    def chunk(self, documents: list[Document]) -> list[Chunk]:
        result: list[Chunk] = []
        for doc in documents:
            if not doc.content.strip():
                continue
            raw = self._chunker(doc.content)
            for i, rc in enumerate(raw):
                meta = _build_chunk_metadata(doc, ChunkStrategy.SENTENCE.value)
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


__all__ = ["SentenceChunker"]
