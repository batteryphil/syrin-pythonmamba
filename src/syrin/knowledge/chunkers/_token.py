"""Token-based chunker using chonkie (fixed token windows)."""

from __future__ import annotations

from typing import cast

from syrin.knowledge._chunker import Chunk, ChunkConfig, ChunkStrategy
from syrin.knowledge._document import Document
from syrin.knowledge.chunkers._base import (
    _CHONKIE_IMPORT_ERROR,
    _build_chunk_metadata,
    _filter_by_min_size,
)


def _get_token_chunker() -> type:
    """Lazy import chonkie.TokenChunker."""
    try:
        from chonkie import TokenChunker as ChonkieTokenChunker

        return cast(type, ChonkieTokenChunker)
    except ImportError as e:
        raise ImportError(_CHONKIE_IMPORT_ERROR) from e


class TokenChunker:
    """Chunk by fixed token-count windows. Uses chonkie TokenChunker."""

    def __init__(self, config: ChunkConfig) -> None:
        if config.strategy != ChunkStrategy.TOKEN:
            raise ValueError("TokenChunker requires strategy=ChunkStrategy.TOKEN")
        self._config = config
        Klass = _get_token_chunker()
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
                meta = _build_chunk_metadata(doc, ChunkStrategy.TOKEN.value)
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


__all__ = ["TokenChunker"]
