"""Page-level chunker: one chunk per document (for PDF-style, one doc per page)."""

from __future__ import annotations

from syrin.knowledge._chunker import Chunk, ChunkConfig, ChunkStrategy
from syrin.knowledge._document import Document
from syrin.knowledge.chunkers._base import (
    _build_chunk_metadata,
    _estimate_tokens,
    _filter_by_min_size,
)


class PageChunker:
    """One chunk per document. Best when each Document is one page (e.g. PDFLoader)."""

    def __init__(self, config: ChunkConfig) -> None:
        if config.strategy != ChunkStrategy.PAGE:
            raise ValueError("PageChunker requires strategy=ChunkStrategy.PAGE")
        self._config = config

    def chunk(self, documents: list[Document]) -> list[Chunk]:
        result: list[Chunk] = []
        for i, doc in enumerate(documents):
            if not doc.content.strip():
                continue
            page = doc.metadata.get("page")
            chunk_index = (int(page) - 1) if isinstance(page, (int, float)) else i
            token_count = _estimate_tokens(doc.content)
            meta = _build_chunk_metadata(doc, ChunkStrategy.PAGE.value)
            result.append(
                Chunk(
                    content=doc.content,
                    metadata=meta,
                    document_id=doc.source,
                    chunk_index=chunk_index,
                    token_count=token_count,
                )
            )
        return _filter_by_min_size(result, self._config.min_chunk_size)

    async def achunk(self, documents: list[Document]) -> list[Chunk]:
        return self.chunk(documents)


__all__ = ["PageChunker"]
