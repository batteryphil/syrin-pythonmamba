"""Code chunker using chonkie (AST-aware via tree-sitter)."""

from __future__ import annotations

from typing import cast

from syrin.knowledge._chunker import Chunk, ChunkConfig, ChunkStrategy
from syrin.knowledge._document import Document
from syrin.knowledge.chunkers._base import (
    _build_chunk_metadata,
    _filter_by_min_size,
)


def _get_code_chunker() -> type:
    """Lazy import chonkie.CodeChunker (requires chonkie[code] for tree-sitter)."""
    try:
        from chonkie import CodeChunker as ChonkieCodeChunker

        return cast(type, ChonkieCodeChunker)
    except ImportError as e:
        raise ImportError(
            "CodeChunker requires chonkie[code] (tree-sitter). "
            "Install with: uv pip install 'chonkie[code]'"
        ) from e


class CodeChunker:
    """Chunk source code by AST (tree-sitter). Best for Python, JavaScript, etc."""

    def __init__(self, config: ChunkConfig) -> None:
        if config.strategy != ChunkStrategy.CODE:
            raise ValueError("CodeChunker requires strategy=ChunkStrategy.CODE")
        self._config = config
        Klass = _get_code_chunker()
        lang = config.language or "python"
        self._chunker = Klass(
            tokenizer="character",
            chunk_size=config.chunk_size,
            language=lang,
        )

    def chunk(self, documents: list[Document]) -> list[Chunk]:
        result: list[Chunk] = []
        for doc in documents:
            if not doc.content.strip():
                continue
            raw = self._chunker(doc.content)
            for i, rc in enumerate(raw):
                meta = _build_chunk_metadata(doc, ChunkStrategy.CODE.value)
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


__all__ = ["CodeChunker"]
