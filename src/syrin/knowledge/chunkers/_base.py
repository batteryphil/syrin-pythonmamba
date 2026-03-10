"""Shared helpers for chunker implementations."""

from __future__ import annotations

from syrin.knowledge._chunker import Chunk, ChunkMetadata
from syrin.knowledge._document import Document

_CHONKIE_IMPORT_ERROR = (
    "chonkie is required for this chunker. Install with: uv pip install syrin[knowledge]"
)


def _estimate_tokens(text: str) -> int:
    """Estimate token count (~4 chars per token)."""
    return max(1, len(text) // 4)


def _build_chunk_metadata(
    doc: Document,
    chunk_strategy: str,
    extra: ChunkMetadata | None = None,
) -> ChunkMetadata:
    """Merge document metadata with chunk-specific keys."""
    meta: ChunkMetadata = {
        **doc.metadata,
        "chunk_strategy": chunk_strategy,
        "source": doc.source,
        "source_type": doc.source_type,
    }
    if extra:
        meta = {**meta, **extra}
    return meta


def _filter_by_min_size(chunks: list[Chunk], min_chunk_size: int) -> list[Chunk]:
    """Drop chunks with token_count < min_chunk_size."""
    if min_chunk_size <= 0:
        return chunks
    return [c for c in chunks if c.token_count >= min_chunk_size]
