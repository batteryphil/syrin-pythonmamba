"""Semantic chunker: split at similarity drops. Requires EmbeddingProvider (async)."""

from __future__ import annotations

import asyncio
import re
from typing import TYPE_CHECKING

from syrin.knowledge._chunker import Chunk, ChunkConfig, ChunkStrategy
from syrin.knowledge._document import Document
from syrin.knowledge.chunkers._base import (
    _build_chunk_metadata,
    _estimate_tokens,
    _filter_by_min_size,
)

if TYPE_CHECKING:
    from syrin.embedding._protocol import EmbeddingProvider

_SENTENCE_DELIM = re.compile(r"(?<=[.!?])\s+|\n+")


def _sentences(text: str) -> list[str]:
    """Split text into sentences (simple regex)."""
    parts = _SENTENCE_DELIM.split(text.strip())
    return [p.strip() for p in parts if p.strip()]


def _cosine_sim(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two vectors."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    val: float = dot / (norm_a * norm_b)
    return max(0.0, min(1.0, val))


class SemanticChunker:
    """Chunk by semantic similarity: split when similarity between consecutive sentences drops.

    Requires config.embedding (EmbeddingProvider). Use achunk() for async pipelines.
    """

    def __init__(self, config: ChunkConfig) -> None:
        if config.strategy != ChunkStrategy.SEMANTIC:
            raise ValueError("SemanticChunker requires strategy=ChunkStrategy.SEMANTIC")
        if config.embedding is None:
            raise ValueError("SemanticChunker requires config.embedding (EmbeddingProvider)")
        self._config = config
        self._embedding: EmbeddingProvider = config.embedding

    def chunk(self, documents: list[Document]) -> list[Chunk]:
        """Sync entry point: runs achunk in event loop."""
        return asyncio.run(self.achunk(documents))

    async def achunk(self, documents: list[Document]) -> list[Chunk]:
        result: list[Chunk] = []
        for doc in documents:
            if not doc.content.strip():
                continue
            sentences = _sentences(doc.content)
            if not sentences:
                token_count = _estimate_tokens(doc.content)
                meta = _build_chunk_metadata(doc, ChunkStrategy.SEMANTIC.value)
                result.append(
                    Chunk(
                        content=doc.content,
                        metadata=meta,
                        document_id=doc.source,
                        chunk_index=0,
                        token_count=token_count,
                    )
                )
                continue
            embeddings = await self._embedding.embed(sentences)
            if len(embeddings) != len(sentences):
                content = doc.content
                meta = _build_chunk_metadata(doc, ChunkStrategy.SEMANTIC.value)
                result.append(
                    Chunk(
                        content=content,
                        metadata=meta,
                        document_id=doc.source,
                        chunk_index=0,
                        token_count=_estimate_tokens(content),
                    )
                )
                continue
            groups: list[list[str]] = []
            current: list[str] = [sentences[0]]
            for i in range(1, len(sentences)):
                sim = _cosine_sim(embeddings[i - 1], embeddings[i])
                if sim >= self._config.similarity_threshold:
                    current.append(sentences[i])
                else:
                    groups.append(current)
                    current = [sentences[i]]
            if current:
                groups.append(current)
            chunk_index = 0
            for group in groups:
                content = " ".join(group)
                token_count = _estimate_tokens(content)
                if token_count > self._config.chunk_size:
                    for j in range(0, len(group), 5):
                        sub = group[j : j + 5]
                        sub_content = " ".join(sub)
                        sub_tokens = _estimate_tokens(sub_content)
                        meta = _build_chunk_metadata(doc, ChunkStrategy.SEMANTIC.value)
                        result.append(
                            Chunk(
                                content=sub_content,
                                metadata=meta,
                                document_id=doc.source,
                                chunk_index=chunk_index,
                                token_count=sub_tokens,
                            )
                        )
                        chunk_index += 1
                else:
                    meta = _build_chunk_metadata(doc, ChunkStrategy.SEMANTIC.value)
                    result.append(
                        Chunk(
                            content=content,
                            metadata=meta,
                            document_id=doc.source,
                            chunk_index=chunk_index,
                            token_count=token_count,
                        )
                    )
                    chunk_index += 1
        return _filter_by_min_size(result, self._config.min_chunk_size)


__all__ = ["SemanticChunker"]
