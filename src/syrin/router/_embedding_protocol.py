"""Embedding provider protocol — pluggable embeddings without sentence-transformers."""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Protocol for embedding providers. Use OpenAI, Cohere, or custom instead of sentence-transformers.

    This is the sync protocol for router classification. For async embedding (Knowledge, Memory),
    use the new syrin.embedding module instead.
    """

    def encode(self, texts: list[str]) -> list[list[float]]:
        """Encode texts to embeddings. One embedding per text."""
        ...

    def encode_async(self, texts: list[str]) -> list[list[float]]:
        """Async encode. Falls back to sync encode() if not implemented."""
        ...
