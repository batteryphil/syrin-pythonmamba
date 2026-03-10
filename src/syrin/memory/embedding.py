"""EmbeddingConfig - pluggable embeddings for vector memory backends."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=False)
class EmbeddingConfig:
    """Configuration for embedding model used by Qdrant/Chroma backends.

    Use embedding_provider for async embedding (recommended), or custom_fn for sync.
    When embedding_provider is set, it is used for async operations.
    When custom_fn is set, it is used for sync operations.
    When neither is set, backends fall back to MD5 pseudo-embeddings.

    Example:
        # Using new async provider (recommended)
        from syrin.embedding import Embedding
        config = EmbeddingConfig(
            embedding_provider=Embedding.OpenAI(dimensions=256),
            dimensions=256,
        )

        # Using custom sync function (legacy)
        def my_embed(text: str) -> list[float]:
            return [0.1] * 256
        config = EmbeddingConfig(custom_fn=my_embed, dimensions=256)
    """

    model: str = "text-embedding-3-small"
    provider: str = "openai"
    dimensions: int = 1536
    batch_size: int = 100
    api_key: str | None = None
    custom_fn: Callable[[str], list[float]] | None = field(default=None, repr=False)
    embedding_provider: Any = field(default=None, repr=True)

    def embed(self, text: str) -> list[float]:
        """Compute embedding for text. Uses custom_fn if set; else raises.

        Caller (QdrantBackend/ChromaBackend) falls back to MD5 when custom_fn is None.
        """
        if self.embedding_provider is not None:
            raise ValueError(
                "EmbeddingConfig.embed is sync. Use embed_async() with embedding_provider, "
                "or set custom_fn for sync embeddings."
            )
        if self.custom_fn is None:
            raise ValueError(
                "EmbeddingConfig.embed requires custom_fn. "
                "Set embedding_config=EmbeddingConfig(custom_fn=...) for custom embeddings, "
                "or omit embedding_config to use MD5 fallback."
            )
        result = self.custom_fn(text)
        if len(result) != self.dimensions:
            raise ValueError(
                f"EmbeddingConfig custom_fn returned {len(result)} dimensions, "
                f"expected {self.dimensions}"
            )
        return result

    async def embed_async(self, text: str) -> list[float]:
        """Compute embedding for text asynchronously using embedding_provider.

        Requires embedding_provider to be set. This is the recommended way
        for async memory backends.
        """
        if self.embedding_provider is None:
            raise ValueError(
                "EmbeddingConfig.embed_async requires embedding_provider. "
                "Set embedding_provider=Embedding.OpenAI(...) for async embeddings."
            )
        results: list[list[float]] = await self.embedding_provider.embed([text])
        if not results:
            raise ValueError("Embedding provider returned empty result")
        return results[0]
