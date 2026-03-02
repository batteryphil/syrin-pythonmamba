"""EmbeddingConfig - pluggable embeddings for vector memory backends."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field


@dataclass(frozen=False)
class EmbeddingConfig:
    """Configuration for embedding model used by Qdrant/Chroma backends.

    Use custom_fn for full control; provider/model for built-in providers (OpenAI, etc.).
    When custom_fn is set, it is used; otherwise backends fall back to MD5 pseudo-embeddings.
    """

    model: str = "text-embedding-3-small"
    provider: str = "openai"
    dimensions: int = 1536
    batch_size: int = 100
    api_key: str | None = None
    custom_fn: Callable[[str], list[float]] | None = field(default=None, repr=False)

    def embed(self, text: str) -> list[float]:
        """Compute embedding for text. Uses custom_fn if set; else raises.

        Caller (QdrantBackend/ChromaBackend) falls back to MD5 when custom_fn is None.
        """
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
