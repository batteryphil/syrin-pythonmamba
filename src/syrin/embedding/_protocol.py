"""Embedding provider protocol — async embeddings for Knowledge and Memory."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from syrin.budget import BudgetTracker


class EmbeddingProvider(Protocol):
    """Protocol for async text embedding providers.

    Use with Knowledge (RAG) and Memory (semantic recall) modules.
    Built-in providers: OpenAI, Ollama, LiteLLM.

    Example:
        provider = OpenAIEmbedding()
        embeddings = await provider.embed(["hello world"])
    """

    @property
    def dimensions(self) -> int:
        """Vector dimensionality (e.g., 1536 for OpenAI, 768 for Ollama)."""
        ...

    @property
    def model_id(self) -> str:
        """Model identifier for cost tracking (e.g., 'text-embedding-3-small')."""
        ...

    async def embed(
        self,
        texts: list[str],
        budget_tracker: BudgetTracker | None = None,
    ) -> list[list[float]]:
        """Embed a batch of texts into vectors.

        Args:
            texts: List of text strings to embed.
            budget_tracker: Optional BudgetTracker for cost tracking.

        Returns:
            List of embedding vectors, one per input text.
        """
        ...
