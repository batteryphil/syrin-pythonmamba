"""OpenAI embedding provider."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from openai import AsyncOpenAI

# Default dimensions per model
_MODEL_DIMENSIONS: dict[str, int] = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}

# Valid dimension reductions for embedding-3 models
_VALID_DIMENSIONS = {256, 512, 768, 1024, 1536, 2048, 3072}


class OpenAIEmbedding:
    """OpenAI embedding provider (text-embedding-3-small, text-embedding-3-large).

    Supports dimension reduction for text-embedding-3 models.
    Uses OPENAI_API_KEY environment variable if api_key not provided.

    Example:
        provider = OpenAIEmbedding()
        embeddings = await provider.embed(["hello world"])

        # With dimension reduction
        provider = OpenAIEmbedding(dimensions=256)
        embeddings = await provider.embed(["hello world"])  # 256-dim vectors
    """

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        api_key: str | None = None,
        dimensions: int | None = None,
    ) -> None:
        """Initialize OpenAI embedding provider.

        Args:
            model: Model name. Defaults to text-embedding-3-small.
            api_key: OpenAI API key. Defaults to OPENAI_API_KEY env var.
            dimensions: Optional dimension reduction (256, 512, 768, 1024, 1536, 2048, 3072).
                       Only supported for text-embedding-3 models.
        """
        self._model = model
        self._api_key = api_key or os.getenv("OPENAI_API_KEY")

        if not self._api_key:
            raise ValueError(
                "API key required. Set OPENAI_API_KEY environment variable "
                "or pass api_key explicitly."
            )

        # Validate dimensions
        if dimensions is not None:
            if dimensions not in _VALID_DIMENSIONS:
                raise ValueError(
                    f"dimensions must be one of {_VALID_DIMENSIONS} "
                    f"for text-embedding-3 models, got {dimensions}"
                )
            if model not in ("text-embedding-3-small", "text-embedding-3-large"):
                raise ValueError(
                    f"dimension reduction only supported for text-embedding-3 models, got {model}"
                )
            self._dimensions = dimensions
        else:
            self._dimensions = _MODEL_DIMENSIONS.get(model, 1536)

        self._client: AsyncOpenAI | None = None

    @property
    def dimensions(self) -> int:
        """Vector dimensionality."""
        return self._dimensions

    @property
    def model(self) -> str:
        """Model name."""
        return self._model

    @property
    def model_id(self) -> str:
        """Model identifier for cost tracking."""
        return self._model

    def _get_client(self) -> AsyncOpenAI:
        """Get or create async OpenAI client."""
        if self._client is None:
            from openai import AsyncOpenAI

            self._client = AsyncOpenAI(api_key=self._api_key)
        return self._client

    async def embed(
        self,
        texts: list[str],
        budget_tracker: Any | None = None,
    ) -> list[list[float]]:
        """Embed texts into vectors.

        Args:
            texts: List of text strings to embed.
            budget_tracker: Optional BudgetTracker for cost tracking.

        Returns:
            List of embedding vectors.
        """
        if not texts:
            return []

        client = self._get_client()

        # Build request kwargs
        kwargs: dict[str, Any] = {
            "model": self._model,
            "input": texts,
        }

        # Add dimensions if reduced
        if self._dimensions != _MODEL_DIMENSIONS.get(self._model, 1536):
            kwargs["dimensions"] = self._dimensions

        response = await client.embeddings.create(**kwargs)

        # Extract embeddings
        embeddings = [item.embedding for item in response.data]

        # Track cost if budget_tracker provided
        if budget_tracker is not None and response.usage is not None:
            from syrin.cost import calculate_embedding_cost

            token_count = response.usage.prompt_tokens
            cost = calculate_embedding_cost(self._model, token_count)
            budget_tracker.record_external(
                service="openai_embedding",
                cost_usd=cost,
                metadata={
                    "model": self._model,
                    "token_count": token_count,
                },
            )

        return embeddings
