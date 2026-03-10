"""Ollama embedding provider."""

from __future__ import annotations

from typing import Any

import httpx

# Default dimensions per model
_MODEL_DIMENSIONS: dict[str, int] = {
    "nomic-embed-text": 768,
    "mxbai-embed-large": 1024,
    "bge-m3": 1024,
    "bge-large": 1024,
    "bge-small": 384,
    "gte-qwen2": 1024,
    "snowflake-arctic-embed": 384,
}


class OllamaEmbedding:
    """Local embedding via Ollama (nomic-embed-text, mxbai-embed-large, etc.).

    Requires Ollama server running. Default: http://localhost:11434

    Example:
        provider = OllamaEmbedding()
        embeddings = await provider.embed(["hello world"])

        # Custom model
        provider = OllamaEmbedding(model="mxbai-embed-large")
    """

    def __init__(
        self,
        model: str = "nomic-embed-text",
        api_base: str = "http://localhost:11434",
    ) -> None:
        """Initialize Ollama embedding provider.

        Args:
            model: Ollama model name. Defaults to nomic-embed-text.
            api_base: Ollama server URL. Defaults to http://localhost:11434.
        """
        self._model = model
        self._api_base = api_base.rstrip("/")
        self._dimensions = _MODEL_DIMENSIONS.get(model, 768)

    @property
    def dimensions(self) -> int:
        """Vector dimensionality."""
        return self._dimensions

    @property
    def model(self) -> str:
        """Model name."""
        return self._model

    @property
    def api_base(self) -> str:
        """API base URL."""
        return self._api_base

    @property
    def model_id(self) -> str:
        """Model identifier for cost tracking."""
        return f"ollama:{self._model}"

    async def embed(
        self,
        texts: list[str],
        budget_tracker: Any | None = None,
    ) -> list[list[float]]:
        """Embed texts into vectors.

        Args:
            texts: List of text strings to embed.
            budget_tracker: Optional BudgetTracker (not used for local provider).

        Returns:
            List of embedding vectors.
        """
        if not texts:
            return []

        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self._api_base}/api/embeddings",
                json={
                    "model": self._model,
                    "prompt": texts[0] if len(texts) == 1 else "\n".join(texts),
                },
                timeout=30.0,
            )
            response.raise_for_status()
            data = response.json()

            # Single prompt returns single embedding
            if len(texts) == 1:
                return [data["embedding"]]

            # For batch, we need to call individually (Ollama doesn't support batch)
            embeddings: list[list[float]] = []
            for text in texts:
                resp = await client.post(
                    f"{self._api_base}/api/embeddings",
                    json={"model": self._model, "prompt": text},
                    timeout=30.0,
                )
                resp.raise_for_status()
                embeddings.append(resp.json()["embedding"])

            return embeddings
