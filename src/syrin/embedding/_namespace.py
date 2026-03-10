"""Embedding namespace - convenient access to embedding providers."""

from __future__ import annotations

from typing import Any

from syrin.embedding._enum import EmbeddingBackend
from syrin.embedding._ollama import OllamaEmbedding
from syrin.embedding._openai import OpenAIEmbedding
from syrin.embedding._protocol import EmbeddingProvider


class Embedding:
    """Namespace for embedding providers.

    Provides convenient access to built-in embedding providers.
    Use like: provider = Embedding.OpenAI("text-embedding-3-small")

    Example:
        # OpenAI (default: text-embedding-3-small, 1536 dims)
        provider = Embedding.OpenAI()
        provider = Embedding.OpenAI("text-embedding-3-large")

        # Ollama (local)
        provider = Embedding.Ollama()
        provider = Embedding.Ollama("mxbai-embed-large")

        # LiteLLM (any provider)
        provider = Embedding.LiteLLM("cohere/embed-english-v3.0")
    """

    @staticmethod
    def OpenAI(
        model: str = "text-embedding-3-small",
        api_key: str | None = None,
        dimensions: int | None = None,
    ) -> OpenAIEmbedding:
        """Create OpenAI embedding provider.

        Args:
            model: Model name. Defaults to text-embedding-3-small.
            api_key: OpenAI API key. Defaults to OPENAI_API_KEY env var.
            dimensions: Optional dimension reduction.

        Returns:
            OpenAIEmbedding provider.
        """
        return OpenAIEmbedding(model=model, api_key=api_key, dimensions=dimensions)

    @staticmethod
    def Ollama(
        model: str = "nomic-embed-text",
        api_base: str = "http://localhost:11434",
    ) -> OllamaEmbedding:
        """Create Ollama embedding provider.

        Args:
            model: Ollama model name. Defaults to nomic-embed-text.
            api_base: Ollama server URL.

        Returns:
            OllamaEmbedding provider.
        """
        return OllamaEmbedding(model=model, api_base=api_base)

    @staticmethod
    def LiteLLM(
        model: str,
        api_key: str | None = None,
    ) -> Any:
        """Create LiteLLM embedding provider.

        Args:
            model: LiteLLM model string (e.g., 'cohere/embed-english-v3.0').
            api_key: Optional API key. Defaults to LITELLM_API_KEY.

        Returns:
            LiteLLMEmbedding provider.
        """
        from syrin.embedding._litellm import LiteLLMEmbedding

        return LiteLLMEmbedding(model=model, api_key=api_key)

    @staticmethod
    def from_backend(
        backend: EmbeddingBackend,
        model: str | None = None,
        **kwargs: Any,
    ) -> EmbeddingProvider:
        """Create embedding provider from backend enum.

        Args:
            backend: EmbeddingBackend enum value.
            model: Model name (backend-specific defaults if None).
            **kwargs: Additional provider-specific arguments.

        Returns:
            EmbeddingProvider implementation.
        """
        if backend == EmbeddingBackend.OPENAI:
            provider: EmbeddingProvider = OpenAIEmbedding(
                model=model or "text-embedding-3-small",
                **kwargs,
            )
            return provider
        elif backend == EmbeddingBackend.OLLAMA:
            provider = OllamaEmbedding(
                model=model or "nomic-embed-text",
                **kwargs,
            )
            return provider
        elif backend == EmbeddingBackend.LITELLM:
            if not model:
                raise ValueError("model required for LiteLLM backend")
            from syrin.embedding._litellm import LiteLLMEmbedding

            provider = LiteLLMEmbedding(model=model, **kwargs)
            return provider
        else:
            raise ValueError(f"Unknown embedding backend: {backend}")
