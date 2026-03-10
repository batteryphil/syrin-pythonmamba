"""Embedding providers for Knowledge and Memory modules.

This module provides async embedding providers for text vectorization.
Used by Knowledge (RAG) and Memory (semantic recall) backends.

Example:
    from syrin.embedding import Embedding, EmbeddingProvider

    # Quick start
    provider = Embedding.OpenAI()

    # Or use directly
    from syrin.embedding import OpenAIEmbedding, OllamaEmbedding, LiteLLMEmbedding

    provider = OpenAIEmbedding(dimensions=256)
    embeddings = await provider.embed(["hello world"])
"""

from __future__ import annotations

from typing import Any

from syrin.embedding._enum import EmbeddingBackend
from syrin.embedding._namespace import Embedding
from syrin.embedding._ollama import OllamaEmbedding
from syrin.embedding._openai import OpenAIEmbedding
from syrin.embedding._protocol import EmbeddingProvider

__all__ = [
    "Embedding",
    "EmbeddingBackend",
    "EmbeddingProvider",
    "OllamaEmbedding",
    "OpenAIEmbedding",
]


def __getattr__(name: str) -> Any:
    """Lazy import for optional providers."""
    if name == "LiteLLMEmbedding":
        from syrin.embedding._litellm import LiteLLMEmbedding

        return LiteLLMEmbedding
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
