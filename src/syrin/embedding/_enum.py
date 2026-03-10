"""Embedding backend types."""

from __future__ import annotations

from enum import StrEnum


class EmbeddingBackend(StrEnum):
    """Embedding backend selection.

    Attributes:
        OPENAI: OpenAI text embeddings (text-embedding-3-small/large).
        OLLAMA: Local embeddings via Ollama server.
        LITELLM: Any embedding provider via LiteLLM (Cohere, Voyage, Mistral).
    """

    OPENAI = "openai"
    OLLAMA = "ollama"
    LITELLM = "litellm"
