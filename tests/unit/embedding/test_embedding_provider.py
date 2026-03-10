"""Tests for EmbeddingProvider Protocol and implementations."""

from __future__ import annotations

import os
from unittest.mock import AsyncMock, patch

import pytest


class TestEmbeddingProviderProtocol:
    """Tests for the EmbeddingProvider Protocol interface."""

    def test_protocol_signature(self) -> None:
        """Protocol defines required methods and properties."""
        from syrin.embedding._protocol import EmbeddingProvider

        assert hasattr(EmbeddingProvider, "embed")
        assert hasattr(EmbeddingProvider, "dimensions")
        assert hasattr(EmbeddingProvider, "model_id")

    @pytest.mark.asyncio
    async def test_protocol_embed_returns_list_of_lists(self) -> None:
        """embed() should return list[list[float]]."""

        class MockProvider:
            @property
            def dimensions(self) -> int:
                return 1536

            @property
            def model_id(self) -> str:
                return "test-model"

            async def embed(self, texts: list[str]) -> list[list[float]]:
                return [[0.1] * 1536 for _ in texts]

        from syrin.embedding._protocol import EmbeddingProvider

        provider: EmbeddingProvider = MockProvider()
        result = await provider.embed(["test"])
        assert isinstance(result, list)
        assert isinstance(result[0], list)
        assert len(result[0]) == 1536

    @pytest.mark.asyncio
    async def test_protocol_embed_single_text(self) -> None:
        """embed() works with single text in list."""

        class MockProvider:
            @property
            def dimensions(self) -> int:
                return 768

            @property
            def model_id(self) -> str:
                return "mock"

            async def embed(self, texts: list[str]) -> list[list[float]]:
                return [[float(i) / 100] * 768 for i in range(len(texts))]

        from syrin.embedding._protocol import EmbeddingProvider

        provider: EmbeddingProvider = MockProvider()
        result = await provider.embed(["hello world"])
        assert len(result) == 1
        assert len(result[0]) == 768

    @pytest.mark.asyncio
    async def test_protocol_embed_empty_list(self) -> None:
        """embed() with empty list returns empty list."""

        class MockProvider:
            @property
            def dimensions(self) -> int:
                return 1536

            @property
            def model_id(self) -> str:
                return "mock"

            async def embed(self, texts: list[str]) -> list[list[float]]:
                return []

        from syrin.embedding._protocol import EmbeddingProvider

        provider: EmbeddingProvider = MockProvider()
        result = await provider.embed([])
        assert result == []


class TestEmbeddingProviderInvalidInputs:
    """Tests for invalid inputs to EmbeddingProvider."""

    @pytest.mark.asyncio
    async def test_embed_with_none_raises(self) -> None:
        """embed(None) should raise TypeError."""

        class MockProvider:
            @property
            def dimensions(self) -> int:
                return 1536

            @property
            def model_id(self) -> str:
                return "mock"

            async def embed(self, texts: list[str]) -> list[list[float]]:
                if texts is None:
                    raise TypeError("texts cannot be None")
                return [[0.0] * 1536 for _ in texts]

        from syrin.embedding._protocol import EmbeddingProvider

        provider: EmbeddingProvider = MockProvider()
        with pytest.raises(TypeError, match="cannot be None"):
            await provider.embed(None)  # type: ignore[arg-type]

    @pytest.mark.asyncio
    async def test_embed_with_non_string_raises(self) -> None:
        """embed() with non-string items should handle gracefully."""

        class MockProvider:
            @property
            def dimensions(self) -> int:
                return 1536

            @property
            def model_id(self) -> str:
                return "mock"

            async def embed(self, texts: list[str]) -> list[list[float]]:
                for text in texts:
                    if not isinstance(text, str):
                        raise TypeError(f"Expected str, got {type(text)}")
                return [[0.0] * 1536 for _ in texts]

        from syrin.embedding._protocol import EmbeddingProvider

        provider: EmbeddingProvider = MockProvider()
        with pytest.raises(TypeError, match="Expected str"):
            await provider.embed([123])  # type: ignore[list-item]


class TestOpenAIEmbedding:
    """Tests for OpenAIEmbedding provider."""

    @pytest.fixture
    def mock_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Set up mock environment."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-key-123")

    def test_defaults(self, mock_env: None) -> None:
        """Test default configuration."""
        from syrin.embedding._openai import OpenAIEmbedding

        provider = OpenAIEmbedding()
        assert provider.model == "text-embedding-3-small"
        assert provider.dimensions == 1536

    def test_custom_model(self, mock_env: None) -> None:
        """Test custom model configuration."""
        from syrin.embedding._openai import OpenAIEmbedding

        provider = OpenAIEmbedding(model="text-embedding-3-large")
        assert provider.model == "text-embedding-3-large"
        assert provider.dimensions == 3072

    def test_dimension_reduction(self, mock_env: None) -> None:
        """Test dimension reduction."""
        from syrin.embedding._openai import OpenAIEmbedding

        provider = OpenAIEmbedding(dimensions=256)
        assert provider.dimensions == 256

    def test_explicit_api_key(self, mock_env: None) -> None:
        """Test explicit API key overrides env var."""
        from syrin.embedding._openai import OpenAIEmbedding

        provider = OpenAIEmbedding(api_key="explicit-key")
        assert provider._api_key == "explicit-key"

    def test_missing_api_key_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Missing API key without env var raises ValueError."""
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)

        from syrin.embedding._openai import OpenAIEmbedding

        with pytest.raises(ValueError, match="API key required"):
            OpenAIEmbedding()

    @pytest.mark.asyncio
    async def test_embed_returns_correct_count(self, mock_env: None) -> None:
        """embed() returns correct number of embeddings."""
        from syrin.embedding._openai import OpenAIEmbedding

        with patch("openai.AsyncOpenAI") as mock_client:
            mock_instance = AsyncMock()
            mock_instance.embeddings.create = AsyncMock(return_value=_make_mock_response(3, 1536))
            mock_client.return_value = mock_instance

            provider = OpenAIEmbedding()
            result = await provider.embed(["a", "b", "c"])

            assert len(result) == 3
            assert len(result[0]) == 1536

    @pytest.mark.asyncio
    async def test_embed_single_text(self, mock_env: None) -> None:
        """embed() works with single text."""
        from syrin.embedding._openai import OpenAIEmbedding

        with patch("openai.AsyncOpenAI") as mock_client:
            mock_instance = AsyncMock()
            mock_instance.embeddings.create = AsyncMock(return_value=_make_mock_response(1, 1536))
            mock_client.return_value = mock_instance

            provider = OpenAIEmbedding()
            result = await provider.embed(["hello"])

            assert len(result) == 1

    @pytest.mark.asyncio
    async def test_embed_empty_list(self, mock_env: None) -> None:
        """embed() with empty list returns empty list."""
        from syrin.embedding._openai import OpenAIEmbedding

        provider = OpenAIEmbedding()
        result = await provider.embed([])
        assert result == []


class TestOpenAIEmbeddingDimensionReduction:
    """Tests for OpenAIEmbedding dimension reduction feature."""

    @pytest.fixture
    def mock_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Set up mock environment."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-key-123")

    def test_dimension_reduction_256(self, mock_env: None) -> None:
        """Test dimension reduction to 256."""
        from syrin.embedding._openai import OpenAIEmbedding

        provider = OpenAIEmbedding(dimensions=256)
        assert provider.dimensions == 256

    def test_dimension_reduction_1024(self, mock_env: None) -> None:
        """Test dimension reduction to 1024."""
        from syrin.embedding._openai import OpenAIEmbedding

        provider = OpenAIEmbedding(dimensions=1024)
        assert provider.dimensions == 1024

    def test_dimension_reduction_invalid_raises(self, mock_env: None) -> None:
        """Invalid dimension raises ValueError."""
        from syrin.embedding._openai import OpenAIEmbedding

        with pytest.raises(ValueError, match="dimensions must be"):
            OpenAIEmbedding(dimensions=100)  # Invalid: not in valid set


class TestOllamaEmbedding:
    """Tests for OllamaEmbedding provider."""

    def test_defaults(self) -> None:
        """Test default configuration."""
        from syrin.embedding._ollama import OllamaEmbedding

        provider = OllamaEmbedding()
        assert provider.model == "nomic-embed-text"
        assert provider.api_base == "http://localhost:11434"
        assert provider.dimensions == 768

    def test_custom_model(self) -> None:
        """Test custom model."""
        from syrin.embedding._ollama import OllamaEmbedding

        provider = OllamaEmbedding(model="mxbai-embed-large")
        assert provider.model == "mxbai-embed-large"

    def test_custom_api_base(self) -> None:
        """Test custom API base."""
        from syrin.embedding._ollama import OllamaEmbedding

        provider = OllamaEmbedding(api_base="http://192.168.1.100:11434")
        assert provider.api_base == "http://192.168.1.100:11434"

    @pytest.mark.asyncio
    async def test_embed_returns_correct_dimensions(self) -> None:
        """embed() returns correct dimensions."""
        from syrin.embedding._ollama import OllamaEmbedding

        provider = OllamaEmbedding()

        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = AsyncMock()
            mock_response = AsyncMock()
            mock_response.status_code = 200
            mock_response.json = lambda: {"embedding": [0.1] * 768}
            mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_instance.__aexit__ = AsyncMock(return_value=None)
            mock_instance.post = AsyncMock(return_value=mock_response)
            mock_client.return_value = mock_instance

            result = await provider.embed(["hello"])

            assert len(result) == 1
            assert len(result[0]) == 768

    @pytest.mark.asyncio
    async def test_embed_batch(self) -> None:
        """embed() works with multiple texts."""
        from syrin.embedding._ollama import OllamaEmbedding

        provider = OllamaEmbedding()

        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = AsyncMock()

            # First call: batch request (joined text)
            mock_response_batch = AsyncMock()
            mock_response_batch.status_code = 200
            mock_response_batch.json = lambda: {"embedding": [0.1] * 768}

            # Second call: "hello"
            mock_response_hello = AsyncMock()
            mock_response_hello.status_code = 200
            mock_response_hello.json = lambda: {"embedding": [0.1] * 768}

            # Third call: "world"
            mock_response_world = AsyncMock()
            mock_response_world.status_code = 200
            mock_response_world.json = lambda: {"embedding": [0.2] * 768}

            mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_instance.__aexit__ = AsyncMock(return_value=None)
            mock_instance.post = AsyncMock(
                side_effect=[mock_response_batch, mock_response_hello, mock_response_world]
            )
            mock_client.return_value = mock_instance

            result = await provider.embed(["hello", "world"])

            assert len(result) == 2
            assert len(result[0]) == 768


class TestLiteLLMEmbedding:
    """Tests for LiteLLMEmbedding provider."""

    @pytest.fixture
    def litellm_available(self) -> bool:
        """Check if litellm is available."""
        import importlib.util

        return importlib.util.find_spec("litellm") is not None

    def test_defaults(self, litellm_available: bool) -> None:
        """Test default configuration."""
        if not litellm_available:
            pytest.skip("litellm not installed")
        from syrin.embedding._litellm import LiteLLMEmbedding

        provider = LiteLLMEmbedding(model="cohere/embed-english-v3.0")
        assert provider.model == "cohere/embed-english-v3.0"
        assert provider.dimensions == 1024  # Cohere default

    def test_custom_model(self, litellm_available: bool) -> None:
        """Test custom model pass-through."""
        if not litellm_available:
            pytest.skip("litellm not installed")
        from syrin.embedding._litellm import LiteLLMEmbedding

        provider = LiteLLMEmbedding(model="mistral/mistral-embed")
        assert provider.model == "mistral/mistral-embed"

    @pytest.mark.asyncio
    async def test_embed(self, litellm_available: bool) -> None:
        """embed() works."""
        if not litellm_available:
            pytest.skip("litellm not installed")
        from syrin.embedding._litellm import LiteLLMEmbedding

        provider = LiteLLMEmbedding(model="cohere/embed-english-v3.0")

        with patch("syrin.embedding._litellm.litellm") as mock_litellm:
            mock_litellm.aembedding = AsyncMock(
                return_value={
                    "data": [{"embedding": [0.1] * 1024}] * 2,
                    "usage": {"prompt_tokens": 10},
                }
            )

            result = await provider.embed(["hello", "world"])
            assert len(result) == 2
            assert len(result[0]) == 1024


class TestEmbeddingNamespace:
    """Tests for Embedding namespace class."""

    def test_openai_provider(self) -> None:
        """Embedding.OpenAI() returns OpenAIEmbedding."""
        from syrin.embedding import Embedding

        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            provider = Embedding.OpenAI("text-embedding-3-small")
            from syrin.embedding._openai import OpenAIEmbedding

            assert isinstance(provider, OpenAIEmbedding)
            assert provider.model == "text-embedding-3-small"

    def test_ollama_provider(self) -> None:
        """Embedding.Ollama() returns OllamaEmbedding."""
        from syrin.embedding import Embedding

        provider = Embedding.Ollama("nomic-embed-text")
        from syrin.embedding._ollama import OllamaEmbedding

        assert isinstance(provider, OllamaEmbedding)
        assert provider.model == "nomic-embed-text"

    def test_litellm_provider(self) -> None:
        """Embedding.LiteLLM() returns LiteLLMEmbedding."""
        import importlib.util

        if importlib.util.find_spec("litellm") is None:
            pytest.skip("litellm not installed")

        from syrin.embedding import Embedding

        provider = Embedding.LiteLLM("cohere/embed-english-v3.0")
        from syrin.embedding._litellm import LiteLLMEmbedding

        assert isinstance(provider, LiteLLMEmbedding)
        assert provider.model == "cohere/embed-english-v3.0"


class TestEmbeddingCostTracking:
    """Tests for embedding cost tracking."""

    @pytest.mark.asyncio
    async def test_cost_tracking_with_budget(self) -> None:
        """Embedding costs are tracked in BudgetTracker."""
        from unittest.mock import MagicMock

        from syrin.embedding._openai import OpenAIEmbedding

        budget_tracker = MagicMock()
        budget_tracker.record_external = MagicMock()

        with (
            patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}),
            patch("openai.AsyncOpenAI") as mock_client,
        ):
            mock_instance = AsyncMock()
            mock_response = _make_mock_response(1, 1536, token_count=10)
            mock_response.usage = type("Usage", (), {"prompt_tokens": 10})()
            mock_instance.embeddings.create = AsyncMock(return_value=mock_response)
            mock_client.return_value = mock_instance

            provider = OpenAIEmbedding()
            await provider.embed(["test"], budget_tracker=budget_tracker)

            budget_tracker.record_external.assert_called_once()

    @pytest.mark.asyncio
    async def test_cost_tracking_without_budget(self) -> None:
        """Works without BudgetTracker (no tracking)."""
        from syrin.embedding._openai import OpenAIEmbedding

        with (
            patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}),
            patch("openai.AsyncOpenAI") as mock_client,
        ):
            mock_instance = AsyncMock()
            mock_instance.embeddings.create = AsyncMock(return_value=_make_mock_response(1, 1536))
            mock_client.return_value = mock_instance

            provider = OpenAIEmbedding()
            result = await provider.embed(["test"], budget_tracker=None)
            assert len(result) == 1


class TestEmbeddingBackendEnum:
    """Tests for EmbeddingBackend enum."""

    def test_enum_values(self) -> None:
        """EmbeddingBackend has correct values."""
        from syrin.embedding import EmbeddingBackend

        assert EmbeddingBackend.OPENAI == "openai"
        assert EmbeddingBackend.OLLAMA == "ollama"
        assert EmbeddingBackend.LITELLM == "litellm"

    def test_enum_is_strenum(self) -> None:
        """EmbeddingBackend is a StrEnum."""
        from enum import StrEnum

        from syrin.embedding import EmbeddingBackend

        assert issubclass(EmbeddingBackend, StrEnum)


class TestEmbeddingCostCalculation:
    """Tests for embedding cost calculation."""

    def test_calculate_embedding_cost_small(self) -> None:
        """Cost calculation for text-embedding-3-small."""
        from syrin.cost import calculate_embedding_cost

        cost = calculate_embedding_cost("text-embedding-3-small", 1_000_000)
        assert cost == 0.02  # $0.02 per 1M tokens

    def test_calculate_embedding_cost_large(self) -> None:
        """Cost calculation for text-embedding-3-large."""
        from syrin.cost import calculate_embedding_cost

        cost = calculate_embedding_cost("text-embedding-3-large", 1_000_000)
        assert cost == 0.13  # $0.13 per 1M tokens

    def test_calculate_embedding_cost_ada(self) -> None:
        """Cost calculation for text-embedding-ada-002."""
        from syrin.cost import calculate_embedding_cost

        cost = calculate_embedding_cost("text-embedding-ada-002", 1_000_000)
        assert cost == 0.10  # $0.10 per 1M tokens

    def test_calculate_embedding_cost_unknown(self) -> None:
        """Unknown model returns 0 cost."""
        from syrin.cost import calculate_embedding_cost

        cost = calculate_embedding_cost("unknown-model", 1_000_000)
        assert cost == 0.0

    def test_calculate_embedding_cost_zero_tokens(self) -> None:
        """Zero tokens returns 0 cost."""
        from syrin.cost import calculate_embedding_cost

        cost = calculate_embedding_cost("text-embedding-3-small", 0)
        assert cost == 0.0


# Helper function to create mock OpenAI responses
def _make_mock_response(
    num_embeddings: int,
    dimensions: int,
    token_count: int | None = None,
) -> object:
    """Create a mock OpenAI embeddings response."""

    class MockEmbeddingData:
        def __init__(self, embedding: list[float]) -> None:
            self.embedding = embedding

    class MockResponse:
        def __init__(
            self,
            num_embeddings: int,
            dimensions: int,
            token_count: int | None,
        ) -> None:
            if token_count is not None:
                self.usage = type("Usage", (), {"prompt_tokens": token_count})()
            self.data = [MockEmbeddingData([0.1] * dimensions) for _ in range(num_embeddings)]

    return MockResponse(num_embeddings, dimensions, token_count)
