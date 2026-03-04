"""Core Model class and base functionality."""

from __future__ import annotations

import os
import re
from typing import TYPE_CHECKING, Any, Literal, TypeVar, cast, overload

from pydantic import BaseModel

from syrin.cost import ModelPricing
from syrin.exceptions import ModelNotFoundError, ProviderError
from syrin.tool import ToolSpec
from syrin.types import (
    Message,
    ModelConfig,
    ProviderResponse,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Iterator

    from syrin.providers.base import Provider
else:
    from collections.abc import Iterator  # Runtime for cast()

T = TypeVar("T", bound=BaseModel)

_PROVIDER_PREFIXES = [
    ("anthropic/", "anthropic"),
    ("openai/", "openai"),
    ("google/", "google"),
    ("ollama/", "ollama"),
    ("azure/", "azure"),
    ("cohere/", "cohere"),
    ("deepseek/", "deepseek"),
    ("kimi/", "kimi"),
    ("sarvam/", "sarvam"),
]
# Patterns for bare model names (without prefix)
_PROVIDER_PATTERNS = [
    (re.compile(r"^gpt-", re.IGNORECASE), "openai"),
    (re.compile(r"^claude-", re.IGNORECASE), "anthropic"),
    (re.compile(r"^gemini-", re.IGNORECASE), "google"),
    (re.compile(r"^llama-", re.IGNORECASE), "ollama"),
]


def _resolve_env_var(value: str) -> str:
    """Resolve $VAR or ${VAR} to environment variable if present."""
    if not value or value[0] != "$":
        return value
    name = value[1:].strip("{}")
    return os.environ.get(name, value)


def detect_provider(model_id: str) -> str:
    """Detect provider from model_id prefix or pattern. Public API."""
    resolved = _resolve_env_var(model_id)
    for prefix, provider in _PROVIDER_PREFIXES:
        if resolved.lower().startswith(prefix):
            return provider
    for pattern, provider in _PROVIDER_PATTERNS:
        if pattern.match(resolved):
            return provider
    return "litellm"


_detect_provider = detect_provider  # Internal alias for backward compatibility


class ModelVersion:
    """Version tracking for models.

    Use when you need to track model versions (e.g., for caching or A/B testing).
    Not required for typical usage.
    """

    def __init__(self, major: int = 1, minor: int = 0, patch: int = 0) -> None:
        self.major = major
        self.minor = minor
        self.patch = patch

    def __str__(self) -> str:
        return f"{self.major}.{self.minor}.{self.patch}"

    def bump_major(self) -> ModelVersion:
        return ModelVersion(self.major + 1, 0, 0)

    def bump_minor(self) -> ModelVersion:
        return ModelVersion(self.major, self.minor + 1, 0)

    def bump_patch(self) -> ModelVersion:
        return ModelVersion(self.major, self.minor, self.patch + 1)


class ModelVariable:
    """Metadata about a model configuration parameter.

    Used internally for introspection (e.g., UI or validation).
    Developers rarely need to construct this directly.

    Attributes:
        name: Parameter name (e.g., "temperature", "max_tokens").
        type_hint: Expected type (float, int, str, etc.).
        default: Default value if not provided.
        description: Human-readable description for docs/UI.
        required: Whether the parameter must be provided.
    """

    def __init__(
        self,
        name: str,
        type_hint: type,
        default: Any = None,
        description: str = "",
        required: bool = False,
    ) -> None:
        self.name = name
        self.type_hint = type_hint
        self.default = default
        self.description = description
        self.required = required


class ModelSettings:
    """Model-level settings: temperature, tokens, context window, etc.

    Accessed via ``model.settings``. Use these to inspect or validate
    what parameters will be sent to the LLM.

    Attributes:
        context_window: Max input tokens the model supports. None = provider default.
        max_output_tokens: Max completion tokens per call. None = provider default.
        default_reserve_tokens: Tokens reserved for output in context budget.
        temperature: Sampling temperature (0–2). Higher = more random.
        top_p: Nucleus sampling. None = provider default.
        top_k: Top-k sampling. None = provider default.
        stop: Stop sequences. Generation stops when any appears.
        extra: Additional provider-specific settings.
    """

    def __init__(
        self,
        context_window: int | None = None,
        max_output_tokens: int | None = None,
        default_reserve_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        stop: list[str] | None = None,
        **extra: Any,
    ) -> None:
        self.context_window = context_window
        self.max_output_tokens = max_output_tokens
        self.default_reserve_tokens = default_reserve_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.stop = stop
        self.extra = extra


class Middleware:
    """Hook for transforming requests/responses.

    Users can subclass this to add custom transformation layers.

    Example:
        class MyMiddleware(Middleware):
            def transform_request(self, messages, **kwargs):
                return messages, kwargs

            def transform_response(self, response):
                return response
    """

    def transform_request(
        self,
        messages: list[Message],
        **kwargs: Any,
    ) -> tuple[list[Message], dict[str, Any]]:
        return messages, kwargs

    def transform_response(self, response: ProviderResponse) -> ProviderResponse:
        return response


class Model:
    """
    Extensible LLM model — the backend your agents use or call directly.

    You can use a Model in two ways:
    1. **With an Agent**: Pass to ``Agent(model=...)`` — Agent handles tools, memory, budget.
    2. **Directly**: Call ``model.complete(messages)`` or ``await model.acomplete(messages)``
       for simple completions without an Agent.

    Create models via provider namespaces:
        model = Model.OpenAI("gpt-4o-mini", api_key=...)
        model = Model.Anthropic("claude-sonnet", api_key=...)
        model = Model.Custom("deepseek-chat", api_base="...", api_key=...)

    All constructors accept tweakable properties: temperature, max_tokens, context_window,
    output (structured output type), fallback, etc. See the Models guide in the docs.
    """

    # Provider namespace - use Model.OpenAI("gpt-4o"), Model.Anthropic("claude"), etc.
    # These are defined as static methods below for IDE support

    @staticmethod
    def OpenAI(
        model_name: str,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        max_output_tokens: int | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        stop: list[str] | None = None,
        api_key: str | None = None,
        api_base: str | None = None,
        context_window: int | None = None,
        output: type | None = None,
        input_price: float | None = None,
        output_price: float | None = None,
        fallback: list[Model] | None = None,
        **kwargs: Any,
    ) -> Model:
        """Create an OpenAI model.

        Args:
            model_name: Model name (e.g., "gpt-4o", "gpt-4o-mini", "o1").
            temperature: Sampling temperature (0.0–2.0). Higher = more creative.
            max_tokens: Maximum output tokens.
            api_key: API key. Required; pass explicitly (e.g. os.getenv("OPENAI_API_KEY")).
            api_base: Custom base URL for proxies or compatible APIs.

        Returns:
            Model instance configured for OpenAI.

        Example:
            model = Model.OpenAI("gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"))
            response = model.complete(messages)
        """
        import os

        return Model(
            model_id=f"openai/{model_name}",
            name=model_name,
            provider="openai",
            api_base=api_base or os.getenv("OPENAI_BASE_URL") or "https://api.openai.com/v1",
            api_key=api_key,
            context_window=context_window or 128000,
            temperature=temperature,
            max_tokens=max_tokens,
            max_output_tokens=max_output_tokens,
            top_p=top_p,
            top_k=top_k,
            stop=stop,
            output=output,
            input_price=input_price,
            output_price=output_price,
            fallback=fallback,
            **kwargs,
        )

    @staticmethod
    def Anthropic(
        model_name: str,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        max_output_tokens: int | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        stop: list[str] | None = None,
        api_key: str | None = None,
        api_base: str | None = None,
        context_window: int | None = None,
        output: type | None = None,
        input_price: float | None = None,
        output_price: float | None = None,
        fallback: list[Model] | None = None,
        **kwargs: Any,
    ) -> Model:
        """Create an Anthropic Claude model.

        Args:
            model_name: Model name (e.g., "claude-sonnet-4-5", "claude-opus-4-5").
            temperature: Sampling temperature (0.0–1.0).
            api_key: API key. Required; pass explicitly.
            api_base: Custom base URL.

        Returns:
            Model instance configured for Anthropic.
        """
        import os

        return Model(
            model_id=f"anthropic/{model_name}",
            name=model_name,
            provider="anthropic",
            api_base=api_base or os.getenv("ANTHROPIC_BASE_URL") or "https://api.anthropic.com",
            api_key=api_key,
            context_window=context_window or 200000,
            temperature=temperature,
            max_tokens=max_tokens,
            max_output_tokens=max_output_tokens,
            top_p=top_p,
            top_k=top_k,
            stop=stop,
            output=output,
            input_price=input_price,
            output_price=output_price,
            fallback=fallback,
            **kwargs,
        )

    @staticmethod
    def Ollama(
        model_name: str,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        max_output_tokens: int | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        stop: list[str] | None = None,
        api_key: str | None = None,
        api_base: str | None = None,
        context_window: int | None = None,
        output: type | None = None,
        input_price: float | None = None,
        output_price: float | None = None,
        fallback: list[Model] | None = None,
        **kwargs: Any,
    ) -> Model:
        """Create an Ollama (local) model. No API key needed.

        Args:
            model_name: Model name (e.g., "llama3", "mistral").
            api_base: Base URL (default: http://localhost:11434).

        Returns:
            Model instance configured for Ollama.
        """
        import os

        return Model(
            model_id=f"ollama/{model_name}",
            name=model_name,
            provider="ollama",
            api_base=api_base or os.getenv("OLLAMA_BASE_URL") or "http://localhost:11434",
            api_key=api_key,
            context_window=context_window or 8192,
            temperature=temperature,
            max_tokens=max_tokens,
            max_output_tokens=max_output_tokens,
            top_p=top_p,
            top_k=top_k,
            stop=stop,
            output=output,
            input_price=input_price,
            output_price=output_price,
            fallback=fallback,
            **kwargs,
        )

    @staticmethod
    def Google(
        model_name: str,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        max_output_tokens: int | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        stop: list[str] | None = None,
        api_key: str | None = None,
        api_base: str | None = None,
        context_window: int | None = None,
        output: type | None = None,
        input_price: float | None = None,
        output_price: float | None = None,
        fallback: list[Model] | None = None,
        **kwargs: Any,
    ) -> Model:
        """Create a Google Gemini model.

        Args:
            model_name: Model name (e.g., "gemini-2.0-flash", "gemini-1.5-pro").
            api_key: API key. Required; pass explicitly.
            api_base: Custom base URL.

        Returns:
            Model instance configured for Google.
        """
        import os

        return Model(
            model_id=model_name,
            name=model_name,
            provider="google",
            api_base=api_base
            or os.getenv("GOOGLE_BASE_URL")
            or "https://generativelanguage.googleapis.com/v1beta",
            api_key=api_key,
            context_window=context_window or 1048576,
            temperature=temperature,
            max_tokens=max_tokens,
            max_output_tokens=max_output_tokens,
            top_p=top_p,
            top_k=top_k,
            stop=stop,
            output=output,
            input_price=input_price,
            output_price=output_price,
            fallback=fallback,
            **kwargs,
        )

    @staticmethod
    def LiteLLM(
        model_name: str,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        max_output_tokens: int | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        stop: list[str] | None = None,
        api_key: str | None = None,
        api_base: str | None = None,
        context_window: int | None = None,
        output: type | None = None,
        input_price: float | None = None,
        output_price: float | None = None,
        fallback: list[Model] | None = None,
        **kwargs: Any,
    ) -> Model:
        """Create a LiteLLM model. Supports 100+ providers via unified interface.

        Args:
            model_name: Full model ID (e.g., "openai/gpt-4o", "anthropic/claude-3-5-sonnet").
            api_key: API key. Required for most providers.
            api_base: Custom base URL.

        Returns:
            Model instance routed through LiteLLM.
        """
        import os

        name = model_name.split("/")[-1] if "/" in model_name else model_name
        return Model(
            model_id=model_name,
            name=name,
            provider="litellm",
            api_base=api_base or os.getenv("LITELLM_BASE_URL") or "https://api.litellm.ai",
            api_key=api_key,
            context_window=context_window,
            temperature=temperature,
            max_tokens=max_tokens,
            max_output_tokens=max_output_tokens,
            top_p=top_p,
            top_k=top_k,
            stop=stop,
            output=output,
            input_price=input_price,
            output_price=output_price,
            fallback=fallback,
            **kwargs,
        )

    @staticmethod
    def Custom(
        model_id: str,
        *,
        api_base: str,
        provider: str = "openai",
        api_key: str | None = None,
        name: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        max_output_tokens: int | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        stop: list[str] | None = None,
        context_window: int | None = None,
        output: type | None = None,
        input_price: float | None = None,
        output_price: float | None = None,
        fallback: list[Model] | None = None,
        **kwargs: Any,
    ) -> Model:
        """Create a model for any OpenAI-compatible or custom API endpoint.

        Use for third-party providers (DeepSeek, KIMI, Grok, etc.) that expose
        OpenAI-compatible APIs. Default provider is \"openai\"; use \"litellm\"
        for providers routed through LiteLLM.

        Usage:
            Model.Custom("deepseek-chat", api_base="https://api.deepseek.com/v1", api_key="...")
            Model.Custom("grok-3-mini", api_base="https://api.x.ai/v1", api_key="...")

            # Tweak properties (temperature, max_tokens, context_window) - same as Model.OpenAI
            Model.Custom("grok-3", api_base="https://api.x.ai/v1", api_key="...",
                        temperature=0.7, max_tokens=2048, context_window=8192)

        Args:
            model_id: Model identifier (e.g., "deepseek-chat", "grok-3-mini")
            api_base: API base URL (required)
            provider: Provider to use; "openai" for OpenAI-compatible APIs (default)
            api_key: API key (required for most providers)
            name: Display name; derived from model_id if not provided
            **kwargs: Additional Model parameters

        Returns:
            Model instance
        """
        if not model_id or not str(model_id).strip():
            raise ValueError("model_id is required and cannot be empty")
        if not api_base or not str(api_base).strip():
            raise ValueError("api_base is required and cannot be empty")
        provider = (provider or "openai").strip().lower()
        display_name = name or (model_id.split("/")[-1] if "/" in model_id else model_id)
        return Model(
            model_id=model_id,
            name=display_name,
            provider=provider,
            api_base=api_base.strip(),
            api_key=api_key,
            context_window=context_window,
            temperature=temperature,
            max_tokens=max_tokens,
            max_output_tokens=max_output_tokens,
            top_p=top_p,
            top_k=top_k,
            stop=stop,
            output=output,
            input_price=input_price,
            output_price=output_price,
            fallback=fallback,
            **kwargs,
        )

    @staticmethod
    def Almock(
        *,
        pricing_tier: str | None = None,
        context_window: int | None = 8192,
        response_mode: str = "lorem",
        custom_response: str | None = None,
        lorem_length: int = 100,
        latency_min: float = 1.0,
        latency_max: float = 3.0,
        latency_seconds: float | None = None,
        **kwargs: Any,
    ) -> Model:
        """Create an Almock (An LLM Mock) model — no API calls, for testing and development.

        Returns configurable Lorem Ipsum or custom text, optional latency, and pricing
        tiers so you can test budgeting and run examples without an API key.

        Args:
            pricing_tier: One of "low", "medium", "high", "ultra_high" for cost testing.
            context_window: Simulated context window size (default 8192).
            response_mode: "lorem" = Lorem Ipsum of lorem_length; "custom" = custom_response.
            custom_response: Used when response_mode == "custom".
            lorem_length: Output length in characters when response_mode == "lorem".
            latency_min: Min delay in seconds (default 1); ignored if latency_seconds set.
            latency_max: Max delay in seconds (default 3); ignored if latency_seconds set.
            latency_seconds: Fixed delay in seconds; must be > 0. Overrides min/max.

        Returns:
            Model instance that uses AlmockProvider (no API key required).

        Example:
            model = Model.Almock(pricing_tier="medium", lorem_length=50)
            agent = Agent(model=model)
            r = agent.response("Hello")
        """
        from syrin.enums import AlmockPricing
        from syrin.providers.almock import ALMOCK_PRICING

        tier: AlmockPricing = AlmockPricing.MEDIUM
        if isinstance(pricing_tier, AlmockPricing):
            tier = pricing_tier
        elif isinstance(pricing_tier, str):
            tier = AlmockPricing(pricing_tier.lower())
        inp, out = ALMOCK_PRICING.get(tier, ALMOCK_PRICING[AlmockPricing.MEDIUM])

        return Model(
            model_id="almock/default",
            name="almock",
            provider="almock",
            context_window=context_window or 8192,
            input_price=inp,
            output_price=out,
            latency_min=latency_min,
            latency_max=latency_max,
            latency_seconds=latency_seconds,
            response_mode=response_mode,
            custom_response=custom_response,
            lorem_length=lorem_length,
            **kwargs,
        )

    def __init__(
        self,
        model_id: str | None = None,
        *,
        provider: str | None = None,
        name: str | None = None,
        description: str = "",
        version: ModelVersion | None = None,
        fallback: list[Model] | None = None,
        output: type | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        max_output_tokens: int | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        stop: list[str] | None = None,
        api_key: str | None = None,
        api_base: str | None = None,
        context_window: int | None = None,
        default_reserve_tokens: int | None = None,
        pricing: ModelPricing | None = None,
        input_price: float | None = None,
        output_price: float | None = None,
        transformer: Middleware | None = None,
        **provider_kwargs: Any,
    ) -> None:
        # Check if this is a subclass (for custom LLM providers)
        is_subclass = type(self) is not Model

        # If no model_id provided but this is a subclass, allow it
        # (subclasses may override __init__ differently)
        if model_id is None and not is_subclass:
            raise TypeError(
                "Model requires either model_id or provider. "
                "Usage: Model(provider='openai', model_id='gpt-4o') "
                "or Model.Provider('gpt-4o', provider='openai') "
                "or inherit from Model for custom LLM providers."
            )

        self._model_id = _resolve_env_var(model_id) if model_id else ""

        # Use provided provider or detect from model_id
        if provider:
            self._provider = provider.lower()
        else:
            self._provider = _detect_provider(model_id) if model_id else "litellm"

        self._name = (
            name or (self._model_id.split("/")[-1] if "/" in self._model_id else self._model_id)
            if self._model_id
            else ""
        )
        self._description = description
        self._version = version or ModelVersion(1, 0, 0)

        self._api_key = api_key
        self._api_base = api_base

        # Handle pricing
        self._pricing: ModelPricing | None
        if pricing is not None:
            self._pricing = pricing
        elif input_price is not None or output_price is not None:
            self._pricing = ModelPricing(
                input_per_1m=input_price or 0.0,
                output_per_1m=output_price or 0.0,
            )
        else:
            self._pricing = None
        self._output_type = output
        self._transformer = transformer
        self._fallback: list[Model] = list(fallback) if fallback else []

        self._settings = ModelSettings(
            context_window=context_window,
            max_output_tokens=max_output_tokens or max_tokens,
            default_reserve_tokens=default_reserve_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            stop=stop,
        )

        # Strip _internal — must not be forwarded to provider APIs
        provider_kwargs = {k: v for k, v in provider_kwargs.items() if k != "_internal"}
        self._provider_kwargs = provider_kwargs
        self._variables = self._extract_variables()

    def _extract_variables(self) -> list[ModelVariable]:
        """Extract configuration parameters."""
        variables = []

        if self._settings.temperature is not None:
            variables.append(
                ModelVariable(
                    name="temperature",
                    type_hint=float,
                    default=self._settings.temperature,
                    description="Sampling temperature (0.0-2.0)",
                    required=False,
                )
            )

        if self._settings.max_output_tokens is not None:
            variables.append(
                ModelVariable(
                    name="max_tokens",
                    type_hint=int,
                    default=self._settings.max_output_tokens,
                    description="Maximum tokens to generate",
                    required=False,
                )
            )

        if self._settings.top_p is not None:
            variables.append(
                ModelVariable(
                    name="top_p",
                    type_hint=float,
                    default=self._settings.top_p,
                    description="Nucleus sampling parameter",
                    required=False,
                )
            )

        if self._settings.context_window is not None:
            variables.append(
                ModelVariable(
                    name="context_window",
                    type_hint=int,
                    default=self._settings.context_window,
                    description="Maximum context window size",
                    required=False,
                )
            )

        return variables

    @property
    def model_id(self) -> str:
        """Model identifier (e.g., ``openai/gpt-4o``, ``anthropic/claude-sonnet``).

        Used by providers to select the right model. May include provider prefix.
        """
        return self._model_id

    @property
    def name(self) -> str:
        """Human-readable model name for display or logging.

        Usually the short name (e.g., ``gpt-4o``) without provider prefix.
        """
        return self._name

    @property
    def provider(self) -> str:
        """Provider identifier: ``openai``, ``anthropic``, ``ollama``, ``litellm``, etc.

        Determines which backend handles the completion request.
        """
        return self._provider

    @property
    def description(self) -> str:
        """Optional description of the model. For documentation or UI."""
        return self._description

    @property
    def version(self) -> ModelVersion:
        """Version info (major, minor, patch). For versioning or caching."""
        return self._version

    @property
    def metadata(self) -> dict[str, Any]:
        """Metadata dict: model_id, provider, context_window, has_fallback, etc.

        Use for logging, analytics, or passing context to downstream systems.
        """
        return {
            "model_id": self._model_id,
            "provider": self._provider,
            "name": self._name,
            "description": self._description,
            "version": str(self._version),
            "has_fallback": len(self._fallback) > 0,
            "has_output_type": self._output_type is not None,
            "context_window": self._settings.context_window,
            "max_output_tokens": self._settings.max_output_tokens,
        }

    @property
    def variables(self) -> list[ModelVariable]:
        """Extracted configuration parameters (temperature, max_tokens, etc.).

        Used for introspection, validation, or UI generation.
        """
        return self._variables

    @property
    def fallback(self) -> list[Model]:
        """Fallback models used when the primary fails or is rate-limited.

        Set via ``model.with_fallback(other_model, ...)``. Tried in order on error.
        """
        return list(self._fallback)

    @property
    def output_type(self) -> type | None:
        """Pydantic type for structured output, or None for plain text.

        When set, the model requests JSON matching this schema and parses the response.
        Use with ``model.with_output(MyPydanticModel)`` or the ``output=`` constructor arg.
        """
        return self._output_type

    @property
    def settings(self) -> ModelSettings:
        """Model-level settings: temperature, max_output_tokens, context_window, etc.

        Inspect or validate parameters sent to the LLM. Modify via ``with_params()``.
        """
        return self._settings

    @property
    def pricing(self) -> ModelPricing | None:
        """Pricing per 1M tokens (input/output). Used for budget tracking.

        Set via ``input_price``/``output_price`` constructor args or ``ModelPricing``.
        """
        return self._pricing

    @property
    def api_base(self) -> str | None:
        """API base URL. Overrides the default for the provider.

        Use for custom endpoints (e.g., proxies) or third-party APIs.
        """
        return self._api_base

    @property
    def api_key(self) -> str | None:
        """API key for authentication. Must be passed explicitly; never auto-read from env.

        Pass when creating the model, e.g. ``api_key=os.getenv("OPENAI_API_KEY")``.
        """
        return self._api_key

    def to_config(self) -> ModelConfig:
        """Convert to ModelConfig for provider use. Called internally; rarely needed by users."""
        return ModelConfig(
            name=self._name,
            provider=self._provider,
            model_id=self._model_id,
            api_key=self._api_key,
            base_url=self._api_base,
            output=self._output_type,
        )

    def with_params(
        self,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        max_output_tokens: int | None = None,
        default_reserve_tokens: int | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        stop: list[str] | None = None,
        context_window: int | None = None,
        output: type | None = None,
        **kwargs: Any,
    ) -> Model:
        """Return a copy of this model with overridden parameters.

        Use when you need a variant (e.g., different temperature) without mutating the original.

        Returns:
            New Model instance with the given params; others unchanged.
        """
        return Model(
            model_id=self._model_id,
            name=self._name,
            description=self._description,
            version=self._version,
            fallback=self._fallback.copy() if self._fallback else None,
            output=output or self._output_type,
            temperature=temperature if temperature is not None else self._settings.temperature,
            max_tokens=max_tokens,
            max_output_tokens=max_output_tokens or max_tokens or self._settings.max_output_tokens,
            top_p=top_p if top_p is not None else self._settings.top_p,
            top_k=top_k if top_k is not None else self._settings.top_k,
            stop=stop if stop is not None else self._settings.stop,
            api_key=self._api_key,
            api_base=self._api_base,
            context_window=context_window
            if context_window is not None
            else self._settings.context_window,
            default_reserve_tokens=default_reserve_tokens
            if default_reserve_tokens is not None
            else self._settings.default_reserve_tokens,
            pricing=self._pricing,
            transformer=self._transformer,
            _internal=True,
            **self._provider_kwargs,
            **kwargs,
        )

    def with_fallback(self, *models: Model) -> Model:
        """Return a copy with fallback models. Tried in order when the primary fails.

        Use for resilience: e.g. primary Claude, fallback GPT-4o, then local Ollama.

        Returns:
            New Model instance with fallbacks appended.
        """
        new_fallback = self._fallback.copy()
        for m in models:
            new_fallback.append(m)

        return Model(
            model_id=self._model_id,
            name=self._name,
            description=self._description,
            version=self._version,
            fallback=new_fallback,
            output=self._output_type,
            temperature=self._settings.temperature,
            max_output_tokens=self._settings.max_output_tokens,
            top_p=self._settings.top_p,
            top_k=self._settings.top_k,
            stop=self._settings.stop,
            api_key=self._api_key,
            api_base=self._api_base,
            context_window=self._settings.context_window,
            pricing=self._pricing,
            transformer=self._transformer,
            _internal=True,
            **self._provider_kwargs,
        )

    def with_output(
        self,
        output: type,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> Model:
        """Return a copy configured for structured output (Pydantic schema).

        The LLM response will be parsed into the given Pydantic model.

        Returns:
            New Model instance with output type set.
        """
        return self.with_params(
            output=output,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    def with_middleware(self, middleware: Middleware) -> Model:
        """Return a copy with a request/response middleware. For custom transforms."""
        return Model(
            model_id=self._model_id,
            provider=self._provider,
            name=self._name,
            description=self._description,
            version=self._version,
            fallback=self._fallback.copy() if self._fallback else None,
            output=self._output_type,
            temperature=self._settings.temperature,
            max_output_tokens=self._settings.max_output_tokens,
            top_p=self._settings.top_p,
            top_k=self._settings.top_k,
            stop=self._settings.stop,
            api_key=self._api_key,
            api_base=self._api_base,
            context_window=self._settings.context_window,
            pricing=self._pricing,
            transformer=middleware,
            _internal=True,
            **self._provider_kwargs,
        )

    def _get_provider_instance(self) -> Any:
        """Get the provider instance for this model."""
        if self._provider == "almock":
            from syrin.providers.almock import AlmockProvider

            return AlmockProvider()
        if self._provider == "anthropic":
            from syrin.providers.anthropic import AnthropicProvider

            return AnthropicProvider()
        if self._provider == "openai":
            from syrin.providers.openai import OpenAIProvider

            return OpenAIProvider()
        if self._provider in ("ollama", "litellm"):
            from syrin.providers.litellm import LiteLLMProvider

            return LiteLLMProvider()
        from syrin.providers.litellm import LiteLLMProvider

        return LiteLLMProvider()

    def get_provider(self) -> Provider:
        """Return the Provider instance for this model.

        Use when you need the low-level Provider (e.g. Agent uses this to call complete).
        """
        from syrin.providers.base import Provider as ProviderCls

        return cast(ProviderCls, self._get_provider_instance())

    def complete(
        self,
        messages: list[Message],
        *,
        tools: list[ToolSpec] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        stream: bool = False,
        **kwargs: Any,
    ) -> ProviderResponse | Iterator[ProviderResponse]:
        """Send messages to the LLM and return a response (sync).

        Use this when you want to call the model directly without an Agent.
        Pass a list of ``Message`` objects; returns a ``ProviderResponse``.

        Args:
            messages: Conversation messages (system, user, assistant, tool).
            tools: Optional tool specs for function calling.
            temperature: Override sampling temperature.
            max_tokens: Override max output tokens.
            stream: If True, returns an iterator of response chunks.

        Returns:
            ProviderResponse with content, tool_calls, token_usage; or iterator if stream=True.

        Example:
            response = model.complete([
                Message(role=MessageRole.USER, content="Hello"),
            ])
            print(response.content)
        """
        provider = self._get_provider_instance()

        settings = {
            "temperature": temperature if temperature is not None else self._settings.temperature,
            "max_tokens": max_tokens or self._settings.max_output_tokens,
            "top_p": self._settings.top_p,
            "stop": self._settings.stop,
            **self._provider_kwargs,
            **kwargs,
        }

        transformer_result = self._apply_transformer("request", messages, **settings)
        if isinstance(transformer_result, tuple):
            messages, settings = transformer_result
        else:
            # Unexpected single response during request phase
            raise ProviderError(
                "Transformer returned unexpected response type during request phase"
            )

        try:
            if stream:
                return cast(
                    Iterator[ProviderResponse],
                    provider.stream_sync(messages, self.to_config(), tools, **settings),
                )

            response = provider.complete_sync(
                messages=messages,
                model=self.to_config(),
                tools=tools,
                **settings,
            )

            if response is None:
                raise ProviderError(f"Provider {self._provider} returned no response")

            transformer_response = self._apply_transformer("response", response)
            if isinstance(transformer_response, tuple):
                raise ProviderError("Transformer returned tuple instead of response")
            response = transformer_response

            if self._output_type and response.content:
                response = self._parse_structured_output(response)

            return response
        except Exception:
            if self._fallback:
                return self._try_fallback(messages, tools=tools, **kwargs)
            raise

    async def acomplete(
        self,
        messages: list[Message],
        *,
        tools: list[ToolSpec] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        stream: bool = False,
        **kwargs: Any,
    ) -> ProviderResponse | AsyncIterator[ProviderResponse]:
        """Send messages to the LLM and return a response (async).

        Async variant of ``complete()``. Use ``await model.acomplete(messages)``.

        Args:
            messages: Conversation messages (system, user, assistant, tool).
            tools: Optional tool specs for function calling.
            temperature: Override sampling temperature.
            max_tokens: Override max output tokens.
            stream: If True, returns an async iterator of response chunks.

        Returns:
            ProviderResponse; or async iterator if stream=True.
        """
        if stream:
            return self._astream_internal(
                messages, tools=tools, temperature=temperature, max_tokens=max_tokens, **kwargs
            )

        provider = self._get_provider_instance()

        settings = {
            "temperature": temperature if temperature is not None else self._settings.temperature,
            "max_tokens": max_tokens or self._settings.max_output_tokens,
            "top_p": self._settings.top_p,
            "stop": self._settings.stop,
            **self._provider_kwargs,
            **kwargs,
        }

        transformer_result = self._apply_transformer("request", messages, **settings)
        if isinstance(transformer_result, tuple):
            messages, settings = transformer_result
        else:
            # Unexpected single response during request phase
            raise ProviderError(
                "Transformer returned unexpected response type during request phase"
            )

        try:
            response = await provider.complete(
                messages=messages,
                model=self.to_config(),
                tools=tools,
                **settings,
            )

            transformer_response = self._apply_transformer("response", response)
            if isinstance(transformer_response, tuple):
                raise ProviderError("Transformer returned tuple instead of response")
            response = transformer_response

            if self._output_type and response.content:
                response = self._parse_structured_output(response)

            return response
        except Exception:
            if self._fallback:
                return await self._atry_fallback(messages, tools=tools, **kwargs)
            raise

    async def _astream_internal(
        self,
        messages: list[Message],
        *,
        tools: list[ToolSpec] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[ProviderResponse]:
        """Internal async streaming implementation."""
        provider = self._get_provider_instance()

        settings = {
            "temperature": temperature if temperature is not None else self._settings.temperature,
            "max_tokens": max_tokens or self._settings.max_output_tokens,
            "top_p": self._settings.top_p,
            "stop": self._settings.stop,
            **self._provider_kwargs,
            **kwargs,
        }

        transformer_result = self._apply_transformer("request", messages, **settings)
        if isinstance(transformer_result, tuple):
            messages, settings = transformer_result
        else:
            # Unexpected single response during request phase
            raise ProviderError(
                "Transformer returned unexpected response type during request phase"
            )

        try:
            async for chunk in provider.stream(messages, self.to_config(), tools, **settings):
                yield chunk
        except Exception:
            if self._fallback:
                async for chunk in self._astream_fallback(messages, tools=tools, **kwargs):
                    yield chunk
            else:
                raise

    async def astream(
        self,
        messages: list[Message],
        *,
        tools: list[ToolSpec] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[ProviderResponse]:
        """Stream response chunks asynchronously. Yields ProviderResponse per chunk."""
        async for chunk in self._astream_internal(
            messages,
            tools=tools,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        ):
            yield chunk

    def count_tokens(self, text: str) -> int:
        """Count tokens for the given text. Used for budget and context limits."""
        from syrin.cost import count_tokens

        return count_tokens(text, self._model_id)

    def get_pricing(self) -> ModelPricing | None:
        """Return pricing info (USD per 1M tokens). Used for budget calculations."""
        if self._pricing is not None:
            return self._pricing

        from syrin.cost import _resolve_pricing

        inp, out = _resolve_pricing(self._model_id)
        if inp > 0 or out > 0:
            return ModelPricing(input_per_1m=inp, output_per_1m=out)
        return None

    def estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Estimate cost in USD for a call with given token counts.

        Use before calling the LLM to check affordability. Uses model pricing.

        Args:
            input_tokens: Number of input tokens.
            output_tokens: Number of output (completion) tokens.

        Returns:
            Estimated cost in USD. 0.0 if pricing unknown.

        Example:
            >>> cost = model.estimate_cost(1000, 500)
        """
        pricing = self.get_pricing()
        if pricing is None:
            return 0.0
        return round(
            (input_tokens / 1_000_000) * pricing.input_per_1m
            + (output_tokens / 1_000_000) * pricing.output_per_1m,
            6,
        )

    @overload
    def _apply_transformer(
        self,
        phase: Literal["request"],
        messages_or_response: list[Message],
        **kwargs: Any,
    ) -> tuple[list[Message], dict[str, Any]]: ...

    @overload
    def _apply_transformer(
        self,
        phase: Literal["response"],
        messages_or_response: ProviderResponse,
        **kwargs: Any,
    ) -> ProviderResponse: ...

    def _apply_transformer(
        self,
        phase: str,
        messages_or_response: list[Message] | ProviderResponse,
        **kwargs: Any,
    ) -> tuple[list[Message], dict[str, Any]] | ProviderResponse:
        """Apply response transformer if set."""
        if self._transformer is None:
            if phase == "request":
                # Cast to list[Message] since phase is "request"
                return cast(list[Message], messages_or_response), kwargs
            # Cast to ProviderResponse since phase is "response"
            return cast(ProviderResponse, messages_or_response)

        if phase == "request":
            return self._transformer.transform_request(
                cast(list[Message], messages_or_response), **kwargs
            )
        else:
            return self._transformer.transform_response(
                cast(ProviderResponse, messages_or_response)
            )

    def _try_fallback(
        self,
        messages: list[Message],
        *,
        tools: list[ToolSpec] | None = None,
        **kwargs: Any,
    ) -> ProviderResponse | Iterator[ProviderResponse]:
        """Try fallback models in order."""
        for fb in self._fallback:
            try:
                return fb.complete(messages, tools=tools, **kwargs)
            except Exception:
                continue
        raise ProviderError("All fallback models failed")

    async def _atry_fallback(
        self,
        messages: list[Message],
        *,
        tools: list[ToolSpec] | None = None,
        **kwargs: Any,
    ) -> ProviderResponse | AsyncIterator[ProviderResponse]:
        """Try fallback models in order (async)."""
        for fb in self._fallback:
            try:
                return await fb.acomplete(messages, tools=tools, **kwargs)
            except Exception:
                continue
        raise ProviderError("All fallback models failed")

    async def _astream_fallback(
        self,
        messages: list[Message],
        *,
        tools: list[ToolSpec] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[ProviderResponse]:
        """Try fallback models in order for streaming."""
        for fb in self._fallback:
            try:
                async for chunk in fb.astream(messages, tools=tools, **kwargs):
                    yield chunk
                return
            except Exception:
                continue
        raise ProviderError("All fallback models failed")

    def _parse_structured_output(self, response: ProviderResponse) -> ProviderResponse:
        """Parse response content into structured output type."""
        if not self._output_type or not response.content:
            return response

        try:
            import json

            content = response.content.strip()

            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
                content = content.strip()

            data = json.loads(content)

            # Cast to BaseModel type since we've checked _output_type is not None
            output_type = cast(type[BaseModel], self._output_type)
            if isinstance(data, dict):
                parsed = output_type.model_validate(data)
            else:
                parsed = output_type.model_validate_json(response.content)

            response.content = parsed.model_dump_json()
            response.raw_response = parsed

        except Exception as e:
            raise ProviderError(
                f"Failed to parse structured output: {e}. "
                f"Expected {getattr(self._output_type, '__name__', str(self._output_type))}, got: {response.content[:200]}"
            ) from e

        return response

    @classmethod
    def _create(cls, **kwargs: Any) -> Model:
        """Internal method for creating Model instances (for testing).

        Use provider namespaces or inherit from Model instead:
            Model.OpenAI('gpt-4o')
            Model.Anthropic('claude-sonnet')

            class MyModel(Model):
                ...
        """
        return cls(**kwargs, _internal=True)

    def __repr__(self) -> str:
        fallback_info = f", {len(self._fallback)} fallbacks" if self._fallback else ""
        output_info = ""
        if self._output_type:
            if hasattr(self._output_type, "__name__"):
                output_info = f", output={self._output_type.__name__}"
            else:
                output_info = f", output={type(self._output_type).__name__}"
        return f"Model({self._model_id!r}, provider={self._provider!r}{fallback_info}{output_info})"


class ModelRegistry:
    """Singleton registry for named models. Use for dynamic lookup by name.

    Register models once, then get them by string (e.g., from config). Useful when
    the model choice is driven by config or feature flags.

    Example:
        >>> registry = ModelRegistry()
        >>> registry.register("default", Model.OpenAI("gpt-4o-mini"))
        >>> registry.register("fallback", Model.Anthropic("claude-3-haiku"))
        >>> model = registry.get("default")
    """

    _instance: ModelRegistry | None = None
    _models: dict[str, Model]

    def __new__(cls) -> ModelRegistry:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._models = {}
        return cls._instance

    def register(self, name: str, model: Model) -> None:
        """Register a model under a name for later lookup."""
        self._models[name] = model

    def get(self, name: str) -> Model:
        """Return the model registered under name. Raises ModelNotFoundError if missing."""
        if name not in self._models:
            raise ModelNotFoundError(f"Model not found: {name}")
        return self._models[name]

    def list_names(self) -> list[str]:
        """Return all registered model names."""
        return list(self._models)

    def clear(self) -> None:
        """Remove all registered models. Mainly for tests."""
        self._models.clear()


__all__ = [
    "Model",
    "ModelVersion",
    "ModelVariable",
    "ModelSettings",
    "ModelRegistry",
    "Middleware",
]
