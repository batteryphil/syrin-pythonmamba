"""Cost calculation and token counting for budget tracking."""

from __future__ import annotations

from collections.abc import Callable

from syrin.types import Message, TokenUsage

# USD per 1M tokens (input, output). Keys: model_id or prefix pattern (first match wins).
# Sources: Anthropic/OpenAI public pricing; update as needed.
# IMPORTANT: More specific models must come before general ones!
MODEL_PRICING: dict[str, tuple[float, float]] = {
    # OpenAI - specific models first (more specific to less specific)
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (5.0, 15.0),
    "gpt-4-turbo": (10.0, 30.0),
    "gpt-4": (30.0, 60.0),
    "gpt-3.5-turbo": (0.50, 1.50),
    "o1-preview": (15.0, 60.0),
    "o1-mini": (3.0, 12.0),
    # Anthropic - specific models first
    "claude-opus-4-6": (15.0, 75.0),
    "claude-opus-4-5": (15.0, 75.0),
    "claude-opus-4": (15.0, 75.0),
    "claude-sonnet-4-5": (3.0, 15.0),
    "claude-sonnet-4": (3.0, 15.0),
    "claude-3-7-sonnet": (3.0, 15.0),
    "claude-3-5-sonnet": (3.0, 15.0),
    "claude-3-5-haiku": (1.0, 5.0),
    "claude-haiku-4": (1.0, 5.0),
    "claude-3-opus": (15.0, 75.0),
    "claude-3-sonnet": (3.0, 15.0),
    "claude-3-haiku": (0.25, 1.25),
    # Google
    "gemini-2.0-flash": (0.0, 0.0),  # Check current pricing
    "gemini-1.5-pro": (1.25, 5.0),
    "gemini-1.5-flash": (0.075, 0.30),
    "gemini-3-flash-preview": (0.075, 0.30),  # gemini-3-flash pricing
}

# Per-image USD. Used by image generation providers to populate GenerationResult.metadata.
# Keys: model_id or prefix (first match wins). Sources: OpenAI, Google public pricing.
IMAGE_PRICING: dict[str, float] = {
    "dall-e-3": 0.08,  # 1024x1024 standard
    "imagen-4.0-generate-001": 0.067,  # 1024px default
    "imagen-3": 0.04,  # legacy
    "gpt-image-1": 0.10,
    "gpt-image-1-mini": 0.02,
}

# Per-second USD for video. Used by video generation providers.
VIDEO_PRICING: dict[str, float] = {
    "veo-2.0-generate-001": 0.35,  # Gemini API
    "veo-2": 0.35,
}

# Per 1M tokens USD for embeddings. Used by embedding providers.
# Keys: model_id or prefix (first match wins). Sources: OpenAI public pricing.
EMBEDDING_PRICING: dict[str, float] = {
    "text-embedding-3-small": 0.02,  # $0.02 per 1M tokens
    "text-embedding-3-large": 0.13,  # $0.13 per 1M tokens
    "text-embedding-ada-002": 0.10,  # $0.10 per 1M tokens
}

# Default video duration (seconds) when provider does not report it
_DEFAULT_VIDEO_DURATION_SECONDS = 5.0


def _resolve_image_pricing(model_id: str) -> float:
    """Return USD per image for model_id. Strips provider prefix."""
    if not model_id:
        return 0.0
    normalized = model_id.split("/")[-1] if "/" in model_id else model_id
    for key, price in IMAGE_PRICING.items():
        if normalized.startswith(key):
            return price
    return 0.0


def _resolve_video_pricing(model_id: str) -> float:
    """Return USD per second for model_id."""
    if not model_id:
        return 0.0
    normalized = model_id.split("/")[-1] if "/" in model_id else model_id
    for key, price in VIDEO_PRICING.items():
        if normalized.startswith(key):
            return price
    return 0.0


def _resolve_embedding_pricing(model_id: str) -> float:
    """Return USD per 1M tokens for embedding model_id."""
    if not model_id:
        return 0.0
    normalized = model_id.split("/")[-1] if "/" in model_id else model_id
    for key, price in EMBEDDING_PRICING.items():
        if normalized.startswith(key):
            return price
    return 0.0


def calculate_image_cost(
    model_id: str,
    number_of_images: int = 1,
) -> float:
    """Compute cost in USD for image generation.

    Used by providers to populate GenerationResult.metadata["cost_usd"].
    Returns 0.0 if model is unknown or number_of_images <= 0.

    Args:
        model_id: Model identifier (e.g. dall-e-3, imagen-4.0-generate-001).
        number_of_images: Number of images generated.

    Returns:
        Cost in USD, rounded to 6 decimal places.
    """
    if number_of_images <= 0:
        return 0.0
    price = _resolve_image_pricing(model_id)
    return round(price * number_of_images, 6)


def calculate_video_cost(
    model_id: str,
    duration_seconds: float = _DEFAULT_VIDEO_DURATION_SECONDS,
) -> float:
    """Compute cost in USD for video generation.

    Used by providers to populate GenerationResult.metadata["cost_usd"].
    Returns 0.0 if model is unknown or duration <= 0.

    Args:
        model_id: Model identifier (e.g. veo-2.0-generate-001).
        duration_seconds: Video length in seconds. Default 5.0 when unknown.

    Returns:
        Cost in USD, rounded to 6 decimal places.
    """
    if duration_seconds <= 0:
        return 0.0
    price = _resolve_video_pricing(model_id)
    return round(price * duration_seconds, 6)


def calculate_embedding_cost(
    model_id: str,
    token_count: int,
) -> float:
    """Compute cost in USD for embedding generation.

    Used by embedding providers to track costs in BudgetTracker.
    Returns 0.0 if model is unknown or token_count <= 0.

    Args:
        model_id: Model identifier (e.g. text-embedding-3-small).
        token_count: Number of tokens processed.

    Returns:
        Cost in USD, rounded to 6 decimal places.

    Example:
        >>> calculate_embedding_cost("text-embedding-3-small", 1_000_000)
        0.02
    """
    if token_count <= 0:
        return 0.0
    price = _resolve_embedding_pricing(model_id)
    return round(price * (token_count / 1_000_000), 6)


class ModelPricing:
    """Custom pricing per 1M tokens (input, output) in USD.

    Override MODEL_PRICING for custom models or when you have negotiated rates.
    Pass to Model(pricing=ModelPricing(...)) or calculate_cost(pricing_override=...).

    Attributes:
        input_per_1m: USD per 1M input/prompt tokens.
        output_per_1m: USD per 1M output/completion tokens.
    """

    def __init__(
        self,
        input_per_1m: float = 0.0,
        output_per_1m: float = 0.0,
    ) -> None:
        self.input_per_1m = input_per_1m
        self.output_per_1m = output_per_1m


Pricing = ModelPricing


def _resolve_pricing(model_id: str) -> tuple[float, float]:
    """Return (input_per_1m, output_per_1m) for model_id. Strips provider prefix."""
    import warnings

    normalized = model_id.split("/")[-1] if "/" in model_id else model_id
    for key, (inp, out) in MODEL_PRICING.items():
        if normalized.startswith(key):
            return (inp, out)
    warnings.warn(
        f"No pricing data for model '{model_id}'. Cost tracking will show $0.00. "
        "Pass input_price/output_price to Model() or ModelPricing to set pricing.",
        stacklevel=2,
    )
    return (0.0, 0.0)


def calculate_cost(
    model_id: str,
    token_usage: TokenUsage,
    pricing_override: Pricing | None = None,
    pricing_resolver: Callable[[str], tuple[float, float]] | None = None,
) -> float:
    """Compute cost in USD for the given token usage and model.

    Args:
        model_id: Model identifier (e.g. openai/gpt-4o-mini). Used for MODEL_PRICING lookup.
        token_usage: Input/output token counts (from ProviderResponse or accumulated).
        pricing_override: Per-call fixed pricing (ModelPricing). Overridden by resolver.
        pricing_resolver: Optional callable(model_id) -> (input_per_1m, output_per_1m).
            When provided, used instead of pricing_override and MODEL_PRICING.

    Returns:
        Cost in USD, rounded to 6 decimal places. 0.0 if pricing unknown.

    Example:
        >>> from syrin.types import TokenUsage
        >>> calculate_cost("openai/gpt-4o-mini", TokenUsage(input_tokens=100, output_tokens=50))
        0.000045
    """
    if pricing_resolver is not None:
        inp_p, out_p = pricing_resolver(model_id)
    elif pricing_override is not None:
        inp_p, out_p = pricing_override.input_per_1m, pricing_override.output_per_1m
    else:
        inp_p, out_p = _resolve_pricing(model_id)
    input_cost = (token_usage.input_tokens / 1_000_000) * inp_p
    output_cost = (token_usage.output_tokens / 1_000_000) * out_p
    return round(input_cost + output_cost, 6)


def count_tokens(text: str, model_id: str) -> int:
    """Estimate token count for text.

    Uses tiktoken for OpenAI-style models (gpt-4, gpt-3.5) when available;
    otherwise falls back to ~4 characters per token for English.

    Args:
        text: Text to count.
        model_id: Model ID (e.g. openai/gpt-4o-mini). Used to select encoding.

    Returns:
        Estimated token count.

    Example:
        >>> count_tokens("Hello, world!", "openai/gpt-4o-mini")
        4
    """
    if not text:
        return 0
    normalized = model_id.split("/")[-1] if "/" in model_id else model_id
    try:
        import tiktoken
    except ImportError:
        return _estimate_tokens(text)
    encoding_name = "cl100k_base"
    if "gpt-4" in normalized or "gpt-3.5" in normalized or "gpt-4o" in normalized:
        try:
            enc = tiktoken.encoding_for_model("gpt-4" if "gpt-4" in normalized else "gpt-3.5-turbo")
            return len(enc.encode(text))
        except Exception:
            enc = tiktoken.get_encoding(encoding_name)
            return len(enc.encode(text))
    try:
        enc = tiktoken.get_encoding(encoding_name)
        return len(enc.encode(text))
    except Exception:
        return _estimate_tokens(text)


def _estimate_tokens(text: str) -> int:
    """Fallback: ~4 characters per token for English."""
    return max(0, (len(text) + 3) // 4)


def estimate_cost_for_call(
    model_id: str,
    messages: list[Message],
    max_output_tokens: int | None = 1024,
    pricing_override: Pricing | None = None,
    pricing_resolver: Callable[[str], tuple[float, float]] | None = None,
) -> float:
    """Estimate cost in USD for a single LLM call (best-effort).

    Counts input tokens from message contents via count_tokens; uses max_output_tokens
    for output. Actual cost may differ. Use for pre-call budget checks (e.g. before
    calling the LLM to see if the run would exceed budget).

    Args:
        model_id: Model identifier (e.g. openai/gpt-4o-mini).
        messages: List of message-like objects with .content (str).
        max_output_tokens: Assumed max completion tokens (default 1024).
        pricing_override: Optional ModelPricing. Overrides MODEL_PRICING.
        pricing_resolver: Optional callable(model_id) -> (input_per_1m, output_per_1m).

    Returns:
        Estimated cost in USD. 0.0 if pricing unknown.
    """
    out_tokens = 1024 if max_output_tokens is None else max_output_tokens
    input_tokens = 0
    for m in messages:
        content = getattr(m, "content", None) or ""
        if isinstance(content, str):
            input_tokens += count_tokens(content, model_id)
    usage = TokenUsage(
        input_tokens=input_tokens,
        output_tokens=out_tokens,
        total_tokens=input_tokens + out_tokens,
    )
    return calculate_cost(
        model_id,
        usage,
        pricing_override=pricing_override,
        pricing_resolver=pricing_resolver,
    )


__all__ = [
    "EMBEDDING_PRICING",
    "IMAGE_PRICING",
    "ModelPricing",
    "MODEL_PRICING",
    "Pricing",
    "VIDEO_PRICING",
    "calculate_cost",
    "calculate_embedding_cost",
    "calculate_image_cost",
    "calculate_video_cost",
    "count_tokens",
    "estimate_cost_for_call",
]
