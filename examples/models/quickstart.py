"""
Syrin Model Examples - Complete Guide to Creating and Using Models

This examples file demonstrates:
1. Basic model creation with provider namespaces
2. Structured output for type-safe responses
3. Fallback chains for reliability
4. Response transformers for custom processing
5. Creating custom model classes for new LLMs

Run: python -m examples.models.quickstart

Note: API keys must be passed explicitly. Examples use os.getenv for convenience.
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

# =============================================================================
# Organized Imports - By Category
# =============================================================================

# Core
from syrin import Model

# Structured Output


# =============================================================================
# Example 1: Basic Model Creation with Provider Namespaces
# =============================================================================


def example_basic_models():
    """The simplest way to create a model - use provider namespaces."""

    print("=" * 60)
    print("Example 1: Basic Model Creation")
    print("=" * 60)

    # OpenAI models - pass api_key explicitly
    model = Model.OpenAI("gpt-4o", api_key=os.getenv("OPENAI_API_KEY"))
    print(f"OpenAI gpt-4o: {model}")
    print(f"  - Provider: {model.provider}")
    print(f"  - Model ID: {model.model_id}")
    print(f"  - Context Window: {model.settings.context_window}")

    # Anthropic models
    model = Model.Anthropic("claude-sonnet-4-5-20241022", api_key=os.getenv("ANTHROPIC_API_KEY"))
    print(f"\nAnthropic claude-sonnet: {model}")
    print(f"  - Provider: {model.provider}")
    print(f"  - Model ID: {model.model_id}")

    # Ollama (local)
    model = Model.Ollama("llama3")
    print(f"\nOllama llama3: {model}")
    print(f"  - Provider: {model.provider}")

    # Google
    model = Model.Google("gemini-2.0-flash", api_key=os.getenv("GOOGLE_API_KEY"))
    print(f"\nGoogle gemini-2.0-flash: {model}")
    print(f"  - Provider: {model.provider}")


# =============================================================================
# Example 2: Model with Configuration
# =============================================================================


def example_model_config():
    """Create model with custom configuration."""

    print("\n" + "=" * 60)
    print("Example 2: Model with Configuration")
    print("=" * 60)

    # Tweak properties - Model.OpenAI (and Model.Anthropic, Model.Custom, etc.) support:
    # temperature, max_tokens, top_p, top_k, stop, context_window, output, input_price, output_price
    model = Model.OpenAI(
        "gpt-4o",
        api_key=os.getenv("OPENAI_API_KEY"),
        temperature=0.7,
        max_tokens=2048,
        context_window=128000,
    )
    print(f"Model.OpenAI with tweaked properties: {model}")
    print(f"  - Temperature: {model.settings.temperature}")
    print(f"  - Max tokens: {model.settings.max_output_tokens}")
    print(f"  - Context window: {model.settings.context_window}")

    # With custom API key and base URL
    model = Model.OpenAI(
        "gpt-4o", api_key="sk-custom-key", api_base="https://custom.endpoint.com/v1"
    )
    print(f"\nWith custom API: {model}")
    print(f"  - API Key: {model.api_key[:10]}...")
    print(f"  - API Base: {model.api_base}")

    # Third-party OpenAI-compatible APIs (DeepSeek, KIMI, Grok, etc.) - use Model.Custom
    model = Model.Custom(
        "deepseek-chat",
        api_base="https://api.deepseek.com/v1",
        api_key=os.getenv("DEEPSEEK_API_KEY"),
    )
    print(f"\nModel.Custom (third-party): {model}")

    # Model.Custom with tweaked properties (same as Model.OpenAI)
    model = Model.Custom(
        "grok-3-mini",
        api_base="https://api.x.ai/v1",
        api_key=os.getenv("XAI_API_KEY"),
        temperature=0.7,
        max_tokens=2048,
        context_window=8192,
    )
    print(f"\nModel.Custom with params: {model}")
    print(f"  - Temperature: {model.settings.temperature}")
    print(f"  - Max tokens: {model.settings.max_output_tokens}")


# =============================================================================
# Example 3: Structured Output
# =============================================================================


def example_structured_output():
    """Use structured output for type-safe responses."""

    print("\n" + "=" * 60)
    print("Example 3: Structured Output")
    print("=" * 60)

    from pydantic import BaseModel

    class SentimentAnalysis(BaseModel):
        sentiment: str
        confidence: float

    # With structured output
    model = Model.OpenAI("gpt-4o", output=SentimentAnalysis, api_key=os.getenv("OPENAI_API_KEY"))
    print(f"Structured output: {model}")
    print(f"  - Output type: {model.output_type}")


# =============================================================================
# Example 4: Fallback Chains
# =============================================================================


def example_fallback():
    """Create fallback chains for reliability."""

    print("\n" + "=" * 60)
    print("Example 4: Fallback Chains")
    print("=" * 60)

    # Using with_fallback
    openai_key = os.getenv("OPENAI_API_KEY")
    model = Model.Anthropic("claude-sonnet", api_key=os.getenv("ANTHROPIC_API_KEY")).with_fallback(
        Model.OpenAI("gpt-4o", api_key=openai_key),
        Model.OpenAI("gpt-4o-mini", api_key=openai_key),  # Cheaper fallback
        Model.Ollama("llama3"),  # Local fallback - no api_key
    )
    print(f"With fallback: {model}")
    print(f"  - Fallbacks: {len(model.fallback)}")

    # Fallback with structured output
    class SentimentAnalysis(BaseModel):
        sentiment: str

    model = Model.Anthropic(
        "claude-sonnet", output=SentimentAnalysis, api_key=os.getenv("ANTHROPIC_API_KEY")
    ).with_fallback(
        Model.OpenAI("gpt-4o", output=SentimentAnalysis, api_key=os.getenv("OPENAI_API_KEY"))
    )
    print(f"\nWith fallback + structured output: {model}")


# =============================================================================
# Example 5: Custom Model via Inheritance
# =============================================================================


def example_custom_model():
    """Create custom model for new LLM providers."""

    print("\n" + "=" * 60)
    print("Example 5: Custom Model via Inheritance")
    print("=" * 60)

    class MyCustomModel(Model):
        """Custom model for any LLM API."""

        def complete(self, messages, **kwargs):
            # Your custom implementation here
            print(f"  - Called with {len(messages)} messages")
            return None

    model = MyCustomModel("my-model")
    print(f"Custom model: {model}")
    print(f"  - Provider: {model.provider}")


# =============================================================================
# Main
# =============================================================================


if __name__ == "__main__":
    example_basic_models()
    example_model_config()
    example_structured_output()
    example_fallback()
    example_custom_model()
