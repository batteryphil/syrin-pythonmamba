"""Tests for Model and ModelRegistry (model.py)."""

from __future__ import annotations

import pytest

from syrin.cost import ModelPricing
from syrin.exceptions import ModelNotFoundError
from syrin.model import Model, ModelRegistry
from syrin.types import ModelConfig


def test_model_creation_and_provider_detection() -> None:
    m = Model("anthropic/claude-3-5-sonnet-20241022")
    assert m.model_id == "anthropic/claude-3-5-sonnet-20241022"
    assert m.provider == "anthropic"


def test_model_openai_prefix() -> None:
    m = Model("openai/gpt-4")
    assert m.provider == "openai"


def test_model_gpt_pattern() -> None:
    m = Model("gpt-4-turbo")
    assert m.provider == "openai"


def test_model_ollama() -> None:
    m = Model("ollama/llama2")
    assert m.provider == "ollama"


def test_model_fallback_litellm() -> None:
    m = Model("custom/unknown-model")
    assert m.provider == "litellm"


def test_model_to_config() -> None:
    m = Model("anthropic/claude-3", api_key="sk-fake", base_url=None)
    cfg = m.to_config()
    assert isinstance(cfg, ModelConfig)
    assert cfg.provider == "anthropic"
    assert cfg.model_id == "anthropic/claude-3"
    assert cfg.api_key == "sk-fake"


def test_model_repr() -> None:
    m = Model("openai/gpt-4")
    r = repr(m)
    assert "openai/gpt-4" in r
    assert "openai" in r


def test_model_env_var_resolution(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MY_MODEL", "anthropic/claude-3-5-sonnet")
    m = Model("$MY_MODEL")
    assert m.model_id == "anthropic/claude-3-5-sonnet"
    assert m.provider == "anthropic"


def test_model_env_var_unset_returns_raw() -> None:
    m = Model("$NONEXISTENT_VAR_12345")
    assert m.model_id == "$NONEXISTENT_VAR_12345"


def test_registry_singleton() -> None:
    r1 = ModelRegistry()
    r2 = ModelRegistry()
    assert r1 is r2


def test_registry_register_and_get() -> None:
    reg = ModelRegistry()
    reg._models.clear()
    m = Model("anthropic/claude-3")
    reg.register("default", m)
    got = reg.get("default")
    assert got is m
    assert reg.list_names() == ["default"]


def test_registry_get_missing_raises() -> None:
    reg = ModelRegistry()
    reg._models.clear()
    with pytest.raises(ModelNotFoundError):
        reg.get("nonexistent")


def test_model_estimate_cost_returns_float() -> None:
    """estimate_cost(input_tokens, output_tokens) returns USD cost."""
    m = Model("openai/gpt-4o-mini", pricing=ModelPricing(input_per_1m=0.15, output_per_1m=0.60))
    cost = m.estimate_cost(1000, 500)
    assert isinstance(cost, float)
    assert cost >= 0.0


def test_model_estimate_cost_zero_tokens() -> None:
    """estimate_cost(0, 0) returns 0.0."""
    m = Model("openai/gpt-4", pricing=ModelPricing(input_per_1m=1.0, output_per_1m=2.0))
    assert m.estimate_cost(0, 0) == 0.0


def test_model_estimate_cost_no_pricing_returns_zero() -> None:
    """estimate_cost when pricing unknown returns 0.0."""
    m = Model("unknown/foo-model-no-pricing")
    assert m.estimate_cost(1000, 500) == 0.0


# =============================================================================
# AGGRESSIVE EDGE CASES - TRY TO BREAK MODEL
# =============================================================================


def test_model_bare_claude_name() -> None:
    """Bare claude model names should detect provider."""
    m = Model("claude-4")
    assert m.provider == "anthropic"


def test_model_bare_gemini_name() -> None:
    """Bare gemini model names should detect provider."""
    m = Model("gemini-2")
    assert m.provider == "google"


def test_model_bare_gpt_name() -> None:
    """Bare gpt model names should detect provider."""
    m = Model("gpt-5")
    assert m.provider == "openai"


def test_model_with_very_long_name() -> None:
    """Very long model names should work."""
    long_name = "a" * 1000
    m = Model(long_name)
    assert m.model_id == long_name


def test_model_special_chars_in_name() -> None:
    """Special characters in model names."""
    m = Model("model-v2.0_alpha-test")
    assert m.provider == "litellm"  # Falls back to litellm


def test_model_case_sensitivity() -> None:
    """Provider detection should be case-sensitive or handle properly."""
    m = Model("GPT-4")
    assert m.provider == "openai"  # Should normalize or handle


def test_model_unicode_in_name() -> None:
    """Unicode characters in model names."""
    m = Model("模型-4")
    assert m.provider == "litellm"  # Falls back


def test_registry_with_many_models() -> None:
    """Registry handles many registered models."""
    reg = ModelRegistry()
    reg._models.clear()
    for i in range(100):
        reg.register(f"model_{i}", Model(f"openai/gpt-{i}"))
    assert len(reg.list_names()) == 100


def test_model_config_with_none_values() -> None:
    """ModelConfig should handle None values gracefully."""
    cfg = ModelConfig(
        name="test",
        provider="openai",
        model_id="gpt-4",
        api_key=None,
    )
    assert cfg.api_key is None


def test_model_with_temperature_extreme_values() -> None:
    """Temperature at extreme bounds."""
    m1 = Model(provider="openai", model_id="gpt-4", temperature=0.0)
    assert m1.settings.temperature == 0.0

    m2 = Model(provider="openai", model_id="gpt-4", temperature=2.0)
    assert m2.settings.temperature == 2.0

    m3 = Model(provider="openai", model_id="gpt-4", temperature=1.0)
    assert m3.settings.temperature == 1.0


# =============================================================================
# Model.Custom() tests
# =============================================================================


def test_model_custom_valid_openai_compatible() -> None:
    """Model.Custom creates a valid model for OpenAI-compatible APIs."""
    m = Model.Custom(
        "deepseek-chat",
        api_base="https://api.deepseek.com/v1",
        api_key="sk-fake",
    )
    assert m.model_id == "deepseek-chat"
    assert m.provider == "openai"
    assert m.api_base == "https://api.deepseek.com/v1"
    assert m.api_key == "sk-fake"
    assert m.name == "deepseek-chat"


def test_model_custom_valid_with_model_id_slash() -> None:
    """Model.Custom derives name from model_id with slash."""
    m = Model.Custom(
        "deepseek/deepseek-chat",
        api_base="https://api.deepseek.com/v1",
        api_key="sk-fake",
    )
    assert m.model_id == "deepseek/deepseek-chat"
    assert m.name == "deepseek-chat"


def test_model_custom_valid_explicit_name() -> None:
    """Model.Custom accepts explicit name override."""
    m = Model.Custom(
        "grok-3-mini",
        api_base="https://api.x.ai/v1",
        api_key="xai-fake",
        name="Grok Mini",
    )
    assert m.name == "Grok Mini"
    assert m.model_id == "grok-3-mini"


def test_model_custom_valid_provider_litellm() -> None:
    """Model.Custom with provider=litellm uses LiteLLM."""
    m = Model.Custom(
        "custom/foobar",
        api_base="https://custom.api/v1",
        provider="litellm",
        api_key="sk-fake",
    )
    assert m.provider == "litellm"
    assert m.api_base == "https://custom.api/v1"


def test_model_custom_valid_with_optional_params() -> None:
    """Model.Custom accepts temperature, context_window, etc."""
    m = Model.Custom(
        "kimi-v1",
        api_base="https://api.moonshot.ai/v1",
        api_key="ms-fake",
        temperature=0.7,
        context_window=8192,
        max_tokens=2048,
    )
    assert m.settings.temperature == 0.7
    assert m.settings.context_window == 8192
    assert m.settings.max_output_tokens == 2048


def test_model_custom_to_config() -> None:
    """Model.Custom to_config returns correct ModelConfig."""
    m = Model.Custom(
        "deepseek-reasoner",
        api_base="https://api.deepseek.com/v1",
        api_key="sk-secret",
    )
    cfg = m.to_config()
    assert cfg.model_id == "deepseek-reasoner"
    assert cfg.provider == "openai"
    assert cfg.base_url == "https://api.deepseek.com/v1"
    assert cfg.api_key == "sk-secret"


def test_model_custom_empty_model_id_raises() -> None:
    """Model.Custom raises ValueError for empty model_id."""
    with pytest.raises(ValueError, match="model_id is required"):
        Model.Custom(
            "",
            api_base="https://api.example.com/v1",
            api_key="sk-fake",
        )


def test_model_custom_whitespace_model_id_raises() -> None:
    """Model.Custom raises ValueError for whitespace-only model_id."""
    with pytest.raises(ValueError, match="model_id is required"):
        Model.Custom(
            "   ",
            api_base="https://api.example.com/v1",
            api_key="sk-fake",
        )


def test_model_custom_missing_api_base_raises() -> None:
    """Model.Custom raises ValueError for empty api_base."""
    with pytest.raises(ValueError, match="api_base is required"):
        Model.Custom(
            "deepseek-chat",
            api_base="",
            api_key="sk-fake",
        )


def test_model_custom_none_api_base_raises() -> None:
    """Model.Custom raises ValueError for None api_base."""
    with pytest.raises(ValueError, match="api_base is required"):
        Model.Custom(
            "deepseek-chat",
            api_base=None,  # type: ignore[arg-type]
            api_key="sk-fake",
        )


def test_model_custom_whitespace_api_base_raises() -> None:
    """Model.Custom raises ValueError for whitespace-only api_base."""
    with pytest.raises(ValueError, match="api_base is required"):
        Model.Custom(
            "deepseek-chat",
            api_base="   ",
            api_key="sk-fake",
        )


def test_model_custom_api_base_normalized() -> None:
    """Model.Custom strips whitespace from api_base."""
    m = Model.Custom(
        "grok",
        api_base="  https://api.x.ai/v1  ",
        api_key="xai-fake",
    )
    assert m.api_base == "https://api.x.ai/v1"


def test_model_custom_provider_normalized_to_lowercase() -> None:
    """Model.Custom normalizes provider to lowercase."""
    m = Model.Custom(
        "grok",
        api_base="https://api.x.ai/v1",
        provider="OPENAI",
        api_key="xai-fake",
    )
    assert m.provider == "openai"


def test_model_custom_provider_default_is_openai() -> None:
    """Model.Custom defaults provider to 'openai'."""
    m = Model.Custom(
        "deepseek-chat",
        api_base="https://api.deepseek.com/v1",
        api_key="sk-fake",
    )
    assert m.provider == "openai"


def test_model_custom_api_key_optional() -> None:
    """Model.Custom allows api_key=None (for local/no-auth endpoints)."""
    m = Model.Custom(
        "local-model",
        api_base="http://localhost:8080/v1",
        api_key=None,
    )
    assert m.api_key is None
