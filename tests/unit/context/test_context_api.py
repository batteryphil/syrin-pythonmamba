"""Context public API: ContextConfig removal and direct Context usage.

ContextConfig is removed from the public API. Users should use Context directly —
it has all the same fields plus more.
"""

from __future__ import annotations

# =============================================================================
# ContextConfig not publicly exported
# =============================================================================


def test_context_config_not_in_syrin_context() -> None:
    """ContextConfig must not be in syrin.context public API."""
    import syrin.context

    assert not hasattr(syrin.context, "ContextConfig"), (
        "ContextConfig should be removed from syrin.context"
    )


def test_context_config_not_in_context_all() -> None:
    """ContextConfig not in syrin.context.__all__."""
    import syrin.context

    assert "ContextConfig" not in syrin.context.__all__


def test_context_config_not_in_syrin_root() -> None:
    """ContextConfig was never in syrin root __all__ (verify still not there)."""
    import syrin

    assert not hasattr(syrin, "ContextConfig")


# =============================================================================
# Context works directly in AgentConfig
# =============================================================================


def test_agent_config_accepts_context_directly() -> None:
    """Agent(context=Context(...)) works directly."""
    from syrin import Agent
    from syrin.context import Context
    from syrin.model import Model

    agent = Agent(model=Model.Almock(), context=Context(max_tokens=4000))
    assert agent is not None


def test_agent_accepts_context_via_config() -> None:
    """Agent accepts context=Context(...) directly."""
    from syrin import Agent
    from syrin.context import Context
    from syrin.model import Model

    agent = Agent(
        model=Model.Almock(),
        system_prompt="test",
        context=Context(max_tokens=8000),
    )
    assert agent is not None


def test_context_has_same_fields_as_context_config() -> None:
    """Context supports all fields ContextConfig had: max_tokens, reserve, thresholds, token_limits, auto_compact_at."""
    from syrin.context import Context

    ctx = Context(
        max_tokens=16000,
        reserve=1000,
        auto_compact_at=0.8,
    )
    assert ctx.max_tokens == 16000
    assert ctx.reserve == 1000
    assert ctx.auto_compact_at == 0.8
