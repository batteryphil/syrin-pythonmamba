"""Tests for dynamic system prompt templates (Step 3)."""

from __future__ import annotations

import pytest

from syrin import Agent, Model, prompt, system_prompt
from syrin.prompt import PromptContext, make_prompt_context


def _almock() -> Model:
    return Model.Almock(latency_seconds=0.01)


def test_agent_static_system_prompt_unchanged() -> None:
    """Agent with static system_prompt works as before."""
    agent = Agent(model=_almock(), system_prompt="You are helpful.")
    vars_ = agent.effective_prompt_vars()
    assert "date" in vars_
    assert "agent_id" in vars_
    assert agent.response("Hi").content is not None


def test_agent_with_prompt_and_prompt_vars() -> None:
    """Agent with @prompt and prompt_vars resolves at run."""

    @prompt
    def persona_prompt(user_name: str, tone: str = "professional") -> str:
        return f"You assist {user_name or 'user'}. Be {tone}."

    agent = Agent(
        model=_almock(),
        system_prompt=persona_prompt,
        prompt_vars={"user_name": "Alice", "tone": "friendly"},
    )
    ctx = make_prompt_context(agent, inject_builtins=True)
    resolved = agent._resolve_system_prompt(agent.effective_prompt_vars(), ctx)
    assert "Alice" in resolved
    assert "friendly" in resolved
    assert agent.response("Hi").content is not None


def test_agent_prompt_vars_merge_class_instance() -> None:
    """Class prompt_vars + instance prompt_vars merge; instance wins."""

    @prompt
    def p(user_name: str, tone: str = "formal") -> str:
        return f"You assist {user_name}. Be {tone}."

    class PersonaAgent(Agent):
        model = _almock()
        system_prompt = p
        prompt_vars = {"tone": "formal"}

    agent = PersonaAgent(prompt_vars={"user_name": "Bob", "tone": "casual"})
    vars_ = agent.effective_prompt_vars()
    assert vars_["user_name"] == "Bob"
    assert vars_["tone"] == "casual"
    ctx = make_prompt_context(agent, inject_builtins=True)
    resolved = agent._resolve_system_prompt(vars_, ctx)
    assert "Bob" in resolved
    assert "casual" in resolved


def test_agent_per_call_prompt_vars() -> None:
    """Per-call prompt_vars override instance vars."""

    @prompt
    def p(user_name: str) -> str:
        return f"You assist {user_name}."

    agent = Agent(
        model=_almock(),
        system_prompt=p,
        prompt_vars={"user_name": "Default"},
    )
    r1 = agent.response("Hi", prompt_vars={"user_name": "Alice"})
    assert r1.content is not None


def test_agent_callable_system_prompt_with_ctx() -> None:
    """Callable system_prompt receives PromptContext."""

    def build_prompt(ctx: PromptContext) -> str:
        assert ctx.agent is not None
        assert ctx.agent_id
        assert ctx.date is not None
        return f"You are agent {ctx.agent_id}. Today is {ctx.date.date()}."

    agent = Agent(model=_almock(), system_prompt=build_prompt)
    r = agent.response("Hi")
    assert r.content is not None


def test_agent_system_prompt_in_class() -> None:
    """@system_prompt method in class defines system prompt."""

    class MyAgent(Agent):
        model = _almock()

        @system_prompt
        def my_prompt(self, user_name: str = "") -> str:
            return f"You assist {user_name or 'user'}."

    agent = MyAgent(prompt_vars={"user_name": "Carol"})
    vars_ = agent.effective_prompt_vars()
    ctx = make_prompt_context(agent, inject_builtins=True)
    resolved = agent._resolve_system_prompt(vars_, ctx)
    assert "Carol" in resolved
    r = agent.response("Hi")
    assert r.content is not None


def test_agent_system_prompt_in_class_no_params() -> None:
    """@system_prompt method with (self) only."""

    class SimpleAgent(Agent):
        model = _almock()

        @system_prompt
        def my_prompt(self) -> str:
            return "You are helpful."

    agent = SimpleAgent()
    r = agent.response("Hi")
    assert r.content is not None


def test_agent_two_system_prompts_raises() -> None:
    """Two @system_prompt methods on same class raises ValueError."""

    with pytest.raises(ValueError, match=r"multiple @system_prompt|only one allowed"):

        class BadAgent(Agent):
            model = _almock()

            @system_prompt
            def prompt_a(self) -> str:
                return "A"

            @system_prompt
            def prompt_b(self) -> str:
                return "B"


def test_agent_get_prompt_builtins() -> None:
    """get_prompt_builtins returns date, agent_id, conversation_id."""
    agent = Agent(model=_almock(), system_prompt="Hi")
    builtins = agent.get_prompt_builtins()
    assert "date" in builtins
    assert "agent_id" in builtins
    assert "conversation_id" in builtins
    assert builtins["conversation_id"] is None


def test_agent_inject_builtins_false() -> None:
    """inject_builtins=False skips built-ins."""
    agent = Agent(model=_almock(), system_prompt="Hi", inject_builtins=False)
    vars_ = agent.effective_prompt_vars()
    assert "date" not in vars_
    assert "agent_id" not in vars_
    assert "conversation_id" not in vars_


def test_agent_prompt_missing_var_raises() -> None:
    """Prompt with required param, prompt_vars omits it, raises clear error."""

    @prompt
    def p(required: str) -> str:
        return f"You use {required}."

    agent = Agent(model=_almock(), system_prompt=p, prompt_vars={})
    with pytest.raises(ValueError, match="required|missing"):
        agent.response("Hi")


def test_agent_system_prompt_int_rejects() -> None:
    """system_prompt=int still raises TypeError."""
    with pytest.raises(TypeError, match=r"system_prompt must be str"):
        Agent(model=_almock(), system_prompt=123)
