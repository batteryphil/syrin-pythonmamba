"""Unit tests for Agent validation — friendly errors on wrong types.

TDD: These tests define expected validation behavior. Fixes remove chaos tests.
"""

from __future__ import annotations

import pytest

from syrin import Agent, Budget, Model


def _almock() -> Model:
    return Model.Almock(latency_seconds=0.01)


# -----------------------------------------------------------------------------
# model validation
# -----------------------------------------------------------------------------


def test_agent_model_str_rejects_with_clear_error() -> None:
    """model=str must raise TypeError with friendly message."""
    with pytest.raises(TypeError, match=r"model must be Model"):
        Agent(model="gpt-4o-mini")


def test_agent_model_int_rejects_with_clear_error() -> None:
    """model=int must raise TypeError with friendly message."""
    with pytest.raises(TypeError, match=r"model must be Model"):
        Agent(model=42)


def test_agent_model_none_rejects_with_clear_error() -> None:
    """model=None must raise TypeError (required)."""
    with pytest.raises(TypeError, match=r"model"):
        Agent(model=None)


# -----------------------------------------------------------------------------
# memory validation
# -----------------------------------------------------------------------------


def test_agent_memory_budget_rejects_with_clear_error() -> None:
    """memory=Budget must raise TypeError."""
    with pytest.raises(TypeError, match=r"memory must be|got Budget"):
        Agent(model=_almock(), memory=Budget(max_cost=1.0))


# -----------------------------------------------------------------------------
# max_tool_iterations validation
# -----------------------------------------------------------------------------


def test_agent_max_tool_iterations_negative_rejects() -> None:
    """max_tool_iterations < 1 must raise ValueError."""
    with pytest.raises(ValueError, match=r"max_tool_iterations must be >= 1"):
        Agent(model=_almock(), max_tool_iterations=-1)


def test_agent_max_tool_iterations_zero_rejects() -> None:
    """max_tool_iterations=0 must raise ValueError."""
    with pytest.raises(ValueError, match=r"max_tool_iterations must be >= 1"):
        Agent(model=_almock(), max_tool_iterations=0)


def test_agent_max_tool_iterations_str_rejects() -> None:
    """max_tool_iterations=str must raise TypeError."""
    with pytest.raises(TypeError, match=r"max_tool_iterations must be int"):
        Agent(model=_almock(), max_tool_iterations="10")


# -----------------------------------------------------------------------------
# system_prompt validation
# -----------------------------------------------------------------------------


def test_agent_system_prompt_int_rejects() -> None:
    """system_prompt=int must raise TypeError at init."""
    with pytest.raises(TypeError, match=r"system_prompt must be str"):
        Agent(model=_almock(), system_prompt=123)


# -----------------------------------------------------------------------------
# tools validation
# -----------------------------------------------------------------------------


def test_agent_tools_str_rejects() -> None:
    """tools=str must raise TypeError."""
    with pytest.raises(TypeError, match=r"tools must be list"):
        Agent(model=_almock(), tools="search")


def test_agent_tools_list_of_strings_rejects() -> None:
    """tools=[str, str] must raise TypeError."""
    with pytest.raises(TypeError, match=r"ToolSpec|tools"):
        Agent(model=_almock(), tools=["search", "calc"])


def test_agent_tools_list_with_none_rejects() -> None:
    """tools=[None, tool] must raise TypeError."""
    from syrin.tool import tool

    @tool
    def search(q: str) -> str:
        return "ok"

    with pytest.raises(TypeError, match=r"tools must not contain None|ToolSpec"):
        Agent(model=_almock(), tools=[None, search])


# -----------------------------------------------------------------------------
# budget validation
# -----------------------------------------------------------------------------


def test_agent_budget_int_rejects() -> None:
    """budget=int must raise TypeError."""
    with pytest.raises(TypeError, match=r"budget must be Budget"):
        Agent(model=_almock(), budget=5)


def test_agent_budget_str_rejects() -> None:
    """budget=str must raise TypeError."""
    with pytest.raises(TypeError, match=r"budget must be Budget"):
        Agent(model=_almock(), budget="0.50")


# -----------------------------------------------------------------------------
# response/run user_input validation
# -----------------------------------------------------------------------------


def test_agent_response_none_rejects() -> None:
    """response(None) must raise TypeError."""
    agent = Agent(model=_almock())
    with pytest.raises(TypeError, match=r"user_input must be str"):
        agent.run(None)


def test_agent_response_int_rejects() -> None:
    """response(42) must raise TypeError."""
    agent = Agent(model=_almock())
    with pytest.raises(TypeError, match=r"user_input must be str"):
        agent.run(42)


def test_agent_response_dict_rejects() -> None:
    """response(dict) must raise TypeError."""
    agent = Agent(model=_almock())
    with pytest.raises(TypeError, match=r"user_input must be str"):
        agent.run({"key": "val"})


def test_agent_response_empty_str_accepts() -> None:
    """response('') accepts."""
    agent = Agent(model=_almock())
    r = agent.run("")
    assert r is not None


def test_agent_passes_provider_kwargs_latency_validation() -> None:
    """Agent passes Model's provider_kwargs to provider; invalid latency_seconds raises at response()."""
    # When Model.Almock(latency_seconds=0), provider rejects; Agent must pass kwargs
    agent = Agent(model=Model.Almock(latency_seconds=0))
    with pytest.raises(ValueError, match=r"latency_seconds must be greater than 0"):
        agent.run("hi")


def test_agent_arun_none_rejects() -> None:
    """arun(None) must raise TypeError."""
    import asyncio

    async def _run() -> None:
        agent = Agent(model=_almock())
        with pytest.raises(TypeError, match=r"user_input must be str"):
            await agent.arun(None)

    asyncio.run(_run())


# -----------------------------------------------------------------------------
# memory=None disables persistent memory
# -----------------------------------------------------------------------------


def test_agent_remember_raises_when_memory_disabled() -> None:
    """agent.remember() raises RuntimeError when memory is disabled."""
    agent = Agent(model=_almock(), memory=None)
    with pytest.raises(RuntimeError, match=r"[Nn]o persistent memory"):
        agent.remember("test")


def test_agent_recall_raises_when_memory_disabled() -> None:
    """agent.recall() raises RuntimeError when memory is disabled."""
    agent = Agent(model=_almock(), memory=None)
    with pytest.raises(RuntimeError, match=r"[Nn]o persistent memory"):
        agent.recall()


def test_agent_forget_raises_when_memory_disabled() -> None:
    """agent.forget() raises RuntimeError when memory is disabled."""
    agent = Agent(model=_almock(), memory=None)
    with pytest.raises(RuntimeError, match=r"[Nn]o persistent memory"):
        agent.forget("test")


# -----------------------------------------------------------------------------
# report resets between calls
# -----------------------------------------------------------------------------


def test_agent_report_resets_between_calls() -> None:
    """agent.report from a previous run is cleared at the start of a new response()."""
    from syrin import Memory, MemoryType
    from syrin.response import GuardrailReport

    agent = Agent(model=_almock(), memory=Memory())
    agent.remember("test data", memory_type=MemoryType.FACTS)
    agent.run("first")
    # Report should have memory stores from remember()
    _ = agent.report

    agent.run("second")
    report2 = agent.report
    # Report should be fresh (reset at start of each response call)
    assert report2 is not None
    assert isinstance(report2.guardrail, GuardrailReport)
    # Guardrail should be a fresh default (not carried over)
    assert report2.guardrail.blocked is False
