"""Tests for dependency injection (Agent deps, RunContext, tool ctx injection)."""

from dataclasses import dataclass

import pytest

from syrin import Agent, Model, RunContext
from syrin.tool import tool

# -----------------------------------------------------------------------------
# Dependencies
# -----------------------------------------------------------------------------


@dataclass
class DIExampleDeps:
    value: str
    counter: list[int]


# -----------------------------------------------------------------------------
# Tools
# -----------------------------------------------------------------------------


@tool
def tool_without_ctx(query: str) -> str:
    """Tool without RunContext."""
    return f"Result: {query}"


@tool
def tool_with_ctx(ctx: RunContext[DIExampleDeps], key: str) -> str:
    """Tool with RunContext - uses ctx.deps."""
    ctx.deps.counter.append(1)
    return f"{ctx.deps.value}:{key}"


# -----------------------------------------------------------------------------
# Agent
# -----------------------------------------------------------------------------


class DIAgent(Agent):
    model = Model.Almock(latency_min=0, latency_max=0, lorem_length=50)
    system_prompt = "Use tools."
    tools = [tool_without_ctx, tool_with_ctx]


# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------


def test_tool_without_ctx_no_deps_needed() -> None:
    """Tool without ctx works without deps."""
    agent = DIAgent()
    result = agent._execute_tool("tool_without_ctx", {"query": "hello"})
    assert "Result: hello" in result


def test_tool_with_ctx_receives_deps() -> None:
    """Tool with ctx receives RunContext with deps."""
    counter: list[int] = []
    deps = DIExampleDeps(value="injected", counter=counter)
    agent = DIAgent(dependencies=deps)
    result = agent._execute_tool("tool_with_ctx", {"key": "foo"})
    assert len(counter) >= 1
    assert result == "injected:foo"


def test_tool_with_ctx_no_deps_raises() -> None:
    """Tool with ctx but agent has no deps raises clear error."""
    agent = DIAgent()
    from syrin.exceptions import ToolExecutionError

    with pytest.raises(ToolExecutionError) as exc_info:
        agent._execute_tool("tool_with_ctx", {"key": "foo"})
    assert "expects ctx: RunContext" in str(exc_info.value)
    assert "no deps" in str(exc_info.value).lower() or "Pass dependencies=" in str(exc_info.value)


def test_tool_schema_excludes_ctx() -> None:
    """Tool with ctx has schema without ctx param."""
    assert "ctx" not in tool_with_ctx.parameters_schema.get("properties", {})
    assert "ctx" not in tool_with_ctx.parameters_schema.get("required", [])
    assert "key" in tool_with_ctx.parameters_schema.get("properties", {})
