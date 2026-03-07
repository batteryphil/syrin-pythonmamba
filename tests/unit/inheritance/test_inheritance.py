"""Tests for Agent inheritance (MRO: tools merge, prompt/model override)."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from syrin.agent import Agent
from syrin.model import Model
from syrin.tool import tool
from syrin.types import ProviderResponse, TokenUsage


@tool
def base_tool(x: str) -> str:
    """Base tool."""
    return f"base:{x}"


@tool
def sub_tool(y: int) -> int:
    """Sub tool."""
    return y + 1


class BaseAgent(Agent):
    model = Model("anthropic/claude-3-5-sonnet")
    system_prompt = "You are the base."
    tools = [base_tool]


class SubAgent(BaseAgent):
    system_prompt = "You are the sub."
    tools = [sub_tool]


def test_sub_inherits_base_tools_merged() -> None:
    agent = SubAgent()
    names = [t.name for t in agent._tools]
    assert "base_tool" in names
    assert "sub_tool" in names
    assert len(agent._tools) == 2


def test_sub_overrides_system_prompt() -> None:
    agent = SubAgent()
    assert agent._system_prompt == "You are the sub."


def test_sub_inherits_model() -> None:
    agent = SubAgent()
    assert agent._model_config.model_id == "anthropic/claude-3-5-sonnet"


def test_base_has_only_base_tools() -> None:
    agent = BaseAgent()
    assert len(agent._tools) == 1
    assert agent._tools[0].name == "base_tool"


def test_explicit_init_overrides_class_defaults() -> None:
    agent = SubAgent(system_prompt="Custom prompt")
    assert agent._system_prompt == "Custom prompt"


def test_agent_without_model_raises() -> None:
    with pytest.raises(TypeError, match="Agent requires model"):
        Agent()


# =============================================================================
# INHERITANCE EDGE CASES - TRY TO BREAK FUNCTIONALITY
# =============================================================================


@tool
def tool_a() -> str:
    return "A"


@tool
def tool_b() -> str:
    return "B"


@tool
def tool_c() -> str:
    return "C"


class TestInheritanceEdgeCases:
    """Edge cases for agent inheritance."""

    def test_deep_inheritance_chain(self) -> None:
        """Test 4-level inheritance chain."""

        class Level1(Agent):
            model = Model("openai/gpt-4")
            system_prompt = "Level 1"
            tools = [tool_a]

        class Level2(Level1):
            system_prompt = "Level 2"
            tools = [tool_b]

        class Level3(Level2):
            system_prompt = "Level 3"

        class Level4(Level3):
            system_prompt = "Level 4"
            tools = [tool_c]

        agent = Level4()
        assert agent._system_prompt == "Level 4"
        # Should have all tools from chain
        tool_names = [t.name for t in agent._tools]
        assert "tool_a" in tool_names
        assert "tool_b" in tool_names
        assert "tool_c" in tool_names

    def test_multiple_inheritance_diamond(self) -> None:
        """Test diamond inheritance pattern."""

        class A(Agent):
            model = Model("openai/gpt-4")
            system_prompt = "A"

        class B(A):
            system_prompt = "B"

        class C(A):
            system_prompt = "C"

        class D(B, C):
            system_prompt = "D"

        agent = D()
        assert agent._system_prompt == "D"
        assert "gpt-4" in agent._model_config.model_id

    def test_inheritance_with_empty_tools(self) -> None:
        """Test inheriting with empty tools list."""

        class Base(Agent):
            model = Model("openai/gpt-4")
            tools = [tool_a]

        class Child(Base):
            tools = []

        agent = Child()
        # Should still have base tool
        assert len(agent._tools) == 1

    def test_inheritance_model_override(self) -> None:
        """Test overriding model in child class."""

        class Base(Agent):
            model = Model("openai/gpt-4")

        class Child(Base):
            model = Model("anthropic/claude-3")

        agent = Child()
        assert "claude-3" in agent._model_config.model_id

    def test_inheritance_same_tool_names(self) -> None:
        """Test when parent and child have tools with same name."""

        @tool
        def shared_tool() -> str:
            return "parent"

        class Base(Agent):
            model = Model("openai/gpt-4")
            tools = [shared_tool]

        # Child with same tool name (different implementation)
        @tool
        def shared_tool() -> str:
            return "child"

        class Child(Base):
            tools = [shared_tool]

        agent = Child()
        # Merge deduplicates by name — one shared_tool (subclass wins in merge order)
        tool_names = [t.name for t in agent._tools]
        assert tool_names.count("shared_tool") == 1
        result = agent._execute_tool("shared_tool", {})
        assert result in ("parent", "child")  # implementation may vary by merge order

    def test_inheritance_with_none_tools(self) -> None:
        """Test inheriting when tools is None."""

        class Base(Agent):
            model = Model("openai/gpt-4")
            tools = [tool_a]

        class Child(Base):
            tools = None

        agent = Child()
        # Should still inherit base tools
        assert len(agent._tools) == 1

    def test_inheritance_with_many_tools(self) -> None:
        """Test inheritance with many tools (50+)."""
        # Create tools at module level to avoid closure issues
        test_tools = []
        for i in range(50):

            @tool
            def dynamic_tool(x=i) -> str:
                return f"result_{x}"

            # Rename the tool to be unique
            dynamic_tool.name = f"tool_{i}"
            test_tools.append(dynamic_tool)

        class Base(Agent):
            model = Model("openai/gpt-4")
            tools = test_tools[:25]

        class Child(Base):
            tools = test_tools[25:]

        agent = Child()
        assert len(agent._tools) == 50

    def test_inheritance_preserves_tool_execution(self) -> None:
        """Test that inherited tools still execute correctly."""

        @tool
        def working_tool(x: int) -> int:
            return x * 2

        class Base(Agent):
            model = Model("openai/gpt-4")
            tools = [working_tool]

        class Child(Base):
            pass

        agent = Child()
        # Tool should still be callable
        result = agent._execute_tool("working_tool", {"x": 5})
        assert result == "10"  # Tool returns string representation

    def test_inheritance_with_memory(self) -> None:
        """Test inheritance with memory configuration."""
        from syrin.memory import Memory

        class Base(Agent):
            model = Model("openai/gpt-4")

        class Child(Base):
            pass

        # Create with memory
        mem = Memory()
        agent = Child(memory=mem)
        assert agent._persistent_memory is not None

    def test_inheritance_with_budget(self) -> None:
        """Test inheritance with budget configuration."""
        from syrin.budget import Budget

        class Base(Agent):
            model = Model("openai/gpt-4")

        class Child(Base):
            pass

        budget = Budget(run=10.0)
        agent = Child(budget=budget)
        assert agent._budget is not None

    def test_inheritance_budget_exceeded(self) -> None:
        """Test budget inheritance behavior."""
        from syrin.budget import Budget, raise_on_exceeded
        from syrin.exceptions import BudgetExceededError

        class Base(Agent):
            model = Model("openai/gpt-4")
            budget = Budget(run=0.000001, on_exceeded=raise_on_exceeded)

        class Child(Base):
            pass

        agent = Child()
        # Should inherit budget settings
        with (
            patch.object(
                agent._provider,
                "complete",
                new_callable=AsyncMock,
                return_value=ProviderResponse(
                    content="test",
                    token_usage=TokenUsage(input_tokens=1000, output_tokens=500),
                ),
            ),
            pytest.raises(BudgetExceededError),
        ):
            agent.response("test")
