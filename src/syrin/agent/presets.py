"""Agent presets — preconfigured agents for common use cases.

Usage:
    >>> from syrin import Agent
    >>> agent = Agent.basic(Model.OpenAI("gpt-4o-mini"), system_prompt="You are helpful.")
    >>> agent = Agent.with_memory(Model.OpenAI("gpt-4o-mini"))
    >>> agent = Agent.with_budget(Model.OpenAI("gpt-4o-mini"), budget=Budget(max_cost=0.25))
    >>> agent = Agent.presets.research()
    >>> agent = Agent.presets.assistant()
    >>> agent = Agent.presets.code_helper()
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from syrin.agent import Agent
    from syrin.budget import Budget
    from syrin.memory import Memory
    from syrin.model import Model


def basic(model: Model, *, system_prompt: str = "You are helpful.") -> Agent:
    """Create a minimal agent: model + system prompt, no memory, no budget.

    Use when you want the simplest possible agent for quick tasks or experimentation.

    Args:
        model: LLM to use (required). Use Model.OpenAI, Model.Anthropic, etc.
        system_prompt: Instructions for the agent. Default: "You are helpful."

    Returns:
        Agent with no memory, no budget, SINGLE_SHOT loop (no tools).

    Example:
        >>> from syrin import Agent
        >>> from syrin.model import Model
        >>> agent = Agent.basic(Model.OpenAI("gpt-4o-mini"), system_prompt="You are concise.")
        >>> agent.run("What is 2+2?")
    """
    from syrin.agent import Agent

    return Agent(
        model=model,
        system_prompt=system_prompt,
        memory=None,
    )


def with_memory(
    model: Model,
    *,
    system_prompt: str = "You are helpful.",
    memory: Memory | None = None,
) -> Agent:
    """Create an agent with persistent memory (remember/recall/forget).

    Use for multi-turn conversations or when the agent needs to remember facts across turns.

    Args:
        model: LLM to use (required).
        system_prompt: Instructions for the agent. Default: "You are helpful."
        memory: Memory config. Default: Memory() with core+episodic, top_k=10.

    Returns:
        Agent with memory enabled, REACT loop.

    Example:
        >>> agent = Agent.with_memory(Model.OpenAI("gpt-4o-mini"))
        >>> agent.run("Remember my name is Alice.")
        >>> agent.run("What's my name?")  # "Alice"
    """
    from syrin.agent import Agent
    from syrin.enums import MemoryType
    from syrin.memory import Memory

    return Agent(
        model=model,
        system_prompt=system_prompt,
        memory=memory
        if memory is not None
        else Memory(types=[MemoryType.FACTS, MemoryType.HISTORY], top_k=10),
    )


def with_budget(
    model: Model,
    *,
    system_prompt: str = "You are helpful.",
    budget: Budget | None = None,
) -> Agent:
    """Create an agent with cost budget control.

    Use when you need to cap spend per run or per period.

    Args:
        model: LLM to use (required).
        system_prompt: Instructions for the agent. Default: "You are helpful."
        budget: Budget config. Default: Budget(max_cost=0.25) ($0.25 per run).

    Returns:
        Agent with budget, REACT loop.

    Example:
        >>> from syrin import Agent, Budget
        >>> agent = Agent.with_budget(Model.OpenAI("gpt-4o-mini"), budget=Budget(max_cost=0.50))
        >>> agent.run("Summarize this long document")
    """
    from syrin.agent import Agent
    from syrin.budget import Budget

    return Agent(
        model=model,
        system_prompt=system_prompt,
        budget=budget if budget is not None else Budget(max_cost=0.25),
    )


def research() -> Agent:
    """Create an agent preset for research-style workflows.

    - REACT loop (tool use)
    - Run budget: $0.50
    - Memory: core + episodic
    - Higher tool iterations (15) for multi-step reasoning

    Example:
        >>> agent = Agent.presets.research()
        >>> agent.run("Summarize the latest papers on RAG")
    """
    from syrin.agent import Agent
    from syrin.budget import Budget
    from syrin.enums import MemoryType
    from syrin.memory import Memory

    return Agent(
        model=_default_model(),
        system_prompt="You are a research assistant. Use tools to search and cite sources. Be thorough and accurate.",
        budget=Budget(max_cost=0.50),
        memory=Memory(types=[MemoryType.FACTS, MemoryType.HISTORY], top_k=15),
        max_tool_iterations=15,
    )


def assistant() -> Agent:
    """Create an agent preset for conversational assistants.

    - REACT loop
    - Run budget: $0.25
    - Memory: core + episodic
    - Conversational system prompt

    Example:
        >>> agent = Agent.presets.assistant()
        >>> agent.run("What can you help me with?")
    """
    from syrin.agent import Agent
    from syrin.budget import Budget
    from syrin.enums import MemoryType
    from syrin.memory import Memory

    return Agent(
        model=_default_model(),
        system_prompt="You are a helpful assistant. Be concise and friendly.",
        budget=Budget(max_cost=0.25),
        memory=Memory(types=[MemoryType.FACTS, MemoryType.HISTORY], top_k=10),
    )


def code_helper() -> Agent:
    """Create an agent preset for code-related tasks.

    - REACT loop (for running tools, executing code)
    - Run budget: $0.50
    - Memory: core + episodic
    - System prompt oriented toward code clarity

    Example:
        >>> agent = Agent.presets.code_helper()
        >>> agent.run("Refactor this function to use async")
    """
    from syrin.agent import Agent
    from syrin.budget import Budget
    from syrin.enums import MemoryType
    from syrin.memory import Memory

    return Agent(
        model=_default_model(),
        system_prompt="You are a code assistant. Provide clear, idiomatic code. Prefer standard library and minimal dependencies.",
        budget=Budget(max_cost=0.50),
        memory=Memory(types=[MemoryType.FACTS, MemoryType.HISTORY], top_k=10),
    )


def _default_model() -> Model:
    """Default model for presets when none is configured."""
    from syrin.model import Model

    try:
        return Model.OpenAI("gpt-4o-mini")
    except Exception:
        try:
            return Model.Anthropic("claude-3-haiku-20240307")
        except Exception:
            return Model.Ollama("llama3.2")
