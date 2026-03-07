"""Agent builder — fluent API for constructing agents with many optional params.

Usage:
    >>> from syrin import Agent, Budget
    >>> agent = (
    ...     Agent.builder(Model.OpenAI("gpt-4o-mini"))
    ...     .with_system_prompt("You are helpful.")
    ...     .with_budget(Budget(run=0.5))
    ...     .with_tools([search, calculate])
    ...     .build()
    ... )
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from syrin.agent import Agent
    from syrin.budget import Budget
    from syrin.context import Context
    from syrin.memory import Memory
    from syrin.model import Model


class AgentBuilder:
    """Fluent builder for Agent instances. Use Agent.builder(model).with_*(...).build()."""

    def __init__(self, model: Model) -> None:
        """Start builder with required model.

        Args:
            model: LLM to use (required).
        """
        self._model = model
        self._system_prompt: str | Any = ""
        self._prompt_vars: dict[str, Any] = {}
        self._tools: list[Any] = []
        self._budget: Budget | None = None
        self._output: Any = None
        self._max_tool_iterations: int = 10
        self._budget_store: Any = None
        self._budget_store_key: str = "default"
        self._memory: Memory | None = None
        self._loop_strategy: Any = None
        self._loop: Any = None
        self._guardrails: Any = None
        self._context: Context | None = None
        self._rate_limit: Any = None
        self._checkpoint: Any = None
        self._debug: bool = False
        self._tracer: Any = None
        self._bus: Any = None

    def with_system_prompt(self, prompt: str | Any) -> AgentBuilder:
        """Set system prompt (str, Prompt from @prompt, or callable)."""
        self._system_prompt = prompt if prompt is not None else ""
        return self

    def with_prompt_vars(self, vars: dict[str, Any]) -> AgentBuilder:
        """Set prompt vars for dynamic system prompts."""
        self._prompt_vars = dict(vars) if vars else {}
        return self

    def with_tools(self, tools: list[Any]) -> AgentBuilder:
        """Set tools list."""
        self._tools = list(tools) if tools else []
        return self

    def with_budget(self, budget: Budget) -> AgentBuilder:
        """Set budget (run/period limits)."""
        self._budget = budget
        return self

    def with_output(self, output: Any) -> AgentBuilder:
        """Set structured output config."""
        self._output = output
        return self

    def with_max_tool_iterations(self, n: int) -> AgentBuilder:
        """Set max tool-call iterations per response."""
        self._max_tool_iterations = n
        return self

    def with_budget_store(self, store: Any, key: str = "default") -> AgentBuilder:
        """Set budget store for persistence."""
        self._budget_store = store
        self._budget_store_key = key
        return self

    def with_memory(self, memory: Memory) -> AgentBuilder:
        """Set conversation or persistent memory."""
        self._memory = memory
        return self

    def with_loop_strategy(self, strategy: Any) -> AgentBuilder:
        """Set loop strategy (REACT, SINGLE_SHOT, etc.)."""
        self._loop_strategy = strategy
        return self

    def with_loop(self, loop: Any) -> AgentBuilder:
        """Set custom Loop instance."""
        self._loop = loop
        return self

    def with_guardrails(self, guardrails: Any) -> AgentBuilder:
        """Set guardrails."""
        self._guardrails = guardrails
        return self

    def with_context(self, context: Context) -> AgentBuilder:
        """Set context config (max_tokens, thresholds)."""
        self._context = context
        return self

    def with_rate_limit(self, rate_limit: Any) -> AgentBuilder:
        """Set rate limit config."""
        self._rate_limit = rate_limit
        return self

    def with_checkpoint(self, checkpoint: Any) -> AgentBuilder:
        """Set checkpoint config."""
        self._checkpoint = checkpoint
        return self

    def with_debug(self, debug: bool = True) -> AgentBuilder:
        """Enable debug (print events to console)."""
        self._debug = debug
        return self

    def with_tracer(self, tracer: Any) -> AgentBuilder:
        """Set custom tracer."""
        self._tracer = tracer
        return self

    def with_bus(self, bus: Any) -> AgentBuilder:
        """Set event bus for typed domain events."""
        self._bus = bus
        return self

    def build(self) -> Agent:
        """Build and return the Agent."""
        from syrin.agent import Agent

        return Agent(
            model=self._model,
            system_prompt=self._system_prompt,
            prompt_vars=self._prompt_vars,
            tools=self._tools,
            budget=self._budget,
            output=self._output,
            max_tool_iterations=self._max_tool_iterations,
            budget_store=self._budget_store,
            budget_store_key=self._budget_store_key,
            memory=self._memory,
            loop_strategy=self._loop_strategy,
            loop=self._loop,
            guardrails=self._guardrails,
            context=self._context,
            rate_limit=self._rate_limit,
            checkpoint=self._checkpoint,
            debug=self._debug,
            tracer=self._tracer,
            bus=self._bus,
        )
