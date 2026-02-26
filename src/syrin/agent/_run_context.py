"""Agent run context: narrow interface used by Loop implementations.

Loops depend only on AgentRunContext (Protocol), not on Agent. The single place
that builds the message list for the LLM is syrin.agent._context_builder.build_messages;
DefaultAgentRunContext.build_messages delegates to it via the agent.

- Refactoring Agent internals does not break Loop implementations.
- The contract is explicit (Protocol) and minimal (ISP).

This module is internal; the public API is Agent and Loop.
"""

from __future__ import annotations

from typing import Any, Protocol, cast

# Hook is used in emit_event; avoid circular import by importing inside methods
# or use TYPE_CHECKING. We need it at runtime for the Protocol.
from syrin.enums import Hook
from syrin.events import EventContext
from syrin.tool import ToolSpec
from syrin.types import Message, ProviderResponse, TokenUsage


class AgentRunContext(Protocol):
    """Narrow interface for running an agent loop.

    Implemented by DefaultAgentRunContext (wrapping Agent). Custom loops
    should type-hint run(self, ctx: AgentRunContext, user_input: str)
    and use only the methods and properties defined here.
    """

    # ---- Message and completion ----
    def build_messages(self, user_input: str) -> list[Message]:
        """Build the message list for the next LLM call (memory + context + user)."""
        ...

    async def complete(
        self, messages: list[Message], tools: list[ToolSpec] | None = None
    ) -> ProviderResponse:
        """Call the LLM with messages and optional tools."""
        ...

    async def execute_tool(self, name: str, arguments: dict[str, Any]) -> str:
        """Execute a tool by name with the given arguments."""
        ...

    # ---- Events ----
    def emit_event(self, hook: Hook, ctx: EventContext) -> None:
        """Emit a lifecycle hook with the given context."""
        ...

    # ---- Budget and rate limits ----
    def check_and_apply_budget(self) -> None:
        """Check budget/token limits and apply threshold actions (e.g. switch model)."""
        ...

    def check_and_apply_rate_limit(self) -> None:
        """Check rate limits and apply threshold actions (e.g. wait, switch)."""
        ...

    def pre_call_budget_check(self, messages: list[Any], *, max_output_tokens: int = 1024) -> None:
        """Run budget/rate checks before an LLM call. Call once per request."""
        ...

    def record_rate_limit_usage(self, token_usage: TokenUsage) -> None:
        """Record token usage for rate limit tracking after an LLM call."""
        ...

    def record_cost(self, token_usage: TokenUsage, model_id: str) -> None:
        """Record cost and update budget state after an LLM call."""
        ...

    # ---- Read-only properties for loops ----
    @property
    def model_id(self) -> str:
        """Model ID for cost calculation and events (e.g. openai/gpt-4o)."""
        ...

    @property
    def tools(self) -> list[ToolSpec] | None:
        """Tool specs to pass to complete(); None if no tools."""
        ...

    @property
    def max_output_tokens(self) -> int:
        """Max output tokens for this model (from metadata or default 1024)."""
        ...

    @property
    def has_budget(self) -> bool:
        """True if the agent has a budget (cost tracking)."""
        ...

    @property
    def has_rate_limit(self) -> bool:
        """True if the agent has rate limiting enabled."""
        ...

    @property
    def pricing_override(self) -> Any:
        """Optional pricing override from the model for cost calculation."""
        ...

    @property
    def approval_gate(self) -> Any:
        """Optional ApprovalGate for HITL. None = no approval required."""
        ...

    @property
    def hitl_timeout(self) -> int:
        """Timeout in seconds for HITL approval. Default 300."""
        ...

    @property
    def tracer(self) -> Any:
        """Optional tracer for observability; when set, loop creates LLM/tool spans."""
        ...


class DefaultAgentRunContext:
    """Implements AgentRunContext by delegating to an Agent instance.

    Used internally so that Loop.run() receives a narrow interface
    instead of the full Agent.
    """

    def __init__(self, agent: Any) -> None:
        self._agent = agent

    def build_messages(self, user_input: str) -> list[Message]:
        return cast(list[Message], self._agent._build_messages(user_input))

    async def complete(
        self, messages: list[Message], tools: list[ToolSpec] | None = None
    ) -> ProviderResponse:
        return cast(ProviderResponse, await self._agent.complete(messages, tools))

    async def execute_tool(self, name: str, arguments: dict[str, Any]) -> str:
        return cast(str, await self._agent.execute_tool(name, arguments))

    def emit_event(self, hook: Hook, ctx: EventContext) -> None:
        self._agent._emit_event(hook, ctx)

    def check_and_apply_budget(self) -> None:
        self._agent._check_and_apply_budget()

    def check_and_apply_rate_limit(self) -> None:
        self._agent._check_and_apply_rate_limit()

    def pre_call_budget_check(self, messages: list[Any], *, max_output_tokens: int = 1024) -> None:
        self._agent._pre_call_budget_check(messages, max_output_tokens=max_output_tokens)

    def record_rate_limit_usage(self, token_usage: TokenUsage) -> None:
        self._agent._record_rate_limit_usage(token_usage)

    def record_cost(self, token_usage: TokenUsage, model_id: str) -> None:
        self._agent._record_cost(token_usage, model_id)

    @property
    def model_id(self) -> str:
        return cast(str, self._agent._model_config.model_id)

    @property
    def tools(self) -> list[ToolSpec] | None:
        return self._agent._tools if self._agent._tools else None

    @property
    def max_output_tokens(self) -> int:
        if getattr(self._agent, "_model", None) is None:
            return 1024
        meta = getattr(self._agent._model, "metadata", None) or {}
        return cast(int, meta.get("max_output_tokens", 1024))

    @property
    def has_budget(self) -> bool:
        return self._agent._budget is not None

    @property
    def has_rate_limit(self) -> bool:
        return getattr(self._agent, "_rate_limit_manager_internal", None) is not None

    @property
    def pricing_override(self) -> Any:
        return (
            getattr(self._agent._model, "pricing", None)
            if getattr(self._agent, "_model", None) is not None
            else None
        )

    @property
    def approval_gate(self) -> Any:
        return getattr(self._agent, "_approval_gate", None)

    @property
    def hitl_timeout(self) -> int:
        return getattr(self._agent, "_hitl_timeout", 300)

    @property
    def tracer(self) -> Any:
        """Return agent's tracer so loops can create LLM/tool child spans."""
        return getattr(self._agent, "_tracer", None)
