"""Domain events — typed events for key lifecycle moments.

Domain events complement the Hook system: Hooks carry EventContext dicts;
domain events are typed dataclasses that observability and other consumers
can subscribe to without parsing strings.

Usage:
    >>> from syrin import Agent
    >>> from syrin.domain_events import BudgetThresholdReached, ContextCompacted, EventBus
    >>>
    >>> bus = EventBus()
    >>> bus.subscribe(BudgetThresholdReached, lambda e: print(f"Budget at {e.percentage}%"))
    >>> bus.subscribe(ContextCompacted, lambda e: print(f"Compacted: {e.method}"))
    >>>
    >>> agent = Agent(model=..., event_bus=bus)
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Generic, TypeVar

__all__ = [
    "AgentRunStarted",
    "AgentRunEnded",
    "BudgetExceeded",
    "BudgetThresholdReached",
    "ContextCompacted",
    "DomainEvent",
    "EventBus",
    "GuardrailBlocked",
    "HandoffCompleted",
    "HandoffStarted",
    "LLMRequestCompleted",
    "LLMRequestStarted",
    "ToolCallCompleted",
    "ToolCallFailed",
]

T = TypeVar("T", bound="DomainEvent")


@dataclass(frozen=True)
class DomainEvent:
    """Base for domain events. All domain events are immutable dataclasses.

    Use EventBus to subscribe and emit. Pass event_bus=EventBus() to Agent.
    """


@dataclass(frozen=True)
class BudgetThresholdReached(DomainEvent):
    """Emitted when a budget threshold is crossed (e.g. 80% of run budget)."""

    percentage: int
    """Utilization percentage (0-100) that triggered the threshold."""
    current_value: float
    """Current cost or token count."""
    limit_value: float
    """Limit or cap (e.g. run budget, max tokens)."""
    metric: str = "cost"
    """Metric type: 'cost' or 'tokens'."""
    action_taken: str | None = None
    """Optional: action executed by threshold (e.g. 'warn', 'switch_model')."""


@dataclass(frozen=True)
class ContextCompacted(DomainEvent):
    """Emitted when context window compaction runs (truncation/summarization)."""

    method: str
    """Compaction method (e.g. 'middle_out', 'truncate')."""
    tokens_before: int
    """Token count before compaction."""
    tokens_after: int
    """Token count after compaction."""
    messages_before: int = 0
    """Number of messages before compaction."""
    messages_after: int = 0
    """Number of messages after compaction."""


@dataclass(frozen=True)
class AgentRunStarted(DomainEvent):
    """Emitted when an agent begins processing user input."""

    input: str
    """User input text (or stringified multimodal input)."""
    model: str
    """Model ID being used."""
    iteration: int
    """Current iteration count."""


@dataclass(frozen=True)
class AgentRunEnded(DomainEvent):
    """Emitted when an agent finishes and returns a response."""

    content: str
    """Response content."""
    cost: float
    """Total cost in USD for this run."""
    tokens: int
    """Total tokens consumed."""
    duration: float
    """Wall-clock duration in seconds."""
    stop_reason: str
    """Why the run stopped (end_turn, tool_use, max_tokens, etc.)."""
    iteration: int
    """Final iteration count."""


@dataclass(frozen=True)
class LLMRequestStarted(DomainEvent):
    """Emitted before an LLM API call is sent."""

    iteration: int
    """Current iteration."""
    tool_count: int = 0
    """Number of tools available for this call."""


@dataclass(frozen=True)
class LLMRequestCompleted(DomainEvent):
    """Emitted after an LLM API response is received."""

    content: str
    """Response content text."""
    iteration: int
    """Current iteration."""


@dataclass(frozen=True)
class ToolCallCompleted(DomainEvent):
    """Emitted after a tool executes successfully."""

    tool_name: str
    """Name of the tool that was called."""
    duration_ms: float
    """Execution time in milliseconds."""


@dataclass(frozen=True)
class ToolCallFailed(DomainEvent):
    """Emitted when a tool execution raises an error."""

    tool_name: str
    """Name of the tool that failed."""
    error: str
    """Error message."""
    iteration: int
    """Current iteration."""


@dataclass(frozen=True)
class BudgetExceeded(DomainEvent):
    """Emitted when a hard budget limit is exceeded."""

    used: float
    """Amount used (cost in USD or token count)."""
    limit: float
    """The limit that was exceeded."""
    exceeded_by: float
    """How much over the limit."""


@dataclass(frozen=True)
class GuardrailBlocked(DomainEvent):
    """Emitted when a guardrail blocks input or output."""

    stage: str
    """'input' or 'output'."""
    reason: str
    """Human-readable reason for blocking."""
    guardrail_names: list[str]
    """Names of guardrails that triggered the block."""


@dataclass(frozen=True)
class HandoffStarted(DomainEvent):
    """Emitted when an agent begins a handoff to another agent."""

    target_agent: str
    """Name of the target agent."""
    task: str
    """Task being delegated."""


@dataclass(frozen=True)
class HandoffCompleted(DomainEvent):
    """Emitted when a handoff completes and result is received."""

    target_agent: str
    """Name of the target agent."""
    success: bool
    """Whether the handoff was successful."""


class EventBus(Generic[T]):
    """Event bus for typed domain events. Subscribe and emit typed events.

    Use this when you need typed, structured event handling (e.g. for metrics,
    observability, or custom pipelines).
    """

    def __init__(self) -> None:
        self._listeners: dict[type[DomainEvent], list[Callable[[DomainEvent], None]]] = {}

    def subscribe(
        self,
        event_type: type[T],
        handler: Callable[[T], None],
    ) -> None:
        """Subscribe to a domain event type."""
        if event_type not in self._listeners:
            self._listeners[event_type] = []
        self._listeners[event_type].append(handler)  # type: ignore[arg-type]

    def on(
        self,
        event_type: type[T],
        handler: Callable[[T], None],
    ) -> None:
        """Alias for subscribe(). Register handler for a domain event type."""
        self.subscribe(event_type, handler)

    def emit(self, event: DomainEvent) -> None:
        """Emit a domain event to all subscribers."""
        t = type(event)
        for base in t.__mro__:
            if base in self._listeners:
                for h in self._listeners[base]:
                    h(event)
