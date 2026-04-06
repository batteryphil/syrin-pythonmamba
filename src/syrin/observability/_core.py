"""Enhanced observability system with spans, semantic attributes, and OpenTelemetry support.

This module provides:
- Span-based distributed tracing with parent-child relationships
- Semantic attributes for LLM, tool, memory, and budget events
- Session tracking across multiple agent calls
- Multiple export backends (Console, OTLP, File)
- Debug mode with full introspection

Example:
    >>> from syrin.observability import trace, SpanKind
    >>>
    >>> # Basic span usage
    >>> with trace.span("agent.run", kind=SpanKind.AGENT) as span:
    ...     span.set_attribute("agent.name", "my_agent")
    ...     result = agent.run("Hello")
    ...     span.set_attribute("llm.tokens.total", result.tokens.total_tokens)
    >>>
    >>> # Session tracking
    >>> with trace.session("user_123"):
    ...     for msg in conversation:
    ...         response = agent.run(msg)
    >>>
    >>> # Debug mode
    >>> agent = Agent(..., debug=True)  # Captures full context
"""

from __future__ import annotations

import contextvars
import json
import logging
import threading
import uuid
from abc import ABC, abstractmethod
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum

_log = logging.getLogger(__name__)

# Context variables for trace propagation
_current_span: contextvars.ContextVar[Span | None] = contextvars.ContextVar(
    "syrin_current_span", default=None
)
_current_session: contextvars.ContextVar[Session | None] = contextvars.ContextVar(
    "syrin_current_session", default=None
)


class SpanKind(StrEnum):
    """Types of spans in the system."""

    AGENT = "agent"  # Agent execution
    LLM = "llm"  # LLM completion call
    TOOL = "tool"  # Tool execution
    MEMORY = "memory"  # Memory operation
    BUDGET = "budget"  # Budget check/operation
    GUARDRAIL = "guardrail"  # Guardrail check
    HANDOFF = "handoff"  # Agent handoff
    WORKFLOW = "workflow"  # User-defined workflow
    INTERNAL = "internal"  # Internal framework operation


class SpanStatus(StrEnum):
    """Status of a span."""

    OK = "ok"
    ERROR = "error"
    CANCELLED = "cancelled"
    PENDING = "pending"


@dataclass
class SpanContext:
    """Context for span propagation (trace_id, span_id, flags)."""

    trace_id: str
    span_id: str
    parent_span_id: str | None = None
    flags: int = 1  # 1 = sampled, 0 = not sampled

    @classmethod
    def create(cls, parent: SpanContext | None = None) -> SpanContext:
        """Create a new span context, optionally as child of parent."""
        if parent:
            return cls(
                trace_id=parent.trace_id,
                span_id=_generate_id(),
                parent_span_id=parent.span_id,
                flags=parent.flags,
            )
        return cls(
            trace_id=_generate_id(),
            span_id=_generate_id(),
            parent_span_id=None,
            flags=1,
        )

    def is_sampled(self) -> bool:
        """Check if this trace should be sampled."""
        return self.flags & 1 == 1


@dataclass
class Session:
    """A session groups related traces together (e.g., a conversation).

    span_count is updated when spans are exported while this session is the current
    context (i.e. spans created inside ``with trace.session(...):``).
    """

    id: str
    start_time: datetime = field(default_factory=lambda: datetime.now(UTC))
    metadata: dict[str, object] = field(default_factory=dict)
    span_count: int = 0

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "start_time": self.start_time.isoformat(),
            "metadata": self.metadata,
            "span_count": self.span_count,
        }


@dataclass
class Span:
    """A span represents a single operation within a trace.

    Spans form a tree structure via parent-child relationships.
    Each span captures timing, attributes, and events.
    """

    # Identification
    name: str
    kind: SpanKind
    context: SpanContext
    session_id: str | None = None

    # Timing
    start_time: datetime = field(default_factory=lambda: datetime.now(UTC))
    end_time: datetime | None = None

    # Status and result
    status: SpanStatus = SpanStatus.PENDING
    status_message: str | None = None

    # Data
    attributes: dict[str, object] = field(default_factory=dict)
    events: list[dict[str, object]] = field(default_factory=list)
    children: list[Span] = field(default_factory=list)

    # Parent reference (not in serialization)
    parent: Span | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Set parent reference on children."""
        for child in self.children:
            child.parent = self

    @property
    def duration_ms(self) -> float:
        """Calculate duration in milliseconds."""
        end = self.end_time or datetime.now(UTC)
        return (end - self.start_time).total_seconds() * 1000

    @property
    def trace_id(self) -> str:
        return self.context.trace_id

    @property
    def span_id(self) -> str:
        return self.context.span_id

    @property
    def parent_span_id(self) -> str | None:
        return self.context.parent_span_id

    def set_attribute(self, key: str, value: object) -> None:
        """Set a semantic attribute on this span."""
        self.attributes[key] = value

    def set_attributes(self, attrs: dict[str, object]) -> None:
        """Set multiple attributes at once."""
        self.attributes.update(attrs)

    def add_event(self, name: str, attributes: dict[str, object] | None = None) -> None:
        """Add a timed event to this span."""
        self.events.append(
            {
                "name": name,
                "timestamp": datetime.now(UTC).isoformat(),
                "attributes": attributes or {},
            }
        )

    def set_status(self, status: SpanStatus, message: str | None = None) -> None:
        """Set the span status."""
        self.status = status
        if message:
            self.status_message = message

    def record_exception(self, exception: Exception) -> None:
        """Record an exception on this span."""
        self.set_status(SpanStatus.ERROR, str(exception))
        self.add_event(
            "exception",
            {
                "exception.type": type(exception).__name__,
                "exception.message": str(exception),
            },
        )

    def end(self, status: SpanStatus | None = None, message: str | None = None) -> None:
        """Mark the span as ended."""
        self.end_time = datetime.now(UTC)
        if status:
            self.set_status(status, message)
        elif self.status == SpanStatus.PENDING:
            self.set_status(SpanStatus.OK)

    def add_child(self, span: Span) -> None:
        """Add a child span."""
        span.parent = self
        self.children.append(span)

    def walk(self) -> Iterator[Span]:
        """Iterate over this span and all descendants (pre-order)."""
        yield self
        for child in self.children:
            yield from child.walk()

    def to_dict(self) -> dict[str, object]:
        """Convert span to dictionary for serialization."""
        return {
            "name": self.name,
            "kind": self.kind.value,
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "parent_span_id": self.parent_span_id,
            "session_id": self.session_id,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_ms": self.duration_ms,
            "status": self.status.value,
            "status_message": self.status_message,
            "attributes": self.attributes,
            "events": self.events,
            "children": [c.to_dict() for c in self.children],
        }


class Tracer:
    """Main tracer class for creating and managing spans.

    This is a singleton-like class accessed via get_tracer() or the module-level
    trace object.
    """

    def __init__(self) -> None:
        self._exporters: list[SpanExporter] = []
        self._spans: list[Span] = []  # Root spans
        self._lock = threading.Lock()
        self._debug_mode = False
        self._sampler = None  # Optional sampler
        self._collect_metrics = True  # Whether to collect metrics

    def add_exporter(self, exporter: SpanExporter) -> None:
        """Add a span exporter."""
        self._exporters.append(exporter)

    def set_debug_mode(self, enabled: bool) -> None:
        """Enable or disable debug mode (captures full context)."""
        self._debug_mode = enabled

    @property
    def debug_mode(self) -> bool:
        return self._debug_mode

    def set_sampler(self, sampler: object) -> None:
        """Set a sampler to control which spans are recorded."""
        self._sampler = sampler  # type: ignore[assignment]

    def set_collect_metrics(self, enabled: bool) -> None:
        """Enable or disable metrics collection."""
        self._collect_metrics = enabled

    @contextmanager
    def span(
        self,
        name: str,
        kind: SpanKind = SpanKind.INTERNAL,
        attributes: dict[str, object] | None = None,
    ) -> Iterator[Span]:
        """Create a new span as a context manager.

        Automatically sets parent from current context.
        """
        parent = _current_span.get()
        context = SpanContext.create(parent.context if parent else None)

        session = _current_session.get()
        span = Span(
            name=name,
            kind=kind,
            context=context,
            session_id=session.id if session else None,
        )

        if attributes:
            span.set_attributes(attributes)

        # Set parent relationship
        if parent:
            parent.add_child(span)

        # Track root spans
        if parent is None:
            with self._lock:
                self._spans.append(span)

        # Set as current span for duration of context
        token = _current_span.set(span)

        try:
            yield span
            span.end(status=SpanStatus.OK)
        except Exception as e:
            span.record_exception(e)
            span.end(status=SpanStatus.ERROR)
            raise
        finally:
            _current_span.reset(token)
            # Export completed span
            self._export(span)

    @contextmanager
    def session(self, session_id: str | None = None, **metadata: object) -> Iterator[Session]:
        """Create a new session context.

        All spans created within this context will be associated with the session.
        """
        session = Session(
            id=session_id or _generate_id(),
            metadata=metadata,
        )
        token = _current_session.set(session)

        try:
            yield session
        finally:
            _current_session.reset(token)

    def get_current_span(self) -> Span | None:
        """Get the current span from context."""
        return _current_span.get()

    def get_current_session(self) -> Session | None:
        """Get the current session from context."""
        return _current_session.get()

    def _export(self, span: Span) -> None:
        """Export a completed span to all exporters."""
        # Check sampling policy
        if self._sampler is not None:
            try:
                should_sample = self._sampler.should_sample(span)
            except Exception:
                should_sample = True  # Default to sampling on error
            if not should_sample:
                return

        # Update session span_count when span is associated with current session
        session = _current_session.get()
        if session is not None and span.session_id and session.id == span.session_id:
            session.span_count += 1

        # Record metrics
        if self._collect_metrics:
            try:
                from syrin.observability.metrics import record_span_metrics

                record_span_metrics(span)
            except Exception as e:
                _log.warning("Failed to record metrics: %s", e)

        # Export to all exporters
        for exporter in self._exporters:
            try:
                exporter.export(span)
            except Exception as e:
                _log.warning("Failed to export span to %s: %s", type(exporter).__name__, e)

    def clear(self) -> None:
        """Clear all stored spans (mainly for testing)."""
        with self._lock:
            self._spans.clear()


class SpanExporter(ABC):
    """Abstract base class for span exporters."""

    @abstractmethod
    def export(self, span: Span) -> None:
        """Export a completed span."""
        ...


class ConsoleExporter(SpanExporter):
    """Export spans to console in a human-readable tree format."""

    def __init__(self, pretty: bool = True, colors: bool = True, verbose: bool = False) -> None:
        self.pretty = pretty
        self.colors = colors
        self.verbose = verbose

    def export(self, span: Span) -> None:
        """Export span to console."""
        tracer = get_tracer()
        is_debug = tracer.debug_mode if hasattr(tracer, "debug_mode") else False
        is_verbose = is_debug or self.verbose

        if span.parent_span_id is None or is_verbose:
            # In verbose/debug mode, each span is printed when exported; avoid
            # printing children again when printing the root (they were already printed).
            include_children = not is_verbose or span.parent_span_id is not None
            print(self._format_span(span, include_children=include_children))

    def _format_span(self, span: Span, indent: int = 0, include_children: bool = True) -> str:
        """Format a span and optionally its children as a tree."""
        prefix = "  " * indent

        # Color codes
        gray = "\033[90m" if self.colors else ""
        green = "\033[92m" if self.colors else ""
        red = "\033[91m" if self.colors else ""
        yellow = "\033[93m" if self.colors else ""
        reset = "\033[0m" if self.colors else ""

        # Status color
        status_color = (
            green
            if span.status == SpanStatus.OK
            else red
            if span.status == SpanStatus.ERROR
            else yellow
        )

        lines = []
        lines.append(f"{prefix}{span.kind.value}: {span.name}")
        lines.append(
            f"{prefix}  {gray}trace_id={span.trace_id[:8]} span_id={span.span_id[:8]}{reset}"
        )
        lines.append(
            f"{prefix}  duration={span.duration_ms:.2f}ms "
            f"status={status_color}{span.status.value}{reset}"
        )

        # Key attributes
        attrs = self._format_attributes(span.attributes, indent + 2)
        if attrs:
            lines.append(f"{prefix}  attributes:")
            lines.extend(attrs)

        # Events (fallback, provider_error, etc.)
        if span.events:
            lines.append(f"{prefix}  events:")
            for ev in span.events:
                name = ev.get("name", "event")
                ev_attrs = ev.get("attributes") or {}
                ev_str = ", ".join(f"{k}={v}" for k, v in ev_attrs.items() if v is not None)  # type: ignore[attr-defined]
                lines.append(
                    f"{prefix}    - {name}: {ev_str}" if ev_str else f"{prefix}    - {name}"
                )

        # Children (only when not in verbose mode for root, to avoid duplicate output)
        if include_children:
            for child in span.children:
                lines.append(self._format_span(child, indent + 1, include_children=True))

        return "\n".join(lines)

    def _format_attributes(self, attrs: dict[str, object], indent: int) -> list[str]:
        """Format all attributes without filtering. Cost keys shown as decimal with $."""
        prefix = "  " * indent
        lines = []

        cost_keys = ("cost.usd", "llm.cost", "cost")
        for key, value in attrs.items():
            if isinstance(value, str) and len(value) > 200:
                value = value[:197] + "..."
            elif key in cost_keys and isinstance(value, (int, float)):
                value = f"${value:.6f}".rstrip("0").rstrip(".")
            lines.append(f"{prefix}{key}={value}")

        return lines


class JSONLExporter(SpanExporter):
    """Export spans to a JSONL file."""

    def __init__(self, filepath: str) -> None:
        self.filepath = filepath

    def export(self, span: Span) -> None:
        """Append span to JSONL file."""
        with open(self.filepath, "a") as f:
            f.write(json.dumps(span.to_dict(), default=str) + "\n")


class InMemoryExporter(SpanExporter):
    """Store spans in memory for testing/debugging."""

    def __init__(self) -> None:
        self.spans: list[Span] = []

    def export(self, span: Span) -> None:
        self.spans.append(span)

    def get_root_spans(self) -> list[Span]:
        """Get all root spans (those without parents)."""
        return [s for s in self.spans if s.parent_span_id is None]

    def clear(self) -> None:
        """Clear all stored spans."""
        self.spans.clear()


# Singleton tracer instance
_tracer: Tracer | None = None


def get_tracer() -> Tracer:
    """Get the global tracer instance."""
    global _tracer
    if _tracer is None:
        _tracer = Tracer()
    return _tracer


def _generate_id() -> str:
    """Generate a unique ID (16 hex chars for OTel compatibility)."""
    return uuid.uuid4().hex[:16]


# Module-level convenience methods
trace = get_tracer()


def span(name: str, kind: SpanKind = SpanKind.INTERNAL, **attributes: object) -> object:
    """Create a span context manager (convenience method)."""
    return trace.span(name, kind, attributes)


def session(session_id: str | None = None, **metadata: object) -> object:
    """Create a session context manager (convenience method)."""
    return trace.session(session_id, **metadata)


def current_span() -> Span | None:
    """Get current span (convenience method)."""
    return trace.get_current_span()


def current_session() -> Session | None:
    """Get current session (convenience method)."""
    return trace.get_current_session()


def set_debug(enabled: bool) -> None:
    """Enable/disable debug mode globally."""
    trace.set_debug_mode(enabled)


# =============================================================================
# Convenience Decorators and Context Managers
# =============================================================================


def llm_span(
    model: str,
    operation: str = "complete",
    **attributes: object,
) -> object:
    """Create an LLM span with common attributes pre-set.

    Args:
        model: Model identifier (e.g., "gpt-4o")
        operation: Operation type ("complete", "stream", "embed")
        **attributes: Additional attributes

    Example:
        >>> with llm_span("gpt-4o", prompt="Hello") as span:
        ...     response = call_llm(prompt)
        ...     span.set_attribute("completion", response)
    """
    return trace.span(
        f"llm.{operation}",
        kind=SpanKind.LLM,
        attributes={
            SemanticAttributes.LLM_MODEL: model,
            **attributes,
        },
    )


def tool_span(
    tool_name: str,
    input: dict[str, object] | None = None,
    **attributes: object,
) -> object:
    """Create a tool span with common attributes pre-set.

    Args:
        tool_name: Name of the tool being called
        input: Tool input arguments
        **attributes: Additional attributes

    Example:
        >>> with tool_span("calculator", input={"x": 1, "y": 2}) as span:
        ...     result = calculator(x, y)
        ...     span.set_attribute("output", result)
    """
    return trace.span(
        f"tool.{tool_name}",
        kind=SpanKind.TOOL,
        attributes={
            SemanticAttributes.TOOL_NAME: tool_name,
            SemanticAttributes.TOOL_INPUT: json.dumps(input) if input else "{}",
            **attributes,
        },
    )


def memory_span(
    operation: str,
    memory_type: str | None = None,
    query: str | None = None,
    **attributes: object,
) -> object:
    """Create a memory span with common attributes pre-set.

    Args:
        operation: Memory operation ("recall", "store", "forget", "consolidate")
        memory_type: Type of memory (core, episodic, semantic, procedural)
        query: Query string for recall operations
        **attributes: Additional attributes

    Example:
        >>> with memory_span("recall", memory_type="history", query="user preferences") as span:
        ...     results = memory recall(query)
        ...     span.set_attribute("results_count", len(results))
    """
    attrs = {
        SemanticAttributes.MEMORY_OPERATION: operation,
        **attributes,
    }
    if memory_type:
        attrs[SemanticAttributes.MEMORY_TYPE] = memory_type
    if query:
        attrs[SemanticAttributes.MEMORY_QUERY] = query

    return trace.span(
        f"memory.{operation}",
        kind=SpanKind.MEMORY,
        attributes=attrs,
    )


def budget_span(
    operation: str,
    limit: float | None = None,
    used: float | None = None,
    **attributes: object,
) -> object:
    """Create a budget span with common attributes pre-set.

    Args:
        operation: Budget operation ("check", "threshold", "exceeded")
        limit: Budget limit
        used: Amount used so far
        **attributes: Additional attributes

    Example:
        >>> with budget_span("check", limit=10.0, used=5.0) as span:
        ...     remaining = limit - used
        ...     span.set_attribute("remaining", remaining)
    """
    attrs = {**attributes}
    if limit is not None:
        attrs[SemanticAttributes.BUDGET_LIMIT] = limit
    if used is not None:
        attrs[SemanticAttributes.BUDGET_USED] = used

    return trace.span(
        f"budget.{operation}",
        kind=SpanKind.BUDGET,
        attributes=attrs,
    )


def guardrail_span(
    name: str,
    stage: str,
    **attributes: object,
) -> object:
    """Create a guardrail span with common attributes pre-set.

    Args:
        name: Name of the guardrail
        stage: Stage ("input" or "output")
        **attributes: Additional attributes

    Example:
        >>> with guardrail_span("content_filter", stage="output") as span:
        ...     result = guardrail.check(content)
        ...     span.set_attribute("passed", result.passed)
    """
    return trace.span(
        f"guardrail.{name}",
        kind=SpanKind.GUARDRAIL,
        attributes={
            SemanticAttributes.GUARDRAIL_NAME: name,
            SemanticAttributes.GUARDRAIL_STAGE: stage,
            **attributes,
        },
    )


def handoff_span(
    source_agent: str,
    target_agent: str,
    **attributes: object,
) -> object:
    """Create a handoff span with common attributes pre-set.

    Args:
        source_agent: Source agent name
        target_agent: Target agent name
        **attributes: Additional attributes

    Example:
        >>> with handoff_span(" triage", "specialist") as span:
        ...     span.set_attribute("memories_transferred", 5)
    """
    return trace.span(
        "agent.handoff",
        kind=SpanKind.HANDOFF,
        attributes={
            SemanticAttributes.HANDOFF_SOURCE: source_agent,
            SemanticAttributes.HANDOFF_TARGET: target_agent,
            **attributes,
        },
    )


def agent_span(
    name: str,
    **attributes: object,
) -> object:
    """Create an agent span with common attributes pre-set.

    Args:
        name: Agent name
        **attributes: Additional attributes

    Example:
        >>> with agent_span("research_agent", user_id="123") as span:
        ...     result = agent.run(query)
    """
    return trace.span(
        f"agent.{name}",
        kind=SpanKind.AGENT,
        attributes={
            SemanticAttributes.AGENT_NAME: name,
            **attributes,
        },
    )


# =============================================================================
# Semantic Attributes
# =============================================================================


class SemanticAttributes:
    """Standard semantic attribute keys for consistent observability."""

    # Agent attributes
    AGENT_NAME = "agent.name"
    AGENT_CLASS = "agent.class"
    AGENT_ITERATION = "agent.iteration"

    # LLM attributes
    LLM_MODEL = "llm.model"
    LLM_PROVIDER = "llm.provider"
    LLM_PROMPT = "llm.prompt"
    LLM_COMPLETION = "llm.completion"
    LLM_TOKENS_INPUT = "llm.tokens.input"
    LLM_TOKENS_OUTPUT = "llm.tokens.output"
    LLM_TOKENS_TOTAL = "llm.tokens.total"
    LLM_COST = "llm.cost"
    LLM_TEMPERATURE = "llm.temperature"
    LLM_STOP_REASON = "llm.stop_reason"

    # Tool attributes
    TOOL_NAME = "tool.name"
    TOOL_INPUT = "tool.input"
    TOOL_OUTPUT = "tool.output"
    TOOL_ERROR = "tool.error"
    TOOL_DURATION_MS = "tool.duration_ms"

    # Memory attributes
    MEMORY_OPERATION = "memory.operation"  # recall, store, forget
    MEMORY_TYPE = "memory.type"  # core, episodic, semantic, procedural
    MEMORY_QUERY = "memory.query"
    MEMORY_RESULTS_COUNT = "memory.results.count"

    # Budget attributes
    BUDGET_LIMIT = "budget.limit"
    BUDGET_USED = "budget.used"
    BUDGET_REMAINING = "budget.remaining"
    BUDGET_PERCENTAGE = "budget.percentage"

    # Guardrail attributes
    GUARDRAIL_NAME = "guardrail.name"
    GUARDRAIL_STAGE = "guardrail.stage"  # input, output
    GUARDRAIL_PASSED = "guardrail.passed"
    GUARDRAIL_VIOLATION = "guardrail.violation"

    # Handoff attributes
    HANDOFF_SOURCE = "handoff.source"
    HANDOFF_TARGET = "handoff.target"
    HANDOFF_MEMORIES_TRANSFERRED = "handoff.memories_transferred"

    # Session attributes
    SESSION_ID = "session.id"
    SESSION_MESSAGE_COUNT = "session.message_count"

    # Error attributes
    ERROR_TYPE = "error.type"
    ERROR_MESSAGE = "error.message"

    # Fallback and provider error attributes (used in span events)
    LLM_FALLBACK_FROM = "llm.fallback.from_model"
    LLM_FALLBACK_TO = "llm.fallback.to_model"
    LLM_PROVIDER_ERROR_MODEL = "llm.provider_error.model"


__all__ = [
    # Core classes
    "Span",
    "SpanKind",
    "SpanStatus",
    "SpanContext",
    "Session",
    "Tracer",
    "SpanExporter",
    "ConsoleExporter",
    "JSONLExporter",
    "InMemoryExporter",
    "SemanticAttributes",
    # Convenience functions
    "trace",
    "span",
    "session",
    "current_span",
    "current_session",
    "set_debug",
    "get_tracer",
    # Convenience decorators
    "llm_span",
    "tool_span",
    "memory_span",
    "budget_span",
    "guardrail_span",
    "handoff_span",
    "agent_span",
]
