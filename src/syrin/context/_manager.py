"""Default context manager implementation."""

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

from syrin.context.compactors import CompactionResult, ContextCompactor, ContextCompactorProtocol
from syrin.context.config import Context, ContextStats, ContextWindowCapacity
from syrin.context.counter import TokenCounter, get_counter
from syrin.enums import Hook, ThresholdMetric
from syrin.threshold import ThresholdContext


@dataclass
class ContextPayload:
    """The prepared context for an LLM call.

    Attributes:
        messages: Messages ready for the model.
        system_prompt: System prompt.
        tools: Tool definitions.
        tokens: Total token count.
    """

    messages: list[dict[str, Any]]
    system_prompt: str
    tools: list[dict[str, Any]]
    tokens: int


class _NullSpan:
    """Null context manager for when no tracer is set."""

    def __enter__(self) -> "_NullSpan":
        return self

    def __exit__(self, *args: Any) -> None:
        pass

    def set_attribute(self, key: str, value: Any) -> None:
        pass


class _ContextSpan:
    """Wrapper for tracer spans."""

    def __init__(self, tracer: Any, name: str, **attrs: Any):
        self._tracer = tracer
        self._name = name
        self._attrs = attrs
        self._span: Any | None = None

    def __enter__(self) -> Any:
        if hasattr(self._tracer, "span"):
            self._span = self._tracer.span(self._name)
            for k, v in self._attrs.items():
                if self._span is not None:
                    self._span.set_attribute(k, v)
            if self._span is not None:
                return self._span.__enter__()
        return self

    def __exit__(self, *args: Any) -> None:
        if self._span is not None:
            self._span.__exit__(*args)

    def set_attribute(self, key: str, value: Any) -> None:
        if self._span is not None:
            self._span.set_attribute(key, value)


class ContextManager(Protocol):
    """Protocol for custom context management strategies.

    Implement this protocol to create custom context management strategies.
    prepare() may accept context for per-call override; ignore if not used.
    """

    def prepare(
        self,
        messages: list[dict[str, Any]],
        system_prompt: str,
        tools: list[dict[str, Any]],
        memory_context: str,
        capacity: ContextWindowCapacity,
        context: Context | None = None,
    ) -> ContextPayload:
        """Prepare context for LLM call."""
        ...

    def on_compact(self, _event: CompactionResult) -> None:
        """Called after compaction."""
        ...


@dataclass
class DefaultContextManager:
    """Default context manager with on-demand compaction.

    Features:
    - Automatic token counting (encoding from Context.encoding)
    - Compaction via ctx.compact() in threshold actions
    - Pluggable compactor (Context.compactor or default ContextCompactor)
    - Full observability via events and spans
    """

    context: Context = field(default_factory=Context)
    _counter: TokenCounter = field(default_factory=get_counter)
    _compactor: ContextCompactorProtocol = field(default_factory=ContextCompactor)
    _stats: ContextStats = field(default_factory=ContextStats)
    _compaction_count: int = 0
    _emit_fn: Callable[[str, dict[str, Any]], None] | None = field(default=None, repr=False)
    _tracer: Any = field(default=None, repr=False)
    _current_messages: list[dict[str, Any]] | None = field(default=None, repr=False)
    _current_available: int = 0
    _did_compact: bool = False
    _last_compaction_method: str | None = field(default=None, repr=False)
    _current_compact_fn: Callable[[], None] | None = field(default=None, repr=False)
    _run_compaction_count: int = field(default=0, repr=False)

    def __post_init__(self) -> None:
        """Apply Context.encoding and Context.compactor when set."""
        if getattr(self.context, "encoding", None) is not None:
            self._counter = TokenCounter(encoding=self.context.encoding)
        compactor = self.context.compactor
        if compactor is not None:
            self._compactor = compactor

    def _emit(self, event: Hook | str, ctx: dict[str, Any]) -> None:
        """Emit an event if emit_fn is configured."""
        if self._emit_fn:
            event_str = event.value if hasattr(event, "value") else str(event)
            self._emit_fn(event_str, ctx)

    def _span(self, name: str, **attrs: Any) -> _ContextSpan | _NullSpan:
        """Create a span if tracer is configured."""
        if self._tracer is None:
            return _NullSpan()
        return _ContextSpan(self._tracer, name, **attrs)

    def set_emit_fn(self, emit_fn: Callable[[str, dict[str, Any]], None]) -> None:
        """Set the event emit function for lifecycle hooks."""
        self._emit_fn = emit_fn

    def set_tracer(self, tracer: Any) -> None:
        """Set the tracer for observability."""
        self._tracer = tracer

    def prepare(
        self,
        messages: list[dict[str, Any]],
        system_prompt: str,
        tools: list[dict[str, Any]],
        memory_context: str = "",
        capacity: ContextWindowCapacity | None = None,
        context: Context | None = None,
    ) -> ContextPayload:
        """Prepare context for LLM call with automatic management.

        Args:
            messages: Conversation messages
            system_prompt: System prompt
            tools: Tool definitions
            memory_context: Injected memory context
            capacity: Context window capacity (auto-created if not provided)
            context: Optional Context for this call only (overrides agent's context; capacity and thresholds).
                When set, use its get_capacity() if capacity not provided, and its thresholds.

        Returns:
            ContextPayload ready for LLM call
        """
        effective_context = context if context is not None else self.context
        capacity = capacity or effective_context.get_capacity()

        with self._span("context.prepare") as span:
            if span:
                span.set_attribute("context.max_tokens", capacity.max_tokens)
                span.set_attribute("context.available", capacity.available)

            memory_msg: dict[str, Any] | None = None
            if memory_context:
                memory_msg = {
                    "role": "system",
                    "content": f"[Memory]\n{memory_context}",
                }

            final_messages = list(messages)

            if memory_msg:
                final_messages = [memory_msg] + final_messages

            system_msg: dict[str, Any] | None = None
            if system_prompt:
                system_msg = {"role": "system", "content": system_prompt}

            all_messages = final_messages
            if system_msg:
                first = final_messages[0] if final_messages else {}
                if first.get("role") != "system" or first.get("content") != system_msg.get(
                    "content"
                ):
                    all_messages = [system_msg] + all_messages

            tokens_before = self._counter.count_messages(all_messages).total
            tools_tokens = self._counter.count_tools(tools)
            available_for_messages = max(0, capacity.available - tools_tokens)

            # Always run thresholds and set stats (no early return when over capacity)
            capacity.used_tokens = tokens_before + tools_tokens
            self._current_messages = all_messages
            self._current_available = available_for_messages
            self._did_compact = False
            self._run_compaction_count = 0

            def _compact_fn() -> None:
                if self._current_messages is None or self._current_available <= 0:
                    return
                result = self._compactor.compact(
                    list(self._current_messages),
                    self._current_available,
                )
                self._current_messages.clear()
                self._current_messages.extend(result.messages)
                self._did_compact = True
                self._last_compaction_method = result.method
                self._compaction_count += 1
                self._run_compaction_count += 1
                compact_event = {
                    "method": result.method,
                    "tokens_before": result.tokens_before,
                    "tokens_after": result.tokens_after,
                    "messages_before": len(all_messages),
                    "messages_after": len(result.messages),
                }
                self._emit("context.compact", compact_event)
                if span:
                    span.set_attribute("context.compacted", True)
                    span.set_attribute("context.compaction_method", result.method)
                    span.set_attribute(
                        "context.tokens_saved", result.tokens_before - result.tokens_after
                    )

            self._current_compact_fn = _compact_fn
            try:
                thresholds_triggered = self._check_thresholds(
                    capacity, _compact_fn, thresholds=effective_context.thresholds
                )
                final_msgs = all_messages
                if system_msg and system_msg not in final_msgs:
                    final_msgs = [system_msg] + final_msgs

                tokens_used = (
                    self._counter.count_messages(
                        [m for m in final_msgs if m.get("role") != "system"],
                        system_prompt,
                    ).total
                    + tools_tokens
                )

                capacity.used_tokens = tokens_used

                self._stats = ContextStats(
                    total_tokens=tokens_used,
                    max_tokens=capacity.max_tokens,
                    utilization=capacity.utilization,
                    compacted=self._did_compact,
                    compact_count=self._run_compaction_count,
                    compact_method=self._last_compaction_method if self._did_compact else None,
                    thresholds_triggered=thresholds_triggered,
                )

                if span:
                    span.set_attribute("context.tokens", tokens_used)
                    span.set_attribute("context.utilization", capacity.utilization)
                    if thresholds_triggered:
                        span.set_attribute("context.thresholds_triggered", thresholds_triggered)

                return ContextPayload(
                    messages=final_msgs,
                    system_prompt=system_prompt,
                    tools=tools,
                    tokens=tokens_used,
                )
            finally:
                self._current_compact_fn = None
                self._current_messages = None

    def on_compact(self, _event: CompactionResult) -> None:
        """Hook called after compaction (e.g. by lifecycle). Count is updated in _compact_fn only."""

    def _check_thresholds(
        self,
        capacity: ContextWindowCapacity,
        compact_fn: Callable[[], None] | None = None,
        thresholds: list[Any] | None = None,
    ) -> list[str]:
        """Check and trigger thresholds using should_trigger (supports at_range).

        Returns list of triggered threshold metrics.
        """
        triggered: list[str] = []
        percent = capacity.percent
        metric = ThresholdMetric.TOKENS
        th_list = thresholds if thresholds is not None else self.context.thresholds

        for threshold in th_list:
            if not threshold.should_trigger(percent, metric):
                continue
            triggered.append(
                threshold.metric.value
                if hasattr(threshold.metric, "value")
                else str(threshold.metric)
            )
            threshold_event = {
                "at": getattr(threshold, "at", None),
                "at_range": getattr(threshold, "at_range", None),
                "percent": percent,
                "metric": threshold.metric,
                "tokens": capacity.used_tokens,
                "max_tokens": capacity.max_tokens,
            }
            self._emit("context.threshold", threshold_event)
            compact = compact_fn if compact_fn is not None else (lambda: None)
            ctx = ThresholdContext(
                percentage=percent,
                metric=threshold.metric,
                current_value=float(capacity.used_tokens),
                limit_value=float(capacity.max_tokens),
                compact=compact,
            )
            threshold.execute(ctx)

        return triggered

    @property
    def stats(self) -> ContextStats:
        """Get context statistics from last call."""
        return self._stats

    @property
    def current_tokens(self) -> int:
        """Get current token count."""
        return self._stats.total_tokens

    def compact(self) -> None:
        """Request compaction of current context (only valid during prepare, e.g. from threshold action).

        Call from a ContextThreshold action via ctx.compact() or agent.context.compact().
        No-op if not currently inside prepare().
        """
        if self._current_compact_fn is not None:
            self._current_compact_fn()


def create_context_manager(
    context: Context | None = None,
    emit_fn: Callable[[str, dict[str, Any]], None] | None = None,
    tracer: Any = None,
) -> DefaultContextManager:
    """Create a default context manager from config.

    Args:
        context: Context configuration (creates default if None)
        emit_fn: Optional event emit function for lifecycle hooks
        tracer: Optional tracer for observability

    Returns:
        Configured DefaultContextManager
    """
    if context is None:
        context = Context()

    manager = DefaultContextManager(context=context)
    if emit_fn:
        manager.set_emit_fn(emit_fn)
    if tracer:
        manager.set_tracer(tracer)
    return manager


__all__ = ["ContextManager", "ContextPayload", "DefaultContextManager", "create_context_manager"]
