"""Default context manager implementation."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol, cast

from syrin.context.compactors import CompactionResult, ContextCompactor, ContextCompactorProtocol
from syrin.context.config import Context, ContextStats, ContextWindowCapacity
from syrin.context.counter import TokenCounter, get_counter
from syrin.context.injection import InjectPlacement, PrepareInput
from syrin.context.map import ContextMap
from syrin.context.snapshot import (
    ContextBreakdown,
    ContextSegmentProvenance,
    ContextSegmentSource,
    ContextSnapshot,
    MessagePreview,
    _context_rot_risk_from_utilization,
)
from syrin.enums import ContextMode, Hook, ThresholdMetric
from syrin.threshold import ThresholdContext

# Max chars for content_snippet in MessagePreview.
_SNIPPET_MAX_CHARS = 80


def _apply_context_mode(
    messages: list[dict[str, Any]],
    mode: ContextMode,
    focused_keep: int,
) -> tuple[list[dict[str, Any]], int]:
    """Apply context mode to messages. Return (filtered_messages, dropped_count).

    Conversation = user/assistant messages before the last (current user). Prefix = leading system.
    """
    if mode == ContextMode.FULL:
        return messages, 0
    if mode == ContextMode.INTELLIGENT:
        raise NotImplementedError(
            "context_mode=intelligent requires a relevance scorer; "
            "use context_mode=focused for now, or wait for pull-based context store (Step 10)"
        )
    if mode != ContextMode.FOCUSED:
        return messages, 0

    if not messages:
        return messages, 0

    # Last message = current user. Prefix = leading system. Conversation = user/assistant before last.
    prefix: list[dict[str, Any]] = []
    i = 0
    while i < len(messages) and messages[i].get("role") == "system":
        prefix.append(messages[i])
        i += 1

    if i >= len(messages):
        return messages, 0

    # conversation + current_user
    rest = messages[i:]
    if len(rest) <= 1:
        return messages, 0  # Only current user or empty after prefix

    conversation = rest[:-1]
    current_user_msg = rest[-1]

    # Last N turns: a turn = user + assistant(s). Find start index for last N user messages.
    user_indices = [j for j, m in enumerate(conversation) if m.get("role") == "user"]
    if not user_indices or focused_keep >= len(user_indices):
        return messages, 0

    start_idx = user_indices[-focused_keep]
    filtered_conv = conversation[start_idx:]
    dropped = len(conversation) - len(filtered_conv)

    result = prefix + filtered_conv + [current_user_msg]
    return result, dropped


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

    def __enter__(self) -> _NullSpan:
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
    _last_snapshot: ContextSnapshot | None = field(default=None, repr=False)
    _injected_indices: set[int] = field(default_factory=set, repr=False)
    _injected_source_detail: str | None = field(default=None, repr=False)
    _last_injected_messages: list[dict[str, Any]] = field(default_factory=list, repr=False)
    _context_mode_dropped_count: int = field(default=0, repr=False)

    def __post_init__(self) -> None:
        """Apply Context.encoding and Context.compactor (or default with compaction_*) when set."""
        if getattr(self.context, "encoding", None) is not None:
            self._counter = TokenCounter(encoding=self.context.encoding)
        compactor = self.context.compactor
        if compactor is not None:
            self._compactor = compactor
        else:
            self._compactor = ContextCompactor(
                compaction_prompt=getattr(self.context, "compaction_prompt", None),
                compaction_system_prompt=getattr(self.context, "compaction_system_prompt", None),
                compaction_model=getattr(self.context, "compaction_model", None),
            )

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

    def _merge_injected(
        self,
        messages: list[dict[str, Any]],
        injected: list[dict[str, Any]],
        placement: InjectPlacement,
    ) -> tuple[list[dict[str, Any]], set[int]]:
        """Merge injected messages into messages by placement. Return (merged, inserted_indices)."""
        # Normalize injected: ensure role and content
        normalized = []
        for m in injected:
            d = dict(m) if isinstance(m, dict) else {}
            role = d.get("role", "user")
            content = d.get("content", "")
            normalized.append({"role": role, "content": content})

        if not normalized:
            return messages, set()

        if placement == InjectPlacement.PREPEND_TO_SYSTEM:
            merged = normalized + messages
            return merged, set(range(len(normalized)))

        if placement == InjectPlacement.AFTER_CURRENT_TURN:
            merged = messages + normalized
            return merged, set(range(len(messages), len(messages) + len(normalized)))

        # BEFORE_CURRENT_TURN: insert before last message (current user)
        if not messages:
            return normalized, set(range(len(normalized)))
        merged = messages[:-1] + normalized + messages[-1:]
        start = len(messages) - 1
        return merged, set(range(start, start + len(normalized)))

    def prepare(
        self,
        messages: list[dict[str, Any]],
        system_prompt: str,
        tools: list[dict[str, Any]],
        memory_context: str = "",
        capacity: ContextWindowCapacity | None = None,
        context: Context | None = None,
        *,
        inject: list[dict[str, Any]] | None = None,
        inject_source_detail: str | None = None,
        pulled_segments: list[dict[str, Any]] | None = None,
        pull_scores: list[float] | None = None,
        output_chunks: list[dict[str, Any]] | None = None,
        output_chunk_scores: list[float] | None = None,
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
            inject: Optional per-call injected messages (RAG, dynamic blocks). When provided,
                runtime_inject callable is not used.
            inject_source_detail: Provenance source_detail for inject (e.g. 'rag').
                Default from Context.inject_source_detail when inject from runtime_inject.

        Returns:
            ContextPayload ready for LLM call
        """
        effective_context = context if context is not None else self.context
        capacity = capacity or effective_context.get_capacity()

        with self._span("context.prepare") as span:
            if span:
                span.set_attribute("context.max_tokens", capacity.max_tokens)
                span.set_attribute("context.available", capacity.available)

            self._injected_indices = set()
            self._injected_source_detail = None
            self._last_injected_messages = []

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

            # Apply context mode (full / focused / intelligent)
            mode = getattr(effective_context, "context_mode", ContextMode.FULL)
            focused_keep = getattr(effective_context, "focused_keep", 10)
            all_messages, self._context_mode_dropped_count = _apply_context_mode(
                all_messages, mode, focused_keep
            )

            # Resolve injected messages: map summary (if inject_map_summary), then inject or runtime_inject
            injected_msgs: list[dict[str, Any]] = []
            if getattr(effective_context, "inject_map_summary", False):
                ctx_map = self.get_map()
                if ctx_map.summary and ctx_map.summary.strip():
                    injected_msgs.append(
                        {
                            "role": "system",
                            "content": f"[Session summary]\n{ctx_map.summary}",
                        }
                    )
            if inject is not None:
                injected_msgs.extend(list(inject))
                self._injected_source_detail = inject_source_detail or getattr(
                    effective_context, "inject_source_detail", "injected"
                )
            elif getattr(effective_context, "runtime_inject", None) is not None:
                runtime_inject_fn = effective_context.runtime_inject
                assert runtime_inject_fn is not None
                last_user = next(
                    (m for m in reversed(messages) if m.get("role") == "user"),
                    None,
                )
                user_input = last_user.get("content", "") if last_user else ""
                prepare_input = PrepareInput(
                    messages=all_messages,
                    system_prompt=system_prompt,
                    tools=tools,
                    memory_context=memory_context,
                    user_input=user_input,
                )
                injected_msgs.extend(runtime_inject_fn(prepare_input))
                self._injected_source_detail = getattr(
                    effective_context, "inject_source_detail", "injected"
                )

            # Merge injected messages by placement
            placement = getattr(
                effective_context, "inject_placement", InjectPlacement.BEFORE_CURRENT_TURN
            )
            injected_tokens_count: int = 0
            if injected_msgs:
                all_messages, inserted_at = self._merge_injected(
                    all_messages, injected_msgs, placement
                )
                self._injected_indices = inserted_at
                self._last_injected_messages = [
                    dict(all_messages[i]) for i in sorted(inserted_at) if i < len(all_messages)
                ]
                for i in inserted_at:
                    if i < len(all_messages):
                        msg = all_messages[i]
                        content = msg.get("content", "")
                        content_str = content if isinstance(content, str) else str(content)
                        role = msg.get("role", "user")
                        injected_tokens_count += self._counter.count(
                            content_str
                        ) + self._counter._role_overhead(role)

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
                # Proactive compaction: when utilization >= auto_compact_at, compact once before thresholds
                auto_at = getattr(effective_context, "auto_compact_at", None)
                if auto_at is not None and capacity.utilization >= auto_at:
                    _compact_fn()
                    # Recompute token count after compaction so thresholds see updated utilization
                    tokens_after_compact = (
                        self._counter.count_messages(
                            [m for m in all_messages if m.get("role") != "system"],
                            system_prompt,
                        ).total
                        + tools_tokens
                    )
                    capacity.used_tokens = tokens_after_compact

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

                breakdown = self._counter.count_breakdown(
                    system_prompt=system_prompt,
                    memory_context=memory_context,
                    tools=tools,
                    tokens_used=tokens_used,
                    injected_tokens=injected_tokens_count,
                )
                self._stats = ContextStats(
                    total_tokens=tokens_used,
                    max_tokens=capacity.max_tokens,
                    utilization=capacity.utilization,
                    compacted=self._did_compact,
                    compact_count=self._run_compaction_count,
                    compact_method=self._last_compaction_method if self._did_compact else None,
                    thresholds_triggered=thresholds_triggered,
                    breakdown=breakdown,
                )

                snap = self._build_snapshot(
                    final_msgs=final_msgs,
                    system_prompt=system_prompt,
                    memory_context=memory_context,
                    tools=tools,
                    tokens_used=tokens_used,
                    capacity=capacity,
                    breakdown=breakdown,
                    context_mode=mode.value,
                    context_mode_dropped_count=self._context_mode_dropped_count,
                    pulled_segments=pulled_segments or [],
                    pull_scores=pull_scores or [],
                    output_chunks=output_chunks or [],
                    output_chunk_scores=output_chunk_scores or [],
                )
                self._last_snapshot = snap
                self._emit(
                    Hook.CONTEXT_SNAPSHOT.value,
                    {"snapshot": snap.to_dict(), "utilization_pct": snap.utilization_pct},
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

    def _build_snapshot(
        self,
        *,
        final_msgs: list[dict[str, Any]],
        system_prompt: str,
        memory_context: str,
        tools: list[dict[str, Any]],
        tokens_used: int,
        capacity: ContextWindowCapacity,
        breakdown: ContextBreakdown,
        context_mode: str = "full",
        context_mode_dropped_count: int = 0,
        pulled_segments: list[dict[str, Any]] | None = None,
        pull_scores: list[float] | None = None,
        output_chunks: list[dict[str, Any]] | None = None,
        output_chunk_scores: list[float] | None = None,
    ) -> ContextSnapshot:
        """Build ContextSnapshot from the last prepare state. Breakdown is computed once in prepare()."""
        import time

        utilization_pct = (
            (capacity.used_tokens / capacity.available * 100.0) if capacity.available > 0 else 0.0
        )
        tokens_available = max(0, capacity.available - capacity.used_tokens)

        previews: list[MessagePreview] = []
        provenances: list[ContextSegmentProvenance] = []
        why_included: list[str] = []

        conversation_reason_added = False
        for i, msg in enumerate(final_msgs):
            role = msg.get("role", "user")
            content = msg.get("content", "")
            content_str = content if isinstance(content, str) else str(content)
            snippet = (
                content_str[:_SNIPPET_MAX_CHARS] + "..."
                if len(content_str) > _SNIPPET_MAX_CHARS
                else content_str
            )
            tok = self._counter.count(content_str) + self._counter._role_overhead(role)

            is_injected = any(
                m.get("role") == role and str(m.get("content", "")) == content_str
                for m in self._last_injected_messages
            )
            if is_injected:
                source = ContextSegmentSource.INJECTED
                detail = self._injected_source_detail or "injected"
                why_included.append(f"injected: {detail}")
            elif system_prompt and content_str.strip() == system_prompt.strip():
                source = ContextSegmentSource.SYSTEM
                why_included.append("system prompt")
            elif memory_context and "[Memory]" in content_str:
                source = ContextSegmentSource.MEMORY
                why_included.append("recalled memory")
            elif i == len(final_msgs) - 1 and role == "user":
                source = ContextSegmentSource.CURRENT_PROMPT
                why_included.append("current user message")
            else:
                source = ContextSegmentSource.CONVERSATION
                if not conversation_reason_added:
                    why_included.append("conversation history")
                    conversation_reason_added = True

            previews.append(
                MessagePreview(
                    role=role,
                    content_snippet=snippet,
                    token_count=tok,
                    source=source,
                )
            )
            prov_detail: str | None = f"message index {i}"
            if source == ContextSegmentSource.INJECTED and self._injected_source_detail:
                prov_detail = self._injected_source_detail
            provenances.append(
                ContextSegmentProvenance(
                    segment_id=str(i),
                    source=source,
                    source_detail=prov_detail,
                )
            )

        if tools:
            why_included.append("tool definitions")
            provenances.append(
                ContextSegmentProvenance(
                    segment_id="tools",
                    source=ContextSegmentSource.TOOLS,
                    source_detail="tool schemas",
                )
            )

        if not why_included:
            why_included = ["context formation"]

        return ContextSnapshot(
            timestamp=time.time(),
            total_tokens=tokens_used,
            max_tokens=capacity.max_tokens,
            tokens_available=tokens_available,
            utilization_pct=utilization_pct,
            breakdown=breakdown,
            compacted=self._did_compact,
            compact_method=self._last_compaction_method if self._did_compact else None,
            messages_count=len(final_msgs),
            message_preview=previews,
            raw_messages=None,
            provenance=provenances,
            why_included=why_included,
            context_rot_risk=_context_rot_risk_from_utilization(utilization_pct),
            context_mode=context_mode,
            context_mode_dropped_count=context_mode_dropped_count,
            pulled_segments=pulled_segments or [],
            pull_scores=pull_scores or [],
            output_chunks=output_chunks or [],
            output_chunk_scores=output_chunk_scores or [],
        )

    def _get_map_backend(self) -> Any:
        """Lazy-init map backend from context. Returns None when map_backend is None."""
        backend = getattr(self.context, "map_backend", None)
        if backend != "file":
            return None
        path = getattr(self.context, "map_path", None) or ""
        if not path.strip():
            return None
        from syrin.context.map import FileContextMapBackend

        return FileContextMapBackend(path)

    def get_map(self) -> ContextMap:
        """Return the persistent context map. Empty if no backend configured."""
        backend = self._get_map_backend()
        if backend is None:
            return ContextMap()
        return cast(ContextMap, backend.load())

    def update_map(self, partial: ContextMap | dict[str, Any]) -> None:
        """Merge partial into the map and persist. No-op if no backend."""
        import time as _time

        backend = self._get_map_backend()
        if backend is None:
            return
        current = backend.load()
        p = ContextMap.from_dict(partial) if isinstance(partial, dict) else partial
        if p.topics:
            current.topics = list(p.topics)
        if p.decisions:
            current.decisions = list(p.decisions)
        if p.segment_ids:
            current.segment_ids = list(p.segment_ids)
        if p.summary or (isinstance(partial, dict) and "summary" in partial):
            current.summary = p.summary
        current.last_updated = _time.time()
        backend.save(current)

    def snapshot(self) -> ContextSnapshot:
        """Return a point-in-time view of the context from the last prepare.

        Before any prepare(), returns an empty snapshot (zeros, low rot risk).
        After prepare(), returns full view: message_preview, provenance, why_included, context_rot_risk.
        """
        if self._last_snapshot is None:
            return ContextSnapshot()
        return self._last_snapshot

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
