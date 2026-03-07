"""Context configuration and stats."""

import dataclasses
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from syrin.threshold import ContextThreshold

if TYPE_CHECKING:
    from syrin.context.injection import PrepareInput
    from syrin.model import Model

from syrin.budget import TokenLimits
from syrin.context.compactors import ContextCompactor, ContextCompactorProtocol
from syrin.context.injection import InjectPlacement
from syrin.context.snapshot import ContextBreakdown


@dataclass
class ContextStats:
    """Statistics about context usage after an LLM call.

    All fields reflect the last prepare (or the call when using result.context_stats).
    """

    total_tokens: int = 0
    """Total tokens used in the last run (messages + system + tools)."""
    max_tokens: int = 0
    """Context window size (max_tokens) used for that run."""
    utilization: float = 0.0
    """Used tokens / available (0.0-1.0). Capped at 1.0 when over budget."""
    compacted: bool = False
    """True if compaction ran during this prepare."""
    compact_count: int = 0
    """Number of compactions in this run (this prepare) only."""
    compact_method: str | None = None
    """Method used (e.g. 'middle_out_truncate', 'summarize') or None if no compaction."""
    thresholds_triggered: list[str] = field(default_factory=list)
    """List of threshold metric names that fired (e.g. ['tokens'])."""
    breakdown: ContextBreakdown | None = None
    """Token counts by component (system, tools, memory, messages). Set after prepare(); None before any prepare."""


@dataclass
class ContextWindowCapacity:
    """Internal context window capacity used during prepare (max tokens, reserve, utilization).

    Not for end users: use Context and TokenLimits (token caps) instead.
    Compaction is not automatic; use ctx.compact() in a ContextThreshold action
    or agent.context.compact() during prepare (e.g. from a threshold action).
    """

    max_tokens: int
    """Maximum context window size (tokens)."""
    reserve: int = 2000
    """Tokens reserved for model output; subtracted from max_tokens to get available."""
    _used_tokens: int = 0

    @property
    def available(self) -> int:
        """Tokens available for context (excluding response reserve)."""
        return max(0, self.max_tokens - self.reserve)

    @property
    def used_tokens(self) -> int:
        """Tokens used in this prepare (set by manager)."""
        return self._used_tokens

    @used_tokens.setter
    def used_tokens(self, value: int) -> None:
        self._used_tokens = value

    @property
    def utilization(self) -> float:
        """Current utilization as a fraction (0-1). Capped at 1.0 when at or over capacity."""
        if self.available <= 0:
            return 1.0 if self._used_tokens > 0 else 0.0
        return min(1.0, self._used_tokens / self.available)

    @property
    def percent(self) -> int:
        """Utilization as percentage (0-100)."""
        return int(self.utilization * 100)

    def reset(self) -> None:
        """Reset used tokens for new call."""
        self._used_tokens = 0


@dataclass
class Context:
    """Context window configuration: limits, compaction triggers, and token caps.

    Provides context window management. Compaction is on-demand: call
    ctx.compact() from a ContextThreshold action (e.g. at 75% to compact).

    **Cost vs tokens:** ``Budget`` = cost limits (USD). ``token_limits`` (TokenLimits) =
    context's token caps (run and/or per period). Same field names (run, per, on_exceeded) for consistency.

    Example:
        >>> from syrin import Agent, Model, Context
        >>> from syrin.threshold import ContextThreshold, compact_if_available
        >>>
        >>> agent = Agent(
        ...     model=Model("openai/gpt-4o"),
        ...     context=Context(
        ...         max_tokens=80000,
        ...         reserve=2000,
        ...         thresholds=[ContextThreshold(at=75, action=compact_if_available)],
        ...     )
        ... )
    """

    max_tokens: int | None = None
    """Max tokens in context window. None = use model's context_window or 128k."""
    reserve: int = 2000
    """Tokens reserved for model output; subtracted from max_tokens to get available. ≥ 0."""
    thresholds: list[ContextThreshold] = field(default_factory=list)
    """When utilization hits these percentages, actions run (e.g. compact at 75%)."""
    token_limits: TokenLimits | None = None
    """Token caps for this context (run and/or per period). Same names as Budget: run, per, on_exceeded."""
    encoding: str = "cl100k_base"
    """TokenCounter encoding. Default context manager uses it for counting."""
    compactor: ContextCompactorProtocol | None = None
    """Custom compactor (compact(messages, budget) -> CompactionResult). Default: ContextCompactor."""
    compaction_prompt: str | None = None
    """User prompt template for summarization (e.g. with {messages}). None = default from prompts.py. Passed to default ContextCompactor."""
    compaction_system_prompt: str | None = None
    """System prompt for summarization. None = default. Passed to default ContextCompactor."""
    compaction_model: "Model | None" = None
    """Model for summarization. None = placeholder (no LLM). Passed to default ContextCompactor."""
    auto_compact_at: float | None = None
    """Proactive compaction: when utilization (0.0–1.0) >= this value, compact once before evaluating thresholds. None = no proactive compaction."""
    runtime_inject: Callable[["PrepareInput"], list[dict[str, Any]]] | None = None
    """Optional callable to inject context at prepare time (e.g. RAG). Receives PrepareInput; returns list of message dicts. Not called when prepare(inject=...) is provided."""
    inject_placement: InjectPlacement = InjectPlacement.BEFORE_CURRENT_TURN
    """Where to place injected messages: prepend_to_system, before_current_turn (default), after_current_turn."""
    inject_source_detail: str = "injected"
    """Provenance source_detail for injected messages (e.g. 'rag', 'dynamic_rules')."""

    def __post_init__(self) -> None:
        if self.reserve < 0:
            raise ValueError(f"Context reserve must be >= 0, got {self.reserve}")
        if self.max_tokens is not None and self.max_tokens <= 0:
            raise ValueError(f"Context max_tokens must be > 0 when set, got {self.max_tokens}")
        if self.auto_compact_at is not None and not (0.0 <= self.auto_compact_at <= 1.0):
            raise ValueError(
                "auto_compact_at must be between 0 and 1 (fraction of context window), "
                f"got {self.auto_compact_at}"
            )
        self._validate_thresholds()

    def _validate_thresholds(self) -> None:
        """Validate that only ContextThreshold is used."""
        for th in self.thresholds:
            if not isinstance(th, ContextThreshold):
                raise ValueError(
                    f"Context thresholds only accept ContextThreshold, got {type(th).__name__}"
                )

    def get_capacity(self, model: "Model | None" = None) -> ContextWindowCapacity:
        """Get a ContextWindowCapacity for this configuration.

        Args:
            model: Optional model to auto-detect context window from.

        Returns:
            ContextWindowCapacity configured for this context.
        """
        max_tokens = self.max_tokens

        if max_tokens is None and model is not None:
            from syrin.model import Model as ModelClass
            from syrin.model.core import ModelSettings

            if isinstance(model, ModelClass):
                settings = model.settings
                if isinstance(settings, ModelSettings) and settings.context_window:
                    max_tokens = settings.context_window

        if max_tokens is None:
            max_tokens = 128000

        if max_tokens <= 0:
            raise ValueError(f"Resolved max_tokens must be > 0, got {max_tokens}")

        reserve_val = self.reserve
        if model is not None:
            from syrin.model.core import ModelSettings

            model_settings = getattr(model, "settings", None)
            if isinstance(model_settings, ModelSettings):
                default_reserve = getattr(model_settings, "default_reserve_tokens", None)
                if default_reserve is not None:
                    reserve_val = default_reserve

        return ContextWindowCapacity(max_tokens=max_tokens, reserve=reserve_val)

    def apply(
        self,
        messages: list[Any],
        model: "Model | None" = None,
        max_tokens: int | None = None,
    ) -> list[dict[str, Any]]:
        """Apply compaction to messages so they fit within the context budget.

        Uses the context compactor (or default). Use before sending to the LLM
        when you need to trim context manually.

        Args:
            messages: List of Message or dict with "role" and "content".
            model: Optional model to resolve max_tokens from.
            max_tokens: Override available tokens; if None, uses context limit.

        Returns:
            List of message dicts (role, content) after compaction.

        Example:
            >>> compacted = context.apply(messages, max_tokens=4000)
        """
        capacity = self.get_capacity(model)
        available = max_tokens if max_tokens is not None else capacity.available
        if available <= 0:
            return []
        msgs: list[dict[str, Any]] = []
        for m in messages:
            if hasattr(m, "model_dump"):
                d = m.model_dump()
                msgs.append({"role": d.get("role"), "content": d.get("content", "")})
            elif isinstance(m, dict):
                msgs.append({"role": m.get("role"), "content": m.get("content", "")})
            else:
                msgs.append(
                    {"role": getattr(m, "role", "user"), "content": str(getattr(m, "content", ""))}
                )
        compactor = self.compactor
        if compactor is None:
            compactor = ContextCompactor(
                compaction_prompt=self.compaction_prompt,
                compaction_system_prompt=self.compaction_system_prompt,
                compaction_model=self.compaction_model,
            )
        result = compactor.compact(msgs, available)
        return result.messages

    def get_remote_config_schema(self, section_key: str) -> tuple[Any, dict[str, object]]:
        """RemoteConfigurable: return (schema, current_values) for the context section."""
        from syrin.remote._schema import build_section_schema_from_obj
        from syrin.remote._types import ConfigSchema

        if section_key != "context":
            return (ConfigSchema(section="context", class_name="Context", fields=[]), {})
        return build_section_schema_from_obj(self, "context", "Context")

    def apply_remote_overrides(
        self,
        agent: Any,
        pairs: list[tuple[str, object]],
        section_schema: Any,
    ) -> None:
        """RemoteConfigurable: apply context overrides to agent._context.context."""
        from syrin.context import DefaultContextManager
        from syrin.remote._resolver_helpers import build_nested_update

        update = build_nested_update(section_schema, pairs, "context")
        if not update:
            return
        ctx_manager = getattr(agent, "_context", None)
        if not isinstance(ctx_manager, DefaultContextManager):
            return
        current = getattr(ctx_manager, "context", None)
        if current is None:
            return
        new_ctx = dataclasses.replace(current, **update)
        object.__setattr__(ctx_manager, "context", new_ctx)


__all__ = [
    "Context",
    "ContextStats",
    "ContextWindowCapacity",
]
