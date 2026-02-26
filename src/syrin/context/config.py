"""Context configuration and stats."""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from syrin.threshold import ContextThreshold

if TYPE_CHECKING:
    from syrin.model import Model

from syrin.budget import TokenLimits
from syrin.context.compactors import ContextCompactor, ContextCompactorProtocol


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


@dataclass
class ContextWindowBudget:
    """Internal window capacity used during context prepare (max tokens, reserve, utilization).

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
        """Current utilization as a fraction (0-1). Capped at 1.0 when over budget."""
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

    **Budget vs token caps:** ``Budget`` = cost limits (USD). ``budget`` (TokenLimits) =
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
    budget: TokenLimits | None = None
    """Context's token caps (run and/or per period). Same names as Budget: run, per, on_exceeded."""
    encoding: str = "cl100k_base"
    """TokenCounter encoding. Default context manager uses it for counting."""
    compactor: ContextCompactorProtocol | None = None
    """Custom compactor (compact(messages, budget) -> CompactionResult). Default: ContextCompactor."""

    def __post_init__(self) -> None:
        if self.reserve < 0:
            raise ValueError(f"Context reserve must be >= 0, got {self.reserve}")
        if self.max_tokens is not None and self.max_tokens <= 0:
            raise ValueError(f"Context max_tokens must be > 0 when set, got {self.max_tokens}")
        self._validate_thresholds()

    def _validate_thresholds(self) -> None:
        """Validate that only ContextThreshold is used."""
        for th in self.thresholds:
            if not isinstance(th, ContextThreshold):
                raise ValueError(
                    f"Context thresholds only accept ContextThreshold, got {type(th).__name__}"
                )

    def get_budget(self, model: "Model | None" = None) -> ContextWindowBudget:
        """Get a ContextWindowBudget for this configuration.

        Args:
            model: Optional model to auto-detect context window from.

        Returns:
            ContextWindowBudget configured for this context.
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

        return ContextWindowBudget(max_tokens=max_tokens, reserve=reserve_val)

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
        budget = self.get_budget(model)
        available = max_tokens if max_tokens is not None else budget.available
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
        compactor = self.compactor if self.compactor is not None else ContextCompactor()
        result = compactor.compact(msgs, available)
        return result.messages


__all__ = [
    "Context",
    "ContextStats",
    "ContextWindowBudget",
]
