"""Response object returned by agent.run() with content, cost, and metadata."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Generic, TypeVar

if TYPE_CHECKING:
    from syrin.context import Context, ContextStats

from syrin.enums import StopReason
from syrin.output_format._citation import Citation
from syrin.types import TokenUsage
from syrin.types.validation import ValidationAttempt

T = TypeVar("T")

__all__ = [
    "MediaAttachment",
    "StructuredOutput",
    "TraceStep",
    "Response",
    "BudgetStatus",
    "GuardrailReport",
    "GroundingReport",
    "ContextReport",
    "MemoryReport",
    "TokenReport",
    "OutputReport",
    "RateLimitReport",
    "CheckpointReport",
    "AgentReport",
]


@dataclass
class StructuredOutput:
    """Wrapper for structured output responses.

    Provides easy access to parsed content and raw data with full validation tracking.

    Usage:
        result = agent.run("What is 2+2?", output=MathResult)
        result.structured.result  # Access parsed field
        result.structured.raw    # Raw JSON string

    With validation tracking:
        result.structured.validation_attempts  # All validation attempts
        result.structured.final_error          # Error if validation failed
        result.structured.is_valid             # Whether validation succeeded
        result.structured.all_errors           # List of all errors

    Example:
        # Check if validation succeeded
        if result.structured.is_valid:
            print(result.structured.parsed)
        else:
            print(f"Failed: {result.structured.final_error}")
            # Debug: see all attempts
            for attempt in result.structured.validation_attempts:
                print(f"Attempt {attempt.attempt}: {attempt.error}")
    """

    raw: str = ""
    parsed: object = None
    _data: dict[str, object] = field(default_factory=dict)
    validation_attempts: list[ValidationAttempt] = field(default_factory=list)
    final_error: Exception | None = None
    tool_name: str | None = None

    def __getattr__(self, name: str) -> object:
        """Allow accessing parsed fields directly."""
        if name.startswith("_") or name in (
            "raw",
            "parsed",
            "_data",
            "validation_attempts",
            "final_error",
            "tool_name",
        ):
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
        # First try _data
        if name in self._data:
            return self._data[name]
        # Then try parsed object
        if self.parsed is not None and hasattr(self.parsed, name):
            return getattr(self.parsed, name)
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    @property
    def is_valid(self) -> bool:
        """Whether validation succeeded."""
        return self.final_error is None and self.parsed is not None

    @property
    def last_error(self) -> str | None:
        """Last error message if any."""
        if self.validation_attempts:
            return self.validation_attempts[-1].error
        return str(self.final_error) if self.final_error else None

    @property
    def all_errors(self) -> list[str]:
        """All error messages from all attempts."""
        return [a.error for a in self.validation_attempts if a.error]

    @property
    def parse_error(self) -> Exception | None:
        """JSON parsing error if any."""
        if self.final_error and "JSON" in str(self.final_error):
            return self.final_error
        return None

    def __str__(self) -> str:
        if self.parsed is not None:
            return str(self.parsed)
        return self.raw

    def __repr__(self) -> str:
        return (
            f"StructuredOutput(raw={self.raw!r}, parsed={self.parsed!r}, is_valid={self.is_valid})"
        )


@dataclass
class TraceStep:
    """A single step in an execution trace (LLM call, tool call, etc.).

    Attributes:
        step_type: Type (llm_call, tool_call, etc.).
        timestamp: Unix timestamp.
        model, tokens, cost_usd, latency_ms: Step metrics.
        extra: Additional key-value data.
    """

    step_type: str
    timestamp: float
    model: str = ""
    tokens: int = 0
    cost_usd: float = 0.0
    latency_ms: float = 0.0
    extra: dict[str, object] = field(default_factory=dict)


@dataclass
class MediaAttachment:
    """Media attachment (image, video, audio) in agent responses.

    Use content_bytes for inline binary data, or url for references.
    content_type is the MIME type (e.g. image/png, video/mp4).

    Attributes:
        type: Media type — image, video, or audio.
        content_bytes: Inline binary content. None if url is used.
        url: URL reference. None if content_bytes is used.
        content_type: MIME type (e.g. image/png, video/mp4, audio/wav).
    """

    type: str  # image, video, audio
    content_type: str
    content_bytes: bytes | None = None
    url: str | None = None


@dataclass
class StreamChunk:  # type: ignore[explicit-any]
    """One chunk from agent.stream() or agent.astream().

    Why: Build real-time UIs. text is the delta; accumulated_text is full so far.
    cost_so_far and tokens_so_far update as the stream progresses.

    Fields:
        index: Zero-based chunk index in this stream.
        text: New text in this chunk (delta).
        accumulated_text: Full text received so far.
        cost_so_far: Total cost up to this chunk (USD).
        tokens_so_far: TokenUsage (input/output/total) so far.
    """

    index: int = 0
    text: str = ""
    accumulated_text: str = ""
    cost_so_far: float = 0.0
    tokens_so_far: TokenUsage = field(default_factory=TokenUsage)
    is_final: bool = False
    response: Response[Any] | None = None  # type: ignore[explicit-any]


@dataclass
class BudgetStatus:
    """Budget status. Access via response.budget or response.report.budget.

    Attributes:
        remaining: Remaining budget (USD). None if unlimited.
        used: Spent this run (USD).
        total: Total limit (USD). None if unlimited.
        cost: Same as used.
    """

    remaining: float | None
    used: float
    total: float | None
    cost: float

    def __str__(self) -> str:
        def _fmt(val: float | None) -> str:
            if val is None:
                return "N/A"
            if val < 0.0001:
                return f"{val:.6f}"
            return f"{val:.4f}"

        if self.total is not None:
            return f"BudgetStatus(remaining=${_fmt(self.remaining)}, used=${_fmt(self.used)}, total=${_fmt(self.total)})"
        return f"BudgetStatus(used=${_fmt(self.used)}, unlimited)"


@dataclass
class GuardrailReport:
    """Report of guardrail evaluations for a single run.

    Attributes:
        input_passed: Whether input guardrails passed.
        input_reason: Reason if input guardrail failed.
        input_guardrails: List of input guardrails run.
        output_passed: Whether output guardrails passed.
        output_reason: Reason if output guardrail failed.
        output_guardrails: List of output guardrails run.
        blocked: Whether request was blocked.
        blocked_stage: Stage at which blocked (input/output).
    """

    input_passed: bool = True
    input_reason: str | None = None
    input_guardrails: list[str] = field(default_factory=list)
    output_passed: bool = True
    output_reason: str | None = None
    output_guardrails: list[str] = field(default_factory=list)
    blocked: bool = False
    blocked_stage: str | None = None

    @property
    def passed(self) -> bool:
        return self.input_passed and self.output_passed and not self.blocked


@dataclass
class ContextReport:
    """Report of context usage for a single run.

    Attributes:
        initial_tokens: Tokens before compaction.
        final_tokens: Tokens after compaction.
        max_tokens: Context window max.
        compressions: Number of compactions.
        offloads: Number of offloads (if used).
    """

    initial_tokens: int = 0
    final_tokens: int = 0
    max_tokens: int = 0
    compressions: int = 0
    offloads: int = 0


@dataclass
class MemoryReport:
    """Report of memory operations for a single run.

    Attributes:
        recalls: Number of recall operations.
        stores: Number of store operations.
        forgets: Number of forget operations.
        consolidated: Number of consolidations.
    """

    recalls: int = 0
    stores: int = 0
    forgets: int = 0
    consolidated: int = 0


@dataclass
class TokenReport:
    """Report of token usage for a single run.

    Attributes:
        input_tokens: Input token count.
        output_tokens: Output token count.
        total_tokens: Total tokens.
        cost_usd: Total cost (USD).
    """

    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float = 0.0


@dataclass
class OutputReport:
    """Report of output validation for a single run.

    Attributes:
        validated: Whether output was validated.
        attempts: Number of validation attempts.
        is_valid: Final validation result.
        final_error: Error message if validation failed.
    """

    validated: bool = False
    attempts: int = 0
    is_valid: bool = True
    final_error: str | None = None


@dataclass
class RateLimitReport:
    """Report of rate limit checks for a single run.

    Attributes:
        checks: Number of rate limit checks.
        throttles: Number of throttle events.
        exceeded: Whether limit was exceeded.
    """

    checks: int = 0
    throttles: int = 0
    exceeded: bool = False


@dataclass
class CheckpointReport:
    """Report of checkpoint operations for a single run.

    Attributes:
        saves: Number of checkpoint saves.
        loads: Number of checkpoint loads.
    """

    saves: int = 0
    loads: int = 0


@dataclass
class GroundingReport:
    """Report of grounding (verified facts) for a single run.

    Populated when the agent uses knowledge search with grounding enabled.
    None when no grounding was used.

    Attributes:
        verified_count: Number of facts verified against sources.
        total_facts: Total facts extracted (before verification filter).
        sources: Unique source document identifiers used.
    """

    verified_count: int = 0
    total_facts: int = 0
    sources: list[str] = field(default_factory=list)


@dataclass
class AgentReport:
    """Aggregated report of all agent operations for a single run.

    Access via response.report or agent.report:

        result = agent.run("Hello")
        result.report.guardrail       # GuardrailReport
        result.report.context         # ContextReport
        result.report.memory         # MemoryReport
        result.report.budget         # BudgetReport (same as response.budget)
        result.report.tokens         # TokenReport
        result.report.output         # OutputReport
        result.report.ratelimits     # RateLimitReport
        result.report.checkpoints    # CheckpointReport
        result.report.grounding      # GroundingReport | None
    """

    guardrail: GuardrailReport = field(default_factory=GuardrailReport)
    context: ContextReport = field(default_factory=ContextReport)
    memory: MemoryReport = field(default_factory=MemoryReport)
    budget_remaining: float | None = None
    budget_used: float | None = None
    tokens: TokenReport = field(default_factory=TokenReport)
    output: OutputReport = field(default_factory=OutputReport)
    ratelimits: RateLimitReport = field(default_factory=RateLimitReport)
    checkpoints: CheckpointReport = field(default_factory=CheckpointReport)
    grounding: GroundingReport | None = None

    @property
    def budget(self) -> BudgetStatus:
        return BudgetStatus(
            remaining=self.budget_remaining,
            used=self.budget_used or 0.0,
            total=self.budget_remaining + self.budget_used
            if self.budget_remaining is not None and self.budget_used is not None
            else None,
            cost=self.budget_used or 0.0,
        )


@dataclass
class Response(Generic[T]):
    """Result returned by agent.run(), agent.arun().

    Single object for content, cost, tokens, model, and full report. Use
    str(response) for quick printing (returns content).

    Attributes:
        content: Main reply text (or parsed type if output= set).
        raw: Raw string from the model before parsing.
        cost: USD cost of this run.
        tokens: TokenUsage (input_tokens, output_tokens, total_tokens).
        model: Model ID used (e.g. "openai/gpt-4o-mini").
        duration: Run duration in seconds.
        budget_remaining: Remaining budget (USD) if budget configured. None if unlimited.
        budget_used: Spent this run (USD).
        trace: Execution trace steps (LLM calls, tool calls, etc.).
        tool_calls: Tool calls requested by the model (if any).
        stop_reason: Why the run ended (END_TURN, BUDGET, MAX_ITERATIONS, etc.).
        structured: StructuredOutput if output= configured; else None.
        iterations: Number of LLM/tool iterations.
        report: Full AgentReport (guardrails, memory, budget, etc.).
        context_stats: Context usage stats for this call.
        context: Context used (when overridden per-call).
        raw_response: Provider-specific raw response (parsed Pydantic for structured output).
        routing_reason: RoutingReason when model routing was used. None if single model.
        model_used: Actual model ID from provider (e.g. OpenRouter header). Same as model if not set.
        task_type: TaskType used for routing when applicable.
        actual_cost: Actual cost from provider (e.g. OpenRouter). Same as cost if not set.
        cost_estimated: Pre-call estimated cost (before the LLM call). None if not estimated.
        cache_hit: Whether the response was served from provider cache (e.g. prompt cache).
        cache_savings: USD saved due to cache (input_tokens * price_per_token). 0.0 if no cache.
        template_data: Slot values used to render template (when output_config.template is set).
        file: Path to generated file when output_config.format produces a file (PDF, DOCX, etc.).
        file_bytes: Raw bytes of generated file (for streaming/API). None when no file generated.
        citations: Parsed citations when output_config.citation is set. Empty list when none.

    For structured output (output=Output(MyModel)):
        result.output — Typed model instance (canonical)
        result.structured.parsed — Pydantic instance
        result.structured.is_valid — Validation succeeded

    Example:
        >>> r = agent.run("What is 2+2?")
        >>> print(r.content)
        4
        >>> print(r.cost, r.tokens.total_tokens)
    """

    content: T
    raw: str = ""
    attachments: list[MediaAttachment] = field(default_factory=list)
    cost: float = 0.0
    tokens: TokenUsage = field(default_factory=TokenUsage)
    model: str = ""
    duration: float = 0.0
    budget_remaining: float | None = None
    budget_used: float | None = None
    trace: list[TraceStep] = field(default_factory=list)
    tool_calls: list[object] = field(default_factory=list)
    stop_reason: StopReason = StopReason.END_TURN
    structured: StructuredOutput | None = None
    iterations: int = 1
    report: AgentReport = field(default_factory=AgentReport)
    context_stats: ContextStats | None = None  # per-call context stats
    context: Context | None = None  # Context used for this call (overrides agent's when passed)
    raw_response: object = None  # Provider raw response; parsed Pydantic for structured output
    routing_reason: object = None  # RoutingReason when routing was used
    model_used: str | None = None  # Actual model from provider (e.g. OpenRouter header)
    task_type: object = None  # TaskType used for routing
    actual_cost: float | None = None  # Actual cost from provider (e.g. OpenRouter header)
    cost_estimated: float | None = None  # Pre-call estimated cost (before the LLM call)
    cache_hit: bool = False  # Whether the response was served from provider cache
    cache_savings: float = 0.0  # USD saved due to cache (input_tokens * price_per_token)
    template_data: dict[str, object] | None = None  # Slot values when output_config.template used
    file: Path | None = None  # Path to generated file when output_config produces file
    file_bytes: bytes | None = None  # Raw bytes of generated file
    citations: list[Citation] = field(default_factory=list)  # When output_config.citation set

    @property
    def output(self) -> object:
        """The main output of this response.

        For **structured output** (``Output(MyModel)`` configured): returns the typed
        ``MyModel`` instance once validation succeeds, or ``None`` if validation failed.
        Check ``response.is_valid`` before accessing typed fields.

        For **plain text** agents (no output type configured): returns ``response.content``
        — the text string generated by the LLM.

        This is the canonical, model-agnostic way to retrieve "what the agent produced".
        You can always call ``result.output`` regardless of whether the agent uses
        structured output or plain text::

            # Plain agent — output is the text string
            result = agent.run("Summarize this")
            print(result.output)  # same as result.content

            # Structured agent — output is the typed model instance
            result = agent.run("Parse this", output=MyModel)
            if result.is_valid:
                print(result.output.field)

        For per-attempt parsing details, access ``response.structured`` directly.
        """
        if self.structured is not None:
            return self.structured.parsed
        return self.content

    @property
    def total_tokens(self) -> int:
        """Total tokens used (input + output). Convenience for ``response.tokens.total_tokens``."""
        return self.tokens.total_tokens

    @property
    def is_valid(self) -> bool:
        """Whether the response is valid.

        For structured output (``Output(MyModel)`` configured): ``True`` when validation
        succeeded. For plain text responses: always ``True`` — nothing to validate.

        Use this as the canonical check before accessing ``response.output``::

            if response.is_valid:
                print(response.output.my_field)
        """
        if self.structured is not None:
            return self.structured.is_valid
        return True

    @property
    def validation_attempts(self) -> int:
        """Number of structured output validation attempts made during this run.

        Returns 0 for plain text responses (no structured output configured).
        Returns > 1 when the model failed to produce valid output on the first try
        and validation retries were configured (``Output(MyModel, validation_retries=N)``).

        Use this to detect and debug unexpected retry costs::

            if response.validation_attempts > 1:
                print(f"Needed {response.validation_attempts} attempts to get valid output")
        """
        if self.structured is not None:
            return len(self.structured.validation_attempts)
        return 0

    @property
    def blocked(self) -> bool:
        """Whether this response was blocked by a guardrail.

        ``True`` when an input or output guardrail rejected the request.
        Check ``response.block_reason`` for the human-readable rejection message
        and ``response.report.guardrail`` for full per-stage details.

        Example::

            result = agent.run("...")
            if result.blocked:
                print(f"Request blocked: {result.block_reason}")
        """
        return self.report.guardrail.blocked

    @property
    def block_reason(self) -> str | None:
        """Human-readable reason why this response was blocked, or ``None`` if not blocked.

        Returns the reason from whichever guardrail stage fired first
        (input before output). ``None`` when ``response.blocked`` is ``False``.
        """
        g = self.report.guardrail
        if not g.blocked:
            return None
        return g.input_reason or g.output_reason

    def __str__(self) -> str:
        return str(self.content)

    def __bool__(self) -> bool:
        """True if the response completed successfully."""
        return self.stop_reason == StopReason.END_TURN

    @property
    def budget(self) -> BudgetStatus:
        """Get the budget object associated with this response.

        This is a property that returns the full budget object with current status.
        The budget object includes:
        - budget.remaining: remaining budget amount
        - budget.max_cost: total budget cap
        - budget.cost: alias for response.cost (convenience)
        """
        # Return a BudgetStatus object with convenience properties
        return BudgetStatus(
            remaining=self.budget_remaining,
            used=self.budget_used or self.cost,
            total=self.budget_remaining + (self.budget_used or self.cost)
            if self.budget_remaining is not None
            else None,
            cost=self.cost,
        )
