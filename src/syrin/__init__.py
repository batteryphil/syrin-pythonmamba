"""Syrin — Python library for building AI agents with budget management and DSL codegen.

Quick start::

    from syrin import Agent
    from syrin.model import Model

    agent = Agent(
        model=Model.OpenAI("gpt-4o-mini", api_key="..."),
        system_prompt="You are a helpful assistant.",
        budget=Budget(run=0.50),
    )
    r = agent.response("Hello!")
    print(r.content, r.cost)

Key exports: Agent, Model, Budget, Memory, Context, CheckpointConfig, Guardrails.
See docs/ and examples/ for full guides.
"""

import atexit
import sys
from typing import Any, cast

_trace_enabled = False


def _trace_summary_on_exit() -> None:
    """Print trace summary at process exit when --trace was used (built-in)."""
    if not _trace_enabled:
        return
    try:
        from syrin.observability.metrics import get_metrics

        metrics = get_metrics()
        summary = cast(dict[str, object], metrics.get_summary())
        agent_raw = summary.get("agent")
        agent_data: dict[str, object] = (
            cast(dict[str, object], agent_raw) if isinstance(agent_raw, dict) else {}
        )
        llm_raw = summary.get("llm")
        llm_data: dict[str, object] = (
            cast(dict[str, object], llm_raw) if isinstance(llm_raw, dict) else {}
        )
        agent_cost = agent_data.get("cost")
        llm_cost = llm_data.get("cost")
        total_cost = float(cast(float | int, agent_cost or llm_cost or 0))
        runs = agent_data.get("runs")
        errors = agent_data.get("errors")
        tokens = llm_data.get("tokens_total")
        cost_str = f"${total_cost:.6f}".rstrip("0").rstrip(".")
        if cost_str == "$":
            cost_str = "$0"
        runs_int = int(cast(float | int, runs)) if runs is not None else 0
        errors_int = int(cast(float | int, errors)) if errors is not None else 0
        tokens_int = int(cast(float | int, tokens)) if tokens is not None else 0
        print("\n" + "=" * 60)
        print(" TRACE SUMMARY (--trace)")
        print("=" * 60)
        print(f"  Agent runs:    {runs_int}")
        print(f"  Errors:        {errors_int}")
        print(f"  Total tokens:  {tokens_int}")
        print(f"  Total cost:    {cost_str}")
        print("=" * 60 + "\n")
    except Exception:
        pass


def _auto_trace_check() -> None:
    """Check for --trace flag and auto-enable observability."""
    global _trace_enabled
    if _trace_enabled or "--trace" not in sys.argv:
        return

    _trace_enabled = True
    sys.argv.remove("--trace")

    try:
        from syrin.observability import ConsoleExporter, get_tracer

        tracer = get_tracer()
        tracer.add_exporter(ConsoleExporter(colors=True, verbose=True))
        tracer.set_debug_mode(True)
        atexit.register(_trace_summary_on_exit)

        print("\n" + "=" * 60)
        print(" Syrin Tracing Enabled (--trace flag detected)")
        print("=" * 60 + "\n")
    except ImportError:
        pass


_auto_trace_check()

del _auto_trace_check

from syrin.agent import Agent
from syrin.agent.multi_agent import (
    AgentTeam,
    DynamicPipeline,
    Pipeline,
    PipelineRun,
    parallel,
    sequential,
)
from syrin.audit import (
    AuditBackendProtocol,
    AuditEntry,
    AuditFilters,
    AuditLog,
    JsonlAuditBackend,
)
from syrin.budget import (
    Budget,
    BudgetExceededContext,
    BudgetLimitType,
    BudgetState,
    BudgetThreshold,
    RateLimit,
    TokenLimits,
    TokenRateLimit,
    raise_on_exceeded,
    stop_on_exceeded,
    warn_on_exceeded,
)
from syrin.budget_store import BudgetStore, FileBudgetStore, InMemoryBudgetStore
from syrin.checkpoint import (
    CheckpointConfig,
    Checkpointer,
    CheckpointState,
    CheckpointTrigger,
)
from syrin.circuit import CircuitBreaker

# =============================================================================
# CLI & Observability (New)
# =============================================================================
from syrin.cli import (
    WorkflowDebugger,
    auto_trace,
    check_for_trace_flag,
    remove_trace_flag,
)
from syrin.config import configure, get_config
from syrin.context import (
    Context,
    ContextManager,
    ContextStats,
    DefaultContextManager,
    TokenCounter,
)
from syrin.domain_events import (
    BudgetThresholdReached,
    ContextCompacted,
    DomainEvent,
    EventBus,
)
from syrin.enums import (
    AlmockPricing,
    AuditBackend,
    AuditEventType,
    CheckpointBackend,
    CheckpointStrategy,
    CircuitState,
    ContentType,
    ContextMode,
    ContextStrategy,
    DecayStrategy,
    DocFormat,
    FormationMode,
    GuardrailStage,
    Hook,
    InjectionStrategy,
    LoopStrategy,
    MemoryBackend,
    MemoryScope,
    MemoryType,
    MessageRole,
    OffloadBackend,
    OutputChunkStrategy,
    ProgressStatus,
    RateWindow,
    RetryBackoff,
    SandboxRuntime,
    ServeProtocol,
    StepType,
    StopReason,
    SwitchReason,
    ThresholdMetric,
    ThresholdWindow,
    TraceLevel,
    TracingBackend,
)
from syrin.events import EventContext, Events
from syrin.exceptions import (
    CircuitBreakerOpenError,
    HandoffBlockedError,
    HandoffRetryRequested,
    ValidationError,
)
from syrin.guardrails import (
    ContentFilter,
    Guardrail,
    GuardrailChain,
    GuardrailResult,
    LengthGuardrail,
)
from syrin.hitl import ApprovalGate, ApprovalGateProtocol
from syrin.loop import (
    CODE_ACTION,
    HITL,
    PLAN_EXECUTE,
    REACT,
    SINGLE_SHOT,
    CodeActionLoop,
    HumanInTheLoop,
    Loop,
    LoopResult,
    PlanExecuteLoop,
    ReactLoop,
    SingleShotLoop,
    ToolApprovalFn,
)
from syrin.mcp import MCP, MCPClient
from syrin.memory import (
    Decay,
    Memory,
    MemoryBudget,
    MemoryEntry,
)

# =============================================================================
# Model - Core
# =============================================================================
# =============================================================================
# Model - Structured Output
# =============================================================================
# =============================================================================
# Model - Provider Namespaces
# =============================================================================
from syrin.model import (
    Anthropic,
    Google,
    LiteLLM,
    Middleware,
    # Core
    Model,
    ModelRegistry,
    ModelSettings,
    ModelVariable,
    ModelVersion,
    Ollama,
    # Provider namespaces
    OpenAI,
    OutputType,
    # Structured output
    StructuredOutput,
    output,
    structured,
)

# =============================================================================
# Observability (New)
# =============================================================================
from syrin.observability import (
    ConsoleExporter,
    InMemoryExporter,
    JSONLExporter,
    SemanticAttributes,
    Session,
    Span,
    SpanContext,
    SpanExporter,
    SpanKind,
    SpanStatus,
    current_session,
    current_span,
    session,
    set_debug,
    span,
    trace,
)
from syrin.observability import (
    get_tracer as get_observability_tracer,
)
from syrin.output import Output
from syrin.pipe import Pipe, pipe
from syrin.prompt import (
    Prompt,
    PromptContext,
    make_prompt_context,
    prompt,
    system_prompt,
    validated,
)
from syrin.remote import init
from syrin.response import (
    AgentReport,
    BudgetStatus,
    CheckpointReport,
    ContextReport,
    GuardrailReport,
    MemoryReport,
    OutputReport,
    RateLimitReport,
    Response,
    TokenReport,
)
from syrin.run_context import RunContext
from syrin.serve import (
    AgentCard,
    AgentCardAuth,
    AgentCardProvider,
    AgentRouter,
    ServeConfig,
    build_agent_card_json,
    build_router,
)
from syrin.task import task
from syrin.threshold import (
    ContextThreshold,
    RateLimitThreshold,
    ThresholdContext,
    compact_if_available,
)
from syrin.tool import ToolSpec, tool
from syrin.validation import ValidationPipeline, validate_output

__version__ = "0.5.0"


def run(
    input: str,
    model: str | Model | None = None,
    *,
    system_prompt: str | None = None,
    tools: list[ToolSpec] | None = None,
    budget: Budget | None = None,
    prompt_vars: dict[str, Any] | None = None,
    **kwargs: Any,  # pyright: ignore[reportAny]
) -> Response[str]:
    """Run a one-shot completion with an agent.

    This is a convenience function for simple one-off LLM calls without
    needing to create an Agent instance.

    Args:
        input: The user input/message
        model: Model identifier (e.g., "openai/gpt-4o", "anthropic/claude-sonnet")
               or Model instance. Uses config default if not specified.
        system_prompt: Optional system prompt
        tools: Optional list of tools
        budget: Optional budget
        **kwargs: Additional arguments passed to Agent

    Returns:
        Response object with content, cost, tokens, etc.

    Example:
        >>> import syrin
        >>> result = syrin.run("What is 2+2?", model="openai/gpt-4o")
        >>> print(result.content)
        4

        >>> result = syrin.run("Summarize this", model=syrin.Model.Anthropic("claude-sonnet"))
    """
    from syrin.model.core import Model as ModelClass
    from syrin.model.core import detect_provider

    # Resolve model to Model instance
    if model is None:
        config = get_config()
        default = config.default_model
        if default is not None:
            model_obj = ModelClass(provider=default.provider, model_id=default.model_id)
        else:
            model_obj = ModelClass(provider="litellm", model_id="gpt-4o")
    elif isinstance(model, str):
        import os

        provider = detect_provider(model)
        api_key = None
        if provider == "openai":
            api_key = os.getenv("OPENAI_API_KEY") or get_config().default_api_key
        elif provider == "anthropic":
            api_key = os.getenv("ANTHROPIC_API_KEY") or get_config().default_api_key
        elif provider == "google":
            api_key = os.getenv("GOOGLE_API_KEY") or get_config().default_api_key
        else:
            api_key = get_config().default_api_key or os.getenv("OPENAI_API_KEY")
        model_obj = ModelClass(provider=provider, model_id=model, api_key=api_key)
    else:
        model_obj = model

    agent = Agent(
        model=model_obj,
        system_prompt=system_prompt or "",
        tools=tools or [],
        budget=budget,
        **kwargs,  # pyright: ignore[reportAny]
    )
    return agent.response(input, prompt_vars=prompt_vars)


__all__ = [
    # =============================================================================
    # Core
    # =============================================================================
    "__version__",
    "Agent",
    "run",
    "configure",
    "get_config",
    "init",
    # =============================================================================
    # Model - Core
    # =============================================================================
    "Model",
    "ModelRegistry",
    "ModelSettings",
    "ModelVariable",
    "ModelVersion",
    "Middleware",
    # =============================================================================
    # Model - Structured Output
    # =============================================================================
    "StructuredOutput",
    "structured",
    "OutputType",
    "output",
    # =============================================================================
    # Model - Provider Namespaces
    # =============================================================================
    "OpenAI",
    "Anthropic",
    "Ollama",
    "Google",
    "LiteLLM",
    # =============================================================================
    # Budget
    # =============================================================================
    "Budget",
    "BudgetExceededContext",
    "BudgetState",
    "RateLimit",
    "TokenLimits",
    "TokenRateLimit",
    "BudgetThreshold",
    "raise_on_exceeded",
    "stop_on_exceeded",
    "warn_on_exceeded",
    "BudgetStore",
    "InMemoryBudgetStore",
    "FileBudgetStore",
    # =============================================================================
    # Threshold
    # =============================================================================
    "ThresholdContext",
    "ContextThreshold",
    "compact_if_available",
    "RateLimitThreshold",
    # =============================================================================
    # Memory
    # =============================================================================
    "Memory",
    "MemoryEntry",
    "MemoryBudget",
    "Decay",
    # Context
    # =============================================================================
    "Context",
    "ContextStats",
    "ContextManager",
    "DefaultContextManager",
    "TokenCounter",
    # =============================================================================
    # Pipeline
    # =============================================================================
    "Pipe",
    "pipe",
    "prompt",
    "system_prompt",
    "validated",
    "Prompt",
    "PromptContext",
    "make_prompt_context",
    "Response",
    "Output",
    "ValidationPipeline",
    "validate_output",
    "ValidationError",
    "ApprovalGate",
    "ApprovalGateProtocol",
    "CircuitBreaker",
    "CircuitBreakerOpenError",
    "HandoffBlockedError",
    "HandoffRetryRequested",
    "Pipeline",
    "PipelineRun",
    "AgentTeam",
    "DynamicPipeline",
    "parallel",
    "sequential",
    # =============================================================================
    # Reports
    # =============================================================================
    "AgentReport",
    "GuardrailReport",
    "ContextReport",
    "MemoryReport",
    "TokenReport",
    "OutputReport",
    "RateLimitReport",
    "CheckpointReport",
    "BudgetStatus",
    # =============================================================================
    # Tool & Task
    # =============================================================================
    "tool",
    "ToolSpec",
    "task",
    "RunContext",
    "ServeConfig",
    "AgentCard",
    "AgentCardAuth",
    "AgentCardProvider",
    "AgentRouter",
    "build_agent_card_json",
    "build_router",
    "MCP",
    "MCPClient",
    # =============================================================================
    # Enums
    # =============================================================================
    "StopReason",
    "LoopStrategy",
    "ContextMode",
    "ContextStrategy",
    "FormationMode",
    "OutputChunkStrategy",
    "TracingBackend",
    "TraceLevel",
    "MessageRole",
    "StepType",
    "GuardrailStage",
    "SwitchReason",
    "RateWindow",
    "AlmockPricing",
    "AuditBackend",
    "AuditBackendProtocol",
    "AuditEntry",
    "AuditEventType",
    "AuditFilters",
    "AuditLog",
    "ContentType",
    "JsonlAuditBackend",
    "SandboxRuntime",
    "ServeProtocol",
    "Hook",
    "DocFormat",
    "MemoryType",
    "MemoryBackend",
    "MemoryScope",
    "DecayStrategy",
    "InjectionStrategy",
    "CheckpointStrategy",
    "CheckpointBackend",
    "CheckpointTrigger",
    "OffloadBackend",
    "RetryBackoff",
    "CircuitState",
    "ProgressStatus",
    "BudgetLimitType",
    "ThresholdMetric",
    "ThresholdWindow",
    "Events",
    "EventContext",
    # Loop
    "Loop",
    "LoopResult",
    "ReactLoop",
    "SingleShotLoop",
    "HumanInTheLoop",
    "PlanExecuteLoop",
    "CodeActionLoop",
    "ToolApprovalFn",
    "REACT",
    "SINGLE_SHOT",
    "PLAN_EXECUTE",
    "CODE_ACTION",
    "HITL",
    # =============================================================================
    # Observability
    # =============================================================================
    "Span",
    "SpanKind",
    "SpanStatus",
    "SpanContext",
    "Session",
    "SpanExporter",
    "ConsoleExporter",
    "JSONLExporter",
    "InMemoryExporter",
    "SemanticAttributes",
    "trace",
    "span",
    "session",
    "current_span",
    "current_session",
    "set_debug",
    "get_observability_tracer",
    # =============================================================================
    # CLI & Observability Helpers
    # =============================================================================
    "WorkflowDebugger",
    "auto_trace",
    "check_for_trace_flag",
    "remove_trace_flag",
    "Checkpointer",
    "CheckpointConfig",
    "CheckpointState",
    "BudgetThresholdReached",
    "ContextCompacted",
    "DomainEvent",
    "EventBus",
    "ContentFilter",
    "Guardrail",
    "GuardrailChain",
    "GuardrailResult",
    "LengthGuardrail",
]
