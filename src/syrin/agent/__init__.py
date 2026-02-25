"""Agent base class and response loop with tool execution and budget."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator, Callable, Iterator
from typing import Any, cast

from syrin.budget import (
    Budget,
    BudgetExceededContext,
    BudgetLimitType,
    BudgetStatus,
    BudgetTracker,
    CheckBudgetResult,
)
from syrin.budget_store import BudgetStore
from syrin.checkpoint import CheckpointConfig, Checkpointer, CheckpointTrigger
from syrin.context import Context, DefaultContextManager
from syrin.context.config import ContextStats


class _ContextFacade:
    """Facade for agent.context: config attributes + compact() during prepare."""

    def __init__(self, config: Context, manager: DefaultContextManager) -> None:
        self._config = config
        self._manager = manager

    def compact(self) -> None:
        """Request context compaction (valid during prepare, e.g. from threshold action)."""
        self._manager.compact()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._config, name)


from syrin.agent._context_builder import build_messages as build_messages_for_llm
from syrin.agent._run_context import DefaultAgentRunContext
from syrin.cost import calculate_cost, estimate_cost_for_call
from syrin.enums import (
    GuardrailStage,
    Hook,
    LoopStrategy,
    MemoryBackend,
    MemoryType,
)
from syrin.events import EventContext, Events
from syrin.exceptions import BudgetExceededError, BudgetThresholdError, ToolExecutionError
from syrin.guardrails import Guardrail, GuardrailChain, GuardrailResult
from syrin.loop import Loop, LoopStrategyMapping, ReactLoop
from syrin.memory import ConversationMemory, Memory
from syrin.memory.backends import InMemoryBackend, get_backend
from syrin.memory.config import MemoryEntry
from syrin.model import Model
from syrin.observability import (
    ConsoleExporter,
    SemanticAttributes,
    SpanKind,
    SpanStatus,
    get_tracer,
)
from syrin.output import Output
from syrin.providers.base import Provider
from syrin.ratelimit import (
    APIRateLimit,
    RateLimitManager,
    RateLimitStats,
    create_rate_limit_manager,
)
from syrin.response import (
    AgentReport,
    Response,
    StreamChunk,
    StructuredOutput,
)
from syrin.types import CostInfo, Message, ModelConfig, ProviderResponse, TokenUsage, ToolSpec

DEFAULT_MAX_TOOL_ITERATIONS = 10
_UNSET: Any = object()
_log = logging.getLogger(__name__)

_agent_loop: asyncio.AbstractEventLoop | None = None


def _get_agent_loop() -> asyncio.AbstractEventLoop:
    """Get or create a persistent event loop for the agent."""
    global _agent_loop
    if _agent_loop is None or _agent_loop.is_closed():
        _agent_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(_agent_loop)
        _agent_loop.set_exception_handler(lambda _loop, _ctx: None)
    return _agent_loop


def _merge_class_attrs(mro: tuple[type, ...], name: str, merge: bool) -> Any:
    """From MRO: for 'merge' (e.g. tools) concatenate lists; else first defined."""
    if merge:
        out: list[Any] = []
        for cls in mro:
            if cls is object:
                continue
            val = cls.__dict__.get(name, _UNSET)
            if val is not _UNSET and val is not None:
                out.extend(val) if isinstance(val, list) else out.append(val)
        return out
    for cls in mro:
        if cls is object:
            continue
        if name in cls.__dict__:
            return cls.__dict__[name]
    return _UNSET


def _resolve_provider(model: Model | None, model_config: ModelConfig) -> Provider:
    """Resolve Provider from Model (preferred) or ModelConfig.provider via registry.

    Canonical path for agent runs: Model.get_provider(). Registry (get_provider(name))
    only when no Model is available (e.g. tests, scripts).
    """
    if model is not None and hasattr(model, "get_provider"):
        return model.get_provider()
    from syrin.providers.registry import get_provider

    return get_provider(model_config.provider, strict=True)


def _emit_domain_event_for_hook(hook: Hook, ctx: EventContext, bus: Any) -> None:
    """Emit domain events for hooks that have typed domain event equivalents."""
    if hook == Hook.BUDGET_THRESHOLD:
        from syrin.domain_events import BudgetThresholdReached

        pct = ctx.get("threshold_percent", 0)
        current = ctx.get("current_value", 0.0)
        limit = ctx.get("limit_value", 0.0)
        metric = ctx.get("metric", "cost")
        bus.emit(BudgetThresholdReached(pct, current, limit, metric))
    elif hook == Hook.CONTEXT_COMPACT:
        from syrin.domain_events import ContextCompacted

        bus.emit(
            ContextCompacted(
                method=ctx.get("method", "unknown"),
                tokens_before=ctx.get("tokens_before", 0),
                tokens_after=ctx.get("tokens_after", 0),
                messages_before=ctx.get("messages_before", 0),
                messages_after=ctx.get("messages_after", 0),
            )
        )


class Agent:
    """AI agent that runs completions, tools, memory, and budget control.

    An Agent is the main interface for talking to an LLM, executing tools, remembering
    facts, and controlling costs. You provide a model (LLM) and optionally tools,
    budget, memory, guardrails, and more.

    Why use an Agent?
        - Run multi-turn conversations with automatic tool-call loops (REACT by default).
        - Keep costs under control with per-run and per-period budgets.
        - Remember facts across sessions with persistent memory.
        - Validate input/output with guardrails.
        - Trace and debug with events and spans.

    How to create one:
        - Pass everything at construction: ``Agent(model=..., system_prompt=...)``
        - Or subclass and set class attributes: ``model = Model.OpenAI(...)``
        - Child classes override parent for model/prompt/budget; tools are merged.

    Attributes:
        events: Lifecycle hooks (before/on/after). Use agent.events.on(Hook.LLM_REQUEST_END, fn).

    Example:
        >>> from syrin import Agent
        >>> from syrin.model import Model
        >>> agent = Agent(
        ...     model=Model.OpenAI("gpt-4o-mini"),
        ...     system_prompt="You are a helpful assistant.",
        ... )
        >>> r = agent.response("What is 2+2?")
        >>> print(r.content)
        2 + 2 equals 4.
    """

    _Syrin_default_model: Model | ModelConfig | None = None
    _Syrin_default_system_prompt: str = ""
    _Syrin_default_tools: list[ToolSpec] = []
    _Syrin_default_budget: Budget | None = None
    _Syrin_default_guardrails: list[Guardrail] = []

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        mro = cls.__mro__
        default_model = _merge_class_attrs(mro, "model", merge=False)
        default_prompt = _merge_class_attrs(mro, "system_prompt", merge=False)
        default_tools = _merge_class_attrs(mro, "tools", merge=True)
        default_budget = _merge_class_attrs(mro, "budget", merge=False)
        default_guardrails = _merge_class_attrs(mro, "guardrails", merge=True)
        cls._Syrin_default_model = default_model if default_model is not _UNSET else None
        cls._Syrin_default_system_prompt = default_prompt if default_prompt is not _UNSET else ""
        cls._Syrin_default_tools = list(default_tools) if default_tools is not _UNSET else []
        cls._Syrin_default_budget = default_budget if default_budget is not _UNSET else None
        cls._Syrin_default_guardrails = (
            list(default_guardrails) if default_guardrails is not _UNSET else []
        )

    def __init__(
        self,
        model: Model | ModelConfig | None = _UNSET,
        system_prompt: str | None = _UNSET,
        tools: list[ToolSpec] | None = _UNSET,
        budget: Budget | None = _UNSET,
        *,
        output: Output | None = None,
        max_tool_iterations: int = DEFAULT_MAX_TOOL_ITERATIONS,
        budget_store: BudgetStore | None = None,
        budget_store_key: str = "default",
        memory: ConversationMemory | Memory | None = None,
        loop_strategy: LoopStrategy = LoopStrategy.REACT,
        loop: Loop | type[Loop] | None = None,
        guardrails: list[Guardrail] | GuardrailChain | None = _UNSET,
        context: Context | DefaultContextManager | None = None,
        rate_limit: APIRateLimit | RateLimitManager | None = None,
        checkpoint: CheckpointConfig | Checkpointer | None = None,
        debug: bool = False,
        tracer: Any = None,
        bus: Any = None,
    ) -> None:
        """Create an agent with model, prompt, tools, and optional config.

        **Required:**
            model: LLM to use. Use Model.OpenAI, Model.Anthropic, etc. The brain of your agent.

        **Core (most users need these):**
            system_prompt: Instructions that define behavior. Sent with every request. Default: empty.
            tools: List of @tool-decorated functions the agent can call. Default: [].
            budget: Cost limits (per run, per period) and threshold actions. Use Budget(run=1.0) for $1/run.

        **Optional (advanced):**
            output: Structured output config (Pydantic model). Validates responses.
            max_tool_iterations: Max tool-call loops per response (default 10).
                Why: Stops infinite tool loops. Increase for complex workflows.
            budget_store: Persist budget across runs (e.g. FileBudgetStore).
                Why: Track spend across restarts. Requires budget_store_key.
            budget_store_key: Key for budget persistence (default "default").
                Why: Isolate budgets per user/session when using budget_store.
            memory: Conversation memory (BufferMemory) or persistent Memory.
                Why: Enables remember/recall/forget or session history.
            loop_strategy: Execution strategy (REACT, SINGLE_SHOT, etc.).
                Why: REACT = tool loop; SINGLE_SHOT = one LLM call, no tools.
            loop: Custom Loop instance. Overrides loop_strategy if set.
            guardrails: List of Guardrail or GuardrailChain. Validate input/output.
                Why: Block harmful content, PII, or policy violations.
            context: Context config (max_tokens, thresholds, budget). Token caps
                go in context.budget (TokenLimits).
            rate_limit: APIRateLimit to enforce RPM/TPM.
                Why: Avoid 429 errors from provider rate limits.
            checkpoint: CheckpointConfig for save/restore state.
                Why: Resume after crashes or save progress in long runs.
            debug: If True, print lifecycle events to console.
                Why: Quick visibility into agent behavior.
            tracer: Custom tracer for observability.
            bus: Optional EventBus for typed domain events (BudgetThresholdReached,
                ContextCompacted). Use when you need structured event handling for
                metrics, observability, or custom pipelines.

        Example:
            >>> agent = Agent(
            ...     model=Model.OpenAI("gpt-4o-mini"),
            ...     system_prompt="You are concise.",
            ...     tools=[search, calculate],
            ...     budget=Budget(run=0.50),
            ...     memory=Memory(top_k=5),
            ... )
        """
        cls = self.__class__
        if model is _UNSET:
            model = getattr(cls, "_Syrin_default_model", None)
        if system_prompt is _UNSET:
            system_prompt = getattr(cls, "_Syrin_default_system_prompt", "") or ""
        if tools is _UNSET:
            tools = getattr(cls, "_Syrin_default_tools", None) or []
        if budget is _UNSET:
            budget = getattr(cls, "_Syrin_default_budget", None)
        if guardrails is _UNSET:
            guardrails = getattr(cls, "_Syrin_default_guardrails", None) or []
        if model is None:
            raise TypeError("Agent requires model (pass explicitly or set class-level model)")
        if isinstance(model, Model):
            self._model: Model | None = model
            self._model_config = model.to_config()
        else:
            self._model = None
            self._model_config = model

        # Handle output configuration
        self._output: Output | None = output
        if output is not None and self._model_config is not None and output.type is not None:
            self._model_config.output = output.type

        self._system_prompt = system_prompt or ""
        self._tools = list(tools) if tools else []
        self._max_tool_iterations = max_tool_iterations
        self._budget = budget
        self._budget_store = budget_store
        self._budget_store_key = budget_store_key
        self._token_limits = None
        self._conversation_memory: ConversationMemory | None = None
        self._persistent_memory: Memory | None = None
        self._memory_backend: InMemoryBackend | None = None
        self._parent_agent: Agent | None = None
        self._budget_tracker_shared: bool = False
        self._provider: Provider

        # Context (and budget from context) set early for budget_store load
        if context is None:
            self._context = DefaultContextManager(Context())
        elif isinstance(context, Context):
            self._context = DefaultContextManager(context)
        else:
            self._context = context
        ctx_config = getattr(self._context, "context", None)
        self._token_limits = getattr(ctx_config, "budget", None) if ctx_config else None

        if memory is None:
            # Default: enable persistent memory with sensible defaults
            self._persistent_memory = Memory(
                types=[MemoryType.CORE, MemoryType.EPISODIC],
                top_k=10,
            )
            self._memory_backend = get_backend(MemoryBackend.MEMORY)
        elif isinstance(memory, Memory):
            self._persistent_memory = memory
            self._memory_backend = get_backend(memory.backend, path=memory.path)
        else:
            self._conversation_memory = memory
        if (budget is not None or self._token_limits is not None) and budget_store and budget:
            loaded = budget_store.load(budget_store_key)
            self._budget_tracker = loaded if loaded is not None else BudgetTracker()
        else:
            self._budget_tracker = BudgetTracker()
        self._provider = _resolve_provider(self._model, self._model_config)
        self._agent_name = self.__class__.__name__
        if self._budget is not None:
            self._budget._consume_callback = self._make_budget_consume_callback()
        if self._budget is not None and self._budget.per is not None and self._budget_store is None:
            _log.warning(
                "Rate limits (hour/day/week/month) are in-memory only; "
                "pass budget_store (e.g. FileBudgetStore) to persist across restarts."
            )
        if self._budget is not None and self._model is not None:
            pricing = getattr(self._model, "pricing", None)
            if pricing is None and hasattr(self._model, "get_pricing"):
                pricing = self._model.get_pricing()
            if pricing is None:
                _log.warning(
                    "Model %r has no pricing; budget cost may be 0 or incorrect. "
                    "Set pricing_override or input_price/output_price on the model.",
                    self._model_config.model_id,
                )

        loop_instance: Loop
        if loop is not None:
            if isinstance(loop, type) and hasattr(loop, "run") and callable(loop.run):
                loop_instance = loop()
            elif hasattr(loop, "run") and callable(loop.run):
                loop_instance = loop  # type: ignore[assignment]
            else:
                loop_instance = ReactLoop(max_iterations=max_tool_iterations)
        else:
            loop_instance = LoopStrategyMapping.create_loop(
                loop_strategy, max_iterations=max_tool_iterations
            )
        self._loop = loop_instance

        # Guardrails setup
        if guardrails is None or (isinstance(guardrails, list) and len(guardrails) == 0):
            self._guardrails = GuardrailChain()
        elif isinstance(guardrails, GuardrailChain):
            self._guardrails = guardrails
        else:
            self._guardrails = GuardrailChain(list(guardrails))

        # Observability setup (before context to pass tracer)
        self._debug = debug
        self._tracer = tracer or get_tracer()
        self._bus = bus
        if debug and not any(isinstance(e, ConsoleExporter) for e in self._tracer._exporters):
            self._tracer.add_exporter(ConsoleExporter())
        if debug:
            self._tracer.set_debug_mode(True)

        # Connect context to events and observability
        if hasattr(self._context, "set_emit_fn"):
            from typing import cast

            self._context.set_emit_fn(cast(Any, self._emit_event))
        if hasattr(self._context, "set_tracer"):
            self._context.set_tracer(self._tracer)

        # Rate limit management setup
        if rate_limit is None:
            self._rate_limit_manager: RateLimitManager | None = None
        elif isinstance(rate_limit, RateLimitManager):
            self._rate_limit_manager = rate_limit
        else:
            self._rate_limit_manager = cast(RateLimitManager, create_rate_limit_manager(rate_limit))

        # Validation settings from Output config
        if self._output is not None:
            self._validation_retries = self._output.validation_retries
            self._validation_context = self._output.context
            self._output_validator = self._output.validator
        else:
            self._validation_retries = 3
            self._validation_context = {}
            self._output_validator = None

        # Connect rate limit to events and observability
        if self._rate_limit_manager and hasattr(self._rate_limit_manager, "set_emit_fn"):
            self._rate_limit_manager.set_emit_fn(self._emit_event)
        if self._rate_limit_manager and hasattr(self._rate_limit_manager, "set_tracer"):
            self._rate_limit_manager.set_tracer(self._tracer)

        self.events = Events(self._emit_event)

        # Initialize run report for tracking metrics across a response() call
        self._run_report: AgentReport = AgentReport()

        # Checkpoint setup
        if checkpoint is None:
            self._checkpoint_config: CheckpointConfig | None = None
            self._checkpointer: Checkpointer | None = None
        elif isinstance(checkpoint, Checkpointer):
            self._checkpoint_config = None
            self._checkpointer = checkpoint
        else:
            self._checkpoint_config = checkpoint
            if checkpoint.enabled:
                from syrin.checkpoint import get_checkpoint_backend

                kwargs = {}
                if checkpoint.path is not None:
                    kwargs["path"] = checkpoint.path
                backend = get_checkpoint_backend(checkpoint.storage, **kwargs)
                self._checkpointer = Checkpointer(
                    max_checkpoints=checkpoint.max_checkpoints, backend=backend
                )
            else:
                self._checkpointer = None

    def save_checkpoint(self, name: str | None = None, reason: str | None = None) -> str | None:
        """Save a snapshot of the agent's current state for later restore.

        Why: Resumes after crashes, saves progress in long runs, or recovers when
        budget is near limit. Essential for production reliability.

        What it saves: Iteration, messages, memory_data, budget_state, reason.

        How to tweak: Pass name to group checkpoints; pass reason for debugging.
        Requires checkpoint=CheckpointConfig(...) at construction.

        Args:
            name: Optional label. Default: agent class name.
            reason: Why saved (e.g. 'step', 'tool', 'budget', 'error').

        Returns:
            Checkpoint ID (str) for load_checkpoint, or None if disabled.

        Example:
            >>> agent = Agent(model=m, checkpoint=CheckpointConfig(storage="memory"))
            >>> cid = agent.save_checkpoint(reason="before_expensive_step")
            >>> agent.load_checkpoint(cid)
        """
        if self._checkpointer is None:
            return None

        agent_name = name or self._agent_name
        state = {
            "iteration": self._run_report.tokens.total_tokens,
            "messages": [],  # Could include conversation history
            "memory_data": {},
            "budget_state": (
                {
                    "remaining": self._budget.remaining,
                    "spent": self._budget._spent,
                    "tracker_state": self._budget_tracker.get_state(),
                }
                if self._budget is not None
                else None
            ),
            "checkpoint_reason": reason,
        }

        checkpoint_id = self._checkpointer.save(agent_name, state)
        self._run_report.checkpoints.saves += 1
        self._emit_event(Hook.CHECKPOINT_SAVE, EventContext(checkpoint_id=checkpoint_id))
        return checkpoint_id

    def _maybe_checkpoint(self, reason: str) -> None:
        """Automatically checkpoint based on trigger configuration.

        Args:
            reason: The reason for checkpointing ('step', 'tool', 'error', 'budget')
        """
        if self._checkpointer is None or self._checkpoint_config is None:
            return

        trigger = self._checkpoint_config.trigger if self._checkpoint_config else None
        if trigger == CheckpointTrigger.MANUAL:
            return

        should_checkpoint = (trigger == CheckpointTrigger.STEP and reason in ("step", "tool")) or (
            trigger is not None and trigger.value == reason
        )
        if should_checkpoint:
            self.save_checkpoint(reason=reason)

    def load_checkpoint(self, checkpoint_id: str) -> bool:
        """Restore agent state from a previously saved checkpoint.

        Why: Resume after failure, replay from a point, or restore budget/memory state.

        Args:
            checkpoint_id: ID returned by save_checkpoint or from list_checkpoints.

        Returns:
            True if loaded; False if ID not found or checkpointing disabled.

        Example:
            >>> ids = agent.list_checkpoints()
            >>> if ids:
            ...     agent.load_checkpoint(ids[-1])
        """
        if self._checkpointer is None:
            return False

        state = self._checkpointer.load(checkpoint_id)
        if state is None:
            return False

        budget_state = getattr(state, "budget_state", None)
        if budget_state is not None and self._budget is not None:
            tracker_state = budget_state.get("tracker_state")
            if tracker_state is not None:
                self._budget_tracker.load_state(tracker_state)
            spent = budget_state.get("spent")
            if spent is not None:
                self._budget._set_spent(spent)

        self._run_report.checkpoints.loads += 1
        self._emit_event(Hook.CHECKPOINT_LOAD, EventContext(checkpoint_id=checkpoint_id))
        return True

    def list_checkpoints(self, name: str | None = None) -> list[str]:
        """List checkpoint IDs for this agent, optionally filtered by name.

        Why: Find which checkpoints exist before loading one.

        Args:
            name: Filter by label. Default: agent class name.

        Returns:
            List of checkpoint IDs (most recent typically last).

        Example:
            >>> ids = agent.list_checkpoints(name="my_agent")
            >>> print(ids)  # ['ckpt_abc123', 'ckpt_def456']
        """
        if self._checkpointer is None:
            return []

        agent_name = name or self._agent_name
        return self._checkpointer.list_checkpoints(agent_name)

    def get_checkpoint_report(self) -> AgentReport:
        """Get the full agent report including checkpoint stats.

        Why: Inspect saves/loads and all other run metrics (guardrails, budget, etc.).

        Returns:
            AgentReport with report.checkpoints (saves, loads) and other sections.

        Example:
            >>> agent.response("Hello")
            >>> r = agent.get_checkpoint_report()
            >>> print(r.checkpoints.saves, r.checkpoints.loads)
        """
        return self._run_report

    def _emit_event(self, hook: Hook | str, ctx: EventContext | dict[str, Any]) -> None:
        """Internal: trigger a hook through the events system.

        Args:
            hook: Hook enum value or string (e.g. "context.compact")
            ctx: EventContext or dict with hook-specific data
        """
        # Map string event names (from context/ratelimit managers) to Hook
        # StrEnum members are also str, so check for Hook first
        if isinstance(hook, str) and not isinstance(hook, Hook):
            _EVENT_TO_HOOK: dict[str, Hook] = {
                "context.compact": Hook.CONTEXT_COMPACT,
                "context.threshold": Hook.CONTEXT_THRESHOLD,
                "ratelimit.threshold": Hook.RATELIMIT_THRESHOLD,
                "ratelimit.exceeded": Hook.RATELIMIT_EXCEEDED,
            }
            resolved = _EVENT_TO_HOOK.get(hook)
            if resolved is None:
                return
            hook = resolved
        if isinstance(ctx, dict):
            ctx = EventContext(ctx)

        # Print event to console when debug=True
        if self._debug:
            self._print_event(hook.value, ctx)

        # Trigger before/on/after handlers
        self.events._trigger_before(hook, ctx)
        self.events._trigger(hook, ctx)
        self.events._trigger_after(hook, ctx)

        # Domain events (observability, typed consumers)
        bus = getattr(self, "_bus", None)
        if bus is not None:
            _emit_domain_event_for_hook(hook, ctx, bus)

    def _print_event(self, event: str, ctx: EventContext) -> None:
        """Print event to console when debug=True."""
        import sys
        from datetime import datetime

        # Check if terminal supports colors
        is_tty = sys.stdout.isatty()

        # Colors
        RESET = "\033[0m" if is_tty else ""
        GREEN = "\033[92m" if is_tty else ""
        BLUE = "\033[94m" if is_tty else ""
        YELLOW = "\033[93m" if is_tty else ""
        CYAN = "\033[96m" if is_tty else ""
        RED = "\033[91m" if is_tty else ""

        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]

        # Determine event type for coloring
        event_str = str(event)
        if "start" in event_str or "init" in event_str:
            color = GREEN
            symbol = "▶"
        elif "end" in event_str or "complete" in event_str:
            color = BLUE
            symbol = "✓"
        elif "tool" in event_str:
            color = YELLOW
            symbol = "🔧"
        elif "llm" in event_str or "request" in event_str:
            color = CYAN
            symbol = "💬"
        elif "error" in event_str:
            color = RED
            symbol = "✗"
        else:
            color = ""
            symbol = "•"

        print(f"{color}{symbol} {timestamp} {event}{RESET}")

        # Print key context info
        indent = "     "

        # Input/task
        if "input" in ctx:
            task = ctx["input"]
            if isinstance(task, str) and len(task) > 60:
                task = task[:57] + "..."
            print(f"{indent}Input: {task}")

        # Model
        if "model" in ctx:
            print(f"{indent}Model: {ctx['model']}")

        # Cost
        if "cost" in ctx and ctx["cost"] is not None:
            cost_val = float(ctx["cost"])
            if cost_val > 0:
                print(f"{indent}Cost: ${cost_val:.6f}")

        # Tokens
        if "tokens" in ctx and ctx["tokens"] is not None:
            print(f"{indent}Tokens: {ctx['tokens']}")

        # Iteration
        if "iteration" in ctx:
            print(f"{indent}Iteration: {ctx['iteration']}")

        # Tool name
        if "name" in ctx:
            print(f"{indent}Tool: {ctx['name']}")

        # Error
        if "error" in ctx:
            print(f"{indent}{RED}Error: {ctx['error']}{RESET}")

        # Duration
        if "duration" in ctx and ctx["duration"] is not None:
            duration_ms = float(ctx["duration"]) * 1000
            print(f"{indent}Duration: {duration_ms:.2f}ms")

        print()

    def switch_model(self, model: Model | ModelConfig) -> None:
        """Change the LLM used by the agent at runtime.

        Why: Switch to a cheaper model when approaching budget, or to a fallback when
        rate limits are hit. Often triggered automatically by BudgetThreshold or
        RateLimitThreshold, or called manually.

        How to tweak: Pass Model.OpenAI("gpt-4o-mini") for cheaper; Model.OpenAI("gpt-4o")
        for higher quality. Use with BudgetThreshold action:
        ``BudgetThreshold(at=80, action=lambda ctx: ctx.parent.switch_model(Model(...)))``

        Args:
            model: New Model or ModelConfig. Must be same provider type.

        Example:
            >>> agent.switch_model(Model.OpenAI("gpt-4o-mini"))
        """
        if isinstance(model, Model):
            self._model = model
            self._model_config = model.to_config()
        else:
            self._model = None
            self._model_config = model
        self._provider = _resolve_provider(self._model, self._model_config)

    @property
    def budget_summary(self) -> dict[str, Any]:
        """Current budget usage across run and rolling windows.

        Why: Monitor spend before/after runs, log for auditing, or show users.

        Returns:
            Dict with current_run_cost, hourly_cost, daily_cost, weekly_cost,
            monthly_cost, entries_count. All costs in USD.

        Example:
            >>> agent.response("Hello")
            >>> print(agent.budget_summary)
            {'current_run_cost': 0.0012, 'hourly_cost': 0.05, ...}
        """
        return self._budget_tracker.get_summary().to_dict()

    def get_budget_tracker(self) -> BudgetTracker | None:
        """Return the budget tracker when this agent has a budget or token_limits.

        Use for reservation (reserve/commit/rollback) or inspection. Returns None
        if the agent has neither budget nor token_limits.

        Example:
            >>> tracker = agent.get_budget_tracker()
            >>> if tracker:
            ...     token = tracker.reserve(estimated_cost)
            ...     try:
            ...         response = await agent.complete(messages, tools)
            ...         token.commit(actual_cost, response.token_usage)
            ...     except Exception:
            ...         token.rollback()
        """
        return (
            self._budget_tracker
            if (self._budget is not None or self._token_limits is not None)
            else None
        )

    @property
    def memory(self) -> ConversationMemory | Memory | None:
        """Active memory: conversation (session) or persistent (remember/recall).

        Why: Inspect what memory type is in use. Use conversation_memory or
        persistent_memory for specific type.

        Returns:
            ConversationMemory if session history; Memory if persistent.
        """
        return self._persistent_memory or self._conversation_memory

    @property
    def conversation_memory(self) -> ConversationMemory | None:
        """Session-only message history (user/assistant turns).

        Why: Set when you pass BufferMemory or WindowMemory as memory=.
        Used to keep multi-turn context without persistent storage.

        Returns:
            The ConversationMemory instance, or None if using persistent memory.
        """
        return self._conversation_memory

    @property
    def persistent_memory(self) -> Memory | None:
        """Persistent memory config (remember/recall/forget).

        Why: Check top_k, types, backend when using persistent memory.
        Enables remember(), recall(), forget().

        Returns:
            Memory if persistent memory enabled; None otherwise.
        """
        return self._persistent_memory

    @property
    def context(self) -> Context | _ContextFacade:
        """Context config (max_tokens, thresholds) and compact().

        Use ctx.compact() in a ContextThreshold action, or agent.context.compact()
        during prepare, to compact context on demand (no auto_compact_at).
        """
        if hasattr(self._context, "context") and hasattr(self._context, "compact"):
            return _ContextFacade(self._context.context, self._context)
        if hasattr(self._context, "context"):
            return self._context.context
        return Context()

    @property
    def context_stats(self) -> ContextStats:
        """Token usage and compaction stats from the last run.

        Why: Debug context size, see if compaction ran, track token growth.
        """
        if hasattr(self._context, "stats"):
            return self._context.stats
        return ContextStats()

    @property
    def _context_manager(self) -> DefaultContextManager:
        """Internal context manager."""
        return self._context

    @property
    def run_context(self) -> DefaultAgentRunContext:
        """Narrow interface for Loop.run(). Used internally; loops receive this instead of Agent."""
        return DefaultAgentRunContext(self)

    @property
    def rate_limit(self) -> APIRateLimit | None:
        """Rate limit config (RPM, TPM, thresholds).

        Why: Inspect limits and thresholds. None if rate_limit not set.
        """
        if self._rate_limit_manager and hasattr(self._rate_limit_manager, "config"):
            return cast(APIRateLimit, self._rate_limit_manager.config)
        return None

    @property
    def rate_limit_stats(self) -> RateLimitStats:
        """Current rate limit usage (RPM/TPM used vs limit).

        Why: Monitor proximity to limits, log for debugging.
        """
        if self._rate_limit_manager and hasattr(self._rate_limit_manager, "stats"):
            return cast(RateLimitStats, self._rate_limit_manager.stats)
        return RateLimitStats()

    @property
    def _rate_limit_manager_internal(self) -> RateLimitManager | None:
        """Internal rate limit manager."""
        return self._rate_limit_manager

    @property
    def report(self) -> AgentReport:
        """Aggregated report of the last run: guardrails, memory, budget, tokens, etc.

        Why: Debug failures, log metrics, inspect guardrail/validation results.
        Reset at the start of each response() or arun().

        Sections:
            report.guardrail  - Input/output guardrail results
            report.context    - Token and compaction stats
            report.memory     - Stores, recalls, forgets
            report.budget     - Budget status
            report.tokens     - Input/output token counts and cost
            report.output     - Structured output validation
            report.ratelimits - Rate limit checks and throttles
            report.checkpoints - Saves and loads

        Example:
            >>> agent.response("Hello")
            >>> print(agent.report.guardrail.input_passed)
            >>> print(agent.report.tokens.total_tokens)
        """
        return self._run_report

    def remember(
        self,
        content: str,
        memory_type: MemoryType = MemoryType.EPISODIC,
        importance: float = 1.0,
        **metadata: Any,
    ) -> str:
        """Store a fact in persistent memory for later recall.

        Why: Let the agent remember user preferences, past events, or learned facts
        across sessions. Recalled automatically before each request based on relevance.

        Memory types: CORE (identity/prefs), EPISODIC (events), SEMANTIC (facts),
        PROCEDURAL (patterns). Importance 0.0–1.0 affects recall ranking.

        Requires persistent memory (Memory). Use memory=False to disable.

        Args:
            content: Text to store (e.g. "User prefers dark mode").
            memory_type: CORE, EPISODIC, SEMANTIC, or PROCEDURAL. Default EPISODIC.
            importance: 0.0–1.0. Higher = more likely to be recalled.
            **metadata: Optional fields (user_id, session_id, etc.).

        Returns:
            Memory ID (str) for forget(memory_id=...).

        Example:
            >>> agent.remember("User name is Alice", memory_type=MemoryType.CORE)
            'uuid-abc-123'
            >>> agent.response("What's my name?")  # Recalls automatically
        """
        import uuid

        if self._memory_backend is None:
            raise RuntimeError("No persistent memory configured")

        with self._tracer.span(
            "memory.store",
            kind=SpanKind.MEMORY,
            attributes={
                SemanticAttributes.MEMORY_OPERATION: "store",
                SemanticAttributes.MEMORY_TYPE: memory_type.value,
            },
        ) as mem_span:
            entry = MemoryEntry(
                id=str(uuid.uuid4()),
                content=content,
                type=memory_type,
                importance=importance,
                metadata=metadata,
            )
            self._memory_backend.add(entry)
            mem_span.set_attribute("memory.id", entry.id)
            # Track in report
            self._run_report.memory.stores += 1
            # Emit hook
            self._emit_event(
                Hook.MEMORY_STORE,
                EventContext(
                    memory_id=entry.id,
                    content=content[:100],  # Truncate for event
                    memory_type=memory_type.value,
                    importance=importance,
                ),
            )
            return entry.id

    def recall(
        self,
        query: str | None = None,
        memory_type: MemoryType | None = None,
        limit: int = 10,
    ) -> list[MemoryEntry]:
        """Retrieve memories by query or type. Used by agent internally; also call manually.

        Why: Inspect what the agent has stored, or manually fetch before custom logic.
        The agent auto-recalls before each request using the user input as query.

        Args:
            query: Search text (e.g. "user preferences"). None = list all.
            memory_type: Filter to CORE, EPISODIC, SEMANTIC, or PROCEDURAL.
            limit: Max results. Default 10.

        Returns:
            List of MemoryEntry (id, content, type, importance, metadata).

        Example:
            >>> entries = agent.recall("name", memory_type=MemoryType.CORE)
            >>> print([e.content for e in entries])
            ['User name is Alice']
        """
        if self._memory_backend is None:
            raise RuntimeError("No persistent memory configured")

        with self._tracer.span(
            "memory.recall",
            kind=SpanKind.MEMORY,
            attributes={
                SemanticAttributes.MEMORY_OPERATION: "recall",
                SemanticAttributes.MEMORY_TYPE: memory_type.value if memory_type else "all",
                SemanticAttributes.MEMORY_QUERY: query or "",
            },
        ) as mem_span:
            if query:
                results = self._memory_backend.search(query, memory_type, limit)
            else:
                results = self._memory_backend.list(memory_type, limit=limit)

            mem_span.set_attribute(
                SemanticAttributes.MEMORY_RESULTS_COUNT,
                len(results),
            )
            # Track in report
            self._run_report.memory.recalls += 1
            # Emit hook
            self._emit_event(
                Hook.MEMORY_RECALL,
                EventContext(
                    query=query,
                    memory_type=memory_type.value if memory_type else "all",
                    results_count=len(results),
                    limit=limit,
                ),
            )
            return results

    def forget(
        self,
        memory_id: str | None = None,
        query: str | None = None,
        memory_type: MemoryType | None = None,
    ) -> int:
        """Delete one or more memories. Use when user requests "forget X" or for GDPR.

        Why: Compliance, correcting wrong facts, clearing obsolete data.

        Provide exactly one of: memory_id, query, or memory_type. memory_id is
        most precise. query deletes entries containing the text. memory_type
        deletes all of that type.

        Args:
            memory_id: ID from remember() return value.
            query: Delete entries whose content contains this text.
            memory_type: Delete all entries of this type.

        Returns:
            Number of memories deleted.

        Example:
            >>> agent.forget(memory_id="uuid-abc-123")
            1
            >>> agent.forget(query="obsolete")
            3
        """
        if self._memory_backend is None:
            raise RuntimeError("No persistent memory configured")

        with self._tracer.span(
            "memory.forget",
            kind=SpanKind.MEMORY,
            attributes={
                SemanticAttributes.MEMORY_OPERATION: "forget",
                SemanticAttributes.MEMORY_TYPE: memory_type.value if memory_type else "all",
                SemanticAttributes.MEMORY_QUERY: query or "",
            },
        ) as mem_span:
            deleted = 0
            if memory_id:
                self._memory_backend.delete(memory_id)
                deleted = 1
            elif query or memory_type:
                memories = self._memory_backend.list(memory_type)
                for mem in memories:
                    if query and query.lower() in mem.content.lower():
                        self._memory_backend.delete(mem.id)
                        deleted += 1

            mem_span.set_attribute("memory.deleted_count", deleted)
            # Track in report
            self._run_report.memory.forgets += deleted
            # Emit hook
            self._emit_event(
                Hook.MEMORY_FORGET,
                EventContext(
                    memory_id=memory_id,
                    query=query,
                    memory_type=memory_type.value if memory_type else "all",
                    deleted_count=deleted,
                ),
            )
            return deleted

    def _run_guardrails(
        self,
        text: str,
        stage: GuardrailStage,
    ) -> GuardrailResult:
        """Run guardrails on text with observability."""
        if len(self._guardrails) == 0:
            return GuardrailResult(passed=True)

        # Emit start hook
        self._emit_event(
            Hook.GUARDRAIL_INPUT if stage == GuardrailStage.INPUT else Hook.GUARDRAIL_OUTPUT,
            EventContext(
                text=text[:200],  # Truncate for event
                stage=stage.value,
                guardrail_count=len(self._guardrails),
            ),
        )

        with self._tracer.span(
            f"guardrails.{stage.value}",
            kind=SpanKind.GUARDRAIL,
            attributes={
                SemanticAttributes.GUARDRAIL_STAGE: stage.value,
            },
        ) as guardrail_span:
            result = self._guardrails.check(text, stage, budget=self._budget, agent=self)

            guardrail_span.set_attribute(
                SemanticAttributes.GUARDRAIL_PASSED,
                result.passed,
            )

            if not result.passed:
                guardrail_span.set_attribute(
                    SemanticAttributes.GUARDRAIL_VIOLATION,
                    result.reason or "Unknown",
                )
                guardrail_span.set_status(SpanStatus.ERROR, result.reason)

                # Emit blocked hook
                self._emit_event(
                    Hook.GUARDRAIL_BLOCKED,
                    EventContext(
                        stage=stage.value,
                        reason=result.reason,
                        guardrail_names=[g.name for g in self._guardrails],
                    ),
                )

                # Store in report
                guardrail_names = [g.name for g in self._guardrails]
                if stage == GuardrailStage.INPUT:
                    self._run_report.guardrail.input_passed = False
                    self._run_report.guardrail.input_reason = result.reason
                    self._run_report.guardrail.input_guardrails = guardrail_names
                    self._run_report.guardrail.blocked = True
                    self._run_report.guardrail.blocked_stage = stage.value
                else:
                    self._run_report.guardrail.output_passed = False
                    self._run_report.guardrail.output_reason = result.reason
                    self._run_report.guardrail.output_guardrails = guardrail_names
                    self._run_report.guardrail.blocked = True
                    self._run_report.guardrail.blocked_stage = stage.value
            else:
                # Store guardrail names that passed
                guardrail_names = [g.name for g in self._guardrails]
                if stage == GuardrailStage.INPUT:
                    self._run_report.guardrail.input_passed = True
                    self._run_report.guardrail.input_guardrails = guardrail_names
                else:
                    self._run_report.guardrail.output_passed = True
                    self._run_report.guardrail.output_guardrails = guardrail_names

            return result

    def handoff(
        self,
        target_agent: type[Agent],
        task: str,
        *,
        transfer_context: bool = True,
        transfer_budget: bool = False,
    ) -> Response[str]:
        """Delegate a task to another agent and return its response.

        Why: Route to specialized agents (e.g. research vs support). Transfers
        memory and optionally budget so the target has full context.

        transfer_context: Copy persistent memories to target so it knows what
        this agent knew. transfer_budget: Share remaining budget with target.

        Args:
            target_agent: Agent class (e.g. ResearchAgent). Instantiated internally.
            task: Task description for the target.
            transfer_context: Copy memories to target. Default True.
            transfer_budget: Give target remaining budget. Default False.

        Returns:
            Response from target_agent.response(task).

        Example:
            >>> r = agent.handoff(SupportAgent, "User needs refund help")
            >>> print(r.content)
        """
        target = target_agent()

        if transfer_budget and self._budget:
            target._budget = self._budget
            target._budget_tracker = self._budget_tracker

        if transfer_context:
            if self._memory_backend is None:
                _log.warning(
                    "handoff: transfer_context=True but source agent has no memory backend. "
                    "Did you set memory=Memory() on the source agent?"
                )
            else:
                memories = self._memory_backend.list()
                if memories:
                    if target._memory_backend is None:
                        target._persistent_memory = Memory(top_k=10, relevance_threshold=0.7)
                        target._memory_backend = get_backend(
                            Memory(top_k=10, relevance_threshold=0.7).backend
                        )
                    for mem in memories:
                        target.remember(
                            mem.content, memory_type=mem.type, importance=mem.importance
                        )
                    _log.debug("handoff: transferred %d memories to target agent", len(memories))

        return target.response(task)

    def spawn(
        self,
        agent_class: type[Agent],
        task: str | None = None,
        *,
        budget: Budget | None = None,
        max_children: int | None = None,
    ) -> Agent | Response[str]:
        """Create a child agent. Optionally run a task and return its response.

        Why: Break work into sub-agents (specialists). Budget flows from parent:
        - Parent has shared budget: child borrows from it.
        - Child gets budget: "pocket money" up to parent's remaining.
        - Child spend is deducted from parent.

        Args:
            agent_class: Agent class to spawn (e.g. ResearchAgent).
            task: If set, run task and return Response. Else return agent instance.
            budget: Child's budget (pocket money). Must not exceed parent remaining.
            max_children: Cap on concurrent children. Default from _max_children or 10.

        Returns:
            Response if task given; else the spawned Agent instance.

        Example:
            >>> r = agent.spawn(ResearchAgent, task="Find papers on X")
            >>> child = agent.spawn(ResearchAgent)  # No task
            >>> child.response("Another task")
        """
        use_instance_limit = max_children is None
        _max_children = getattr(self, "_max_children", 10) if use_instance_limit else max_children

        current_children = getattr(self, "_child_count", 0)

        if _max_children and current_children >= _max_children:
            raise RuntimeError(f"Cannot spawn: max children ({_max_children}) reached")

        if not hasattr(self, "_child_count"):
            self._child_count = 0
        if use_instance_limit:
            self._child_count += 1

        agent_kwargs: dict[str, Any] = {}

        # Handle budget inheritance with borrow mechanism
        if budget is not None:
            # Child has pocket money - validate it doesn't exceed parent's budget
            if self._budget is not None and self._budget.run is not None:
                parent_remaining = self._budget.remaining
                if (
                    parent_remaining is not None
                    and budget.run is not None
                    and budget.run > parent_remaining
                ):
                    raise ValueError(
                        f"Child budget (${budget.run:.2f}) cannot exceed parent's "
                        f"remaining budget (${parent_remaining:.2f}). "
                        "Pocket money must be less than or equal to parent's available funds."
                    )
            # Use child's pocket money
            agent_kwargs["budget"] = budget
        elif self._budget is not None and self._budget.shared:
            # Parent has shared budget - child borrows from parent
            # Create a borrowed budget that shares the parent's budget tracker
            borrowed_budget = Budget(
                run=self._budget.remaining,  # Child can spend up to parent's remaining
                per=self._budget.per,
                on_exceeded=self._budget.on_exceeded,
                thresholds=self._budget.thresholds,
                shared=True,  # Child can also share with its children
            )
            borrowed_budget._parent_budget = self._budget  # Track parent for updates
            agent_kwargs["budget"] = borrowed_budget

        child_agent = agent_class(**agent_kwargs)

        # When shared budget, child uses parent's tracker so parent has live view
        borrowed = agent_kwargs.get("budget")
        if borrowed is not None and getattr(borrowed, "_parent_budget", None) is not None:
            child_agent._parent_agent = self
            child_agent._budget_tracker = self._budget_tracker
            child_agent._budget_tracker_shared = True

        if task:
            result = child_agent.response(task)
            if not getattr(child_agent, "_budget_tracker_shared", False):
                self._update_parent_budget(result.cost)
            return result

        return child_agent

    def _update_parent_budget(self, cost: float) -> None:
        """Update parent's budget when child spends (borrow mechanism)."""
        if self._budget is not None:
            from syrin.types import CostInfo

            model_id = (
                self._model_config.model_id
                if hasattr(self, "_model_config") and self._model_config
                else "unknown"
            )
            cost_info = CostInfo(cost_usd=cost, model_name=model_id)
            self._budget_tracker.record(cost_info)
            self._budget._set_spent(self._budget_tracker.current_run_cost)

    def spawn_parallel(
        self,
        agents: list[tuple[type[Agent], str]],
    ) -> list[Response[str]]:
        """Run multiple agents in parallel, each with its own task.

        Why: Fan-out work (e.g. research + summarization + fact-check in parallel).
        Uses asyncio.gather. Each agent runs independently.

        Args:
            agents: [(AgentClass, task), ...] e.g. [(ResearchAgent, "X"), (Summarizer, "Y")].

        Returns:
            List of Response, one per (agent_class, task), in same order.

        Example:
            >>> results = agent.spawn_parallel([
            ...     (ResearchAgent, "Topic A"),
            ...     (ResearchAgent, "Topic B"),
            ... ])
        """
        import asyncio

        async def spawn_one(
            agent_class: type[Agent],
            task: str,
        ) -> Response[str]:
            child = agent_class()
            return await child.arun(task)

        async def run_all() -> list[Response[str]]:
            tasks = [spawn_one(ac, t) for ac, t in agents]
            return await asyncio.gather(*tasks)

        loop = _get_agent_loop()
        return loop.run_until_complete(run_all())

    def _build_messages(self, user_input: str) -> list[Message]:
        def get_budget() -> Any:
            model_for_context = self._model if self._model is not None else None
            call_ctx = getattr(self, "_call_context", None)
            if call_ctx is not None:
                return call_ctx.get_budget(model_for_context)
            if hasattr(self._context, "context"):
                return self._context.context.get_budget(model_for_context)
            return Context().get_budget(model_for_context)

        return build_messages_for_llm(
            user_input,
            system_prompt=self._system_prompt or "",
            tools=self._tools,
            conversation_memory=self._conversation_memory,
            memory_backend=self._memory_backend,
            persistent_memory=self._persistent_memory,
            context_manager=self._context,
            get_budget=get_budget,
            call_context=getattr(self, "_call_context", None),
            tracer=self._tracer,
        )

    def _build_output(
        self,
        content: str,
        validation_retries: int = 3,
        validation_context: dict[str, Any] | None = None,
        validator: Any = None,
    ) -> StructuredOutput | None:
        """Build structured output from response content with validation.

        Args:
            content: Raw response content from LLM
            validation_retries: Number of validation retries
            validation_context: Context for validation
            validator: Custom output validator
        """
        from syrin.validation import ValidationPipeline

        output_type = getattr(self._model_config, "output", None)
        if output_type is None:
            return None

        # Get the pydantic model from output_type
        pydantic_model = None

        # Check if output_type has _structured_pydantic (from @structured decorator)
        if hasattr(output_type, "_structured_pydantic"):
            pydantic_model = output_type._structured_pydantic

        # Also check if it's a direct BaseModel subclass
        if pydantic_model is None:
            try:
                from pydantic import BaseModel

                if isinstance(output_type, type) and issubclass(output_type, BaseModel):
                    pydantic_model = output_type
            except Exception:
                pass

        if pydantic_model is None:
            # Not a Pydantic model, just return raw
            import json

            try:
                data = json.loads(content)
            except (json.JSONDecodeError, TypeError):
                return StructuredOutput(raw=content, parsed=None, _data={})
            return StructuredOutput(
                raw=content, parsed=data, _data=data if isinstance(data, dict) else {}
            )

        # Use ValidationPipeline for full validation with retries
        context = validation_context or {}

        # Get emit function for hooks
        emit_fn = getattr(self, "_emit_event", None)

        pipeline = ValidationPipeline(
            output_type=pydantic_model,
            max_retries=validation_retries,
            context=context,
            validator=validator,
            emit_fn=emit_fn,
        )

        parsed, attempts, error = pipeline.validate(content)

        # Track in report
        self._run_report.output.validated = True
        self._run_report.output.attempts = len(attempts)
        self._run_report.output.is_valid = error is None and parsed is not None
        self._run_report.output.final_error = str(error) if error else None

        return StructuredOutput(
            raw=content,
            parsed=parsed,
            _data=parsed.model_dump() if parsed else {},
            validation_attempts=attempts,
            final_error=error,
        )

    def _execute_tool(self, name: str, arguments: dict[str, Any]) -> str:
        for spec in self._tools:
            if spec.name == name:
                try:
                    result = spec.func(**arguments)
                    return str(result) if result is not None else ""
                except Exception as e:
                    raise ToolExecutionError(f"Tool {name!r} failed: {e}") from e
        raise ToolExecutionError(f"Unknown tool: {name!r}")

    async def execute_tool(self, name: str, arguments: dict[str, Any]) -> str:
        """Run a tool by name with the given arguments. For custom loops.

        Why: Built-in loops call this automatically. Use when implementing a
        custom Loop to execute tool calls.

        Args:
            name: Tool name (must match a @tool-decorated function).
            arguments: Dict of parameter names to values.

        Returns:
            Tool result as string. Raises ToolExecutionError on failure.

        Example:
            >>> result = await agent.execute_tool("search", {"query": "syrin"})
        """
        return self._execute_tool(name, arguments)

    def estimate_call_cost(
        self,
        messages: list[Any],
        max_output_tokens: int = 1024,
    ) -> float:
        """Estimate cost in USD for the next LLM call (best-effort).

        Uses model pricing and token counts from message contents. Actual cost may
        differ. Useful for pre-call budget checks.

        Args:
            messages: List of Message (role, content) to be sent.
            max_output_tokens: Assumed max completion tokens (default 1024).

        Returns:
            Estimated cost in USD.
        """
        if self._model_config is None:
            return 0.0
        pricing = getattr(self._model, "pricing", None) if self._model is not None else None
        if pricing is None and self._model is not None and hasattr(self._model, "get_pricing"):
            pricing = self._model.get_pricing()
        return estimate_cost_for_call(
            self._model_config.model_id,
            messages,
            max_output_tokens=max_output_tokens,
            pricing_override=pricing,
        )

    def _pre_call_budget_check(
        self,
        messages: list[Any],
        max_output_tokens: int = 1024,
    ) -> None:
        """If run budget would be exceeded after an estimated call, call on_exceeded and raise.

        Best-effort: uses estimated cost; actual cost may differ. Call before complete().
        Skipped when run limit is 0 (post-call check only).
        """
        if self._budget is None or self._budget.run is None:
            return
        effective_run = (
            (self._budget.run - self._budget.reserve)
            if self._budget.run > self._budget.reserve
            else self._budget.run
        )
        if effective_run is not None and effective_run <= 0:
            return
        estimate = self.estimate_call_cost(messages, max_output_tokens=max_output_tokens)
        run_usage = self._budget_tracker.run_usage_with_reserved
        if run_usage + estimate < effective_run:
            return
        on_exceeded = self._budget.on_exceeded
        limit = effective_run
        current = run_usage + estimate
        msg = (
            f"Budget would be exceeded: estimated run cost ${current:.4f} >= ${limit:.4f} "
            "(pre-call estimate)"
        )
        if on_exceeded is not None:
            ctx = BudgetExceededContext(
                current_cost=current,
                limit=limit,
                budget_type=BudgetLimitType.RUN,
                message=msg,
            )
            on_exceeded(ctx)
        raise BudgetExceededError(
            msg, current_cost=current, limit=limit, budget_type=BudgetLimitType.RUN.value
        )

    def _check_and_apply_budget(self) -> None:
        """Raise if budget or token limits exceeded; apply threshold actions (switch, warn). Stop raises."""
        if self._budget is None and self._token_limits is None:
            return
        result: CheckBudgetResult = self._budget_tracker.check_budget(
            self._budget, token_limits=self._token_limits, parent=self
        )
        if result.status == BudgetStatus.THRESHOLD:
            # Emit BUDGET_THRESHOLD hook and domain event for observability
            current = self._budget_tracker.current_run_cost
            limit = (
                (self._budget.run - self._budget.reserve)
                if self._budget is not None
                and self._budget.run is not None
                and self._budget.reserve is not None
                and self._budget.run > self._budget.reserve
                else (self._budget.run if self._budget and self._budget.run else 0.0)
            )
            pct = int((current / limit) * 100) if limit and limit > 0 else 0
            self._emit_event(
                Hook.BUDGET_THRESHOLD,
                EventContext(
                    threshold_percent=pct,
                    current_value=current,
                    limit_value=limit,
                    metric="cost",
                ),
            )
            return
        if result.status != BudgetStatus.EXCEEDED:
            return
        limit_key = result.exceeded_limit or BudgetLimitType.RUN
        on_exceeded = self._budget.on_exceeded if self._budget is not None else None
        if (
            (
                limit_key
                in (
                    BudgetLimitType.RUN_TOKENS,
                    BudgetLimitType.HOUR_TOKENS,
                    BudgetLimitType.DAY_TOKENS,
                    BudgetLimitType.WEEK_TOKENS,
                    BudgetLimitType.MONTH_TOKENS,
                )
                and self._token_limits is not None
                and self._token_limits.on_exceeded is not None
            )
            or on_exceeded is None
            and self._token_limits is not None
        ):
            on_exceeded = self._token_limits.on_exceeded
        if limit_key == BudgetLimitType.RUN:
            current = self._budget_tracker.current_run_cost
            if self._budget is None:
                limit = 0.0
            else:
                effective_run = (
                    (self._budget.run - self._budget.reserve)
                    if self._budget.run is not None and self._budget.run > self._budget.reserve
                    else self._budget.run
                )
                limit = effective_run or 0.0
            msg = f"Budget exceeded: run cost ${current:.4f} >= ${limit:.4f}"
        elif limit_key == BudgetLimitType.RUN_TOKENS:
            current = self._budget_tracker.current_run_tokens
            run_tok = self._token_limits.run if self._token_limits is not None else None
            limit = float(run_tok or 0)
            msg = f"Budget exceeded: run tokens {current} >= {int(limit)}"
        elif limit_key in (
            BudgetLimitType.HOUR_TOKENS,
            BudgetLimitType.DAY_TOKENS,
            BudgetLimitType.WEEK_TOKENS,
            BudgetLimitType.MONTH_TOKENS,
        ):
            token_per = self._token_limits.per if self._token_limits is not None else None
            if limit_key == BudgetLimitType.HOUR_TOKENS and token_per is not None:
                current = float(self._budget_tracker.hourly_tokens)
                limit = float(token_per.hour or 0)
            elif limit_key == BudgetLimitType.DAY_TOKENS and token_per is not None:
                current = float(self._budget_tracker.daily_tokens)
                limit = float(token_per.day or 0)
            elif limit_key == BudgetLimitType.WEEK_TOKENS and token_per is not None:
                current = float(self._budget_tracker.weekly_tokens)
                limit = float(token_per.week or 0)
            elif limit_key == BudgetLimitType.MONTH_TOKENS and token_per is not None:
                current = float(self._budget_tracker.monthly_tokens)
                limit = float(token_per.month or 0)
            else:
                current, limit = 0.0, 0.0
            msg = f"Budget exceeded: {limit_key.value} {int(current)} >= {int(limit)}"
        else:
            per = self._budget.per if self._budget else None
            if limit_key == BudgetLimitType.HOUR and per:
                current, limit = self._budget_tracker.hourly_cost, (per.hour or 0)
            elif limit_key == BudgetLimitType.DAY and per:
                current, limit = self._budget_tracker.daily_cost, (per.day or 0)
            elif limit_key == BudgetLimitType.WEEK and per:
                current, limit = self._budget_tracker.weekly_cost, (per.week or 0)
            elif limit_key == BudgetLimitType.MONTH and per:
                current, limit = self._budget_tracker.monthly_cost, (per.month or 0)
            else:
                current, limit = self._budget_tracker.current_run_cost, 0.0
            msg = f"Budget exceeded: {limit_key.value} cost ${current:.4f} >= ${limit:.4f}"
        if on_exceeded is not None:
            ctx = BudgetExceededContext(
                current_cost=current,
                limit=limit,
                budget_type=limit_key,
                message=msg,
            )
            on_exceeded(ctx)

    def _check_and_apply_rate_limit(self) -> None:
        """Check rate limits and apply threshold actions (switch model, wait, warn, stop).

        Called before each LLM request. This is the main integration point that makes
        rate limiting work automatically during agent execution.
        """
        if self._rate_limit_manager is None:
            return

        manager = self._rate_limit_manager

        allowed, reason = manager.check()

        # Track in report
        self._run_report.ratelimits.checks += 1
        if not allowed:
            self._run_report.ratelimits.exceeded = True
            _log.error("Rate limit exceeded: %s", reason)
            raise RuntimeError(f"Rate limit exceeded: {reason}")

        # Threshold actions (wait, stop, switch model, etc.) are run by the manager
        # in _check_thresholds() via each threshold's action(ctx) callback. The user
        # implements desired behavior in that callback (e.g. raise, time.sleep, switch_model).

    def _record_rate_limit_usage(self, token_usage: TokenUsage) -> None:
        """Record token usage and re-check rate limits after LLM call.

        Called after each LLM request to track usage and trigger threshold checks
        for the next iteration.
        """
        if self._rate_limit_manager is None:
            return

        self._rate_limit_manager.record(tokens_used=token_usage.total_tokens)

    def _record_cost(self, token_usage: TokenUsage, model_id: str) -> None:
        """Compute cost, build CostInfo, record on tracker, sync Budget._spent, then re-check thresholds."""
        pricing = getattr(self._model, "pricing", None) if self._model is not None else None
        cost_usd = calculate_cost(model_id, token_usage, pricing_override=pricing)
        cost_info = CostInfo(
            token_usage=token_usage,
            cost_usd=cost_usd,
            model_name=model_id,
        )
        self._record_cost_info(cost_info)

    def _make_budget_consume_callback(self) -> Callable[[float], None]:
        """Return a callback for Budget.consume() so guardrails can record cost."""

        def _consume(amount: float) -> None:
            model_id = (
                self._model_config.model_id
                if hasattr(self, "_model_config") and self._model_config
                else "unknown"
            )
            self._record_cost_info(CostInfo(cost_usd=amount, model_name=model_id))

        return _consume

    def _record_cost_info(self, cost_info: CostInfo) -> None:
        """Record a CostInfo (e.g. from streaming). Syncs spent and checks budget."""
        self._budget_tracker.record(cost_info)
        if self._budget is not None:
            self._budget._set_spent(self._budget_tracker.current_run_cost)
        if self._budget_store is not None:
            self._budget_store.save(self._budget_store_key, self._budget_tracker)
        self._check_and_apply_budget()

    async def _complete_async(
        self, messages: list[Message], tools: list[ToolSpec] | None
    ) -> ProviderResponse:
        """Async internal method to call provider."""
        return await self._provider.complete(
            messages=messages, model=self._model_config, tools=tools
        )

    async def complete(
        self, messages: list[Message], tools: list[ToolSpec] | None = None
    ) -> ProviderResponse:
        """Call the LLM with messages and optional tools. For custom loops.

        Why: Built-in loops use this internally. Override or use in a custom Loop
        when you need full control over the LLM call (e.g. custom batching).

        Args:
            messages: List of Message (role, content).
            tools: Optional tool specs. None = no tools.

        Returns:
            ProviderResponse (content, tool_calls, token_usage, etc.).
        """
        return await self._complete_async(messages, tools)

    def _with_context_on_response(self, r: Response[str]) -> Response[str]:
        """Attach per-call context_stats and context to a Response."""
        r.context_stats = getattr(self._context, "stats", None)
        r.context = getattr(self, "_call_context", None) or (
            getattr(self._context, "context", None) if hasattr(self._context, "context") else None
        )
        return r

    async def _run_loop_response_async(self, user_input: str) -> Response[str]:
        """Run using the configured loop strategy with full observability (async)."""
        from syrin.agent._run import run_agent_loop_async

        return await run_agent_loop_async(self, user_input)

    def _run_loop_response(self, user_input: str) -> Response[str]:
        """Run using the configured loop strategy (sync wrapper)."""
        return _get_agent_loop().run_until_complete(self._run_loop_response_async(user_input))

    def _stream_response(self, user_input: str) -> Iterator[StreamChunk]:
        """Stream response chunks synchronously. Records cost per chunk and checks budget mid-stream."""
        messages = self._build_messages(user_input)
        tools = self._tools if self._tools else None
        accumulated = ""
        total_cost = 0.0
        total_tokens = TokenUsage()
        prev_cost = 0.0
        prev_tokens = TokenUsage()

        try:
            for chunk in self._provider.stream_sync(messages, self._model_config, tools):
                content = chunk.content or ""
                accumulated += content
                total_cost += chunk.cost_usd if hasattr(chunk, "cost_usd") else 0.0
                total_tokens = TokenUsage(
                    input_tokens=total_tokens.input_tokens
                    + (chunk.token_usage.input_tokens if hasattr(chunk, "token_usage") else 0),
                    output_tokens=total_tokens.output_tokens
                    + (chunk.token_usage.output_tokens if hasattr(chunk, "token_usage") else 0),
                    total_tokens=total_tokens.total_tokens
                    + (chunk.token_usage.total_tokens if hasattr(chunk, "token_usage") else 0),
                )
                if self._budget is not None or self._token_limits is not None:
                    delta_cost = total_cost - prev_cost
                    delta_tokens = TokenUsage(
                        input_tokens=total_tokens.input_tokens - prev_tokens.input_tokens,
                        output_tokens=total_tokens.output_tokens - prev_tokens.output_tokens,
                        total_tokens=total_tokens.total_tokens - prev_tokens.total_tokens,
                    )
                    if delta_cost > 0 or delta_tokens.total_tokens > 0:
                        cost_info = CostInfo(
                            cost_usd=delta_cost,
                            token_usage=delta_tokens,
                            model_name=self._model_config.model_id,
                        )
                        self._budget_tracker.record(cost_info)
                        if self._budget is not None:
                            self._budget._set_spent(self._budget_tracker.current_run_cost)
                        if self._budget_store is not None:
                            self._budget_store.save(self._budget_store_key, self._budget_tracker)
                yield StreamChunk(
                    text=content,
                    accumulated_text=accumulated,
                    cost_so_far=total_cost,
                    tokens_so_far=total_tokens,
                )
                if self._budget is not None or self._token_limits is not None:
                    self._check_and_apply_budget()
                prev_cost = total_cost
                prev_tokens = total_tokens
        except (BudgetExceededError, BudgetThresholdError):
            raise
        except Exception as e:
            raise ToolExecutionError(f"Streaming failed: {e}") from e

    def response(self, user_input: str, context: Context | None = None) -> Response[str]:
        """Run the agent: LLM completion + tool loop. Synchronous.

        Why: Main entry point for getting a reply. Runs the configured loop
        (REACT by default), runs guardrails, records cost, applies budget/rate
        limits. Blocks until complete.

        Args:
            user_input: User message.
            context: Optional Context for this call only. When set, overrides the agent's
                default context (max_tokens, reserve, thresholds, budget). The Context
                used for this call is on ``result.context``; per-call stats on ``result.context_stats``.

        Returns:
            Response with content, cost, tokens, model, stop_reason, structured
            output (if output= set), and report.

        Example:
            >>> r = agent.response("What is 2+2?")
            >>> r = agent.response("Long task...", context=Context(max_tokens=4000))
        """
        self._call_context = context
        try:
            self._run_report = AgentReport()
            if self._budget is not None or self._token_limits is not None:
                self._budget_tracker.reset_run()
                if self._budget is not None:
                    self._budget._set_spent(0)
            try:
                return self._run_loop_response(user_input)
            except (BudgetThresholdError, BudgetExceededError):
                self._maybe_checkpoint("budget")
                raise
            except Exception:
                self._maybe_checkpoint("error")
                raise
        finally:
            self._call_context = None

    async def arun(self, user_input: str, context: Context | None = None) -> Response[str]:
        """Run the agent asynchronously. Same as response() but non-blocking.

        Why: Use in async apps to avoid blocking the event loop. Same behavior
        as response() (guardrails, budget, tools, etc.).

        Args:
            user_input: User message.
            context: Optional Context for this call only (see response()). Overrides
                the agent's context for this call. Used context is on ``result.context``;
                per-call stats on ``result.context_stats``.

        Returns:
            Response (same as response()).

        Example:
            >>> r = await agent.arun("Summarize this")
        """
        self._call_context = context
        try:
            self._run_report = AgentReport()
            if self._budget is not None or self._token_limits is not None:
                self._budget_tracker.reset_run()
                if self._budget is not None:
                    self._budget._set_spent(0)
            try:
                return await self._run_loop_response_async(user_input)
            except (BudgetThresholdError, BudgetExceededError):
                self._maybe_checkpoint("budget")
                raise
            except Exception:
                self._maybe_checkpoint("error")
                raise
        finally:
            self._call_context = None

    def stream(self, user_input: str, context: Context | None = None) -> Iterator[StreamChunk]:
        """Stream response text as it arrives. Synchronous iterator.

        Why: Show tokens in real time (e.g. ChatGPT-style UI). No tool-call loop;
        single completion only.

        Args:
            user_input: User message.
            context: Optional Context for this call only (see response()). Overrides agent's context.

        Yields:
            StreamChunk with text (delta), accumulated_text, cost_so_far,
            tokens_so_far.

        Note:
            Stream does not return a Response; for context stats for this run,
            read ``agent.context_stats`` after the stream completes.

        Example:
            >>> for chunk in agent.stream("Write a poem"):
            ...     print(chunk.text, end="")
        """
        self._call_context = context
        try:
            if self._budget is not None or self._token_limits is not None:
                self._budget_tracker.reset_run()
                if self._budget is not None:
                    self._budget._set_spent(0)
            yield from self._stream_response(user_input)
        finally:
            self._call_context = None

    async def astream(
        self, user_input: str, context: Context | None = None
    ) -> AsyncIterator[StreamChunk]:
        """Stream response text as it arrives. Async iterator.

        Why: Non-blocking streaming for async apps. Same chunks as stream().

        Args:
            user_input: User message.
            context: Optional Context for this call only (see response()). Overrides agent's context.

        Note:
            Astream does not return a Response; for context stats for this run,
            read ``agent.context_stats`` after the stream completes.

        Example:
            >>> async for chunk in agent.astream("Write a poem"):
            ...     print(chunk.text, end="")
        """
        self._call_context = context
        try:
            if self._budget is not None or self._token_limits is not None:
                self._budget_tracker.reset_run()
                if self._budget is not None:
                    self._budget._set_spent(0)
            messages = self._build_messages(user_input)
            tools = self._tools if self._tools else None
            accumulated = ""
            total_cost = 0.0
            total_tokens = TokenUsage()
            prev_cost = 0.0
            prev_tokens = TokenUsage()

            try:
                async for chunk in self._provider.stream(messages, self._model_config, tools):
                    content = chunk.content or ""
                    accumulated += content
                    total_cost += chunk.cost_usd if hasattr(chunk, "cost_usd") else 0.0
                    total_tokens = TokenUsage(
                        input_tokens=total_tokens.input_tokens
                        + (chunk.token_usage.input_tokens if hasattr(chunk, "token_usage") else 0),
                        output_tokens=total_tokens.output_tokens
                        + (chunk.token_usage.output_tokens if hasattr(chunk, "token_usage") else 0),
                        total_tokens=total_tokens.total_tokens
                        + (chunk.token_usage.total_tokens if hasattr(chunk, "token_usage") else 0),
                    )
                    if self._budget is not None or self._token_limits is not None:
                        delta_cost = total_cost - prev_cost
                        delta_tokens = TokenUsage(
                            input_tokens=total_tokens.input_tokens - prev_tokens.input_tokens,
                            output_tokens=total_tokens.output_tokens - prev_tokens.output_tokens,
                            total_tokens=total_tokens.total_tokens - prev_tokens.total_tokens,
                        )
                        if delta_cost > 0 or delta_tokens.total_tokens > 0:
                            cost_info = CostInfo(
                                cost_usd=delta_cost,
                                token_usage=delta_tokens,
                                model_name=self._model_config.model_id,
                            )
                            self._budget_tracker.record(cost_info)
                            if self._budget is not None:
                                self._budget._set_spent(self._budget_tracker.current_run_cost)
                            if self._budget_store is not None:
                                self._budget_store.save(
                                    self._budget_store_key, self._budget_tracker
                                )
                    yield StreamChunk(
                        text=content,
                        accumulated_text=accumulated,
                        cost_so_far=total_cost,
                        tokens_so_far=total_tokens,
                    )
                    if self._budget is not None or self._token_limits is not None:
                        self._check_and_apply_budget()
                    prev_cost = total_cost
                    prev_tokens = total_tokens
            except (BudgetExceededError, BudgetThresholdError):
                raise
            except Exception as e:
                raise ToolExecutionError(f"Streaming failed: {e}") from e
        finally:
            self._call_context = None


# Presets and builder
from syrin.agent import presets as _presets
from syrin.agent.builder import AgentBuilder as _AgentBuilder

Agent.presets = _presets  # type: ignore[attr-defined]
Agent.builder = staticmethod(lambda model: _AgentBuilder(model))  # type: ignore[attr-defined]
