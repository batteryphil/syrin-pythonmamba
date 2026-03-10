"""Agent base class and response loop with tool execution and budget."""

from __future__ import annotations

import logging
import time
from collections.abc import AsyncIterator, Callable, Iterator
from typing import TYPE_CHECKING, Any, ClassVar, cast

if TYPE_CHECKING:
    from syrin.serve.config import ServeConfig  # noqa: F401

from syrin._sentinel import NOT_PROVIDED
from syrin.agent._budget_ops import (
    check_and_apply_budget as _budget_check_and_apply,
)
from syrin.agent._budget_ops import (
    make_budget_consume_callback as _budget_make_consume_callback,
)
from syrin.agent._budget_ops import (
    pre_call_budget_check as _budget_pre_call_check,
)
from syrin.agent._budget_ops import (
    record_cost as _budget_record_cost,
)
from syrin.agent._budget_ops import (
    record_cost_info as _budget_record_cost_info,
)
from syrin.agent._checkpoint import (
    get_checkpoint_report as _checkpoint_get_report,
)
from syrin.agent._checkpoint import (
    list_checkpoints as _checkpoint_list,
)
from syrin.agent._checkpoint import (
    load_checkpoint as _checkpoint_load,
)
from syrin.agent._checkpoint import (
    maybe_checkpoint as _checkpoint_maybe,
)
from syrin.agent._checkpoint import (
    save_checkpoint as _checkpoint_save,
)
from syrin.agent._components import (
    AgentBudgetComponent,
    AgentContextComponent,
    AgentGuardrailsComponent,
    AgentMemoryComponent,
    AgentObservabilityComponent,
)
from syrin.agent._events import print_event as _events_print
from syrin.agent._guardrails import run_guardrails as _guardrails_run
from syrin.agent._handoff import handoff as _handoff_impl
from syrin.agent._helpers import (
    _AgentRuntime,
    _collect_class_tools,
    _collect_system_prompt_method,
    _ContextFacade,
    _emit_domain_event_for_hook,
    _get_system_prompt_method_names,
    _is_mcp,
    _is_valid_system_prompt,
    _make_generate_image_tool,
    _make_generate_video_tool,
    _make_search_knowledge_deep_tool,
    _make_search_knowledge_tool,
    _make_verify_knowledge_tool,
    _merge_class_attrs,
    _normalize_tools,
    _resolve_memory,
    _resolve_provider,
    _validate_agent_media,
    _validate_budget,
    _validate_user_input,
)
from syrin.agent._memory_api import (
    forget as _memory_forget,
)
from syrin.agent._memory_api import (
    recall as _memory_recall,
)
from syrin.agent._memory_api import (
    remember as _memory_remember,
)
from syrin.agent._model_provider import (
    resolve_fallback_provider as _model_resolve_fallback,
)
from syrin.agent._model_provider import (
    switch_model as _model_switch,
)
from syrin.agent._prompt_build import (
    build_messages as _prompt_build_messages,
)
from syrin.agent._prompt_build import (
    build_output as _prompt_build_output,
)
from syrin.agent._prompt_build import (
    effective_template_variables as _prompt_effective_template_variables,
)
from syrin.agent._prompt_build import (
    get_prompt_builtins as _prompt_get_builtins,
)
from syrin.agent._prompt_build import (
    resolve_system_prompt as _prompt_resolve_system_prompt,
)
from syrin.agent._rate_limit import (
    check_and_apply_rate_limit as _rate_limit_check,
)
from syrin.agent._rate_limit import (
    record_rate_limit_usage as _rate_limit_record,
)
from syrin.agent._remote_config import (
    agent_guardrails_schema_and_values as _agent_guardrails_schema_and_values,
)
from syrin.agent._remote_config import (
    agent_mcp_schema_and_values as _agent_mcp_schema_and_values,
)
from syrin.agent._remote_config import (
    agent_template_vars_schema_and_values as _agent_template_vars_schema_and_values,
)
from syrin.agent._remote_config import (
    agent_tools_schema_and_values as _agent_tools_schema_and_values,
)
from syrin.agent._remote_config import (
    apply_guardrails_overrides as _apply_guardrails_overrides,
)
from syrin.agent._remote_config import (
    apply_mcp_overrides as _apply_mcp_overrides,
)
from syrin.agent._remote_config import (
    apply_template_vars_overrides as _apply_template_vars_overrides,
)
from syrin.agent._remote_config import (
    apply_tools_overrides as _apply_tools_overrides,
)
from syrin.agent._response_context import (
    record_conversation_turn as _response_record_conversation_turn,
)
from syrin.agent._response_context import (
    with_context_on_response as _response_with_context,
)
from syrin.agent._run_context import DefaultAgentRunContext
from syrin.agent._spawn import (
    spawn as _spawn_impl,
)
from syrin.agent._spawn import (
    spawn_parallel as _spawn_parallel_impl,
)
from syrin.agent._spawn import (
    update_parent_budget as _spawn_update_parent_budget,
)
from syrin.agent._tool_exec import execute_tool as _tool_execute
from syrin.agent.config import AgentConfig
from syrin.audit import AuditHookHandler, AuditLog
from syrin.budget import (
    Budget,
    BudgetState,
    BudgetTracker,
)
from syrin.budget_store import BudgetStore
from syrin.checkpoint import CheckpointConfig, Checkpointer
from syrin.circuit import CircuitBreaker
from syrin.context import Context, ContextConfig, DefaultContextManager
from syrin.context.config import ContextStats
from syrin.cost import calculate_cost, estimate_cost_for_call
from syrin.domain_events import EventBus
from syrin.enums import (
    CircuitState,
    GuardrailStage,
    Hook,
    LoopStrategy,
    Media,
    MemoryPreset,
    MemoryType,
)
from syrin.events import EventContext, Events
from syrin.exceptions import (
    BudgetExceededError,
    BudgetThresholdError,
    CircuitBreakerOpenError,
    ToolExecutionError,
)
from syrin.guardrails import Guardrail, GuardrailChain, GuardrailResult
from syrin.hitl import ApprovalGate
from syrin.loop import Loop, LoopStrategyMapping, ReactLoop
from syrin.memory import Memory
from syrin.memory.backends import InMemoryBackend
from syrin.memory.config import MemoryEntry
from syrin.model import Model
from syrin.observability import (
    ConsoleExporter,
    SpanKind,
    Tracer,
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
from syrin.router import ModelRouter, RoutingConfig
from syrin.router.agent_integration import build_router_from_models
from syrin.serve.servable import Servable
from syrin.tool import ToolSpec
from syrin.types import CostInfo, Message, ModelConfig, ProviderResponse, TokenUsage

DEFAULT_MAX_TOOL_ITERATIONS = 10
_log = logging.getLogger(__name__)


class _AgentMeta(type):
    """Metaclass that moves name/description to internal attrs so instance property is not shadowed."""

    def __new__(
        mcs,
        name: str,
        bases: tuple[type, ...],
        namespace: dict[str, Any],
        **kwargs: Any,
    ) -> type:
        for attr, internal in (
            ("name", "_syrin_default_name"),
            ("description", "_syrin_default_description"),
        ):
            if attr in namespace:
                val = namespace[attr]
                if not hasattr(val, "__get__"):  # Not a descriptor/property
                    if attr == "name":
                        namespace[internal] = val if isinstance(val, str) else None
                    else:
                        namespace[internal] = val if isinstance(val, str) else ""
                    del namespace[attr]
        # Move "tools" to _syrin_class_tools so subclasses don't shadow Agent.tools property.
        # __init_subclass__ merges from both "tools" and "_syrin_class_tools".
        if "tools" in namespace and not hasattr(namespace["tools"], "__get__"):
            namespace["_syrin_class_tools"] = namespace.pop("tools")
        # Type-checker-friendly ClassVars: copy into internal so __init__ gets them
        if "_agent_name" in namespace:
            val = namespace["_agent_name"]
            if not hasattr(val, "__get__") and isinstance(val, str):
                namespace["_syrin_default_name"] = val
            elif not hasattr(val, "__get__") and val is None:
                namespace["_syrin_default_name"] = None
        if "_agent_description" in namespace:
            val = namespace["_agent_description"]
            if not hasattr(val, "__get__") and isinstance(val, str):
                namespace["_syrin_default_description"] = val
        return super().__new__(mcs, name, bases, namespace, **kwargs)


class Agent(Servable, metaclass=_AgentMeta):
    """AI agent that runs completions, tools, memory, and budget control.

    An Agent is the main interface for talking to an LLM, executing tools, remembering
    facts, and controlling costs. You provide a model (LLM) and optionally tools,
    budget, memory, guardrails, and more.

    Main methods:
        response(user_input) — Sync run; returns Response.
        arun(user_input) — Async run; returns Response.
        stream(user_input) / astream(user_input) — Streaming.
        estimate_cost(messages, max_output_tokens=...) — Estimate USD before calling.
        budget_state — Current budget (limit, remaining, spent, percent_used) or None.
        tools — List of ToolSpec (read-only). model_config — Current ModelConfig or None.

    Why use an Agent?
        - Run multi-turn conversations with automatic tool-call loops (REACT by default).
        - Keep costs under control with per-run and per-period budgets.
        - Remember facts across sessions with persistent memory.
        - Validate input/output with guardrails.
        - Trace and debug with events and spans.

    How to create one:
        - Recommended: ``Agent.builder(model).with_system_prompt(...).with_budget(...).build()``
        - Or presets: ``Agent.basic(model)``, ``Agent.with_memory(model)``
        - Or constructor: ``Agent(model=..., system_prompt=...)``
        - Or subclass and set class attributes: ``model = Model.OpenAI(...)``

    Subclass attributes (set on your Agent subclass; override parent defaults):
        model: Model | None — LLM to use (Model.OpenAI, Model.Anthropic, etc.). Required.
        system_prompt: str — Instructions sent with every request. Default: "".
        name: str | None — Agent identifier for handoffs, discovery. Default: None.
        description: str — Human-readable description (metaclass moves to internal). Default: "".
        _agent_name: ClassVar[str | None] — Same as name; use this to avoid type-checker override warnings.
        Name precedence: constructor name > class _agent_name/name > cls.__name__.lower()
        _agent_description: ClassVar[str] — Same as description; use this to avoid type-checker override warnings.
        tools: list[ToolSpec] — Tools the agent can call. Merged with parent. Default: [].
        budget: Budget | None — Cost limits (run, per-period). Default: None (unlimited).
        memory: Memory | None — Persistent memory config. Default: None.
        guardrails: list[Guardrail] — Input/output guardrails. Merged with parent. Default: [].
        context: Context | None — Context window config. Default: None.
        checkpoint: CheckpointConfig | None — State checkpoint config. Default: None.
        template_variables: dict[str, Any] — Template vars for system prompt (e.g. {"user_name": "Alice"}).
                Merge with inject_template_vars ({date}, {agent_id}, {conversation_id}). Default: {}.

    Instance attributes (read after creation):
        events: Lifecycle hooks. Use agent.events.on(Hook.LLM_REQUEST_END, fn).
        budget_state: BudgetState | None — Current budget state when budget configured.

        Internal: _runtime holds remote config state. Not part of public API.

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

    _syrin_default_model: Model | ModelConfig | None = None
    _syrin_default_memory: Memory | None = None
    _syrin_default_system_prompt: str | Any = ""
    _syrin_system_prompt_method: Any = None  # @system_prompt method if present
    _syrin_default_template_vars: dict[str, Any] = ()  # type: ignore[assignment]
    _syrin_default_tools: list[ToolSpec] = []
    _syrin_default_budget: Budget | None = None
    _syrin_default_guardrails: list[Guardrail] = []
    _syrin_default_name: str | None = None
    _syrin_default_description: str = ""
    _agent_name: ClassVar[str | None] = None
    _agent_description: ClassVar[str] = ""

    # Section key -> attr or path for RemoteConfigurable; None = self (agent section)
    REMOTE_CONFIG_SECTIONS: ClassVar[dict[str, str | tuple[str, ...] | None]] = {
        "agent": None,
        "budget": "_budget",
        "memory": "_persistent_memory",
        "context": ("_context", "context"),
        "checkpoint": "_checkpoint_config",
        "rate_limit": ("_rate_limit_manager", "config"),
        "circuit_breaker": "_circuit_breaker",
        "output": "_output",
        "guardrails": None,
        "template_variables": None,
        "tools": None,
        "mcp": None,
    }

    def get_remote_config_schema(self, section_key: str) -> tuple[Any, dict[str, object]]:
        """RemoteConfigurable: return (schema, current_values) for agent-owned sections."""
        from syrin.remote._schema import get_agent_section_schema_and_values
        from syrin.remote._types import ConfigSchema

        if section_key == "agent":
            return get_agent_section_schema_and_values(self)
        if section_key == "guardrails":
            return _agent_guardrails_schema_and_values(self)
        if section_key == "template_variables":
            return _agent_template_vars_schema_and_values(self)
        if section_key == "tools":
            return _agent_tools_schema_and_values(self)
        if section_key == "mcp":
            return _agent_mcp_schema_and_values(self)
        return (ConfigSchema(section=section_key, class_name="Agent", fields=[]), {})

    def apply_remote_overrides(
        self,
        agent: Any,
        pairs: list[tuple[str, object]],
        section_schema: Any,
    ) -> None:
        """RemoteConfigurable: apply overrides for agent-owned sections."""
        from syrin.remote._resolver_helpers import apply_agent_section_overrides

        section = getattr(section_schema, "section", None)
        if section == "agent":
            apply_agent_section_overrides(agent, pairs, section_schema)
            return
        if section == "guardrails":
            _apply_guardrails_overrides(agent, pairs)
            return
        if section == "template_variables":
            _apply_template_vars_overrides(agent, pairs)
            return
        if section == "tools":
            _apply_tools_overrides(agent, pairs)
            return
        if section == "mcp":
            _apply_mcp_overrides(agent, pairs)
            return

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        mro = cls.__mro__
        default_model = _merge_class_attrs(mro, "model", merge=False)
        default_prompt = _merge_class_attrs(mro, "system_prompt", merge=False)
        default_tools = _merge_class_attrs(mro, "tools", merge=True)
        default_budget = _merge_class_attrs(mro, "budget", merge=False)
        default_guardrails = _merge_class_attrs(mro, "guardrails", merge=True)
        default_memory = _merge_class_attrs(mro, "memory", merge=False)
        default_name = _merge_class_attrs(mro, "_agent_name", merge=False)
        if default_name is NOT_PROVIDED:
            default_name = _merge_class_attrs(mro, "name", merge=False)
        default_description = _merge_class_attrs(mro, "_agent_description", merge=False)
        if default_description is NOT_PROVIDED:
            default_description = _merge_class_attrs(mro, "description", merge=False)
        cls._syrin_default_model = default_model if default_model is not NOT_PROVIDED else None
        # Keep NOT_PROVIDED when no class sets memory so __init__ can default to Memory
        cls._syrin_default_memory = default_memory
        method_names = _get_system_prompt_method_names(cls)
        if len(method_names) > 1:
            names_str = ", ".join(f"'{n}'" for n in method_names)
            raise ValueError(
                f"Agent class {cls.__name__!r} has multiple @system_prompt methods "
                f"(only one allowed): {names_str}. Remove the extras or merge them "
                "into a single @system_prompt method."
            )
        cls._syrin_system_prompt_method = _collect_system_prompt_method(cls)
        cls._syrin_default_system_prompt = (
            default_prompt if default_prompt is not NOT_PROVIDED else ""
        )
        merged_template_vars: dict[str, Any] = {}
        for c in mro:
            if c is object:
                continue
            tv = c.__dict__.get("template_variables", NOT_PROVIDED)
            if tv is not NOT_PROVIDED and isinstance(tv, dict):
                merged_template_vars = {**merged_template_vars, **tv}
        cls._syrin_default_template_vars = merged_template_vars
        # Merge: class @tool methods first, then explicit tools. Explicit overrides by name.
        # MCP and MCPClient kept for init-time expansion; MCP also for co-location.
        class_tools = _collect_class_tools(cls)
        explicit_list = list(default_tools) if default_tools is not NOT_PROVIDED else []
        by_name: dict[str, ToolSpec] = {t.name: t for t in class_tools}
        mcp_sources: list[Any] = []
        for t in explicit_list:
            if isinstance(t, ToolSpec):
                by_name[t.name] = t
            elif isinstance(t, list):
                for s in t:
                    if isinstance(s, ToolSpec):
                        by_name[s.name] = s
            elif hasattr(t, "tools") and callable(getattr(t, "tools", None)):
                mcp_sources.append(t)
        cls._syrin_default_tools = list(by_name.values()) + mcp_sources
        cls._syrin_default_budget = default_budget if default_budget is not NOT_PROVIDED else None
        cls._syrin_default_guardrails = (
            list(default_guardrails) if default_guardrails is not NOT_PROVIDED else []
        )
        if default_name is not NOT_PROVIDED and isinstance(default_name, str):
            cls._syrin_default_name = default_name
        elif default_name is NOT_PROVIDED and "_syrin_default_name" not in cls.__dict__:
            cls._syrin_default_name = None
        if default_description is not NOT_PROVIDED:
            cls._syrin_default_description = default_description
        elif (
            default_description is NOT_PROVIDED and "_syrin_default_description" not in cls.__dict__
        ):
            cls._syrin_default_description = ""

    def __init__(
        self,
        model: Model | ModelConfig | None = NOT_PROVIDED,
        system_prompt: str | None = NOT_PROVIDED,
        tools: list[ToolSpec] | None = NOT_PROVIDED,
        budget: Budget | None = NOT_PROVIDED,
        *,
        output: Output | None = None,
        max_tool_iterations: int = DEFAULT_MAX_TOOL_ITERATIONS,
        budget_store: BudgetStore | None = None,
        budget_store_key: str = "default",
        memory: Memory | MemoryPreset | None = NOT_PROVIDED,
        loop_strategy: LoopStrategy = LoopStrategy.REACT,
        custom_loop: Loop | type[Loop] | None = None,
        guardrails: list[Guardrail] | GuardrailChain | None = NOT_PROVIDED,
        human_approval_timeout: int = 300,
        max_tool_result_length: int = 2000,
        retry_on_transient: bool = True,
        max_retries: int = 3,
        retry_backoff_base: float = 1.0,
        debug: bool = False,
        name: str | None = NOT_PROVIDED,
        description: str | None = NOT_PROVIDED,
        template_variables: dict[str, Any] | None = None,
        inject_template_vars: bool = True,
        max_child_agents: int | None = None,
        config: AgentConfig | None = None,
        model_router: ModelRouter | RoutingConfig | None = None,
        input_media: set[Media] | None = None,
        output_media: set[Media] | None = None,
        input_file_rules: Any = None,
        image_generation: Any = None,
        video_generation: Any = None,
        knowledge: object | None = None,
    ) -> None:
        """Create an agent with model, prompt, tools, and optional config.

        **90% of users need only 3 parameters:**
            >>> agent = Agent(model=claude, system_prompt="You are helpful.", tools=[search])

        **Required:**
            model: LLM to use. Model.OpenAI, Model.Anthropic, etc. The brain of your agent.

        **Essential:**
            system_prompt: Instructions that define behavior. Sent with every request. Default: empty.
            tools: List of @tool-decorated functions the agent can call. Default: [].

        **Cost control:**
            budget: Cost limits (per run, per period) and threshold actions. Use Budget(run=1.0) for $1/run.
            budget_store: Persist budget across runs (e.g. FileBudgetStore).
            budget_store_key: Key for budget persistence (default "default"). Isolate per user/session.

        **Memory:**
            memory: Memory for conversation and optional persistent recall.
                memory=None or MemoryPreset.DISABLED: no memory (stateless).
                memory=MemoryPreset.DEFAULT: core+episodic, top_k=10.
                memory=Memory(): full config.

        **Routing:**
            model_router: RoutingConfig or ModelRouter for model selection when using multiple models.
            input_media: Media types this agent accepts from users (e.g. {Media.TEXT, Media.IMAGE}).
                Validated against model profiles; router only considers models whose input_media >= this.
            output_media: Media types this agent can produce. {Media.IMAGE} enables generate_image tool.

        **Advanced:**
            output: Structured output config (Pydantic model). Validates responses.
            max_tool_iterations: Max tool-call loops per response (default 10).
            loop_strategy: REACT (tool loop) or SINGLE_SHOT. Ignored when custom_loop is set.
            custom_loop: Custom Loop instance or class. Overrides loop_strategy when provided.
                Use only when you implement your own Loop (e.g. HumanInTheLoop); for built-in
                behavior use loop_strategy=LoopStrategy.REACT or LoopStrategy.SINGLE_SHOT.
            guardrails: List of Guardrail or GuardrailChain. Validate input/output.
                Why: Block harmful content, PII, or policy violations.
                When: Production agents handling user input or regulated domains.
            debug: If True, print lifecycle events to console.
                Why: Quick visibility into agent behavior.
                When: Development and debugging.
            human_approval_timeout: Seconds to wait for HITL approval. On timeout, reject. Default 300.
            max_tool_result_length: Max chars for tool results before truncation (default 2000).
            retry_on_transient: Retry tool calls on transient errors (429, 503, timeouts). Default True.
            max_retries: Max retries for transient tool failures (default 3).
            retry_backoff_base: Base delay in seconds for exponential backoff (default 1.0).
            config: AgentConfig for advanced options (context, rate_limit, checkpoint, etc.).
                Use config=AgentConfig(context=Context(max_tokens=100_000)) for context window
                and compaction; prevents unbounded message growth in long conversations.
            template_variables: Template vars for system prompts (e.g. {"user_name": "Alice"}).
                Merged with inject_template_vars ({date}, {agent_id}, {conversation_id}).
            inject_template_vars: If True (default), inject {date}, {agent_id}, {conversation_id}
                into system prompts. Set False if you don't use them.
            max_child_agents: Cap on concurrent child agents when using spawn().
                When exceeded, spawn() raises RuntimeError. Default: 10.
            input_media/output_media: See Routing above.
            input_file_rules: When Media.FILE is in input_media, allowed MIME types and max size.
                Use InputFileRules(allowed_mime_types=[...], max_size_mb=10).
            image_generation: Explicit ImageGenerator for image generation. When set, used instead
                of auto-created default (output_media + API key). Use for custom providers or config.
            video_generation: Explicit VideoGenerator for video generation. When set, used instead
                of auto-created default (output_media + API key). Use for custom providers or config.
            knowledge: Knowledge instance for RAG. Adds search_knowledge tool. Requires embedding
                and sources. Lazy ingest on first search.

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
        if model is NOT_PROVIDED:
            model = getattr(cls, "_syrin_default_model", None)
        if system_prompt is NOT_PROVIDED:
            system_prompt = getattr(cls, "_syrin_default_system_prompt", "") or ""
        # Merge class tools (@tool methods + class tools=[]) with constructor tools (later overrides by name)
        base_tools = getattr(cls, "_syrin_default_tools", None) or []
        if tools is NOT_PROVIDED:
            tools = base_tools
        else:
            if not isinstance(tools, list):
                raise TypeError(
                    f"tools must be list of ToolSpec or None, got {type(tools).__name__}. "
                    "Use @syrin.tool or syrin.tool() to create tools."
                )
            for i, x in enumerate(tools):
                if isinstance(x, ToolSpec) or (
                    isinstance(x, list) and all(isinstance(t, ToolSpec) for t in x)
                ):
                    continue
                if _is_mcp(x) or (hasattr(x, "tools") and hasattr(x, "_url")):
                    continue
                raise TypeError(
                    f"tools[{i}] must be ToolSpec, list of ToolSpec, MCP, or MCPClient, got {type(x).__name__}. "
                    "Use @syrin.tool or syrin.tool() to create tools."
                )
            by_name = {t.name: t for t in base_tools if isinstance(t, ToolSpec)}
            for t in tools:
                if isinstance(t, ToolSpec):
                    by_name[t.name] = t
            tools = list(by_name.values())
        if budget is NOT_PROVIDED:
            budget = getattr(cls, "_syrin_default_budget", None)
        if guardrails is NOT_PROVIDED:
            guardrails = getattr(cls, "_syrin_default_guardrails", None) or []
        if memory is NOT_PROVIDED:
            class_mem = getattr(cls, "_syrin_default_memory", NOT_PROVIDED)
            memory = Memory() if class_mem is NOT_PROVIDED or class_mem is None else class_mem
        if name is NOT_PROVIDED:
            name = getattr(cls, "_syrin_default_name", None)
        if description is NOT_PROVIDED:
            description = getattr(cls, "_syrin_default_description", "") or ""
        if name is None:
            name = cls.__name__.lower()
        if description is None:
            description = ""
        if not isinstance(name, str):
            raise TypeError(
                f"name must be str, got {type(name).__name__}. Example: name='product-agent'"
            )
        if not isinstance(description, str):
            raise TypeError(
                f"description must be str, got {type(description).__name__}. "
                "Example: description='E-commerce product assistant'"
            )
        if not isinstance(max_tool_iterations, int):
            raise TypeError(
                f"max_tool_iterations must be int, got {type(max_tool_iterations).__name__}. "
                "Example: max_tool_iterations=10"
            )
        if max_tool_iterations < 1:
            raise ValueError(
                f"max_tool_iterations must be >= 1, got {max_tool_iterations}. "
                "Use at least 1 to allow at least one LLM call."
            )
        has_system_prompt_method = getattr(cls, "_syrin_system_prompt_method", None)
        if (
            has_system_prompt_method is None
            and system_prompt is not None
            and not _is_valid_system_prompt(system_prompt)
        ):
            raise TypeError(
                f"system_prompt must be str, Prompt, or Callable[[PromptContext], str], "
                f"got {type(system_prompt).__name__}. Example: system_prompt='You are helpful.'"
            )
        if tools is not None and not isinstance(tools, list):
            raise TypeError(
                f"tools must be list of ToolSpec or None, got {type(tools).__name__}. "
                "Use @syrin.tool or syrin.tool() to create tools."
            )
        tools_list = tools if isinstance(tools, list) else []
        tools_final, mcp_instances = _normalize_tools(tools_list, self)
        budget = _validate_budget(budget)
        # Resolve model_router from class when None (so subclasses can set model_router = RoutingConfig(...))
        if model_router is None:
            model_router = getattr(cls, "model_router", None)
        if model is None:
            raise TypeError("Agent requires model (pass explicitly or set class-level model)")
        models_list: list[Model] | None = None
        if isinstance(model, list):
            if not model:
                raise TypeError(
                    "model list cannot be empty. Use model=Model.X() or model=[M1, M2, ...]."
                )
            for i, m in enumerate(model):
                if not isinstance(m, Model):
                    raise TypeError(
                        f"model[{i}] must be Model, got {type(m).__name__}. "
                        "Use Model.OpenAI(), Model.Anthropic(), etc."
                    )
            models_list = model
        elif isinstance(model, Model):
            models_list = [model]
        elif isinstance(model, ModelConfig):
            models_list = None
        else:
            raise TypeError(
                f"model must be Model, list[Model], or ModelConfig, got {type(model).__name__}. "
                "Use Model.OpenAI(), [Model.OpenAI(), Model.Anthropic()], etc."
            )
        # Resolve input_media, output_media, input_file_rules (class or param; default TEXT-only)
        _input_media: set[Media] = (
            input_media
            if input_media is not None
            else getattr(cls, "input_media", None) or {Media.TEXT}
        )
        _output_media: set[Media] = (
            output_media
            if output_media is not None
            else getattr(cls, "output_media", None) or {Media.TEXT}
        )
        _input_file_rules_final = input_file_rules or getattr(cls, "input_file_rules", None)
        if Media.FILE in _input_media:
            if _input_file_rules_final is None:
                raise ValueError(
                    "When Media.FILE is in input_media, provide input_file_rules=InputFileRules(allowed_mime_types=[...], max_size_mb=...)."
                )
            allowed = getattr(_input_file_rules_final, "allowed_mime_types", None) or []
            if not allowed:
                raise ValueError(
                    "When Media.FILE is in input_media, input_file_rules must have non-empty allowed_mime_types."
                )
        if image_generation is not None:
            from syrin.generation import ImageGenerator

            if not isinstance(image_generation, ImageGenerator):
                raise TypeError(
                    f"image_generation must be ImageGenerator or None, got {type(image_generation).__name__}. "
                    "Use ImageGenerator(provider=...) from syrin.generation."
                )
        if video_generation is not None:
            from syrin.generation import VideoGenerator

            if not isinstance(video_generation, VideoGenerator):
                raise TypeError(
                    f"video_generation must be VideoGenerator or None, got {type(video_generation).__name__}. "
                    "Use VideoGenerator(provider=...) from syrin.generation."
                )
        _knowledge = knowledge if knowledge is not None else getattr(cls, "knowledge", None)
        if _knowledge is not None:
            from syrin.knowledge import Knowledge as KnowledgeClass

            if not isinstance(_knowledge, KnowledgeClass):
                raise TypeError(
                    f"knowledge must be Knowledge or None, got {type(_knowledge).__name__}. "
                    "Use Knowledge(sources=[...], embedding=...) from syrin.knowledge."
                )
        self._knowledge = _knowledge
        self._input_media = _input_media
        self._output_media = _output_media
        self._input_file_rules = _input_file_rules_final

        self._router: Any = None
        self._active_model: Model | None = None
        self._active_model_config: ModelConfig | None = None
        self._last_routing_reason: Any = None
        self._last_model_used: str | None = None
        self._last_actual_cost: float | None = None
        self._call_task_override: Any = None
        if models_list is not None:
            if len(models_list) == 1 and model_router is None:
                self._model = models_list[0]
                self._model_config = self._model.to_config()
            else:
                if isinstance(model_router, ModelRouter):
                    self._router = model_router
                else:
                    routing_cfg = model_router if isinstance(model_router, RoutingConfig) else None
                    self._router = build_router_from_models(
                        models_list,
                        routing_config=routing_cfg,
                        budget=budget,
                    )
                self._model = models_list[0]
                self._model_config = self._model.to_config()
                _validate_agent_media(
                    self._router,
                    input_media=_input_media,
                    output_media=_output_media,
                )
        else:
            self._model = None
            self._model_config = model

        # Handle output configuration
        self._output: Output | None = output
        if output is not None and self._model_config is not None and output.type is not None:
            self._model_config.output = output.type

        self._system_prompt_source = (
            system_prompt if system_prompt is not NOT_PROVIDED and system_prompt is not None else ""
        )
        if self._system_prompt_source is NOT_PROVIDED:
            self._system_prompt_source = ""
        class_pv = getattr(cls, "_syrin_default_template_vars", None) or {}
        instance_pv = dict(template_variables or {})
        self._template_vars = {**class_pv, **instance_pv}
        self._inject_template_vars = inject_template_vars
        self._call_template_vars: dict[str, Any] | None = None
        # Wire generation tools from output_media (IMAGE/VIDEO → Gemini when API key available)
        # API key comes only from developer: Google model's api_key or explicit ImageGenerator/VideoGenerator
        _api_key: str | None = None
        if models_list:
            for m in models_list:
                if getattr(m, "_provider", "") == "google":
                    _api_key = getattr(m, "api_key", None) or (
                        m.to_config().api_key if hasattr(m, "to_config") else None
                    )
                    break
        self._generation_api_key: str | None = _api_key
        from syrin.generation import get_default_image_generator, get_default_video_generator

        _img_gen = (
            image_generation
            if image_generation is not None
            else (get_default_image_generator(_api_key) if Media.IMAGE in _output_media else None)
        )
        _vid_gen = (
            video_generation
            if video_generation is not None
            else (get_default_video_generator(_api_key) if Media.VIDEO in _output_media else None)
        )
        self._image_generator = _img_gen
        self._video_generator = _vid_gen
        _tools_list: list[ToolSpec] = list(tools_final) if tools_final else []
        _tool_names = {t.name for t in _tools_list}
        # Add generation tools when: explicit generator, default generator, or output_media declares
        # IMAGE/VIDEO (tool added so model can call it; returns helpful error if no API key).
        _add_image_tool = _img_gen is not None or Media.IMAGE in _output_media
        _add_video_tool = _vid_gen is not None or Media.VIDEO in _output_media
        if _add_image_tool and "generate_image" not in _tool_names:
            _tools_list.append(
                _make_generate_image_tool(
                    get_generator=self._resolve_image_generator,
                    emit=self._emit_event,
                )
            )
        if _add_video_tool and "generate_video" not in _tool_names:
            _tools_list.append(
                _make_generate_video_tool(
                    get_generator=self._resolve_video_generator,
                    emit=self._emit_event,
                )
            )
        if self._knowledge is not None and "search_knowledge" not in _tool_names:

            def _get_bt() -> object | None:
                return self.get_budget_tracker() if hasattr(self, "get_budget_tracker") else None

            def _get_model() -> object | None:
                return getattr(self, "_model", None)

            self._knowledge._attach_to_agent(
                emit=self._emit_event,
                get_budget_tracker=_get_bt,
                get_model=_get_model,
            )
            _tools_list.append(
                _make_search_knowledge_tool(
                    get_knowledge=lambda: self._knowledge,
                    emit=self._emit_event,
                )
            )
            if getattr(self._knowledge, "_agentic", False):
                cfg = getattr(self._knowledge, "_agentic_config", None)
                if cfg is not None:
                    if "search_knowledge_deep" not in _tool_names:
                        _tools_list.append(
                            _make_search_knowledge_deep_tool(
                                get_knowledge=lambda: self._knowledge,
                                get_model=_get_model,
                                get_budget_tracker=_get_bt,
                                emit=self._emit_event,
                            )
                        )
                        _tool_names.add("search_knowledge_deep")
                    if "verify_knowledge" not in _tool_names:
                        _tools_list.append(
                            _make_verify_knowledge_tool(
                                get_knowledge=lambda: self._knowledge,
                                get_model=_get_model,
                                get_budget_tracker=_get_bt,
                                emit=self._emit_event,
                            )
                        )
        self._tools = _tools_list
        self._mcp_instances: list[Any] = mcp_instances
        self._guardrails_disabled: set[str] = set()
        self._tools_disabled: set[str] = set()
        self._mcp_disabled: set[int] = set()
        self._runtime = _AgentRuntime()
        for i, x in enumerate(tools_list):
            if _is_mcp(x) and hasattr(x, "tools") and callable(x.tools):
                for t in x.tools():
                    if isinstance(t, ToolSpec):
                        self._runtime.mcp_tool_indices[t.name] = i
        self._max_tool_iterations = max_tool_iterations
        self._parent_agent: Agent | None = None
        self._provider: Provider

        # Extract advanced options from config
        ctx = config.context if config else None
        rate_limit = config.rate_limit if config else None
        checkpoint = config.checkpoint if config else None
        circuit_breaker = config.circuit_breaker if config else None
        approval_gate = config.approval_gate if config else None
        tracer = config.tracer if config else None
        event_bus = config.event_bus if config else None
        audit = config.audit if config else None
        dependencies = config.dependencies if config else None

        # Context component: manager and token limits
        if ctx is None:
            context_manager = DefaultContextManager(Context())
        elif isinstance(ctx, ContextConfig):
            context_manager = DefaultContextManager(ctx.to_context())
        elif isinstance(ctx, Context):
            context_manager = DefaultContextManager(ctx)
        else:
            context_manager = ctx
        ctx_config = getattr(context_manager, "context", None)
        token_limits = getattr(ctx_config, "token_limits", None) if ctx_config else None
        self._context_component = AgentContextComponent(context_manager, token_limits)

        # Memory component
        persistent_memory, memory_backend = _resolve_memory(memory)
        self._memory_component = AgentMemoryComponent(persistent_memory, memory_backend)

        # Warn when context needs memory but memory is disabled
        if self._memory_component.persistent_memory is None and ctx_config is not None:
            mode = getattr(ctx_config, "context_mode", None)
            if mode is not None and str(getattr(mode, "value", mode)) == "intelligent":
                import warnings

                warnings.warn(
                    "context_mode=intelligent requires memory for relevance filtering; "
                    "provide memory (e.g. memory=Memory()) or use context_mode=focused.",
                    UserWarning,
                    stacklevel=2,
                )

        # Budget component: state and persistence
        self._budget_component = AgentBudgetComponent(
            budget, budget_store, budget_store_key, self._context_component.token_limits
        )
        self._provider = _resolve_provider(self._model, self._model_config)
        object.__setattr__(self, "_agent_name", name)
        self._description = description
        self._dependencies: object | None = dependencies
        if self._budget_component.budget is not None:
            self._budget_component.budget._consume_callback = _budget_make_consume_callback(self)
        if (
            self._budget_component.budget is not None
            and self._budget_component.budget.per is not None
            and self._budget_component.store is None
        ):
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
        if custom_loop is not None:
            if (
                isinstance(custom_loop, type)
                and hasattr(custom_loop, "run")
                and callable(custom_loop.run)
            ):
                loop_instance = custom_loop()
            elif hasattr(custom_loop, "run") and callable(custom_loop.run):
                loop_instance = custom_loop  # type: ignore[assignment]
            else:
                loop_instance = ReactLoop(max_iterations=max_tool_iterations)
        else:
            loop_instance = LoopStrategyMapping.create_loop(
                loop_strategy, max_iterations=max_tool_iterations
            )
        self._loop = loop_instance
        self._last_iteration: int = 0
        self._conversation_id: str | None = None  # Set by caller; scopes state per conversation
        self._child_count: int = 0
        if max_child_agents is not None:
            self._max_child_agents = max_child_agents

        # Guardrails component
        if guardrails is None or (isinstance(guardrails, list) and len(guardrails) == 0):
            _guardrails = GuardrailChain()
        elif isinstance(guardrails, GuardrailChain):
            _guardrails = guardrails
        else:
            _guardrails = GuardrailChain(list(guardrails))
        self._guardrails_component = AgentGuardrailsComponent(_guardrails)

        # Observability component
        self._debug = debug
        _tracer: Tracer = tracer or get_tracer()
        if debug and not any(isinstance(e, ConsoleExporter) for e in _tracer._exporters):
            _tracer.add_exporter(ConsoleExporter())
        if debug:
            _tracer.set_debug_mode(True)
        self._observability_component = AgentObservabilityComponent(_tracer, event_bus, audit)

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

        # Audit logging (compliance)
        if audit is not None:
            if not isinstance(audit, AuditLog):
                raise TypeError(
                    f"audit must be AuditLog or None, got {type(audit).__name__}. "
                    "Use AuditLog(path='./audit.jsonl') for JSONL logging."
                )
            audit_handler = AuditHookHandler(source=cast(str, self._agent_name), config=audit)
            self.events.on_all(audit_handler)

        # Initialize run report for tracking metrics across a response() call
        self._run_report: AgentReport = AgentReport()

        self._approval_gate: ApprovalGate | None = approval_gate
        self._human_approval_timeout = human_approval_timeout
        self._max_tool_result_length = max_tool_result_length
        self._retry_on_transient = retry_on_transient
        self._max_retries = max_retries
        self._retry_backoff_base = retry_backoff_base

        # Circuit breaker
        self._circuit_breaker: CircuitBreaker | None = circuit_breaker
        self._fallback_provider: Provider | None = None
        self._fallback_model_config: ModelConfig | None = None
        if circuit_breaker is not None and not isinstance(circuit_breaker, CircuitBreaker):
            raise TypeError(
                f"circuit_breaker must be CircuitBreaker or None, got {type(circuit_breaker).__name__}"
            )

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

        # Remote config: register with cloud when syrin.init() was called
        from syrin.remote._hooks import on_agent_init as _remote_init

        _remote_init(self)

    @property
    def iteration(self) -> int:
        """Number of loop iterations from the last run (0 before first run or on guardrail block)."""
        return getattr(self, "_last_iteration", 0)

    @property
    def name(self) -> str:
        """Agent name for discovery, routing, and Agent Card. Defaults to lowercase class name."""
        return cast(str, self._agent_name)

    @property
    def description(self) -> str:
        """Agent description for discovery and Agent Card. Defaults to empty string."""
        return self._description

    @property
    def messages(self) -> list[Message]:
        """Current conversation messages from memory, or empty list if none."""
        if self._persistent_memory is not None:
            return self._persistent_memory.get_conversation_messages()
        return []

    @property
    def checkpointer(self) -> Checkpointer | None:
        """Checkpointer for manual save/load. None if checkpointing disabled."""
        return self._checkpointer

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
            >>> agent = Agent(model=m, config=AgentConfig(checkpoint=CheckpointConfig(storage="memory")))
            >>> cid = agent.save_checkpoint(reason="before_expensive_step")
            >>> agent.load_checkpoint(cid)
        """
        return _checkpoint_save(self, name=name, reason=reason)

    def _maybe_checkpoint(self, reason: str) -> None:
        """Automatically checkpoint based on trigger configuration.

        Args:
            reason: The reason for checkpointing ('step', 'tool', 'error', 'budget')
        """
        _checkpoint_maybe(self, reason)

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
        return _checkpoint_load(self, checkpoint_id)

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
        return _checkpoint_list(self, name=name)

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
        return _checkpoint_get_report(self)

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
            _events_print(self, hook.value, ctx)

        # Trigger before/on/after handlers
        self.events._trigger_before(hook, ctx)
        self.events._trigger(hook, ctx)
        self.events._trigger_after(hook, ctx)

        # Domain events (observability, typed consumers)
        event_bus = getattr(self, "_event_bus", None)
        if event_bus is not None:
            _emit_domain_event_for_hook(hook, ctx, event_bus)

        # Record media generation cost into budget (from GenerationResult.metadata)
        if self._budget is not None and self._budget_tracker is not None:
            if hook == Hook.GENERATION_IMAGE_END:
                raw_results = ctx.get("results")
                results = raw_results if isinstance(raw_results, (list, tuple)) else []
                model = ctx.get("model", "")
                for r in results:
                    meta = getattr(r, "metadata", None)
                    if getattr(r, "success", False) and isinstance(meta, dict):
                        cost = meta.get("cost_usd", 0)
                        name = meta.get("model_name", model)
                        if cost and cost > 0:
                            self._record_cost_info(
                                CostInfo(cost_usd=cost, model_name=str(name or "image"))
                            )
            elif hook == Hook.GENERATION_VIDEO_END:
                result = ctx.get("result")
                meta = getattr(result, "metadata", None) if result is not None else None
                if (
                    result is not None
                    and getattr(result, "success", False)
                    and isinstance(meta, dict)
                ):
                    cost = meta.get("cost_usd", 0)
                    name = meta.get("model_name", ctx.get("model", ""))
                    if cost and cost > 0:
                        self._record_cost_info(
                            CostInfo(cost_usd=cost, model_name=str(name or "video"))
                        )

    def _resolve_image_generator(self) -> Any:
        """Resolve image generator. Lazy init from stored key or env if None."""
        if self._image_generator is not None:
            return self._image_generator
        from syrin.generation import get_default_image_generator

        key = getattr(self, "_generation_api_key", None)
        gen = get_default_image_generator(key) if key else get_default_image_generator()
        if gen is not None:
            object.__setattr__(self, "_image_generator", gen)
        return gen

    def _resolve_video_generator(self) -> Any:
        """Resolve video generator. Lazy init from stored key or env if None."""
        if self._video_generator is not None:
            return self._video_generator
        from syrin.generation import get_default_video_generator

        key = getattr(self, "_generation_api_key", None)
        gen = get_default_video_generator(key) if key else get_default_video_generator()
        if gen is not None:
            object.__setattr__(self, "_video_generator", gen)
        return gen

    def _print_event(self, event: str, ctx: EventContext) -> None:
        """Print event to console when debug=True."""
        _events_print(self, event, ctx)

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
        _model_switch(self, model)

    @property
    def budget_state(self) -> BudgetState | None:
        """Current budget state (limit, remaining, spent, percent_used).

        None when agent has no run budget. Use to show users or gate behavior.

        Example:
            >>> agent.response("Hello")
            >>> state = agent.budget_state
            >>> if state:
            ...     print(f"Used {state.percent_used:.1f}%, ${state.remaining:.4f} left")
        """
        if self._budget is None or self._budget.run is None:
            return None
        effective = (
            (self._budget.run - self._budget.reserve)
            if self._budget.run > self._budget.reserve
            else self._budget.run
        )
        spent = self._budget_tracker.current_run_cost
        remaining = max(0.0, effective - spent)
        percent = (spent / effective * 100.0) if effective > 0 else 0.0
        return BudgetState(
            limit=effective,
            remaining=remaining,
            spent=spent,
            percent_used=round(percent, 2),
        )

    # Delegate to budget component (facade)
    @property
    def _budget_tracker(self) -> BudgetTracker:
        return self._budget_component.tracker

    @_budget_tracker.setter
    def _budget_tracker(self, value: BudgetTracker) -> None:
        self._budget_component.set_tracker(value)

    @property
    def _budget(self) -> Budget | None:
        return self._budget_component.budget

    @_budget.setter
    def _budget(self, value: Budget | None) -> None:
        self._budget_component.set_budget(value)

    @property
    def _budget_store(self) -> BudgetStore | None:
        return self._budget_component.store

    @property
    def _budget_store_key(self) -> str:
        return self._budget_component.key

    @property
    def _context(self) -> Any:
        return self._context_component.context_manager

    @property
    def _token_limits(self) -> Any:
        return self._context_component.token_limits

    @property
    def _persistent_memory(self) -> Memory | None:
        return cast("Memory | None", self._memory_component.persistent_memory)

    @_persistent_memory.setter
    def _persistent_memory(self, value: Memory | None) -> None:
        self._memory_component.set_persistent_memory(value)

    @property
    def _memory_backend(self) -> InMemoryBackend | None:
        return cast("InMemoryBackend | None", self._memory_component.memory_backend)

    @_memory_backend.setter
    def _memory_backend(self, value: InMemoryBackend | None) -> None:
        self._memory_component.set_memory_backend(value)

    @property
    def _guardrails(self) -> GuardrailChain:
        return cast(GuardrailChain, self._guardrails_component.guardrails)

    @property
    def _tracer(self) -> Tracer:
        return cast(Tracer, self._observability_component.tracer)

    @property
    def _event_bus(self) -> EventBus[Any] | None:
        return cast("EventBus[Any] | None", self._observability_component.event_bus)

    @property
    def _audit(self) -> AuditLog | None:
        return cast("AuditLog | None", self._observability_component.audit)

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
    def tools(self) -> list[ToolSpec]:
        """Tool specs attached to this agent (read-only). Excludes disabled tools and MCP servers."""
        raw = list(self._tools) if self._tools else []
        tools_disabled = getattr(self, "_tools_disabled", set()) or set()
        mcp_disabled = getattr(self, "_mcp_disabled", set()) or set()
        mcp_indices = self._runtime.mcp_tool_indices
        return [
            t
            for t in raw
            if t.name not in tools_disabled and (mcp_indices.get(t.name) not in mcp_disabled)
        ]

    @property
    def model_config(self) -> ModelConfig | None:
        """Current model config (read-only). None if agent has no model."""
        return self._model_config

    @property
    def memory(self) -> Memory | None:
        """Active memory (conversation and optional persistent recall)."""
        return self._persistent_memory

    @property
    def conversation_memory(self) -> Memory | None:
        """Memory used for conversation (alias for persistent_memory)."""
        return self._persistent_memory

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
            return _ContextFacade(cast(Context, self._context.context), self._context)
        if hasattr(self._context, "context"):
            return cast(Context, self._context.context)
        return Context()

    @property
    def context_stats(self) -> ContextStats:
        """Token usage and compaction stats from the last run.

        Why: Debug context size, see if compaction ran, track token growth.
        """
        if hasattr(self._context, "stats"):
            return cast(ContextStats, self._context.stats)
        return ContextStats()

    @property
    def _context_manager(self) -> DefaultContextManager:
        """Internal context manager."""
        return cast(DefaultContextManager, self._context)

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

        Requires persistent memory (Memory). Use memory=None or MemoryPreset.DISABLED to disable.

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
        return _memory_remember(
            self, content, memory_type=memory_type, importance=importance, **metadata
        )

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
        return _memory_recall(self, query=query, memory_type=memory_type, limit=limit)

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
        return _memory_forget(self, memory_id=memory_id, query=query, memory_type=memory_type)

    def _run_guardrails(
        self,
        text: str,
        stage: GuardrailStage,
    ) -> GuardrailResult:
        """Run guardrails on text with observability. Excludes remotely disabled guardrails."""
        return _guardrails_run(self, text, stage)

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

        Emits HANDOFF_START (before work), HANDOFF_END (after), HANDOFF_BLOCKED
        when blocked by a before-handler raising HandoffBlockedError.
        HandoffRetryRequested from target propagates to caller for retry logic.

        Args:
            target_agent: Agent class (e.g. ResearchAgent). Instantiated internally.
            task: Task description for the target.
            transfer_context: Copy memories to target. Default True.
            transfer_budget: Give target remaining budget. Default False.

        Returns:
            Response from target_agent.response(task).

        Raises:
            ValidationError: task is None or empty.
            HandoffBlockedError: Before-handler blocks handoff.
            HandoffRetryRequested: Target signals invalid data, retry with format_hint.

        Example:
            >>> r = agent.handoff(SupportAgent, "User needs refund help")
            >>> print(r.content)
        """
        return _handoff_impl(
            self,
            target_agent,
            task,
            transfer_context=transfer_context,
            transfer_budget=transfer_budget,
        )

    def spawn(
        self,
        agent_class: type[Agent],
        task: str | None = None,
        *,
        budget: Budget | None = None,
        max_child_agents: int | None = None,
    ) -> Agent | Response[str]:
        """Create a child agent. Optionally run a task and return its response.

        Why: Break work into sub-agents (specialists). Budget flows from parent:
        - Parent has shared budget: child borrows from it.
        - Child gets budget: "pocket money" up to parent's remaining.
        - Child spend is deducted from parent.

        Emits SPAWN_START (before creation), SPAWN_END (after child completes if task given).

        Args:
            agent_class: Agent class to spawn (e.g. ResearchAgent).
            task: If set, run task and return Response. Else return agent instance.
            budget: Child's budget (pocket money). Must not exceed parent remaining.
            max_child_agents: Cap on concurrent children. Default from instance or 10.

        Returns:
            Response if task given; else the spawned Agent instance.

        Example:
            >>> r = agent.spawn(ResearchAgent, task="Find papers on X")
            >>> child = agent.spawn(ResearchAgent)  # No task
            >>> child.response("Another task")
        """
        return _spawn_impl(
            self, agent_class, task, budget=budget, max_child_agents=max_child_agents
        )

    def _update_parent_budget(self, cost: float) -> None:
        """Update parent's budget when child spends (borrow mechanism)."""
        _spawn_update_parent_budget(self, cost)

    def spawn_parallel(
        self,
        agents: list[tuple[type[Agent], str]],
    ) -> list[Response[str]]:
        """Run multiple agents via spawn(), each with its own task.

        Why: Fan-out work (e.g. research + summarization + fact-check).
        Runs sequentially via spawn() to respect parent budget and max_child_agents.
        Emits SPAWN_START/SPAWN_END per child.

        Note: Uses sequential execution to avoid event-loop conflicts with
        sync response() in threaded/async environments. For parallelism,
        use asyncio with agent.arun() directly.

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
        return _spawn_parallel_impl(self, agents)

    @property
    def _system_prompt(self) -> str | Any:
        """Raw system prompt source (str, Prompt, or callable). For introspection.

        Resolved prompt at runtime is built by _resolve_system_prompt.
        """
        method = getattr(self.__class__, "_syrin_system_prompt_method", None)
        return method if method is not None else self._system_prompt_source

    def effective_template_variables(
        self, call_vars: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Return merged template_variables: class + instance + call. For introspection."""
        return _prompt_effective_template_variables(self, call_vars=call_vars)

    def get_prompt_builtins(self) -> dict[str, Any]:
        """Return built-in vars (date, agent_id, conversation_id) that would be injected."""
        return _prompt_get_builtins(self)

    def _resolve_system_prompt(
        self,
        prompt_vars: dict[str, Any],
        ctx: Any,
    ) -> str:
        """Resolve system prompt from source (str, Prompt, callable, or @system_prompt method).

        Override this in subclasses for custom resolution.
        """
        return _prompt_resolve_system_prompt(self, prompt_vars, ctx)

    def _build_messages(self, user_input: str | list[dict[str, Any]]) -> list[Message]:
        return _prompt_build_messages(self, user_input)

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
        return _prompt_build_output(
            self,
            content,
            validation_retries=validation_retries,
            validation_context=validation_context,
            validator=validator,
        )

    def _execute_tool(self, name: str, arguments: dict[str, Any]) -> str:
        return _tool_execute(self, name, arguments)

    async def execute_tool(self, name: str, arguments: dict[str, Any]) -> str:
        """Run a tool by name with the given arguments. For custom loops.

        Why: Built-in loops call this automatically. Use when implementing a
        custom Loop to execute tool calls. Supports both sync and async tools.

        Args:
            name: Tool name (must match a @tool-decorated function).
            arguments: Dict of parameter names to values.

        Returns:
            Tool result as string. Raises ToolExecutionError on failure.

        Example:
            >>> result = await agent.execute_tool("search", {"query": "syrin"})
        """
        import asyncio

        result = self._execute_tool(name, arguments)
        if asyncio.iscoroutine(result):
            return cast(str, await result)
        return result

    def estimate_cost(
        self,
        messages: list[Any],
        max_output_tokens: int = 1024,
    ) -> float:
        """Estimate cost in USD for the next LLM call (best-effort).

        Use before calling the LLM to check affordability. Uses model pricing and
        token counts from message contents. Actual cost may differ.

        Args:
            messages: List of Message (role, content) to be sent.
            max_output_tokens: Assumed max completion tokens (default 1024).

        Returns:
            Estimated cost in USD.

        Example:
            >>> cost = agent.estimate_cost(messages)
            >>> if cost > 0.01:
            ...     print("Call may exceed threshold")
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
        _budget_pre_call_check(self, messages, max_output_tokens=max_output_tokens)

    def _check_and_apply_budget(self) -> None:
        """Raise if budget or token limits exceeded; apply threshold actions (switch, warn). Stop raises."""
        _budget_check_and_apply(self)

    def _check_and_apply_rate_limit(self) -> None:
        """Check rate limits and apply threshold actions (switch model, wait, warn, stop).

        Called before each LLM request. This is the main integration point that makes
        rate limiting work automatically during agent execution.
        """
        _rate_limit_check(self)

    def _record_rate_limit_usage(self, token_usage: TokenUsage) -> None:
        """Record token usage and re-check rate limits after LLM call."""
        _rate_limit_record(self, token_usage)

    def _record_cost(self, token_usage: TokenUsage, model_id: str) -> None:
        """Compute cost, build CostInfo, record on tracker, sync Budget._spent, then re-check thresholds."""
        _budget_record_cost(self, token_usage, model_id)

    def _make_budget_consume_callback(self) -> Callable[[float], None]:
        """Return a callback for Budget.consume() so guardrails can record cost."""
        return _budget_make_consume_callback(self)

    def _record_cost_info(self, cost_info: CostInfo) -> None:
        """Record a CostInfo (e.g. from streaming). Syncs spent and checks budget."""
        _budget_record_cost_info(self, cost_info)

    async def _complete_async(
        self, messages: list[Message], tools: list[ToolSpec] | None
    ) -> ProviderResponse:
        """Async internal method to call LLM. Routes when _router is set."""
        use_model = self._model
        use_config = self._model_config
        use_provider = self._provider
        provider_kwargs: dict[str, Any] = {}

        if self._router is not None:
            prompt = ""
            for m in reversed(messages):
                c = getattr(m, "content", None) or ""
                if isinstance(c, str) and c.strip():
                    prompt = c.strip()
                    break
            ctx: dict[str, object] = {}
            if self._token_limits is not None:
                ctx["max_output_tokens"] = getattr(self.run_context, "max_output_tokens", 1024)
            t0 = time.perf_counter()
            try:
                routed_model, task_type, reason = self._router.route(
                    prompt,
                    tools=tools,
                    messages=messages,
                    context=ctx,
                    task_override=self._call_task_override,
                )
            except Exception:
                raise
            routing_latency_ms = round((time.perf_counter() - t0) * 1000, 2)
            self._active_model = routed_model
            self._active_model_config = routed_model.to_config()
            self._last_routing_reason = reason
            self._emit_event(
                Hook.ROUTING_DECISION,
                EventContext(
                    routing_reason=reason,
                    model=routed_model.model_id,
                    task_type=task_type.value if hasattr(task_type, "value") else str(task_type),
                    prompt=prompt[:200] if prompt else "",
                    routing_latency_ms=routing_latency_ms,
                ),
            )
            with self._tracer.span(
                "routing.decision",
                kind=SpanKind.INTERNAL,
                attributes={
                    "routing.model": reason.selected_model,
                    "routing.model_id": routed_model.model_id,
                    "routing.reason": reason.reason,
                    "routing.task_type": task_type.value
                    if hasattr(task_type, "value")
                    else str(task_type),
                    "routing.cost_estimate": reason.cost_estimate,
                    "routing.confidence": reason.classification_confidence,
                    "routing.alternatives": ",".join(reason.alternatives)
                    if reason.alternatives
                    else "",
                },
            ):
                pass
            use_model = routed_model
            use_config = routed_model.to_config()
            use_provider = routed_model.get_provider()

        if use_model is not None and hasattr(use_model, "_provider_kwargs"):
            provider_kwargs = dict(getattr(use_model, "_provider_kwargs", {}))
        if use_model is not None:
            has_fallback = bool(getattr(use_model, "fallback", None))
            has_transformer = bool(getattr(use_model, "_transformer", None))
            if has_fallback or has_transformer:
                result = await use_model.acomplete(messages, tools=tools, **provider_kwargs)
                resp = cast(ProviderResponse, result)
            else:
                resp = await use_provider.complete(
                    messages=messages, model=use_config, tools=tools, **provider_kwargs
                )
        else:
            resp = await use_provider.complete(
                messages=messages, model=use_config, tools=tools, **provider_kwargs
            )
        meta = getattr(resp, "metadata", None) or {}
        if meta:
            if "model_used" in meta:
                self._last_model_used = meta.get("model_used")
            if "actual_cost" in meta:
                self._last_actual_cost = meta.get("actual_cost")
        return resp

    def _resolve_fallback_provider(self) -> tuple[Provider, ModelConfig]:
        """Resolve fallback model to (provider, config). Cached."""
        return _model_resolve_fallback(self)

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
        cb = self._circuit_breaker
        if cb is not None and not cb.allow_request():
            if cb.fallback is not None:
                prov, cfg = self._resolve_fallback_provider()
                self._emit_event(
                    Hook.LLM_FALLBACK,
                    EventContext(
                        reason="circuit_breaker_open",
                        from_model=getattr(self._model, "_model_id", str(self._model)),
                        to_model=cfg.model_id,
                    ),
                )
                provider_kwargs: dict[str, Any] = {}
                if hasattr(self._model, "_provider_kwargs"):
                    provider_kwargs = dict(getattr(self._model, "_provider_kwargs", {}))
                return await prov.complete(
                    messages=messages, model=cfg, tools=tools, **provider_kwargs
                )
            import time

            state = cb.get_state()
            recovery_at = time.monotonic() + (
                cb.recovery_timeout - (time.monotonic() - (state.last_failure_time or 0))
            )
            fallback_str = str(cb.fallback) if cb.fallback else None
            raise CircuitBreakerOpenError(
                f"Circuit breaker open for agent {self._agent_name!r}. "
                f"Recovery in {cb.recovery_timeout}s.",
                agent_name=cast(str, self._agent_name),
                circuit_state=state,
                recovery_at=recovery_at,
                fallback_model=fallback_str,
            )
        try:
            resp = await self._complete_async(messages, tools)
            if cb is not None:
                was_half_open = cb.get_state().state == CircuitState.HALF_OPEN
                cb.record_success()
                if was_half_open:
                    self._emit_event(Hook.CIRCUIT_RESET, EventContext())
            return resp
        except Exception as e:
            if cb is not None:
                was_closed_or_half = cb.get_state().state in (
                    CircuitState.CLOSED,
                    CircuitState.HALF_OPEN,
                )
                cb.record_failure(e)
                if cb.get_state().state == CircuitState.OPEN and was_closed_or_half:
                    self._emit_event(
                        Hook.CIRCUIT_TRIP,
                        EventContext(
                            error=str(e),
                            failures=cb.get_state().failures,
                            agent_name=cast(str, self._agent_name),
                        ),
                    )
            raise

    def _with_context_on_response(self, r: Response[str]) -> Response[str]:
        """Attach per-call context_stats and context to a Response."""
        return _response_with_context(self, r)

    def record_conversation_turn(
        self, user_input: str | list[dict[str, Any]], assistant_content: str
    ) -> None:
        """Append a user/assistant turn to memory for next context."""
        _response_record_conversation_turn(self, user_input, assistant_content)

    async def _run_loop_response_async(
        self, user_input: str | list[dict[str, Any]]
    ) -> Response[str]:
        """Run using the configured loop strategy with full observability (async)."""
        from syrin.agent._run import run_agent_loop_async

        result = await run_agent_loop_async(self, user_input)
        self.record_conversation_turn(user_input, result.content or "")
        return result

    def _run_loop_response(self, user_input: str | list[dict[str, Any]]) -> Response[str]:
        """Run using the configured loop strategy (sync wrapper)."""
        from syrin._loop import get_loop

        return get_loop().run_until_complete(self._run_loop_response_async(user_input))

    def _stream_response(self, user_input: str | list[dict[str, Any]]) -> Iterator[StreamChunk]:
        """Stream response chunks synchronously. Records cost per chunk and checks budget mid-stream."""
        messages = self._build_messages(user_input)
        tools = self.tools if self.tools else None
        accumulated = ""
        total_cost = 0.0
        total_tokens = TokenUsage()
        prev_cost = 0.0
        prev_tokens = TokenUsage()
        chunk_index = 0

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
                        self._budget_component.save()
                yield StreamChunk(
                    index=chunk_index,
                    text=content,
                    accumulated_text=accumulated,
                    cost_so_far=total_cost,
                    tokens_so_far=total_tokens,
                )
                chunk_index += 1
                if self._budget is not None or self._token_limits is not None:
                    self._check_and_apply_budget()
                prev_cost = total_cost
                prev_tokens = total_tokens
        except (BudgetExceededError, BudgetThresholdError):
            raise
        except Exception as e:
            raise ToolExecutionError(f"Streaming failed: {e}") from e

    def response(
        self,
        user_input: str | list[dict[str, Any]],
        context: Context | None = None,
        template_variables: dict[str, Any] | None = None,
        *,
        inject: list[dict[str, Any]] | None = None,
        inject_source_detail: str | None = None,
        task_type: Any = None,
    ) -> Response[str]:
        """Run the agent: LLM completion + tool loop. Synchronous.

        Why: Main entry point for getting a reply. Runs the configured loop
        (REACT by default), runs guardrails, records cost, applies budget/rate
        limits. Blocks until complete.

        Args:
            user_input: User message.
            context: Optional Context for this call only. When set, overrides the agent's
                default context (max_tokens, reserve, thresholds, budget). The Context
                used for this call is on ``result.context``; per-call stats on ``result.context_stats``.
            template_variables: Optional per-call template vars for dynamic system prompts.
                Overrides instance template_variables for this call only.
            inject: Optional per-call context injection (RAG results, dynamic blocks).
                Each item is a dict with ``role`` and ``content``. Overrides Context.runtime_inject when provided.
            inject_source_detail: Provenance label for inject (e.g. 'rag').
            task_type: Override task type for routing (e.g. TaskType.CODE). Use for ambiguous prompts.

        Returns:
            Response with content, cost, tokens, model, stop_reason, structured
            output (if output= set), and report.

        Example:
            >>> r = agent.response("What is 2+2?")
            >>> r = agent.response("Long task...", context=Context(max_tokens=4000))
        """
        _validate_user_input(user_input, "response")
        self._call_context = context
        self._call_template_vars = dict(template_variables) if template_variables else None
        self._call_inject = inject
        self._call_inject_source_detail = inject_source_detail
        self._call_task_override = task_type
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
            self._call_template_vars = None
            self._call_inject = None
            self._call_inject_source_detail = None
            self._call_task_override = None

    async def arun(
        self,
        user_input: str | list[dict[str, Any]],
        context: Context | None = None,
        template_variables: dict[str, Any] | None = None,
        *,
        inject: list[dict[str, Any]] | None = None,
        inject_source_detail: str | None = None,
        task_type: Any = None,
    ) -> Response[str]:
        """Run the agent asynchronously. Same as response() but non-blocking.

        Why: Use in async apps to avoid blocking the event loop. Same behavior
        as response() (guardrails, budget, tools, etc.).

        Args:
            user_input: User message.
            context: Optional Context for this call only (see response()). Overrides
                the agent's context for this call. Used context is on ``result.context``;
                per-call stats on ``result.context_stats``.
            template_variables: Optional per-call template vars for dynamic system prompts.

        Returns:
            Response (same as response()).

        Example:
            >>> r = await agent.arun("Summarize this")
        """
        _validate_user_input(user_input, "arun")
        self._call_context = context
        self._call_template_vars = dict(template_variables) if template_variables else None
        self._call_inject = inject
        self._call_inject_source_detail = inject_source_detail
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
            self._call_template_vars = None
            self._call_inject = None
            self._call_inject_source_detail = None
            self._call_task_override = None

    def stream(
        self,
        user_input: str | list[dict[str, Any]],
        context: Context | None = None,
        template_variables: dict[str, Any] | None = None,
        *,
        inject: list[dict[str, Any]] | None = None,
        inject_source_detail: str | None = None,
    ) -> Iterator[StreamChunk]:
        """Stream response text as it arrives. Synchronous iterator.

        Why: Show tokens in real time (e.g. ChatGPT-style UI). No tool-call loop;
        single completion only.

        Args:
            user_input: User message.
            context: Optional Context for this call only (see response()). Overrides agent's context.
            template_variables: Optional per-call template vars for dynamic system prompts.

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
        _validate_user_input(user_input, "stream")
        self._call_context = context
        self._call_template_vars = dict(template_variables) if template_variables else None
        self._call_inject = inject
        self._call_inject_source_detail = inject_source_detail
        try:
            if self._budget is not None or self._token_limits is not None:
                self._budget_tracker.reset_run()
                if self._budget is not None:
                    self._budget._set_spent(0)
            yield from self._stream_response(user_input)
        finally:
            self._call_context = None
            self._call_template_vars = None
            self._call_inject = None
            self._call_inject_source_detail = None

    async def astream(
        self,
        user_input: str | list[dict[str, Any]],
        context: Context | None = None,
        template_variables: dict[str, Any] | None = None,
        *,
        inject: list[dict[str, Any]] | None = None,
        inject_source_detail: str | None = None,
    ) -> AsyncIterator[StreamChunk]:
        """Stream response text as it arrives. Async iterator.

        Why: Non-blocking streaming for async apps. Same chunks as stream().

        Args:
            user_input: User message.
            context: Optional Context for this call only (see response()). Overrides agent's context.
            template_variables: Optional per-call template vars for dynamic system prompts.

        Note:
            Astream does not return a Response; for context stats for this run,
            read ``agent.context_stats`` after the stream completes.

        Example:
            >>> async for chunk in agent.astream("Write a poem"):
            ...     print(chunk.text, end="")
        """
        _validate_user_input(user_input, "astream")
        self._call_context = context
        self._call_template_vars = dict(template_variables) if template_variables else None
        self._call_inject = inject
        self._call_inject_source_detail = inject_source_detail
        try:
            if self._budget is not None or self._token_limits is not None:
                self._budget_tracker.reset_run()
                if self._budget is not None:
                    self._budget._set_spent(0)
            messages = self._build_messages(user_input)
            tools = self.tools if self.tools else None
            accumulated = ""
            total_cost = 0.0
            total_tokens = TokenUsage()
            prev_cost = 0.0
            prev_tokens = TokenUsage()
            chunk_index = 0

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
                            # Providers often omit cost_usd on streaming chunks; derive from tokens
                            if delta_cost <= 0 and delta_tokens.total_tokens > 0:
                                pricing = (
                                    getattr(self._model, "pricing", None)
                                    if self._model is not None
                                    else None
                                )
                                delta_cost = calculate_cost(
                                    self._model_config.model_id,
                                    delta_tokens,
                                    pricing_override=pricing,
                                )
                            cost_info = CostInfo(
                                cost_usd=delta_cost,
                                token_usage=delta_tokens,
                                model_name=self._model_config.model_id,
                            )
                            self._budget_tracker.record(cost_info)
                            if self._budget is not None:
                                self._budget._set_spent(self._budget_tracker.current_run_cost)
                            self._budget_component.save()
                    yield StreamChunk(
                        index=chunk_index,
                        text=content,
                        accumulated_text=accumulated,
                        cost_so_far=total_cost,
                        tokens_so_far=total_tokens,
                    )
                    chunk_index += 1
                    if self._budget is not None or self._token_limits is not None:
                        self._check_and_apply_budget()
                    prev_cost = total_cost
                    prev_tokens = total_tokens
            except (BudgetExceededError, BudgetThresholdError):
                raise
            except Exception as e:
                raise ToolExecutionError(f"Streaming failed: {e}") from e
            else:
                # End-of-stream fallback: if we have tokens but never recorded cost
                if (
                    (self._budget is not None or self._token_limits is not None)
                    and total_tokens.total_tokens > 0
                    and self._budget_tracker.current_run_cost <= 0
                ):
                    pricing = (
                        getattr(self._model, "pricing", None) if self._model is not None else None
                    )
                    cost_usd = calculate_cost(
                        self._model_config.model_id,
                        total_tokens,
                        pricing_override=pricing,
                    )
                    if cost_usd > 0:
                        cost_info = CostInfo(
                            cost_usd=cost_usd,
                            token_usage=total_tokens,
                            model_name=self._model_config.model_id,
                        )
                        self._budget_tracker.record(cost_info)
                        if self._budget is not None:
                            self._budget._set_spent(self._budget_tracker.current_run_cost)
                        self._budget_component.save()
                # Auto-store turn when streaming (playground uses /stream)
                from syrin.agent._run import _auto_store_turn

                _auto_store_turn(self, user_input, accumulated)
        finally:
            self._call_context = None
            self._call_template_vars = None
            self._call_inject = None
            self._call_inject_source_detail = None

    def as_router(self, config: Any | None = None, **config_kwargs: Any) -> Any:
        """Return a FastAPI APIRouter for this agent. Mount on your app.

        Use when you want to serve this agent over HTTP. Mount the router on an
        existing FastAPI app, e.g. app.include_router(agent.as_router(), prefix="/agent").

        Requires syrin[serve] (fastapi, uvicorn).

        Args:
            config: Optional ServeConfig. If None, uses defaults.
            **config_kwargs: Override ServeConfig fields (route_prefix, port, etc.).

        Returns:
            FastAPI APIRouter with /chat, /stream, /health, /ready, /budget, /describe.

        Example:
            >>> from fastapi import FastAPI
            >>> app = FastAPI()
            >>> app.include_router(agent.as_router(), prefix="/agent")
        """
        from syrin.serve.config import ServeConfig
        from syrin.serve.http import build_router

        cfg = config if isinstance(config, ServeConfig) else ServeConfig(**config_kwargs)
        return build_router(self, cfg)

    # serve() inherited from Servable — HTTP, CLI, STDIO protocols


# Presets and builder
from syrin.agent import presets as _presets
from syrin.agent.builder import AgentBuilder as _AgentBuilder

Agent.presets = _presets  # type: ignore[attr-defined]
Agent.builder = staticmethod(lambda model: _AgentBuilder(model))  # type: ignore[attr-defined]
Agent.basic = staticmethod(_presets.basic)  # type: ignore[attr-defined]
Agent.with_memory = staticmethod(_presets.with_memory)  # type: ignore[attr-defined]
Agent.with_budget = staticmethod(_presets.with_budget)  # type: ignore[attr-defined]
