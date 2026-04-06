"""Agent class-construction and instance-construction helpers."""

from __future__ import annotations

import logging
import uuid
from collections.abc import Callable
from typing import TYPE_CHECKING, cast

from syrin._sentinel import NOT_PROVIDED
from syrin.agent._budget_ops import make_budget_consume_callback as _budget_make_consume_callback
from syrin.agent._components import (
    AgentBudgetComponent,
    AgentContextComponent,
    AgentGuardrailsComponent,
    AgentMemoryComponent,
    AgentObservabilityComponent,
)
from syrin.agent._helpers import (
    _AgentRuntime,
    _collect_class_tools,
    _collect_system_prompt_method,
    _get_system_prompt_method_names,
    _is_mcp,
    _is_valid_system_prompt,
    _make_generate_image_tool,
    _make_generate_video_tool,
    _make_generate_voice_tool,
    _make_search_knowledge_deep_tool,
    _make_search_knowledge_tool,
    _make_verify_knowledge_tool,
    _merge_class_attrs,
    _normalize_tools,
    _resolve_memory,
    _resolve_provider,
    _validate_agent_media,
    _validate_budget,
)
from syrin.audit import AuditHookHandler, AuditLog
from syrin.budget import Budget
from syrin.budget_store import BudgetStore
from syrin.checkpoint import CheckpointConfig, Checkpointer
from syrin.circuit import CircuitBreaker
from syrin.context import Context, DefaultContextManager
from syrin.context.config import _ContextConfig
from syrin.enums import Media, ToolErrorMode
from syrin.events import Events
from syrin.generation import get_default_image_generator, get_default_video_generator
from syrin.guardrails import Guardrail, GuardrailChain
from syrin.loop import Loop, ReactLoop
from syrin.memory import Memory
from syrin.model import Model
from syrin.observability import ConsoleExporter, Tracer, get_tracer
from syrin.output import Output
from syrin.ratelimit import RateLimitManager, create_rate_limit_manager
from syrin.response import AgentReport
from syrin.router import ModelRouter, RoutingConfig
from syrin.router.agent_integration import build_router_from_models
from syrin.tool import ToolSpec
from syrin.types import ModelConfig

if TYPE_CHECKING:
    from syrin.agent import Agent
    from syrin.domain_events import EventBus
    from syrin.hitl import ApprovalGate
    from syrin.observability import Tracer
    from syrin.ratelimit import APIRateLimit, RateLimitManager

_log = logging.getLogger(__name__)


def init_subclass(cls: type[Agent]) -> None:
    """Populate merged class defaults used by Agent instances."""
    mro = cls.__mro__
    default_model = _merge_class_attrs(mro, "model", merge=False)
    default_prompt = _merge_class_attrs(mro, "system_prompt", merge=False)
    default_tools = _merge_class_attrs(mro, "tools", merge=True)
    default_budget = _merge_class_attrs(mro, "budget", merge=False)
    default_guardrails = _merge_class_attrs(mro, "guardrails", merge=True)
    default_memory = _merge_class_attrs(mro, "memory", merge=False)
    default_output = _merge_class_attrs(mro, "output", merge=False)
    default_name = _merge_class_attrs(mro, "_syrin_default_name", merge=False)
    default_description = _merge_class_attrs(mro, "_syrin_default_description", merge=False)
    cls._syrin_default_model = default_model if default_model is not NOT_PROVIDED else None  # type: ignore[assignment]
    cls._syrin_default_memory = default_memory  # type: ignore[assignment]
    method_names = _get_system_prompt_method_names(cls)
    if len(method_names) > 1:
        names_str = ", ".join(f"'{n}'" for n in method_names)
        raise ValueError(
            f"Agent class {cls.__name__!r} has multiple @system_prompt methods "
            f"(only one allowed): {names_str}. Remove the extras or merge them "
            "into a single @system_prompt method."
        )
    cls._syrin_system_prompt_method = _collect_system_prompt_method(cls)
    cls._syrin_default_system_prompt = default_prompt if default_prompt is not NOT_PROVIDED else ""
    merged_template_vars: dict[str, object] = {}
    for c in mro:
        if c is object:
            continue
        tv = c.__dict__.get("template_variables", NOT_PROVIDED)
        if tv is not NOT_PROVIDED and isinstance(tv, dict):
            merged_template_vars = {**merged_template_vars, **tv}
    cls._syrin_default_template_vars = merged_template_vars
    class_tools = _collect_class_tools(cls)
    explicit_list = list(default_tools) if default_tools is not NOT_PROVIDED else []  # type: ignore[call-overload]
    by_name: dict[str, ToolSpec] = {t.name: t for t in class_tools}
    mcp_sources: list[object] = []
    for t in explicit_list:
        if isinstance(t, ToolSpec):
            by_name[t.name] = t
        elif isinstance(t, list):
            for s in t:
                if isinstance(s, ToolSpec):
                    by_name[s.name] = s
        elif hasattr(t, "tools") and callable(getattr(t, "tools", None)):
            mcp_sources.append(t)
    cls._syrin_default_tools = list(by_name.values()) + mcp_sources  # type: ignore[operator]
    cls._syrin_default_budget = default_budget if default_budget is not NOT_PROVIDED else None  # type: ignore[assignment]
    cls._syrin_default_guardrails = (
        list(default_guardrails) if default_guardrails is not NOT_PROVIDED else []  # type: ignore[call-overload]
    )
    cls._syrin_default_output = default_output if default_output is not NOT_PROVIDED else None
    if default_name is not NOT_PROVIDED and isinstance(default_name, str):
        cls._syrin_default_name = default_name
    elif default_name is NOT_PROVIDED and "_syrin_default_name" not in cls.__dict__:
        cls._syrin_default_name = None
    if default_description is not NOT_PROVIDED:
        cls._syrin_default_description = default_description  # type: ignore[assignment]
    elif default_description is NOT_PROVIDED and "_syrin_default_description" not in cls.__dict__:
        cls._syrin_default_description = ""


def init_agent(
    self: Agent,
    *,
    model: Model | ModelConfig | None,
    system_prompt: str | Callable[[], str] | None,
    tools: list[ToolSpec] | None,
    budget: Budget | None,
    output: type | Output | None,
    max_tool_iterations: int,
    budget_store: BudgetStore | None,
    memory: Memory | None,
    loop: Loop | type[Loop] | None,
    guardrails: list[Guardrail] | GuardrailChain | None,
    human_approval_timeout: int,
    max_tool_result_length: int | None,
    retry_on_transient: bool,
    max_retries: int,
    retry_backoff_base: float,
    max_input_length: int,
    debug: bool,
    name: str | None,
    description: str | None,
    template_variables: dict[str, object] | None,
    inject_template_vars: bool,
    max_child_agents: int | None,
    context: Context | DefaultContextManager | None,
    rate_limit: APIRateLimit | RateLimitManager | None,
    checkpoint: CheckpointConfig | Checkpointer | None,
    circuit_breaker: CircuitBreaker | None,
    approval_gate: ApprovalGate | None,
    tracer: Tracer | None,
    event_bus: EventBus[object] | None,  # type: ignore[type-var]
    audit: AuditLog | None,
    dependencies: object | None,
    spotlight_tool_outputs: bool,
    normalize_inputs: bool,
    tool_error_mode: ToolErrorMode,
    model_router: ModelRouter | RoutingConfig | None,
    input_media: set[Media] | None,
    output_media: set[Media] | None,
    input_file_rules: object,
    image_generation: object,
    video_generation: object,
    voice_generation: object,
    knowledge: object | None,
    output_config: object | None,
) -> None:
    """Initialize an Agent instance."""
    self._watch_protocols = []
    self._watch_concurrency = 1
    self._watch_timeout = None
    self._watch_on_trigger = None
    self._watch_on_result = None
    self._watch_on_error = None
    self._watch_semaphore = None

    try:
        import syrin._runtime_flags as _runtime

        _runtime._auto_trace_check()
        _runtime._auto_debug_check()
        _runtime._auto_log_level_check()
        debug_pry = _runtime.get_debug_pry()
        if debug_pry is not None:
            debug_pry.attach(self)  # type: ignore[attr-defined]
    except AttributeError:
        pass

    cls = self.__class__
    if model is NOT_PROVIDED:
        model = getattr(cls, "_syrin_default_model", None)
    if system_prompt is NOT_PROVIDED:
        system_prompt = getattr(cls, "_syrin_default_system_prompt", "") or ""
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
        memory = (
            Memory() if class_mem is NOT_PROVIDED or class_mem is None else cast(Memory, class_mem)
        )
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
    tools_list = tools if isinstance(tools, list) else []
    tools_final, mcp_instances = _normalize_tools(tools_list, self)  # type: ignore[arg-type]
    budget = _validate_budget(budget)
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
    _input_media = (
        input_media
        if input_media is not None
        else getattr(cls, "input_media", None) or {Media.TEXT}
    )
    _output_media = (
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
    if voice_generation is not None:
        from syrin.generation import VoiceGenerator

        if not isinstance(voice_generation, VoiceGenerator):
            raise TypeError(
                f"voice_generation must be VoiceGenerator or None, got {type(voice_generation).__name__}. "
                "Use VoiceGenerator.OpenAI(api_key=...) or VoiceGenerator.ElevenLabs(api_key=...)."
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
    _output_config_raw = (
        output_config if output_config is not None else getattr(cls, "output_config", None)
    )
    if _output_config_raw is not None:
        from syrin.output_format import OutputConfig, OutputFormat

        if isinstance(_output_config_raw, OutputConfig):
            self._output_config = _output_config_raw
        elif isinstance(_output_config_raw, OutputFormat):
            self._output_config = OutputConfig(format=_output_config_raw)
        else:
            self._output_config = OutputConfig(format=OutputFormat(str(_output_config_raw)))
    else:
        self._output_config = None
    if (
        self._output_config is not None
        and self._output_config.template is not None
        and output is None
    ):
        raise ValueError(
            "output_config with template requires output=Output(SomeModel). "
            "Structured output provides slot values for template rendering."
        )
    self._input_media = _input_media
    self._output_media = _output_media
    self._input_file_rules = _input_file_rules_final
    self._router = None
    self._active_model = None
    self._active_model_config = None
    self._last_routing_reason = None
    self._last_model_used = None
    self._last_actual_cost = None
    self._last_cost_estimated = None
    self._last_cache_hit = False
    self._last_cache_savings = 0.0
    self._call_task_override = None
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
                    models_list, routing_config=routing_cfg, budget=budget
                )
            self._model = models_list[0]
            self._model_config = self._model.to_config()
            _validate_agent_media(
                self._router, input_media=_input_media, output_media=_output_media
            )
    else:
        self._model = None
        self._model_config = cast(ModelConfig, model)
    # Accept bare type — wrap it in Output() automatically
    if output is not None and not isinstance(output, Output):
        if isinstance(output, type):
            output = Output(output)
        else:
            raise TypeError(
                f"output must be a Pydantic model class or Output instance, got {type(output).__name__}. "
                "Example: output=MyModel or output=Output(MyModel, validation_retries=3)"
            )
    if output is None:
        class_output = getattr(cls, "_syrin_default_output", None)
        if isinstance(class_output, Output):
            output = class_output
    self._output = output
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
    self._call_template_vars = None
    _api_key: str | None = None
    if models_list:
        for m in models_list:
            if getattr(m, "_provider", "") == "google":
                _api_key = getattr(m, "api_key", None) or (
                    m.to_config().api_key if hasattr(m, "to_config") else None
                )
                break
    self._generation_api_key = _api_key
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
    _voice_gen = (
        voice_generation if voice_generation is not None else getattr(cls, "voice_generation", None)
    )
    self._image_generator = _img_gen
    self._video_generator = _vid_gen
    self._voice_generator = _voice_gen
    _tools_list = list(tools_final) if tools_final else []
    _tool_names = {t.name for t in _tools_list}
    if (
        _img_gen is not None or Media.IMAGE in _output_media
    ) and "generate_image" not in _tool_names:
        _tools_list.append(
            _make_generate_image_tool(
                get_generator=self._resolve_image_generator, emit=self._emit_event
            )
        )
    if (
        _vid_gen is not None or Media.VIDEO in _output_media
    ) and "generate_video" not in _tool_names:
        _tools_list.append(
            _make_generate_video_tool(
                get_generator=self._resolve_video_generator, emit=self._emit_event
            )
        )
    if (
        _voice_gen is not None or Media.AUDIO in _output_media
    ) and "generate_voice" not in _tool_names:
        _tools_list.append(
            _make_generate_voice_tool(
                get_generator=self._resolve_voice_generator, emit=self._emit_event
            )
        )
    if self._knowledge is not None and "search_knowledge" not in _tool_names:

        def _get_bt() -> object | None:
            return self.get_budget_tracker() if hasattr(self, "get_budget_tracker") else None

        def _get_model() -> object | None:
            return getattr(self, "_model", None)

        self._knowledge._attach_to_agent(
            emit=self._emit_event, get_budget_tracker=_get_bt, get_model=_get_model
        )
        _tools_list.append(
            _make_search_knowledge_tool(
                get_knowledge=lambda: self._knowledge,
                emit=self._emit_event,
                get_model=_get_model,
                get_budget_tracker=_get_bt,
                get_runtime=lambda: self._runtime,
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
                            get_runtime=lambda: self._runtime,
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
    self._mcp_instances = mcp_instances
    self._guardrails_disabled = set()
    self._tools_disabled = set()
    self._mcp_disabled = set()
    self._runtime = _AgentRuntime()
    for i, x in enumerate(tools_list):
        if _is_mcp(x) and hasattr(x, "tools") and callable(x.tools):
            for t in x.tools():
                if isinstance(t, ToolSpec):
                    self._runtime.mcp_tool_indices[t.name] = i
    self._max_tool_iterations = max_tool_iterations
    self._parent_agent = None

    self._spotlight_tool_outputs = spotlight_tool_outputs
    self._normalize_inputs = normalize_inputs
    self._tool_error_mode = tool_error_mode
    ctx = context
    if ctx is None:
        context_manager = DefaultContextManager(Context())
    elif isinstance(ctx, _ContextConfig):
        context_manager = DefaultContextManager(ctx.to_context())
    elif isinstance(ctx, Context):
        context_manager = DefaultContextManager(ctx)
    else:
        context_manager = ctx
    ctx_config = getattr(context_manager, "context", None)
    token_limits = getattr(ctx_config, "token_limits", None) if ctx_config else None
    self._context_component = AgentContextComponent(context_manager, token_limits)
    persistent_memory, memory_backend = _resolve_memory(memory)
    self._memory_component = AgentMemoryComponent(persistent_memory, memory_backend)
    self._budget_component = AgentBudgetComponent(
        budget, budget_store, self._context_component.token_limits
    )
    self._provider = _resolve_provider(self._model, self._model_config)
    object.__setattr__(self, "_agent_name", name)
    object.__setattr__(self, "_agent_instance_id", f"{name}-{uuid.uuid4().hex[:8]}")
    self._description = description
    self._dependencies = dependencies
    if self._budget_component.budget is not None:
        self._budget_component.budget._consume_callback = _budget_make_consume_callback(self)
    if (
        self._budget_component.budget is not None
        and self._budget_component.budget.rate_limits is not None
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
    if loop is not None:
        if isinstance(loop, type) and hasattr(loop, "run") and callable(loop.run):
            loop_instance = loop()
        elif hasattr(loop, "run") and callable(loop.run):
            loop_instance = loop  # type: ignore[assignment]
        else:
            loop_instance = ReactLoop(max_iterations=max_tool_iterations)
    else:
        loop_instance = ReactLoop(max_iterations=max_tool_iterations)
    self._loop = loop_instance
    self._last_iteration = 0
    self._conversation_id = None
    self._child_count = 0
    if max_child_agents is not None:
        self._max_child_agents = max_child_agents
    if guardrails is None or (isinstance(guardrails, list) and len(guardrails) == 0):
        _guardrails = GuardrailChain()
    elif isinstance(guardrails, GuardrailChain):
        _guardrails = guardrails
    else:
        _guardrails = GuardrailChain(list(guardrails))
    self._guardrails_component = AgentGuardrailsComponent(_guardrails)
    self._debug = debug
    _tracer: Tracer = tracer or get_tracer()
    if debug and not any(isinstance(e, ConsoleExporter) for e in _tracer._exporters):
        _tracer.add_exporter(ConsoleExporter())
    if debug:
        _tracer.set_debug_mode(True)
    self._observability_component = AgentObservabilityComponent(_tracer, event_bus, audit)
    if hasattr(self._context, "set_emit_fn"):
        self._context.set_emit_fn(cast(object, self._emit_event))
    if hasattr(self._context, "set_tracer"):
        self._context.set_tracer(self._tracer)
    if rate_limit is None:
        self._rate_limit_manager = None
    elif isinstance(rate_limit, RateLimitManager):
        self._rate_limit_manager = rate_limit
    else:
        self._rate_limit_manager = cast(RateLimitManager, create_rate_limit_manager(rate_limit))
    if self._output is not None:
        self._validation_retries = self._output.validation_retries
        self._validation_context = self._output.context
        self._output_validator = self._output.validator
    else:
        self._validation_retries = 3
        self._validation_context = {}
        self._output_validator = None
    if self._rate_limit_manager and hasattr(self._rate_limit_manager, "set_emit_fn"):
        self._rate_limit_manager.set_emit_fn(self._emit_event)
    if self._rate_limit_manager and hasattr(self._rate_limit_manager, "set_tracer"):
        self._rate_limit_manager.set_tracer(self._tracer)
    self.events = Events(self._emit_event)
    if audit is not None:
        if not isinstance(audit, AuditLog):
            raise TypeError(
                f"audit must be AuditLog or None, got {type(audit).__name__}. "
                "Use AuditLog(path='./audit.jsonl') for JSONL logging."
            )
        audit_handler = AuditHookHandler(source=self._agent_name, config=audit)
        self.events.on_all(audit_handler)
    self._run_report = AgentReport()
    self._approval_gate = approval_gate
    self._human_approval_timeout = human_approval_timeout
    self._max_tool_result_length = max_tool_result_length
    self._retry_on_transient = retry_on_transient
    self._max_retries = max_retries
    self._retry_backoff_base = retry_backoff_base
    self._max_input_length = max_input_length
    self._circuit_breaker = circuit_breaker
    self._fallback_provider = None
    self._fallback_model_config = None
    if circuit_breaker is not None and not isinstance(circuit_breaker, CircuitBreaker):
        raise TypeError(
            f"circuit_breaker must be CircuitBreaker or None, got {type(circuit_breaker).__name__}"
        )
    if checkpoint is None:
        self._checkpoint_config = None
        self._checkpointer = None
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
    from syrin.remote._hooks import on_agent_init as _remote_init

    _remote_init(self)
