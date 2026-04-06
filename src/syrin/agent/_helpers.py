"""Internal agent helpers. Not part of public API."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast

from syrin._sentinel import NOT_PROVIDED
from syrin.budget import Budget
from syrin.context import Context, DefaultContextManager
from syrin.context.snapshot import ContextSnapshot
from syrin.enums import Hook, Media
from syrin.events import EventContext
from syrin.exceptions import ModalityNotSupportedError
from syrin.knowledge._grounding import GroundedFact
from syrin.memory import Memory
from syrin.memory.backends import InMemoryBackend, get_backend
from syrin.model import Model
from syrin.providers.base import Provider
from syrin.tool import ToolSpec
from syrin.types import ModelConfig


class _ContextFacade:
    """Facade for agent.context: config attributes + compact() and snapshot() during/after prepare."""

    def __init__(self, config: Context, manager: DefaultContextManager) -> None:
        self._config = config
        self._manager = manager

    def compact(self) -> None:
        """Request context compaction (valid during prepare, e.g. from threshold action)."""
        self._manager.compact()

    def snapshot(self) -> ContextSnapshot:
        """Return a point-in-time view of the context from the last prepare."""
        return self._manager.snapshot()

    def get_map(self) -> object:
        """Return the persistent context map. Empty if no map backend configured."""
        return self._manager.get_map()

    def update_map(self, partial: object) -> None:
        """Merge partial into the persistent map and persist. No-op if no map backend."""
        return self._manager.update_map(partial)  # type: ignore[arg-type]

    def __getattr__(self, name: str) -> object:
        return getattr(self._config, name)


class _AgentRuntime:
    """Internal runtime state. Not part of public API."""

    __slots__ = (
        "remote_baseline",
        "remote_overrides",
        "mcp_tool_indices",
        "budget_tracker_shared",
        "grounded_facts",
    )

    def __init__(self) -> None:
        self.remote_baseline: dict[str, object] | None = None
        self.remote_overrides: dict[str, object] = {}
        self.mcp_tool_indices: dict[str, int] = {}
        self.budget_tracker_shared: bool = False
        self.grounded_facts: list[GroundedFact] = []


def _validate_agent_media(
    router: object,
    *,
    input_media: set[Media] | None = None,
    output_media: set[Media] | None = None,
) -> None:
    """Validate that router profiles cover declared input/output media. Raise if not."""
    if not input_media and not output_media:
        return
    profiles = getattr(router, "_profiles", []) or []
    if not profiles:
        return
    input_supported: set[Media] = set()
    output_supported: set[Media] = set()
    for p in profiles:
        inp = getattr(p, "input_media", None) or {Media.TEXT}
        out = getattr(p, "output_media", None) or {Media.TEXT}
        input_supported |= inp
        output_supported |= out
    if input_media and not (input_media <= input_supported):
        missing = input_media - input_supported
        raise ModalityNotSupportedError(
            f"Agent input_media {sorted(m.value for m in input_media)} require "
            f"profiles supporting {sorted(m.value for m in missing)}, but router profiles "
            f"only support {sorted(m.value for m in input_supported)}. "
            "Add a profile with input_media including these, or relax input_media.",
            required={m.value for m in input_media},
            supported={m.value for m in input_supported},
        )
    if output_media and not (output_media <= output_supported):
        missing = output_media - output_supported
        raise ModalityNotSupportedError(
            f"Agent output_media {sorted(m.value for m in output_media)} require "
            f"profiles supporting {sorted(m.value for m in missing)}, but router profiles "
            f"only support {sorted(m.value for m in output_supported)}. "
            "Add a profile with output_media including these, or relax output_media.",
            required={m.value for m in output_media},
            supported={m.value for m in output_supported},
        )


def _make_generate_image_tool(  # type: ignore[explicit-any]
    get_generator: Callable[[], Any],
    emit: Callable[[str, dict[str, object]], None] | None = None,
) -> ToolSpec:
    """Build a ToolSpec for generate_image. Uses DI — no closure over agent."""

    def generate_image_tool(prompt: str, aspect_ratio: str = "1:1") -> str:
        img_gen = get_generator()
        if img_gen is None:
            return (
                "Image generation is not available. Provide api_key via a Google model or "
                "image_generation=ImageGenerator.Gemini(api_key=...). Install: pip install syrin[generation]"
            )
        from syrin.enums import AspectRatio

        try:
            ar = AspectRatio(aspect_ratio)
        except (ValueError, KeyError, TypeError):
            ar = AspectRatio.ONE_TO_ONE
        results = img_gen.generate(prompt, aspect_ratio=ar, emit=emit)
        if not results:
            return "Image generation failed."
        r = results[0]
        if r.success and r.url:
            return f"Generated image: {r.url}"
        return r.error or "Image generation failed."

    schema: dict[str, object] = {
        "type": "object",
        "properties": {
            "prompt": {
                "type": "string",
                "description": "Detailed description of the image to generate.",
            },
            "aspect_ratio": {
                "type": "string",
                "enum": ["1:1", "3:4", "4:3", "9:16", "16:9"],
                "default": "1:1",
                "description": "Aspect ratio: 1:1, 3:4, 4:3, 9:16, or 16:9.",
            },
        },
        "required": ["prompt"],
    }
    return ToolSpec(
        name="generate_image",
        description=(
            "Generate an image from a text description. Use when the user asks to "
            "create, draw, or generate an image."
        ),
        parameters_schema=schema,
        func=generate_image_tool,
    )


def _make_generate_video_tool(  # type: ignore[explicit-any]
    get_generator: Callable[[], Any],
    emit: Callable[[str, dict[str, object]], None] | None = None,
) -> ToolSpec:
    """Build a ToolSpec for generate_video. Uses DI — no closure over agent."""

    def generate_video_tool(prompt: str, aspect_ratio: str = "16:9") -> str:
        vid_gen = get_generator()
        if vid_gen is None:
            return (
                "Video generation is not available. Provide api_key via a Google model or "
                "video_generation=VideoGenerator.Gemini(api_key=...). Install: pip install syrin[generation]"
            )
        from syrin.enums import AspectRatio

        try:
            ar = AspectRatio(aspect_ratio)
        except (ValueError, KeyError, TypeError):
            ar = AspectRatio.SIXTEEN_NINE
        result = vid_gen.generate(prompt, aspect_ratio=ar, emit=emit)
        if result.success and result.url:
            return f"Generated video: {result.url}"
        return result.error or "Video generation failed."

    schema: dict[str, object] = {
        "type": "object",
        "properties": {
            "prompt": {
                "type": "string",
                "description": "Detailed description of the video to generate.",
            },
            "aspect_ratio": {
                "type": "string",
                "enum": ["16:9", "9:16"],
                "default": "16:9",
                "description": "Aspect ratio: 16:9 or 9:16.",
            },
        },
        "required": ["prompt"],
    }
    return ToolSpec(
        name="generate_video",
        description=(
            "Generate a short video from a text description. Use when the user asks "
            "to create or generate a video."
        ),
        parameters_schema=schema,
        func=generate_video_tool,
    )


def _make_generate_voice_tool(
    get_generator: Callable[[], object | None],
    emit: Callable[[str, dict[str, object]], None] | None = None,
) -> ToolSpec:
    """Build a ToolSpec for generate_voice. Uses DI — no closure over agent."""

    def generate_voice_tool(
        text: str,
        voice_id: str | None = None,
        speed: float = 1.0,
        language: str | None = None,
    ) -> str:
        gen = get_generator()
        if gen is None:
            return (
                "Voice generation is not available. Pass voice_generation=VoiceGenerator.OpenAI(api_key=...) "
                "or VoiceGenerator.ElevenLabs(api_key=...). Install: pip install syrin[voice] or syrin[openai]"
            )
        generate_fn = getattr(gen, "generate", None)
        if generate_fn is None:
            return "Voice generation failed: invalid generator."
        try:
            result = generate_fn(
                text,
                voice_id=voice_id or "default",
                speed=speed,
                language=language or "en",
                emit=emit,
            )
        except Exception as e:
            return f"Voice generation failed: {e!s}"
        if result.success and result.url:
            return f"Generated audio: {result.url}"
        return result.error or "Voice generation failed."

    schema: dict[str, object] = {
        "type": "object",
        "properties": {
            "text": {
                "type": "string",
                "description": "Text to convert to speech.",
            },
            "voice_id": {
                "type": "string",
                "description": "Optional voice identifier (provider-specific).",
            },
            "speed": {
                "type": "number",
                "description": "Speech rate (e.g. 0.5–2.0). Default 1.0.",
                "default": 1.0,
            },
            "language": {
                "type": "string",
                "description": "Optional language code (e.g. en, hi).",
            },
        },
        "required": ["text"],
    }
    return ToolSpec(
        name="generate_voice",
        description=(
            "Generate speech audio from text. Use when the user asks to speak, "
            "say something aloud, or produce audio."
        ),
        parameters_schema=schema,
        func=generate_voice_tool,
    )


def _make_search_knowledge_tool(
    get_knowledge: Callable[[], object],
    emit: Callable[[str, dict[str, object]], None] | None = None,
    get_model: Callable[[], object | None] | None = None,
    get_budget_tracker: Callable[[], object | None] | None = None,
    get_runtime: Callable[[], _AgentRuntime] | None = None,
) -> ToolSpec:
    """Build a ToolSpec for search_knowledge. Uses DI — no closure over agent."""

    async def search_knowledge_tool(query: str, source: str | None = None) -> str:
        kb = get_knowledge()
        if kb is None:
            return "Knowledge search is not available. No knowledge base configured."
        from typing import cast

        from syrin.knowledge import Knowledge
        from syrin.knowledge._store import MetadataFilter

        k = kb
        if not isinstance(k, Knowledge):
            return "Knowledge search failed: invalid knowledge instance."
        filt: MetadataFilter | None = cast(MetadataFilter, {"source": source}) if source else None
        results = await k.search(query, filter=filt)
        if not results:
            return "No relevant results found."
        gconfig = getattr(k, "_grounding_config", None)
        if gconfig is not None and gconfig.enabled:
            from syrin.knowledge._grounding import apply_grounding

            formatted, facts = await apply_grounding(
                query=query,
                results=results,
                config=gconfig,
                get_model=get_model or (lambda: None),
                emit=emit,
                get_budget_tracker=get_budget_tracker or (lambda: None),
            )
            if get_runtime is not None and facts:
                get_runtime().grounded_facts.extend(facts)
            return formatted
        lines: list[str] = []
        for r in results[:5]:
            lines.append(f"[{r.rank}] (score={r.score:.2f}) {r.chunk.content[:300]}")
        return "\n\n".join(lines)

    schema: dict[str, object] = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search query for the knowledge base.",
            },
            "source": {
                "type": "string",
                "description": "Optional filter by source (e.g. document name).",
            },
        },
        "required": ["query"],
    }
    return ToolSpec(
        name="search_knowledge",
        description=(
            "Search the knowledge base for relevant information. "
            "Use when you need to find facts, documents, or context before answering."
        ),
        parameters_schema=schema,
        func=search_knowledge_tool,
    )


def _make_search_knowledge_deep_tool(
    get_knowledge: Callable[[], object],
    get_model: Callable[[], object | None],
    get_budget_tracker: Callable[[], object | None],
    emit: Callable[[str, dict[str, object]], None] | None = None,
    get_runtime: Callable[[], _AgentRuntime] | None = None,
) -> ToolSpec:
    """Build a ToolSpec for search_knowledge_deep (agentic multi-step retrieval)."""

    async def search_knowledge_deep_tool(query: str, source: str | None = None) -> str:
        kb = get_knowledge()
        if kb is None:
            return "Knowledge search is not available. No knowledge base configured."
        from syrin.knowledge import Knowledge
        from syrin.knowledge._agentic import search_knowledge_deep

        if not isinstance(kb, Knowledge):
            return "Knowledge search failed: invalid knowledge instance."
        config = getattr(kb, "_agentic_config", None)
        if config is None:
            return "search_knowledge_deep requires agentic=True on Knowledge."
        emit_fn = getattr(kb, "_emit", None)
        append_facts: Callable[[list[GroundedFact]], None] | None = None
        if get_runtime is not None:

            def _append(facts: list[GroundedFact]) -> None:
                get_runtime().grounded_facts.extend(facts)

            append_facts = _append
        return await search_knowledge_deep(
            knowledge=kb,
            query=query,
            source=source,
            config=config,
            get_model=get_model,
            emit=emit_fn,
            get_budget_tracker=get_budget_tracker,
            append_grounded_facts=append_facts,
        )

    schema: dict[str, object] = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Complex or multi-part query. Will be decomposed into sub-queries.",
            },
            "source": {
                "type": "string",
                "description": "Optional filter by source (e.g. document name).",
            },
        },
        "required": ["query"],
    }
    return ToolSpec(
        name="search_knowledge_deep",
        description=(
            "Deep search: decomposes complex queries into sub-queries, searches multiple times, "
            "grades and refines results. Use for complex or multi-part questions."
        ),
        parameters_schema=schema,
        func=search_knowledge_deep_tool,
    )


def _make_verify_knowledge_tool(
    get_knowledge: Callable[[], object],
    get_model: Callable[[], object | None],
    get_budget_tracker: Callable[[], object | None],
    emit: Callable[[str, dict[str, object]], None] | None = None,
) -> ToolSpec:
    """Build a ToolSpec for verify_knowledge (claim verification)."""

    async def verify_knowledge_tool(claim: str) -> str:
        kb = get_knowledge()
        if kb is None:
            return "Knowledge verification is not available. No knowledge base configured."
        from syrin.knowledge import Knowledge
        from syrin.knowledge._agentic import verify_knowledge

        if not isinstance(kb, Knowledge):
            return "Knowledge verification failed: invalid knowledge instance."
        config = getattr(kb, "_agentic_config", None)
        if config is None:
            return "verify_knowledge requires agentic=True on Knowledge."
        emit_fn = getattr(kb, "_emit", None)
        return await verify_knowledge(
            knowledge=kb,
            claim=claim,
            config=config,
            get_model=get_model,
            emit=emit_fn,
            get_budget_tracker=get_budget_tracker,
        )

    schema: dict[str, object] = {
        "type": "object",
        "properties": {
            "claim": {
                "type": "string",
                "description": "The claim to verify against the knowledge base.",
            },
        },
        "required": ["claim"],
    }
    return ToolSpec(
        name="verify_knowledge",
        description=(
            "Verify if a specific claim is supported, contradicted, or not found in the knowledge base. "
            "Use to fact-check before responding."
        ),
        parameters_schema=schema,
        func=verify_knowledge_tool,
    )


def _merge_class_attrs(mro: tuple[type, ...], name: str, merge: bool) -> object:
    """From MRO: for 'merge' (e.g. tools) concatenate lists; else first defined."""
    tools_fallback = name == "tools"

    def _get(cls: type, attr: str) -> object:
        val = cls.__dict__.get(attr, NOT_PROVIDED)
        if val is NOT_PROVIDED and tools_fallback:
            val = cls.__dict__.get("_syrin_class_tools", NOT_PROVIDED)
        return val

    if merge:
        out: list[object] = []
        for cls in mro:
            if cls is object:
                continue
            val = _get(cls, name)
            if val is NOT_PROVIDED or val is None:
                continue
            if hasattr(val, "__get__"):
                continue
            if isinstance(val, list):
                out.extend(val)
            else:
                out.append(val)
        return out
    for cls in mro:
        if cls is object:
            continue
        val = _get(cls, name)
        if val is NOT_PROVIDED:
            continue
        if hasattr(val, "__get__"):
            continue
        return val
    return NOT_PROVIDED


def _collect_system_prompt_method(cls: type) -> object:
    """Find @system_prompt-decorated method in MRO. First (subclass) wins. Returns None if none."""
    for c in cls.__mro__:
        if c is object:
            continue
        for _attr_name, val in c.__dict__.items():
            if callable(val) and getattr(val, "_syrin_system_prompt", False):
                return val
    return None


def _get_system_prompt_method_names(cls: type) -> list[str]:
    """Return names of @system_prompt-decorated methods on cls itself (not inherited)."""
    return [
        attr_name
        for attr_name, val in cls.__dict__.items()
        if callable(val) and getattr(val, "_syrin_system_prompt", False)
    ]


def _collect_class_tools(cls: type) -> list[ToolSpec]:
    """Collect ToolSpec from @tool-decorated class methods (MRO order, subclass overrides)."""
    seen: set[str] = set()
    result: list[ToolSpec] = []
    for c in cls.__mro__:
        if c is object:
            continue
        for _attr_name, val in c.__dict__.items():
            if isinstance(val, ToolSpec) and val.name not in seen:
                seen.add(val.name)
                result.append(val)
    return result


def _is_prompt(x: object) -> bool:
    """Return True if x is a Prompt (from @prompt)."""
    return hasattr(x, "variables") and callable(x)


def _is_valid_system_prompt(x: object) -> bool:
    """Return True if x is valid system_prompt: str, Prompt, or callable."""
    if isinstance(x, str):
        return True
    if _is_prompt(x):
        return True
    return callable(x) and not isinstance(x, type)


def _is_mcp(x: object) -> bool:
    """Return True if x is an MCP server instance (has _tool_specs)."""
    return hasattr(x, "_tool_specs") and hasattr(x, "tools")


def _expand_tool_sources(items: list[object]) -> list[ToolSpec]:
    """Expand MCP/MCPClient to ToolSpec; pass through ToolSpec; flatten lists from mcp.select()."""
    out: list[ToolSpec] = []
    for x in items:
        if isinstance(x, ToolSpec):
            out.append(x)
        elif isinstance(x, list):
            out.extend(t for t in x if isinstance(t, ToolSpec))
        elif hasattr(x, "tools") and callable(x.tools):
            out.extend(x.tools())
        elif isinstance(x, ToolSpec):
            out.append(x)
    return out


def _bind_tool_to_instance(spec: ToolSpec, instance: object) -> ToolSpec:
    """If spec.func is an unbound method (has 'self'), bind it to instance."""
    import inspect

    sig = inspect.signature(spec.func)
    params = list(sig.parameters)
    if params and params[0] == "self":
        bound_func = spec.func.__get__(instance, type(instance))
        return ToolSpec(
            name=spec.name,
            description=spec.description,
            parameters_schema=spec.parameters_schema,
            func=bound_func,
            requires_approval=spec.requires_approval,
            inject_run_context=spec.inject_run_context,
        )
    return spec


_MAX_INPUT_LENGTH = 1_000_000  # 1MB default; protects against accidental huge payloads (SEC5)


def _validate_user_input(
    user_input: str | list[dict[str, object]] | None,
    method: str = "response",
    max_input_length: int = _MAX_INPUT_LENGTH,
) -> None:
    """Raise TypeError/InputTooLargeError if user_input is invalid or too large.

    Enforces max_input_length (default 1MB) to prevent sending huge payloads
    to the LLM API.
    """
    from syrin.exceptions import InputTooLargeError

    if user_input is None:
        raise TypeError(
            f"user_input must be str or list[dict] (MultimodalInput), got None. "
            f'Example: agent.{method}("Hello")'
        )
    if isinstance(user_input, str):
        if len(user_input) > max_input_length:
            raise InputTooLargeError(
                f"Input too large: {len(user_input):,} characters exceeds "
                f"max_input_length={max_input_length:,}. "
                "Reduce input size or increase max_input_length on Agent.",
                input_length=len(user_input),
                max_length=max_input_length,
            )
        return
    if isinstance(user_input, list) and all(isinstance(x, dict) for x in user_input):
        return
    got = type(user_input).__name__
    raise TypeError(
        f"user_input must be str or list[dict] (MultimodalInput), got {got}. "
        f'Example: agent.{method}("Hello")'
    )


def _resolve_provider(model: Model | None, model_config: ModelConfig) -> Provider:
    """Resolve Provider from Model (preferred) or ModelConfig.provider via registry."""
    if model is not None and hasattr(model, "get_provider"):
        return model.get_provider()
    from syrin.providers.registry import get_provider

    return get_provider(model_config.provider, strict=True)


def _normalize_tools(
    tools_list: list[object], instance: object
) -> tuple[list[ToolSpec], list[object]]:
    """Expand, validate, and bind tools to agent instance."""
    mcp_instances = [x for x in tools_list if _is_mcp(x)]
    expanded = _expand_tool_sources(tools_list)
    out: list[ToolSpec] = []
    for i, t in enumerate(expanded):
        if t is None:
            raise TypeError(
                "tools must not contain None. Use list of ToolSpec (from @syrin.tool or syrin.tool())."
            )
        if not isinstance(t, ToolSpec):
            raise TypeError(
                f"tools[{i}] must be ToolSpec, got {type(t).__name__}. Use @syrin.tool or syrin.tool()."
            )
        out.append(_bind_tool_to_instance(t, instance))
    return (out, mcp_instances)


def _validate_budget(budget: object) -> Budget | None:
    """Validate budget is Budget or None."""
    if budget is None:
        return None
    if not isinstance(budget, Budget):
        raise TypeError(
            f"budget must be Budget, got {type(budget).__name__}. Use Budget(max_cost=1.0, rate_limits=...)."
        )
    return budget


def _resolve_memory(
    memory: Memory | None,
) -> tuple[Memory | None, InMemoryBackend | None]:
    """Normalize memory argument to (persistent_memory, backend)."""
    if memory is None:
        return (None, None)
    if not isinstance(memory, Memory):
        raise TypeError(
            f"memory must be Memory or None, got {type(memory).__name__}. Use Memory(...) or None."
        )
    return (memory, get_backend(memory.backend, **memory._backend_kwargs()))


def _emit_domain_event_for_hook(hook: Hook, ctx: EventContext, bus: object) -> None:
    """Emit domain events for hooks that have typed domain event equivalents."""
    if hook == Hook.AGENT_RUN_START:
        from syrin.domain_events import AgentRunStarted

        bus.emit(  # type: ignore[attr-defined]
            AgentRunStarted(
                input=cast(str, ctx.get("input", "")),
                model=cast(str, ctx.get("model", "")),
                iteration=cast(int, ctx.get("iteration", 0)),
            )
        )
    elif hook == Hook.AGENT_RUN_END:
        from syrin.domain_events import AgentRunEnded

        bus.emit(  # type: ignore[attr-defined]
            AgentRunEnded(
                content=cast(str, ctx.get("content", "")),
                cost=cast(float, ctx.get("cost", 0.0)),
                tokens=cast(int, ctx.get("tokens", 0)),
                duration=cast(float, ctx.get("duration", 0.0)),
                stop_reason=cast(str, ctx.get("stop_reason", "")),
                iteration=cast(int, ctx.get("iteration", 0)),
            )
        )
    elif hook == Hook.LLM_REQUEST_START:
        from syrin.domain_events import LLMRequestStarted

        tools = ctx.get("tools", [])
        bus.emit(  # type: ignore[attr-defined]
            LLMRequestStarted(
                iteration=cast(int, ctx.get("iteration", 0)),
                tool_count=len(tools) if isinstance(tools, list) else 0,
            )
        )
    elif hook == Hook.LLM_REQUEST_END:
        from syrin.domain_events import LLMRequestCompleted

        bus.emit(  # type: ignore[attr-defined]
            LLMRequestCompleted(
                content=cast(str, ctx.get("content", "")),
                iteration=cast(int, ctx.get("iteration", 0)),
            )
        )
    elif hook == Hook.TOOL_CALL_END:
        from syrin.domain_events import ToolCallCompleted

        bus.emit(  # type: ignore[attr-defined]
            ToolCallCompleted(
                tool_name=cast(str, ctx.get("tool_name", "")),
                duration_ms=cast(float, ctx.get("duration_ms", 0.0)),
            )
        )
    elif hook == Hook.TOOL_ERROR:
        from syrin.domain_events import ToolCallFailed

        bus.emit(  # type: ignore[attr-defined]
            ToolCallFailed(
                tool_name=cast(str, ctx.get("tool_name", "")),
                error=cast(str, ctx.get("error", "")),
                iteration=cast(int, ctx.get("iteration", 0)),
            )
        )
    elif hook == Hook.BUDGET_THRESHOLD:
        from syrin.domain_events import BudgetThresholdReached

        pct = cast(int, ctx.get("threshold_percent", 0))
        current = cast(float, ctx.get("current_value", 0.0))
        limit = cast(float, ctx.get("limit_value", 0.0))
        metric = cast(str, ctx.get("metric", "cost"))
        bus.emit(BudgetThresholdReached(pct, current, limit, metric))  # type: ignore[attr-defined]
    elif hook == Hook.BUDGET_EXCEEDED:
        from syrin.domain_events import BudgetExceeded

        used = cast(float, ctx.get("used", 0.0))
        limit = cast(float, ctx.get("limit", 0.0))
        bus.emit(BudgetExceeded(used=used, limit=limit, exceeded_by=used - limit))  # type: ignore[attr-defined]
    elif hook == Hook.GUARDRAIL_BLOCKED:
        from syrin.domain_events import GuardrailBlocked

        names = ctx.get("guardrail_names", [])
        bus.emit(  # type: ignore[attr-defined]
            GuardrailBlocked(
                stage=cast(str, ctx.get("stage", "")),
                reason=cast(str, ctx.get("reason", "")),
                guardrail_names=list(names) if isinstance(names, list) else [],
            )
        )
    elif hook == Hook.HANDOFF_START:
        from syrin.domain_events import HandoffStarted

        bus.emit(  # type: ignore[attr-defined]
            HandoffStarted(
                target_agent=cast(str, ctx.get("target_agent", "")),
                task=cast(str, ctx.get("task", "")),
            )
        )
    elif hook == Hook.HANDOFF_END:
        from syrin.domain_events import HandoffCompleted

        bus.emit(  # type: ignore[attr-defined]
            HandoffCompleted(
                target_agent=cast(str, ctx.get("target_agent", "")),
                success=bool(ctx.get("success", True)),
            )
        )
    elif hook == Hook.CONTEXT_COMPACT:
        from syrin.domain_events import ContextCompacted

        bus.emit(  # type: ignore[attr-defined]
            ContextCompacted(
                method=cast(str, ctx.get("method", "unknown")),
                tokens_before=cast(int, ctx.get("tokens_before", 0)),
                tokens_after=cast(int, ctx.get("tokens_after", 0)),
                messages_before=cast(int, ctx.get("messages_before", 0)),
                messages_after=cast(int, ctx.get("messages_after", 0)),
            )
        )
