"""Internal agent helpers. Not part of public API."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast

from syrin._sentinel import NOT_PROVIDED
from syrin.budget import Budget
from syrin.context import Context, DefaultContextManager
from syrin.context.snapshot import ContextSnapshot
from syrin.enums import Hook, Media, MemoryBackend, MemoryPreset, MemoryType
from syrin.events import EventContext
from syrin.exceptions import ModalityNotSupportedError
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

    def get_map(self) -> Any:
        """Return the persistent context map. Empty if no map backend configured."""
        return self._manager.get_map()

    def update_map(self, partial: Any) -> None:
        """Merge partial into the persistent map and persist. No-op if no map backend."""
        return self._manager.update_map(partial)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._config, name)


class _AgentRuntime:
    """Internal runtime state. Not part of public API."""

    __slots__ = ("remote_baseline", "remote_overrides", "mcp_tool_indices", "budget_tracker_shared")

    def __init__(self) -> None:
        self.remote_baseline: dict[str, object] | None = None
        self.remote_overrides: dict[str, object] = {}
        self.mcp_tool_indices: dict[str, int] = {}
        self.budget_tracker_shared: bool = False


def _validate_agent_media(
    router: Any,
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


def _make_generate_image_tool(
    get_generator: Callable[[], Any],
    emit: Callable[[str, dict[str, Any]], None] | None = None,
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


def _make_generate_video_tool(
    get_generator: Callable[[], Any],
    emit: Callable[[str, dict[str, Any]], None] | None = None,
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


def _make_search_knowledge_tool(
    get_knowledge: Callable[[], object],
    emit: Callable[[str, dict[str, object]], None] | None = None,
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
        return await search_knowledge_deep(
            knowledge=kb,
            query=query,
            source=source,
            config=config,
            get_model=get_model,
            emit=emit_fn,
            get_budget_tracker=get_budget_tracker,
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


def _merge_class_attrs(mro: tuple[type, ...], name: str, merge: bool) -> Any:
    """From MRO: for 'merge' (e.g. tools) concatenate lists; else first defined."""
    tools_fallback = name == "tools"

    def _get(cls: type, attr: str) -> Any:
        val = cls.__dict__.get(attr, NOT_PROVIDED)
        if val is NOT_PROVIDED and tools_fallback:
            val = cls.__dict__.get("_syrin_class_tools", NOT_PROVIDED)
        return val

    if merge:
        out: list[Any] = []
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


def _collect_system_prompt_method(cls: type) -> Any:
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


def _is_prompt(x: Any) -> bool:
    """Return True if x is a Prompt (from @prompt)."""
    return hasattr(x, "variables") and callable(x)


def _is_valid_system_prompt(x: Any) -> bool:
    """Return True if x is valid system_prompt: str, Prompt, or callable."""
    if isinstance(x, str):
        return True
    if _is_prompt(x):
        return True
    return callable(x) and not isinstance(x, type)


def _is_mcp(x: Any) -> bool:
    """Return True if x is an MCP server instance (has _tool_specs)."""
    return hasattr(x, "_tool_specs") and hasattr(x, "tools")


def _expand_tool_sources(items: list[Any]) -> list[ToolSpec]:
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


def _bind_tool_to_instance(spec: ToolSpec, instance: Any) -> ToolSpec:
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


def _validate_user_input(
    user_input: str | list[dict[str, Any]] | None,
    method: str = "response",
) -> None:
    """Raise TypeError if user_input is not str or list[dict] (MultimodalInput)."""
    if user_input is None:
        raise TypeError(
            f"user_input must be str or list[dict] (MultimodalInput), got None. "
            f'Example: agent.{method}("Hello")'
        )
    if isinstance(user_input, str):
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


def _normalize_tools(tools_list: list[Any], instance: Any) -> tuple[list[ToolSpec], list[Any]]:
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


def _validate_budget(budget: Any) -> Budget | None:
    """Validate budget is Budget or None."""
    if budget is None:
        return None
    if not isinstance(budget, Budget):
        raise TypeError(
            f"budget must be Budget, got {type(budget).__name__}. Use Budget(run=1.0, per=...)."
        )
    return budget


def _resolve_memory(
    memory: Memory | MemoryPreset | None,
) -> tuple[Memory | None, InMemoryBackend | None]:
    """Normalize memory argument to (persistent_memory, backend)."""
    disabled = memory is None or memory is MemoryPreset.DISABLED
    default = memory is MemoryPreset.DEFAULT
    if (
        memory is not None
        and memory is not MemoryPreset.DISABLED
        and memory is not MemoryPreset.DEFAULT
        and not isinstance(memory, Memory)
    ):
        raise TypeError(
            f"memory must be Memory, MemoryPreset.DEFAULT, MemoryPreset.DISABLED, or None, "
            f"got {type(memory).__name__}. Use Memory(...), MemoryPreset.DEFAULT, or None."
        )
    if disabled:
        return (None, None)
    if default:
        return (
            Memory(types=[MemoryType.CORE, MemoryType.EPISODIC], top_k=10),
            get_backend(MemoryBackend.MEMORY),
        )
    assert isinstance(memory, Memory)
    return (memory, get_backend(memory.backend, **memory._backend_kwargs()))


def _emit_domain_event_for_hook(hook: Hook, ctx: EventContext, bus: Any) -> None:
    """Emit domain events for hooks that have typed domain event equivalents."""
    if hook == Hook.BUDGET_THRESHOLD:
        from syrin.domain_events import BudgetThresholdReached

        pct = cast(int, ctx.get("threshold_percent", 0))
        current = cast(float, ctx.get("current_value", 0.0))
        limit = cast(float, ctx.get("limit_value", 0.0))
        metric = cast(str, ctx.get("metric", "cost"))
        bus.emit(BudgetThresholdReached(pct, current, limit, metric))
    elif hook == Hook.CONTEXT_COMPACT:
        from syrin.domain_events import ContextCompacted

        bus.emit(
            ContextCompacted(
                method=cast(str, ctx.get("method", "unknown")),
                tokens_before=cast(int, ctx.get("tokens_before", 0)),
                tokens_after=cast(int, ctx.get("tokens_after", 0)),
                messages_before=cast(int, ctx.get("messages_before", 0)),
                messages_after=cast(int, ctx.get("messages_after", 0)),
            )
        )
