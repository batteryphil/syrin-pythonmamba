"""Tool execution use case: execute tools by name.

Agent delegates to execute_tool. Public API stays on Agent.
Tools may be sync (return str) or async (return coroutine); caller must await.
"""

from __future__ import annotations

import asyncio
import inspect
from typing import TYPE_CHECKING, Any, get_type_hints

if TYPE_CHECKING:
    from syrin.agent import Agent

from syrin.enums import ToolErrorMode
from syrin.exceptions import ToolArgumentError, ToolExecutionError
from syrin.run_context import RunContext
from syrin.tool import ToolSpec


def _validate_and_coerce_args(spec: ToolSpec, arguments: dict[str, object]) -> dict[str, object]:
    """Validate and coerce LLM-generated arguments against tool type hints.

    - Missing required params: uses function default or raises ToolArgumentError.
    - Wrong container type (list vs str, dict vs str, etc.): raises ToolArgumentError.
    - Numeric/bool strings for numeric/bool params: silently coerced.

    Returns a possibly-coerced copy of arguments.
    """
    try:
        hints = get_type_hints(spec.func)
    except Exception:
        return arguments  # If we can't introspect, skip validation

    sig = inspect.signature(spec.func)
    coerced = dict(arguments)

    for param_name, param in sig.parameters.items():
        if param_name in ("ctx", "return"):
            continue  # Skip injected context and return type

        hint = hints.get(param_name)
        if hint is None:
            continue  # No annotation — skip

        # Resolve Optional[T] → (T, allow_none=True)
        origin = getattr(hint, "__origin__", None)
        type_args = getattr(hint, "__args__", ())
        allow_none = type(None) in (type_args or ())
        if allow_none and type_args:
            non_none = [a for a in type_args if a is not type(None)]
            hint = non_none[0] if len(non_none) == 1 else hint
            origin = getattr(hint, "__origin__", None)

        # Determine if param is present
        if param_name not in coerced:
            if param.default is inspect.Parameter.empty:
                raise ToolArgumentError(
                    f"Tool {spec.name!r}: required argument {param_name!r} is missing.",
                    tool_name=spec.name,
                    param_name=param_name,
                    expected_type=str(hint),
                    received_type="missing",
                )
            continue  # Has a default — fine

        value = coerced[param_name]

        if value is None:
            if not allow_none:
                raise ToolArgumentError(
                    f"Tool {spec.name!r}: argument {param_name!r} received None but "
                    f"expected {hint.__name__ if hasattr(hint, '__name__') else hint}.",
                    tool_name=spec.name,
                    param_name=param_name,
                    expected_type=str(hint),
                    received_type="None",
                )
            continue

        # Container type mismatch (list, dict) — not coercible
        if origin is list:
            if not isinstance(value, list):
                raise ToolArgumentError(
                    f"Tool {spec.name!r}: argument {param_name!r} expected list, "
                    f"got {type(value).__name__!r}.",
                    tool_name=spec.name,
                    param_name=param_name,
                    expected_type="list",
                    received_type=type(value).__name__,
                )
            continue

        if origin is dict:
            if not isinstance(value, dict):
                raise ToolArgumentError(
                    f"Tool {spec.name!r}: argument {param_name!r} expected dict, "
                    f"got {type(value).__name__!r}.",
                    tool_name=spec.name,
                    param_name=param_name,
                    expected_type="dict",
                    received_type=type(value).__name__,
                )
            continue

        # Scalar type handling — coerce if possible, raise if not
        if hint in (int, float) and isinstance(value, str):
            try:
                coerced[param_name] = hint(value)
            except ValueError as exc:
                raise ToolArgumentError(
                    f"Tool {spec.name!r}: argument {param_name!r} expected "
                    f"{hint.__name__!r}, cannot coerce {value!r}.",
                    tool_name=spec.name,
                    param_name=param_name,
                    expected_type=hint.__name__,
                    received_type=type(value).__name__,
                ) from exc
            continue

        if hint is bool and isinstance(value, str):
            lower = value.lower()
            if lower in ("true", "1", "yes"):
                coerced[param_name] = True
            elif lower in ("false", "0", "no"):
                coerced[param_name] = False
            else:
                raise ToolArgumentError(
                    f"Tool {spec.name!r}: argument {param_name!r} expected bool, "
                    f"cannot coerce {value!r}.",
                    tool_name=spec.name,
                    param_name=param_name,
                    expected_type="bool",
                    received_type=type(value).__name__,
                )
            continue

        if hint is str and isinstance(value, (int, float, bool)):
            # Numbers sent for a str param — coerce silently
            coerced[param_name] = str(value)
            continue

        # Reject incompatible container/scalar combos: dict/list where scalar expected
        if isinstance(value, (dict, list)) and hint in (str, int, float, bool):
            raise ToolArgumentError(
                f"Tool {spec.name!r}: argument {param_name!r} expected "
                f"{hint.__name__!r}, got {type(value).__name__!r}.",
                tool_name=spec.name,
                param_name=param_name,
                expected_type=hint.__name__,
                received_type=type(value).__name__,
            )

    return coerced


def execute_tool(agent: Agent, name: str, arguments: dict[str, object]) -> str | Any:  # type: ignore[explicit-any]
    """Execute a tool by name. Returns result string or raises ToolExecutionError."""
    # O(1) lookup via tools_map dict
    spec = agent.tools_map.get(name)
    if spec is not None:
        try:
            validated_args = _validate_and_coerce_args(spec, arguments)
            if spec.inject_run_context:
                if agent._dependencies is None:
                    raise ToolExecutionError(
                        f"Tool {name!r} expects ctx: RunContext but Agent has no dependencies. "
                        "Pass dependencies=MyDeps(...) to Agent."
                    )
                ctx = RunContext(
                    deps=agent._dependencies,
                    agent_name=agent._agent_name,
                    conversation_id=getattr(agent, "_conversation_id", None),
                    budget_state=agent.budget_state,
                    retry_count=0,
                )
                result = spec.func(ctx=ctx, **validated_args)
            else:
                result = spec.func(**validated_args)
            if asyncio.iscoroutine(result):
                return result  # Caller must await
            result_str = str(result) if result is not None else ""
            # 6.2: Spotlight tool output if enabled
            if getattr(agent, "_spotlight_tool_outputs", False):
                from syrin.guardrails.injection._spotlight import spotlight_wrap

                result_str = spotlight_wrap(result_str, source=name)
            return result_str
        except (ToolExecutionError, ToolArgumentError):
            raise
        except Exception as e:
            _mode = getattr(agent, "_tool_error_mode", ToolErrorMode.PROPAGATE)
            if _mode == ToolErrorMode.RETURN_AS_STRING:
                return f"Tool error ({name}): {e}"
            elif _mode == ToolErrorMode.STOP:
                raise ToolExecutionError(f"Tool {name!r} failed: {e}") from e
            else:
                raise
    if name == "generate_image":
        return (
            "Image generation is not available. Provide api_key via a Google model or "
            "image_generation=ImageGenerator.Gemini(api_key=...). Install: pip install syrin[generation]"
        )
    if name == "generate_video":
        return (
            "Video generation is not available. Provide api_key via a Google model or "
            "video_generation=VideoGenerator.Gemini(api_key=...). Install: pip install syrin[generation]"
        )
    raise ToolExecutionError(f"Unknown tool: {name!r}")
