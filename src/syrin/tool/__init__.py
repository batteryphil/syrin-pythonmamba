"""Tool decorator and schema generation for agent tools."""

from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import Any, Union, get_type_hints

from pydantic import BaseModel, Field

from syrin.enums import DocFormat
from syrin.tool._schema import schema_to_toon as _schema_to_toon
from syrin.tool._schema import tool_schema_to_format_dict

_TYPE_TO_JSON: dict[type[Any], str] = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
}


class ToolSpec(BaseModel):
    """Spec for a tool the model can call. Usually built via syrin.tool()."""

    name: str = Field(..., description="Tool name (used in tool_calls.name)")
    description: str = Field(
        default="",
        description="Description for the model. Shown in tool list.",
    )
    parameters_schema: dict[str, Any] = Field(
        default_factory=dict,
        description="JSON schema for parameters. Model uses this to generate args.",
    )
    func: Callable[..., Any] = Field(
        ...,
        description="Python function to run. Receives parsed arguments from model.",
    )
    requires_approval: bool = Field(
        default=False,
        description="If True, block execution until human approval via ApprovalGate.",
    )

    model_config = {"arbitrary_types_allowed": True}

    def schema_to_toon(self, indent: int = 0) -> str:
        """Return this tool's parameters schema as TOON (token-efficient) string."""
        return _schema_to_toon(self.parameters_schema or {}, indent)

    def to_format(self, format: DocFormat = DocFormat.TOON) -> dict[str, Any]:
        """Return this tool as a provider-ready schema dict (TOON, JSON, or YAML)."""
        return tool_schema_to_format_dict(
            self.name,
            self.description or "",
            self.parameters_schema or {},
            format,
        )


def _annotation_to_json_schema(annotation: Any) -> dict[str, Any]:
    """Convert a type annotation to a JSON schema fragment."""
    origin = getattr(annotation, "__origin__", None)
    args = getattr(annotation, "__args__", ())

    if annotation is type(None):
        return {"type": "null"}
    if origin is Union or (hasattr(annotation, "__args__") and type(None) in (args or ())):
        non_none = [a for a in (args or ()) if a is not type(None)]
        if len(non_none) == 1:
            return {"oneOf": [_annotation_to_json_schema(non_none[0]), {"type": "null"}]}
        if non_none:
            return {"oneOf": [_annotation_to_json_schema(a) for a in non_none]}
    if origin is list and args:
        return {"type": "array", "items": _annotation_to_json_schema(args[0])}
    if origin is dict and len(args) >= 2:
        value_ann = args[1]  # type: ignore[misc]  # safe after len check; mypy tuple narrow
        if value_ann is not type(None):
            return {
                "type": "object",
                "additionalProperties": _annotation_to_json_schema(value_ann),
            }
    if annotation in _TYPE_TO_JSON:
        return {"type": _TYPE_TO_JSON[annotation]}
    return {"type": "string"}


def _parameters_schema_from_function(func: Callable[..., Any]) -> dict[str, Any]:
    """Build a JSON schema for the function's parameters from type hints."""
    hints = get_type_hints(func) if hasattr(func, "__annotations__") else {}
    sig = inspect.signature(func)
    properties: dict[str, Any] = {}
    required: list[str] = []
    for name, param in sig.parameters.items():
        if name == "self" or name == "cls":
            continue
        ann = hints.get(name, param.annotation)
        if ann is inspect.Parameter.empty:
            ann = Any
        properties[name] = _annotation_to_json_schema(ann)
        if param.default is inspect.Parameter.empty:
            required.append(name)
    return {"type": "object", "properties": properties, "required": required}


def tool(
    func: Callable[..., Any] | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
    requires_approval: bool = False,
) -> Callable[..., Any] | ToolSpec:
    """
    Decorator to register a function as a Syrin tool. Extracts name, docstring,
    and builds a JSON schema from type hints (str, int, float, bool, list, dict).
    """

    def decorator(f: Callable[..., Any]) -> ToolSpec:
        tool_name = name or f.__name__
        desc = description or (inspect.getdoc(f) or "").strip().split("\n")[0] or ""
        params_schema = _parameters_schema_from_function(f)
        return ToolSpec(
            name=tool_name,
            description=desc,
            parameters_schema=params_schema,
            func=f,
            requires_approval=requires_approval,
        )

    if func is not None:
        return decorator(func)
    return decorator


__all__ = ["tool", "ToolSpec"]
