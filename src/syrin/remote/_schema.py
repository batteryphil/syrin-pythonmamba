"""Schema extraction: Pydantic, dataclass, and plain classes → FieldSchema / AgentSchema."""

from __future__ import annotations

import dataclasses
import inspect
from types import UnionType
from typing import Any, cast, get_origin, get_type_hints

from pydantic import BaseModel

from syrin.remote._protocol import RemoteConfigurable
from syrin.remote._types import AgentSchema, ConfigSchema, FieldSchema

_MAX_RECURSION_DEPTH = 10


def _unwrap_optional(annotation: Any) -> Any:
    """Return the non-None part of Optional[X] or X | None; else return annotation."""
    origin = get_origin(annotation)
    if origin is UnionType or (
        hasattr(annotation, "__args__") and type(None) in getattr(annotation, "__args__", ())
    ):
        args = getattr(annotation, "__args__", ())
        for a in args:
            if a is not type(None):
                return a
    return annotation


def _annotation_to_type_string(annotation: Any) -> str:
    """Map Python annotation to normalized type string for FieldSchema.type."""
    if annotation is None or annotation is type(None):
        return "any"
    if isinstance(annotation, str):
        s = annotation.strip()
        for name in ("int", "float", "str", "bool", "list", "dict"):
            if s == name or s.startswith(name + "[") or s.startswith(name + " |"):
                return name if name in ("list", "dict") else name
        return "any"
    annotation = _unwrap_optional(annotation)
    origin = get_origin(annotation)
    if origin is not None:
        if origin is type:
            return "any"  # type[X] → exclude or any
        if origin is list:
            return "list"
        if origin is dict:
            return "dict"
    if hasattr(annotation, "__mro__"):
        try:
            if issubclass(annotation, str):
                return "str"
            if issubclass(annotation, int):
                return "int"
            if issubclass(annotation, float):
                return "float"
            if issubclass(annotation, bool):
                return "bool"
            if issubclass(annotation, BaseModel):
                return "object"
            if dataclasses.is_dataclass(annotation):
                return "object"
            from enum import Enum

            if issubclass(annotation, Enum):
                return "str"
        except TypeError:
            pass
    if "Callable" in str(annotation):
        return "any"
    return "any"


def _constraints_from_metadata(metadata: list[Any]) -> dict[str, float | int | str]:
    """Extract constraints from Pydantic FieldInfo.metadata (Ge, Le, Gt, Lt, MinLen, MaxLen, Pattern)."""
    constraints: dict[str, float | int | str] = {}
    for m in metadata:
        cls_name = type(m).__name__
        if cls_name == "Ge" and hasattr(m, "ge"):
            constraints["ge"] = m.ge
        elif cls_name == "Gt" and hasattr(m, "gt"):
            constraints["gt"] = m.gt
        elif cls_name == "Le" and hasattr(m, "le"):
            constraints["le"] = m.le
        elif cls_name == "Lt" and hasattr(m, "lt"):
            constraints["lt"] = m.lt
        elif cls_name == "MinLen" and hasattr(m, "min_length"):
            constraints["min_length"] = m.min_length
        elif cls_name == "MaxLen" and hasattr(m, "max_length"):
            constraints["max_length"] = m.max_length
        elif cls_name == "Len" and hasattr(m, "min_length"):
            constraints["min_length"] = m.min_length
        if cls_name == "Len" and hasattr(m, "max_length"):
            constraints["max_length"] = m.max_length
        if hasattr(m, "pattern"):
            constraints["pattern"] = m.pattern
    return constraints


def _should_exclude_from_remote(field_name: str, annotation: Any) -> bool:
    """True if field should be remote_excluded (callable, Protocol, type ref, private)."""
    if field_name.startswith("_"):
        return True
    annotation = _unwrap_optional(annotation)
    origin = get_origin(annotation)
    if origin is not None:
        if "Callable" in str(annotation):
            return True
        if origin is type:
            return True
    if hasattr(annotation, "__mro__"):
        try:
            for base in getattr(annotation, "__mro__", ()):
                if getattr(base, "__name__", "") == "Protocol" or "Protocol" in str(base):
                    return True
        except TypeError:
            pass
    ann_str = str(annotation)
    if "Callable" in ann_str or "Protocol" in ann_str:
        return True
    if "type[" in ann_str or "Type[" in ann_str:
        return True
    return "OutputValidator" in ann_str


def _get_enum_values(annotation: Any) -> list[str] | None:
    """If annotation is StrEnum (or Enum), return list of .value strings."""
    if not hasattr(annotation, "__mro__"):
        return None
    try:
        from enum import Enum

        if issubclass(annotation, Enum):
            return [m.value for m in annotation]
    except TypeError:
        pass
    return None


def _get_default_serializable(default: Any) -> Any:
    """Return default if JSON-serializable (int, float, str, bool, None); else None."""
    if default is None:
        return None
    if isinstance(default, (int, float, str, bool)):
        return default
    if hasattr(default, "value"):  # Enum
        return default.value
    return None


def extract_pydantic_schema(cls: type[BaseModel], prefix: str, depth: int = 0) -> list[FieldSchema]:
    """Extract FieldSchema list from a Pydantic model. prefix is the dotted path prefix (e.g. 'budget')."""
    if not prefix or not prefix.strip():
        raise ValueError("prefix must be non-empty")
    if depth > _MAX_RECURSION_DEPTH:
        return []
    fields: list[FieldSchema] = []
    for name, field_info in getattr(cls, "model_fields", {}).items():
        path = f"{prefix}.{name}"
        ann = field_info.annotation
        type_str = _annotation_to_type_string(ann)
        default = (
            _get_default_serializable(field_info.default)
            if field_info.default is not None
            else None
        )
        if default is None and hasattr(field_info, "default") and field_info.default is not None:
            try:
                if callable(field_info.default) and not isinstance(field_info.default, type):
                    default = None
                elif not callable(field_info.default):
                    default = _get_default_serializable(field_info.default)
            except Exception:
                default = None
        description = getattr(field_info, "description", None) or None
        constraints = _constraints_from_metadata(getattr(field_info, "metadata", []) or [])
        enum_values = _get_enum_values(ann)
        remote_excluded = _should_exclude_from_remote(name, ann)
        children: list[FieldSchema] | None = None
        inner_ann = _unwrap_optional(ann)
        if hasattr(inner_ann, "__mro__") and not remote_excluded:
            try:
                if issubclass(inner_ann, BaseModel):
                    children = extract_pydantic_schema(inner_ann, path, depth + 1)
                elif dataclasses.is_dataclass(inner_ann):
                    children = extract_dataclass_schema(cast(type[Any], inner_ann), path, depth + 1)
            except TypeError:
                pass
        fields.append(
            FieldSchema(
                name=name,
                path=path,
                type=type_str,
                default=default,
                description=description,
                constraints=constraints,
                enum_values=enum_values,
                children=children or None,
                remote_excluded=remote_excluded,
            )
        )
    return fields


def extract_dataclass_schema(cls: type[Any], prefix: str, depth: int = 0) -> list[FieldSchema]:
    """Extract FieldSchema list from a dataclass. prefix is the dotted path prefix."""
    if not prefix or not prefix.strip():
        raise ValueError("prefix must be non-empty")
    if depth > _MAX_RECURSION_DEPTH:
        return []
    fields: list[FieldSchema] = []
    try:
        resolved = get_type_hints(cls)
    except Exception:
        resolved = {}
    for dc_field in dataclasses.fields(cls):
        name = dc_field.name
        path = f"{prefix}.{name}"
        ann = resolved.get(name, dc_field.type)
        type_str = _annotation_to_type_string(ann)
        default = (
            _get_default_serializable(dc_field.default)
            if dc_field.default is not dataclasses.MISSING
            else None
        )
        if default is None and dc_field.default is not dataclasses.MISSING:
            default = _get_default_serializable(dc_field.default)
        remote_excluded = _should_exclude_from_remote(name, ann)
        enum_values = _get_enum_values(ann)
        children: list[FieldSchema] | None = None
        inner_ann = _unwrap_optional(ann)
        if hasattr(inner_ann, "__mro__") and not remote_excluded:
            try:
                if issubclass(inner_ann, BaseModel):
                    children = extract_pydantic_schema(inner_ann, path, depth + 1)
                elif dataclasses.is_dataclass(inner_ann):
                    children = extract_dataclass_schema(cast(type[Any], inner_ann), path, depth + 1)
            except TypeError:
                pass
        fields.append(
            FieldSchema(
                name=name,
                path=path,
                type=type_str,
                default=default,
                description=None,
                constraints={},
                enum_values=enum_values,
                children=children or None,
                remote_excluded=remote_excluded,
            )
        )
    return fields


def extract_plain_schema(cls: type[Any], prefix: str, depth: int = 0) -> list[FieldSchema]:
    """Extract FieldSchema list from a plain class using __init__ signature."""
    if not prefix or not prefix.strip():
        raise ValueError("prefix must be non-empty")
    if depth > _MAX_RECURSION_DEPTH:
        return []
    fields: list[FieldSchema] = []
    try:
        sig = inspect.signature(cls.__init__)
    except (ValueError, TypeError):
        return []
    try:
        hints = get_type_hints(cls.__init__)
    except Exception:
        hints = {}
    for param_name, param in sig.parameters.items():
        if param_name == "self":
            continue
        path = f"{prefix}.{param_name}"
        ann = hints.get(
            param_name, param.annotation if param.annotation is not inspect.Parameter.empty else Any
        )
        type_str = _annotation_to_type_string(ann)
        default = None
        if param.default is not inspect.Parameter.empty:
            default = _get_default_serializable(param.default)
        remote_excluded = _should_exclude_from_remote(param_name, ann)
        fields.append(
            FieldSchema(
                name=param_name,
                path=path,
                type=type_str,
                default=default,
                description=None,
                constraints={},
                enum_values=_get_enum_values(ann),
                children=None,
                remote_excluded=remote_excluded,
            )
        )
    return fields


def extract_schema(cls: type[Any], prefix: str) -> list[FieldSchema]:
    """Auto-detect class kind (Pydantic, dataclass, plain) and dispatch to the right extractor."""
    if not prefix or not prefix.strip():
        raise ValueError("prefix must be non-empty")
    if hasattr(cls, "model_fields"):
        return extract_pydantic_schema(cls, prefix)
    if dataclasses.is_dataclass(cls):
        return extract_dataclass_schema(cls, prefix)
    return extract_plain_schema(cls, prefix)


# Agent section: top-level attributes we expose for remote config.
_AGENT_TOP_LEVEL: list[tuple[str, str, type[Any]]] = [
    ("max_tool_iterations", "int", int),
    ("debug", "bool", bool),
    ("loop_strategy", "str", type(None)),  # StrEnum → str
    ("system_prompt", "str", str),
    ("hitl_timeout", "int", int),
]


def _agent_section_schema() -> ConfigSchema:
    """Build the fixed 'agent' section schema (top-level agent params)."""
    from syrin.enums import LoopStrategy

    enum_vals = [m.value for m in LoopStrategy]
    fields: list[FieldSchema] = []
    for name, type_str, _ in _AGENT_TOP_LEVEL:
        path = f"agent.{name}"
        if name == "loop_strategy":
            fields.append(
                FieldSchema(
                    name=name,
                    path=path,
                    type=type_str,
                    default=None,
                    description=None,
                    constraints={},
                    enum_values=enum_vals,
                    children=None,
                    remote_excluded=False,
                )
            )
        else:
            default: object | None = None
            if name == "max_tool_iterations":
                default = 10
            elif name == "debug":
                default = False
            elif name == "system_prompt":
                default = ""
            elif name == "hitl_timeout":
                default = 300
            fields.append(
                FieldSchema(
                    name=name,
                    path=path,
                    type=type_str,
                    default=default,
                    description=None,
                    constraints={},
                    enum_values=None,
                    children=None,
                    remote_excluded=False,
                )
            )
    return ConfigSchema(section="agent", class_name="Agent", fields=fields)


def get_agent_section_schema_and_values(agent: Any) -> tuple[ConfigSchema, dict[str, object]]:
    """Return (schema, current_values) for the agent top-level section. Used by Agent.get_remote_config_schema."""
    agent_cfg = _agent_section_schema()
    current_values: dict[str, object] = {}
    for f in agent_cfg.fields:
        attr = f.name
        if attr == "max_tool_iterations":
            v = getattr(agent, "_max_tool_iterations", None)
        elif attr == "debug":
            v = getattr(agent, "_debug", None)
        elif attr == "system_prompt":
            v = getattr(agent, "_system_prompt_source", "") or ""
        elif attr == "hitl_timeout":
            v = getattr(agent, "_hitl_timeout", None)
        elif attr == "loop_strategy":
            from syrin.enums import LoopStrategy as LS

            loop = getattr(agent, "_loop", None)
            v = getattr(loop, "strategy", None)
            if v is not None and hasattr(v, "value"):
                v = v.value
            elif loop is not None:
                # Loop instances don't have .strategy; infer from class .name (e.g. ReactLoop.name = "react")
                name = getattr(type(loop), "name", None)
                v = name.lower().replace(" ", "_") if isinstance(name, str) else None
            valid = [m.value for m in LS]
            if v is None or (isinstance(v, str) and v not in valid):
                v = LS.REACT.value
        else:
            v = getattr(agent, f"_{attr}", getattr(agent, attr, None))
        if v is not None or f.default is not None:
            current_values[f.path] = v if v is not None else f.default
    return (agent_cfg, current_values)


def _current_value_for_path(obj: Any, path: str) -> Any:
    """Get current value from an object by dotted path. Returns JSON-serializable values."""
    if obj is None:
        return None
    parts = path.split(".", 1)
    key = parts[0]
    if hasattr(obj, key):
        val = getattr(obj, key)
        if len(parts) == 1:
            if hasattr(val, "value") and not hasattr(val, "model_dump"):
                return val.value
            if hasattr(val, "model_dump"):
                return val.model_dump(mode="json")
            return val
        return _current_value_for_path(val, parts[1])
    return None


def _flatten_current_values(
    obj: Any, prefix: str, field_schemas: list[FieldSchema]
) -> dict[str, object]:
    """Recursively build current_values dict from object and field schemas."""
    out: dict[str, object] = {}
    for f in field_schemas:
        if f.remote_excluded:
            continue
        val = _current_value_for_path(
            obj, f.path.replace(prefix + ".", "", 1) if prefix else f.path
        )
        if val is not None or f.default is not None:
            out[f.path] = val if val is not None else f.default
        if f.children:
            child_obj = _current_value_for_path(obj, f.name) if obj is not None else None
            for c in f.children:
                if c.remote_excluded:
                    continue
                cv = _current_value_for_path(child_obj, c.name) if child_obj is not None else None
                if cv is not None or c.default is not None:
                    out[c.path] = cv if cv is not None else c.default
    return out


def build_section_schema_from_obj(
    obj: Any, section_key: str, class_name: str
) -> tuple[ConfigSchema, dict[str, object]]:
    """Build (ConfigSchema, current_values) for a single config object. Used by RemoteConfigurable implementations."""
    fields = extract_schema(type(obj), section_key)
    config_schema = ConfigSchema(section=section_key, class_name=class_name, fields=fields)
    current_values = _flatten_current_values(obj, section_key, fields)
    return (config_schema, current_values)


def _get_configurable(agent: Any, spec: str | tuple[str, ...] | None) -> Any:
    """Resolve configurable from agent by spec: None -> agent, str -> getattr(agent, spec), tuple -> path of attrs."""
    if spec is None:
        return agent
    if isinstance(spec, str):
        return getattr(agent, spec, None)
    obj: Any = agent
    for attr in spec:
        obj = getattr(obj, attr, None)
    return obj


def extract_agent_schema(agent: Any) -> AgentSchema:
    """Build AgentSchema from a live agent via RemoteConfigurable protocol.

    Iterates REMOTE_CONFIG_SECTIONS on the agent's class; for each section
    resolves the configurable (agent attr or nested path) and, if it implements
    RemoteConfigurable, merges its schema and current_values.
    """
    agent_name = (
        getattr(agent, "_agent_name", None) or getattr(agent, "name", None) or type(agent).__name__
    )
    if not isinstance(agent_name, str):
        agent_name = type(agent).__name__
    class_name = type(agent).__name__
    agent_id = f"{agent_name}:{class_name}"

    sections: dict[str, ConfigSchema] = {}
    current_values: dict[str, object] = {}

    remote_sections = getattr(type(agent), "REMOTE_CONFIG_SECTIONS", None) or {}
    for section_key, spec in remote_sections.items():
        configurable = _get_configurable(agent, spec)
        if configurable is None:
            continue
        if not isinstance(configurable, RemoteConfigurable):
            continue
        section_schema, section_values = configurable.get_remote_config_schema(section_key)
        sections[section_key] = section_schema
        current_values.update(section_values)

    return AgentSchema(
        agent_id=agent_id,
        agent_name=agent_name,
        class_name=class_name,
        sections=sections,
        baseline_values={},
        overrides={},
        current_values=current_values,
    )
