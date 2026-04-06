"""Shared helpers for building/merging nested updates and applying agent-section overrides.

Used by per-class RemoteConfigurable implementations. No imports of Budget/Memory/Agent
to avoid circular dependencies.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import cast

from pydantic import BaseModel

from syrin.remote._types import ConfigSchema, FieldSchema


def _normalize_enum_value(path: str, value: str, field: FieldSchema | None) -> str | None:
    """If path is a known enum, try to resolve value by enum name or display format. Returns canonical value or None."""
    if not isinstance(value, str) or not value:
        return None

    # Normalize display-style input: "Plan execute" / "Single Shot" -> "plan_execute" / "single_shot"
    def normalized(s: str) -> str:
        return s.strip().lower().replace(" ", "_")

    if path.startswith("memory.decay.") and field and field.name == "strategy":
        from syrin.enums import DecayStrategy

        if value in (m.value for m in DecayStrategy):
            return value
        try:
            return DecayStrategy[value].value
        except KeyError:
            pass
        norm = normalized(value)
        if norm in (m.value for m in DecayStrategy):
            return norm
        return None
    if path.startswith("checkpoint.") and field and field.name == "trigger":
        from syrin.checkpoint import CheckpointTrigger

        vals = [e.value for e in CheckpointTrigger]
        if value in vals:
            return value
        try:
            return CheckpointTrigger[value].value
        except KeyError:
            pass
        norm = normalized(value)
        if norm in vals:
            return norm
        return None
    if path.startswith("knowledge.chunk_config.") and field and field.name == "strategy":
        from syrin.knowledge._chunker import ChunkStrategy

        if value in (m.value for m in ChunkStrategy):
            return value
        try:
            return ChunkStrategy[value].value
        except KeyError:
            pass
        norm = normalized(value)
        if norm in (m.value for m in ChunkStrategy):
            return norm
        return None
    return None


def _coerce_enum(value: object, field: FieldSchema | None) -> object:
    """If field has enum_values, coerce string to enum; else return value."""
    if field is None or field.enum_values is None or not isinstance(value, str):
        return value
    if value not in field.enum_values:
        return value  # Caller will validate
    path = field.path
    if path.startswith("memory.decay.") and field.name == "strategy":
        from syrin.enums import DecayStrategy

        return DecayStrategy(value)
    if path.startswith("checkpoint.") and field.name == "trigger":
        from syrin.checkpoint import CheckpointTrigger

        return CheckpointTrigger(value)
    if path.startswith("knowledge.chunk_config.") and field.name == "strategy":
        from syrin.knowledge._chunker import ChunkStrategy

        return ChunkStrategy(value)
    return value


def build_nested_update(
    section_schema: ConfigSchema,
    overrides: list[tuple[str, object]],
    prefix: str,
    coerce_fn: Callable[[object, FieldSchema | None], object] = _coerce_enum,
) -> dict[str, object]:
    """Build update dict for a section from list of (path, value). Handles nested paths."""
    update: dict[str, object] = {}
    path_to_field = {f.path: f for f in section_schema.fields}
    for f in section_schema.fields:
        if f.children:
            for c in f.children:
                path_to_field[c.path] = c
    for path, raw_value in overrides:
        if not path.startswith(prefix + "."):
            continue
        suffix = path[len(prefix) + 1 :]
        field = path_to_field.get(path)
        value = coerce_fn(raw_value, field) if field else raw_value
        parts = suffix.split(".")
        if len(parts) == 1:
            update[parts[0]] = value
        else:
            key = parts[0]
            inner = update.setdefault(key, {})
            if not isinstance(inner, dict):
                inner = {}
                update[key] = inner
            if len(parts) == 2:
                inner[parts[1]] = value
            else:
                cur: dict[str, object] = inner
                for i in range(1, len(parts) - 1):
                    nxt = cur.setdefault(parts[i], {})
                    if not isinstance(nxt, dict):
                        nxt = {}
                        cur[parts[i]] = nxt
                    cur = nxt
                cur[parts[-1]] = value
    return update


def _get_model_class_for_field(
    model_class: type[BaseModel], key: str
) -> type[BaseModel] | type[object] | None:
    """Get the model/dataclass type for a field if it's BaseModel or dataclass (handles Optional)."""
    import dataclasses
    import typing

    field = model_class.model_fields.get(key)
    if field is None:
        return None
    ann = field.annotation
    if ann is None:
        return None
    # Handle Optional[X] / X | None
    raw_args = typing.get_args(ann)
    candidates: tuple[object, ...] = raw_args if raw_args else (ann,)
    for a in candidates:
        if a is type(None):
            continue
        try:
            if issubclass(a, BaseModel):  # type: ignore[arg-type]
                return cast(type[BaseModel] | type[object], a)
            if dataclasses.is_dataclass(a):
                return cast(type[BaseModel] | type[object], a)
        except TypeError:
            pass
    try:
        if issubclass(ann, BaseModel):
            return cast(type[BaseModel] | type[object], ann)
        if dataclasses.is_dataclass(ann):
            return cast(type[BaseModel] | type[object], ann)
    except TypeError:
        pass
    return None


def _coerce_enum_for_field(model_class: type[BaseModel], key: str, value: object) -> object:
    """If field expects an enum and value is str, coerce to enum. Else return value."""
    import enum
    import typing

    if not isinstance(value, str):
        return value
    field = model_class.model_fields.get(key)
    if field is None:
        return value
    ann = field.annotation
    if ann is None:
        return value
    args = list(typing.get_args(ann)) or [ann]
    for a in args:
        if a is type(None):
            continue
        try:
            if issubclass(a, enum.Enum) and value in (m.value for m in a):
                return a(value)
        except TypeError:
            pass
    return value


def merge_nested_update(
    current: BaseModel | None,
    update: dict[str, object],
    model_class: type[BaseModel],
) -> BaseModel:
    """Merge nested update dict into current Pydantic model. Re-validates result."""
    if current is None:
        return model_class(**update)
    for key, val in list(update.items()):
        if isinstance(val, dict) and not hasattr(val, "model_fields"):
            child = getattr(current, key, None)
            if child is not None and hasattr(child, "model_copy"):
                from pydantic import BaseModel as BM

                if isinstance(child, BM):
                    child_cls = type(child)
                    merged = merge_nested_update(child, val, child_cls)
                    update[key] = merged
            else:
                # child is None or dataclass: validate dict into model/dataclass if field expects one
                field_cls = _get_model_class_for_field(model_class, key)
                if field_cls is not None:
                    if hasattr(field_cls, "model_validate"):
                        update[key] = field_cls.model_validate(val)
                    else:
                        update[key] = field_cls(**val)
        elif isinstance(val, str):
            # Coerce string to enum when field expects enum (avoids serializer warnings)
            coerced = _coerce_enum_for_field(model_class, key, val)
            if coerced is not val:
                update[key] = coerced
    result = current.model_copy(update=update)
    return model_class.model_validate(result.model_dump())


def apply_agent_section_overrides(
    agent: object,
    pairs: list[tuple[str, object]],
    _section_schema: ConfigSchema,
) -> None:
    """Apply agent-section (agent.*) overrides to the agent."""
    for path, value in pairs:
        if not path.startswith("agent."):
            continue
        key = path.split(".", 1)[1]
        if key == "max_tool_iterations":
            object.__setattr__(
                agent,
                "_max_tool_iterations",
                int(cast(int | float | str, value)) if value is not None else 10,
            )
        elif key == "debug":
            object.__setattr__(agent, "_debug", bool(value))
        elif key == "system_prompt":
            object.__setattr__(
                agent, "_system_prompt_source", str(value) if value is not None else ""
            )
        elif key == "human_approval_timeout":
            object.__setattr__(
                agent,
                "_human_approval_timeout",
                int(cast(int | float | str, value)) if value is not None else 300,
            )
