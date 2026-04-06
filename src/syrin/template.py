"""Template engine for slot-based structured generation.

Uses Mustache-style syntax ({{variable}}, {{#section}}...{{/section}}, {{#list}}{{.}}{{/list}})
to reduce hallucination by constraining LLM output to filling predefined slots.
Supports YAML frontmatter for slot definitions in template files.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import chevron

_log = logging.getLogger(__name__)

__all__ = ["Template", "SlotConfig"]


FRONTMATTER_REGEX = re.compile(
    r"^---\s*\n(.*?)\n---\s*\n(.*)",
    re.DOTALL,
)


def _parse_simple_yaml(content: str) -> dict[str, object] | None:
    """Parse simple YAML frontmatter for slot definitions.

    Supports flat format where each top-level key is a slot name,
    and nested keys are slot properties:
        name:
          type: str
          required: true
        amount:
          type: int
          default: 0
    """
    if not content.strip():
        return None

    lines = content.split("\n")
    slots: dict[str, object] = {}
    current_slot: str | None = None
    current_props: dict[str, object] | None = None
    base_indent = 0

    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue

        if ":" not in line:
            continue

        leading = len(line) - len(line.lstrip())
        key, _, value = line.partition(":")
        key = key.strip()
        value = value.strip()

        if leading == 0:
            if current_slot and current_props:
                slots[current_slot] = current_props
            current_slot = key
            current_props = {"type": "str"}
            base_indent = leading
        elif leading > base_indent and current_props is not None:
            if value:
                current_props[key] = _parse_yaml_value(value)

    if current_slot and current_props:
        slots[current_slot] = current_props

    if not slots:
        return None
    return slots


def _parse_yaml_value(value: str) -> object:
    """Parse a YAML value string into appropriate Python type."""
    value = value.strip()
    if not value:
        return None
    if value.lower() in ("true", "yes", "on"):
        return True
    if value.lower() in ("false", "no", "off"):
        return False
    if value.lower() == "null":
        return None
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        pass
    if (value.startswith("'") and value.endswith("'")) or (
        value.startswith('"') and value.endswith('"')
    ):
        return value[1:-1]
    return value


@dataclass(frozen=True)
class SlotConfig:
    """Configuration for a single template slot.

    Attributes:
        slot_type: Python type name: "str", "int", "float", "bool", "list[str]".
        required: If True, render() raises when slot is missing (strict mode).
        default: Default value when slot is not provided.
    """

    slot_type: str
    required: bool = False
    default: object = None

    def to_json_schema_type(self) -> str:
        """Map slot_type to JSON schema type string."""
        if self.slot_type.startswith("list["):
            return "array"
        mapping: dict[str, str] = {
            "str": "string",
            "int": "integer",
            "float": "number",
            "bool": "boolean",
        }
        return mapping.get(self.slot_type, "string")


def _max_section_depth(content: str) -> int:
    """Compute the maximum Mustache section nesting depth in a template string.

    Counts {{#section}} as depth+1 and {{/section}} as depth-1.
    Returns the maximum depth reached.
    """
    depth = 0
    max_depth = 0
    for token in re.findall(r"\{\{([#^/!>]?)[\w\s.]*\}\}", content):
        if token in ("#", "^"):
            depth += 1
            max_depth = max(max_depth, depth)
        elif token == "/":
            depth = max(0, depth - 1)
    return max_depth


def _coerce_value(value: object, slot_type: str) -> object:
    """Coerce value to slot type for template rendering."""
    if value is None:
        return None
    if slot_type == "str":
        return str(value) if not isinstance(value, str) else value
    if slot_type == "int":
        if isinstance(value, int) and not isinstance(value, bool):
            return value
        try:
            return int(float(str(value).replace(",", "")))
        except (ValueError, TypeError):
            return value
    if slot_type == "float":
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
        try:
            return float(str(value).replace(",", ""))
        except (ValueError, TypeError):
            return value
    if slot_type == "bool":
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.lower() in ("true", "yes", "1", "on")
        return bool(value)
    if slot_type.startswith("list["):
        if isinstance(value, list):
            return value
        if isinstance(value, str):
            try:
                parsed = json.loads(value)
                return parsed if isinstance(parsed, list) else [value]
            except json.JSONDecodeError:
                return [value] if value else []
        return [value] if value is not None else []
    return value


def _prepare_context(
    slots: dict[str, SlotConfig],
    data: dict[str, object],
    strict: bool,
) -> dict[str, object]:
    """Prepare Mustache context from slot config and provided data."""
    ctx: dict[str, object] = {}
    for name, config in slots.items():
        if name in data:
            ctx[name] = _coerce_value(data[name], config.slot_type)
        elif config.default is not None:
            ctx[name] = config.default
        elif config.required and strict:
            raise ValueError(f"Required slot {name!r} is missing")
        else:
            if config.required:
                _log.warning(
                    "Template slot %r is required but was not provided — rendering as empty. "
                    "Pass it via template.render(%s=...) or set strict=True to raise instead.",
                    name,
                    name,
                )
            ctx[name] = None
    unknown = set(data) - set(slots)
    if unknown:
        _log.warning(
            "Template render() received unknown keys %s — these will be ignored. "
            "Declared slots: %s",
            sorted(unknown),
            sorted(slots),
        )
    return ctx


class Template:
    """Structured template with typed slots for constrained generation.

    Templates reduce hallucination by constraining LLM output to filling
    predefined slots rather than generating freeform text. Supports
    Mustache-style syntax: {{var}}, {{#section}}...{{/section}},
    {{#list}}{{.}}{{/list}} for iteration (use {{.}} for current element).

    Example:
        >>> t = Template(
        ...     name="cap",
        ...     content="Capital: {{amount}}",
        ...     slots={"amount": SlotConfig("str", required=True)},
        ... )
        >>> t.render(amount="₹50L")
        'Capital: ₹50L'
    """

    def __init__(
        self,
        name: str,
        content: str,
        *,
        slots: dict[str, SlotConfig | dict[str, object]] | None = None,
        strict: bool = False,
    ) -> None:
        """Create a template.

        Args:
            name: Template identifier.
            content: Mustache template string.
            slots: Slot definitions. Values can be SlotConfig or dict with
                type, required, default keys.
            strict: If True, render() raises when a required slot is missing.
        """
        self._name = name
        self._content = content
        self._strict = strict
        self._slots: dict[str, SlotConfig] = {}
        if slots:
            for k, v in slots.items():
                if isinstance(v, SlotConfig):
                    self._slots[k] = v
                elif isinstance(v, dict):
                    self._slots[k] = SlotConfig(
                        slot_type=str(v.get("type", "str")),
                        required=bool(v.get("required", False)),
                        default=v.get("default"),
                    )
                else:
                    self._slots[k] = SlotConfig(slot_type="str", required=False)

    @property
    def name(self) -> str:
        """Template identifier."""
        return self._name

    @property
    def content(self) -> str:
        """Raw template string."""
        return self._content

    @property
    def slots(self) -> dict[str, SlotConfig]:
        """Slot configuration by name."""
        return dict(self._slots)

    def render(self, max_depth: int = 10, **kwargs: object) -> str:
        """Render template with provided slot values.

        Args:
            max_depth: Maximum Mustache section nesting depth (SEC4). Default 10.
            **kwargs: Slot values by name. Extras are ignored.

        Returns:
            Rendered string.

        Raises:
            ValueError: If strict=True and a required slot is missing.
            ValueError: If template nesting depth exceeds max_depth.
        """
        depth = _max_section_depth(self._content)
        if depth > max_depth:
            raise ValueError(
                f"Template {self._name!r}: Mustache section nesting depth {depth} "
                f"exceeds max_depth={max_depth}. Reduce nesting to prevent runaway rendering."
            )
        ctx = _prepare_context(self._slots, kwargs, self._strict)
        return cast(str, chevron.render(self._content, ctx))

    def slot_schema(self) -> dict[str, object]:
        """Return JSON schema for slots (for LLM extraction)."""
        props: dict[str, object] = {}
        required: list[str] = []
        for name, config in self._slots.items():
            t = config.to_json_schema_type()
            prop: dict[str, object] = {"type": t, "description": f"Slot: {name}"}
            if t == "array":
                prop["items"] = {"type": "string"}
            props[name] = prop
            if config.required:
                required.append(name)
        schema: dict[str, object] = {
            "type": "object",
            "properties": props,
        }
        if required:
            schema["required"] = required
        return schema

    @classmethod
    def from_file(cls, path: str | Path, **kwargs: object) -> Template:
        """Load template from file.

        Supports YAML frontmatter for slot definitions:
            ---
            slots:
              name:
                type: str
                required: true
              amount:
                type: int
                default: 0
            ---
            Hello {{name}}, amount: {{amount}}

        Args:
            path: Path to template file.
            **kwargs: Explicit slots override frontmatter slots.

        Returns:
            Template instance. Name defaults to stem of path.
        """
        import yaml  # type: ignore[import-untyped]

        from syrin.exceptions import TemplateParseError

        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Template file not found: {path}")
        raw_content = p.read_text(encoding="utf-8")

        frontmatter_slots: dict[str, SlotConfig | dict[str, object]] | None = None
        content = raw_content

        match = FRONTMATTER_REGEX.match(raw_content)
        if match:
            yaml_content = match.group(1)
            content = match.group(2)
            try:
                parsed_yaml = yaml.safe_load(yaml_content)
            except yaml.YAMLError as exc:
                line: int | None = None
                if hasattr(exc, "problem_mark") and exc.problem_mark is not None:
                    line = exc.problem_mark.line + 1
                raise TemplateParseError(
                    f"Invalid YAML frontmatter in template file {p.name!r}: {exc}",
                    path=str(p),
                    line=line,
                ) from exc
            if isinstance(parsed_yaml, dict):
                slots_raw = parsed_yaml.get("slots")
                if isinstance(slots_raw, dict):
                    frontmatter_slots = slots_raw
                elif parsed_yaml:
                    # Flat format (no 'slots' key) — treat top-level keys as slots
                    flat = _parse_simple_yaml(yaml_content)
                    if flat:
                        frontmatter_slots = cast("dict[str, SlotConfig | dict[str, object]]", flat)

        explicit_slots = kwargs.pop("slots", None)
        if explicit_slots and frontmatter_slots:
            merged: dict[str, SlotConfig | dict[str, object]] = dict(frontmatter_slots)
            merged.update(explicit_slots)  # type: ignore[call-overload]
            kwargs["slots"] = merged
        elif frontmatter_slots:
            kwargs["slots"] = frontmatter_slots
        elif explicit_slots:
            kwargs["slots"] = explicit_slots

        name = kwargs.pop("name", p.stem)
        return cls(name=name, content=content, **kwargs)  # type: ignore[arg-type]

    @classmethod
    def from_string(cls, content: str, name: str = "unnamed", **kwargs: object) -> Template:
        """Create template from string.

        Args:
            content: Template content.
            name: Template identifier. Default "unnamed".
            **kwargs: Passed to Template constructor (slots, strict).

        Returns:
            Template instance.
        """
        return cls(name=name, content=content, **kwargs)  # type: ignore[arg-type]
