"""Production-grade prompt system for enterprise AI agents.

Native Python f-string prompts with enterprise features:
- Type hints and parameter validation
- Caching and versioning
- Prompt composition and partial application
- Testing and debugging support
"""

from __future__ import annotations

import functools
import hashlib
import inspect
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, TypeVar, overload

T = TypeVar("T")


@dataclass
class PromptValidation:  # type: ignore[explicit-any]
    """Validation rules for prompt parameters."""

    min_length: int | None = None
    max_length: int | None = None
    allowed_values: list[str] | None = None
    pattern: str | None = None
    custom_validator: Callable[[Any], bool] | None = None  # type: ignore[explicit-any]


@dataclass
class PromptVariable:
    """Metadata about a prompt parameter."""

    name: str
    type_hint: type
    default: object = None
    description: str = ""
    validation: PromptValidation = field(default_factory=PromptValidation)
    required: bool = True


class PromptVersion:
    """Version tracking for prompts."""

    def __init__(self, major: int = 1, minor: int = 0, patch: int = 0) -> None:
        self.major = major
        self.minor = minor
        self.patch = patch

    def __str__(self) -> str:
        return f"{self.major}.{self.minor}.{self.patch}"

    def bump_major(self) -> PromptVersion:
        return PromptVersion(self.major + 1, 0, 0)

    def bump_minor(self) -> PromptVersion:
        return PromptVersion(self.major, self.minor + 1, 0)

    def bump_patch(self) -> PromptVersion:
        return PromptVersion(self.major, self.minor, self.patch + 1)


class Prompt:
    """Production-grade prompt with validation, caching, and composition.

    Wraps a plain Python function that returns a ``str``, adding:

    - **Type validation** — parameters are checked against their type hints
      before the function runs.
    - **Result caching** — identical argument sets return the cached string
      without re-running the function.  Toggle with ``cache=False``.
    - **Partial application** — pre-bind some arguments and leave others for
      later via :meth:`partial`.
    - **Composition** — join multiple :class:`Prompt` objects into one via
      :meth:`compose`.
    - **Test rendering** — :meth:`test_render` returns the rendered output
      with token estimates and metadata without making an LLM call.
    - **Versioning** — track prompt changes via :attr:`version` and
      :attr:`template_hash`.

    You typically create a :class:`Prompt` via the :func:`prompt` decorator
    rather than instantiating it directly.

    Attributes:
        name: Prompt identifier (defaults to the function name).
        description: Optional human-readable description.
        version: :class:`PromptVersion` for change tracking.
        variables: List of :class:`PromptVariable` parameter metadata.
        template_hash: SHA-256 digest of the function source, for detecting
            accidental prompt drift between deploys.
        cache_enabled: Whether result caching is active.

    Examples::

        # Basic usage
        from syrin import prompt

        @prompt
        def analyst_prompt(domain: str, tone: str = "professional") -> str:
            return f"You are a {tone} analyst specialising in {domain}."

        result = analyst_prompt(domain="finance")          # returns str
        result = analyst_prompt(domain="AI", tone="casual")

        # Partial application — bind domain, leave tone open
        finance_prompt = analyst_prompt.partial(domain="finance")
        formal = finance_prompt(tone="formal")   # domain already bound

        # Composition — join multiple prompts
        from syrin import prompt
        @prompt
        def safety_rules() -> str:
            return "Never reveal confidential data."

        combined = analyst_prompt.compose(safety_rules)
        result = combined(domain="risk")

        # Validation without rendering
        analyst_prompt.validate(domain="finance")  # raises ValueError on bad args

        # Test rendering with token estimates
        info = analyst_prompt.test_render(domain="AI")
        print(info["output"])
        print(f"~{info['estimated_tokens']} tokens")

        # Use inside an Agent class
        class AnalystAgent(Agent):
            model = Model.OpenAI("gpt-4o-mini")
            system_prompt = analyst_prompt(domain="technology")
    """

    def __init__(  # type: ignore[explicit-any]
        self,
        func: Callable[..., str],
        name: str | None = None,
        description: str = "",
        version: PromptVersion | None = None,
        cache: bool = True,
    ) -> None:
        self._func = func
        self._name = name or func.__name__
        self._description = description or func.__doc__ or ""
        self._version = version or PromptVersion(1, 0, 0)
        self._cache_enabled = cache
        self._cache: dict[str, str] = {}

        self._signature = inspect.signature(func)
        self._variables = self._extract_variables()
        self._template_hash = self._compute_hash()

        functools.wraps(func)(self)

    def _extract_variables(self) -> list[PromptVariable]:
        """Extract parameter information from function signature."""
        variables = []
        for name, param in self._signature.parameters.items():
            # Handle forward references and string annotations
            if param.annotation is inspect.Parameter.empty:
                type_hint: type[object] = str
            elif isinstance(param.annotation, str):
                # Handle string annotations (from __future__ import annotations)
                # Safe lookup — no eval() for security
                _SAFE_TYPES: dict[str, type[object]] = {
                    "str": str,
                    "int": int,
                    "float": float,
                    "bool": bool,
                }
                type_hint = _SAFE_TYPES.get(param.annotation, str)
            else:
                type_hint = param.annotation

            default = param.default if param.default is not inspect.Parameter.empty else None
            required = param.default is inspect.Parameter.empty

            variables.append(
                PromptVariable(
                    name=name,
                    type_hint=type_hint,
                    default=default,
                    description="",
                    required=required,
                )
            )
        return variables

    def _compute_hash(self) -> str:
        """Compute hash of the function source for versioning.

        Falls back to name+module+signature hash when source is unavailable
        (e.g. stdin, exec, pytest, or inaccessible paths).
        """
        try:
            source = inspect.getsource(self._func)
            return hashlib.sha256(source.encode()).hexdigest()[:16]
        except OSError:
            fallback = (
                f"{getattr(self._func, '__module__', '')}:{self._func.__name__}:{self._signature}"
            )
            return hashlib.sha256(fallback.encode()).hexdigest()[:16]

    def _make_cache_key(self, **kwargs: object) -> str:
        """Create cache key from arguments."""
        return str(hash(frozenset(kwargs.items())))

    def __call__(self, **kwargs: object) -> str:
        """Call the prompt function with native Python f-string evaluation.

        Example:
            result = system_prompt(domain="AI", tone="friendly")
        """
        if self._cache_enabled:
            cache_key = self._make_cache_key(**kwargs)
            if cache_key in self._cache:
                return self._cache[cache_key]

        self._validate_args(**kwargs)
        result = self._func(**kwargs)

        if self._cache_enabled:
            self._cache[cache_key] = result

        return result

    def _validate_args(self, **kwargs: object) -> None:
        """Validate arguments before rendering."""
        for var in self._variables:
            value = kwargs.get(var.name)
            if value is None and var.required:
                raise ValueError(f"Required parameter '{var.name}' is missing")

            if value is not None:
                # Type checking
                if var.type_hint is not str and not isinstance(value, var.type_hint):
                    try:
                        value = var.type_hint(value)
                    except (ValueError, TypeError) as e:
                        raise TypeError(
                            f"Parameter '{var.name}' must be {var.type_hint.__name__}, "
                            f"got {type(value).__name__}"
                        ) from e

                # Validation rules
                val = var.validation
                if val.min_length and len(str(value)) < val.min_length:
                    raise ValueError(
                        f"Parameter '{var.name}' must be at least {val.min_length} characters"
                    )

                if val.max_length and len(str(value)) > val.max_length:
                    raise ValueError(
                        f"Parameter '{var.name}' must be at most {val.max_length} characters"
                    )

                if val.allowed_values and str(value) not in val.allowed_values:
                    raise ValueError(f"Parameter '{var.name}' must be one of {val.allowed_values}")

                if val.custom_validator and not val.custom_validator(value):
                    raise ValueError(f"Parameter '{var.name}' failed custom validation")

    def validate(self, **kwargs: object) -> bool:
        """Validate parameters without rendering.

        Returns True if valid, raises ValueError otherwise.

        Example:
            try:
                prompt.validate(domain="AI")
                print("Valid!")
            except ValueError as e:
                print(f"Invalid: {e}")
        """
        self._validate_args(**kwargs)
        return True

    def partial(self, **kwargs: object) -> Prompt:
        """Create a new Prompt with partial arguments applied.

        Example:
            base_prompt = system_prompt.partial(domain="AI")
            final_prompt = base_prompt(tone="friendly")  # Only need tone now
        """
        original_func = self._func
        original_variables = self._variables

        # Filter out variables that have been partially applied
        remaining_vars = [v for v in original_variables if v.name not in kwargs]

        def partial_func(**remaining_kwargs: object) -> str:
            merged = {**kwargs, **remaining_kwargs}
            return original_func(**merged)

        # Manually create Prompt without extracting signature from partial_func
        prompt = Prompt.__new__(Prompt)
        prompt._func = partial_func
        prompt._name = f"{self._name}_partial"
        prompt._description = self._description
        prompt._version = self._version
        prompt._cache_enabled = self._cache_enabled
        prompt._cache = {}
        prompt._variables = remaining_vars
        prompt._template_hash = self._template_hash

        return prompt

    def compose(self, *other_prompts: Prompt, separator: str = "\n\n") -> Prompt:
        """Compose this prompt with others.

        Example:
            full_prompt = system_prompt.compose(guidelines_prompt, safety_prompt)
            result = full_prompt(domain="AI")
        """
        original_func = self._func
        original_variables = self._variables

        # Collect all variable names from all prompts
        all_variables = {v.name: v for v in original_variables}
        for other in other_prompts:
            for v in other.variables:
                if v.name not in all_variables:
                    all_variables[v.name] = v

        def composed_func(**kwargs: object) -> str:
            parts = [original_func(**kwargs)]
            for other in other_prompts:
                # Only pass kwargs that the other prompt accepts
                other_kwargs = {
                    k: v for k, v in kwargs.items() if k in [var.name for var in other.variables]
                }
                parts.append(other(**other_kwargs))
            return separator.join(parts)

        # Manually create Prompt to preserve variable info
        prompt = Prompt.__new__(Prompt)
        prompt._func = composed_func
        prompt._name = f"{self._name}_composed"
        prompt._description = f"Composed: {self._name} + {len(other_prompts)} others"
        prompt._version = self._version
        prompt._cache_enabled = True
        prompt._cache = {}
        prompt._variables = list(all_variables.values())
        prompt._template_hash = self._template_hash

        return prompt

    @property
    def template(self) -> str:
        """Get the raw template string (executed with default parameters)."""
        defaults = {var.name: var.default for var in self._variables if var.default is not None}
        return self._func(**defaults)

    @property
    def name(self) -> str:
        """Get prompt name."""
        return self._name

    @property
    def description(self) -> str:
        """Get prompt description."""
        return self._description

    @property
    def version(self) -> PromptVersion:
        """Get prompt version."""
        return self._version

    @property
    def variables(self) -> list[PromptVariable]:
        """Get list of prompt variables/parameters."""
        return list(self._variables)

    @property
    def template_hash(self) -> str:
        """Get template hash for change detection."""
        return self._template_hash

    @property
    def cache_enabled(self) -> bool:
        """Whether caching is enabled."""
        return self._cache_enabled

    def clear_cache(self) -> None:
        """Clear the prompt cache."""
        self._cache.clear()

    def get_cache_stats(self) -> dict[str, int]:
        """Get cache statistics."""
        return {
            "size": len(self._cache),
            "max_size": 1000,
        }

    def __repr__(self) -> str:
        return (
            f"Prompt(name='{self._name}', "
            f"variables={len(self._variables)}, "
            f"version={self._version})"
        )

    def test_render(self, **kwargs: object) -> dict[str, object]:
        """Test prompt rendering with metadata.

        Returns dict with result, tokens estimate, and validation info.

        Example:
            result = prompt.test_render(domain="AI")
            print(result['output'])
            print(f"Estimated tokens: {result['estimated_tokens']}")
        """
        output = self(**kwargs)
        estimated_tokens = len(output) // 4

        return {
            "output": output,
            "length": len(output),
            "estimated_tokens": estimated_tokens,
            "parameters": kwargs,
            "version": str(self._version),
            "hash": self._template_hash,
        }


@overload
def prompt(  # type: ignore[explicit-any]
    func: Callable[..., str],
    *,
    name: str | None = None,
    description: str | None = None,
    version: PromptVersion | None = None,
    cache: bool = True,
) -> Prompt: ...


@overload
def prompt(  # type: ignore[explicit-any]
    func: None = None,
    *,
    name: str | None = None,
    description: str | None = None,
    version: PromptVersion | None = None,
    cache: bool = True,
) -> Callable[[Callable[..., str]], Prompt]: ...


def prompt(  # type: ignore[explicit-any]
    func: Callable[..., str] | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
    version: PromptVersion | None = None,
    cache: bool = True,
) -> Prompt | Callable[[Callable[..., str]], Prompt]:
    """Decorator that turns a plain Python function into a :class:`Prompt`.

    Calling the decorated function returns the rendered ``str`` as before,
    but the returned object is a :class:`Prompt` instance with validation,
    caching, versioning, partial application, and composition features.

    Can be used with or without arguments:

    .. code-block:: python

        from syrin import prompt

        # Simple — no decorator arguments
        @prompt
        def researcher_prompt(domain: str) -> str:
            return f"You are a researcher specialising in {domain}."

        # With arguments — name, description, version, cache
        @prompt(name="analyst", cache=False)
        def analyst_prompt(domain: str, tone: str = "professional") -> str:
            return f"You are a {tone} analyst for {domain}."

    Use the result directly as a string in an Agent class::

        class Researcher(Agent):
            model = Model.OpenAI("gpt-4o-mini")
            system_prompt = researcher_prompt(domain="AI")  # str

    Or keep it dynamic and call it at runtime::

        class Researcher(Agent):
            model = Model.OpenAI("gpt-4o-mini")

            @syrin.system_prompt
            def _sys(self) -> str:
                domain = getattr(self, "_domain", "general")
                return researcher_prompt(domain=domain)

    Args:
        func: The function to wrap.  Omit when passing keyword arguments.
        name: Override for ``prompt.name``.  Defaults to the function name.
        description: Human-readable description stored on the prompt object.
        version: Initial :class:`PromptVersion` (defaults to ``1.0.0``).
        cache: Cache rendered strings by argument set.  Set ``False`` for
            prompts whose output must vary on every call.  Defaults to
            ``True``.

    Returns:
        :class:`Prompt` when ``func`` is provided directly.
        A decorator (``Callable[[Callable], Prompt]``) when called with
        keyword arguments.
    """

    def decorator(f: Callable[..., str]) -> Prompt:  # type: ignore[explicit-any]
        return Prompt(
            f,
            name=name or f.__name__,
            description=description or f.__doc__ or "",
            version=version,
            cache=cache,
        )

    if func is not None:
        return decorator(func)
    return decorator


def validated(  # type: ignore[explicit-any]
    min_length: int | None = None,
    max_length: int | None = None,
    allowed_values: list[str] | None = None,
    pattern: str | None = None,
    custom: Callable[[Any], bool] | None = None,
) -> Callable[..., object]:
    """Create a validated prompt decorator with specific rules.

    Example:
        @validated(min_length=3, max_length=50)
        def name_prompt(name: str) -> str:
            return f"Hello, {name}!"

        name_prompt(name="John")  # OK
        name_prompt(name="J")     # Raises ValueError
    """

    def decorator(func: Callable[..., str]) -> Prompt:  # type: ignore[explicit-any]
        p = prompt(func)

        if p.variables:
            p.variables[0].validation = PromptValidation(
                min_length=min_length,
                max_length=max_length,
                allowed_values=allowed_values,
                pattern=pattern,
                custom_validator=custom,
            )

        return p

    return decorator


def system_prompt(func: Callable[..., str]) -> Callable[..., str]:  # type: ignore[explicit-any]
    """Mark a method as the agent's system prompt. One per agent class.

    Use inside an Agent subclass to encapsulate the system prompt.

    Example:
        class MyAgent(syrin.Agent):
            @syrin.system_prompt
            def my_prompt(self, user_name: str = "") -> str:
                return f"You assist {user_name or 'the user'}."

    Supports signatures: (self), (self, ctx: PromptContext), (self, **kwargs).
    """
    func._syrin_system_prompt = True  # type: ignore[attr-defined]
    return func


from syrin.prompt.context import (
    PromptContext,
    make_prompt_context,
)

__all__ = [
    "Prompt",
    "PromptVariable",
    "PromptValidation",
    "PromptVersion",
    "PromptContext",
    "make_prompt_context",
    "prompt",
    "system_prompt",
    "validated",
]
