"""Public prompt package facade.

This package exposes Syrin's prompt-definition API for reusable, validated
prompt functions. Import from ``syrin.prompt`` when you need the ``Prompt``
wrapper, prompt metadata models, decorators, or prompt-context helpers used
during agent prompt construction.

Why use this package:
    - Define prompts as native Python functions with validation metadata.
    - Reuse prompt versions and variables across agent workflows.
    - Build prompt contexts consistently for templated system prompts.

Typical usage:
    >>> from syrin.prompt import prompt
    >>> @prompt
    ... def system_prompt(domain: str = "general") -> str:
    ...     return f"You are an expert in {domain}."

Exported surface:
    - ``Prompt`` and related metadata/value objects
    - decorators for defining and validating prompts
    - prompt-context helpers used by higher-level agent code
"""

import sys
import types

from syrin.prompt._core import (
    Prompt,
    PromptContext,
    PromptValidation,
    PromptVariable,
    PromptVersion,
    make_prompt_context,
    system_prompt,
    validated,
)
from syrin.prompt._core import (
    prompt as _prompt_fn,
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

# Re-export for `from syrin.prompt import prompt`
prompt = _prompt_fn


class _CallablePromptModule(types.ModuleType):
    """Module subclass that is itself callable as the ``prompt`` decorator.

    When Python processes ``from syrin.prompt import ...``, it writes
    ``syrin.__dict__['prompt'] = <module syrin.prompt>``, shadowing the
    ``prompt`` function exported by ``syrin``.  Making the module callable
    means ``@prompt`` continues to work correctly even after that side-effect.
    """

    def __call__(self, *args: object, **kwargs: object) -> object:
        return _prompt_fn(*args, **kwargs)  # type: ignore[call-overload]


sys.modules[__name__].__class__ = _CallablePromptModule
