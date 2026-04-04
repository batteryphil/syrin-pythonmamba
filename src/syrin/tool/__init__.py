"""Public tool package facade.

This package exposes the tool decorator and tool specification type used by
agents for callable tool integrations. Import from ``syrin.tool`` for the
end-user tool authoring API.
"""

import sys
import types

from syrin.tool._core import ToolSpec
from syrin.tool._core import tool as _tool_fn

__all__ = ["tool", "ToolSpec"]

# Re-export for `from syrin.tool import tool`
tool = _tool_fn


class _CallableToolModule(types.ModuleType):
    """Module subclass that is itself callable as the ``tool`` decorator.

    When Python processes ``from syrin.tool import tool``, it writes
    ``syrin.__dict__['tool'] = <module syrin.tool>``, shadowing the
    ``tool`` function exported by ``syrin``.  Making the module callable
    means ``@tool`` (i.e. ``syrin.tool(func)``) continues to work
    correctly even after that side-effect occurs.
    """

    def __call__(self, *args: object, **kwargs: object) -> object:
        return _tool_fn(*args, **kwargs)  # type: ignore[arg-type]


# Upgrade this module's class in-place so the change is transparent.
sys.modules[__name__].__class__ = _CallableToolModule
