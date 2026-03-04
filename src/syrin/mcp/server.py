"""MCP Server — syrin.MCP base class with @tool."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, TextIO

if TYPE_CHECKING:
    from syrin.enums import Hook

from syrin.events import EventContext, Events
from syrin.tool import ToolSpec


def _collect_mcp_class_tools(cls: type) -> list[ToolSpec]:
    """Collect ToolSpec from @tool-decorated MCP class methods."""
    seen: set[str] = set()
    result: list[ToolSpec] = []
    for c in cls.__mro__:
        if c is object or c is MCP:
            continue
        for _attr_name, val in c.__dict__.items():
            if isinstance(val, ToolSpec) and val.name not in seen:
                seen.add(val.name)
                result.append(val)
    return result


class MCP:
    """Declarative MCP Server — tools defined with @tool inside class (mandatory).

    Use like Agent: subclass MCP, define @tool methods. Place MCP instance in
    Agent(tools=[mcp]) for co-location — /mcp route is auto-mounted.

    Example:
        >>> class ProductMCP(MCP):
        ...     @tool
        ...     def search_products(self, query: str, limit: int = 10) -> str:
        ...         return catalog.search(query, limit)
        >>> mcp = ProductMCP()
        >>> mcp.tools()  # [ToolSpec(search_products), ...]
    """

    name: str = "mcp"
    description: str = ""

    def __init__(
        self,
        *,
        audit: bool | object = False,
        audit_log: object | None = None,
        guardrails: object | None = None,
    ) -> None:
        """Collect @tool methods from class. Tools must be defined in-class.

        Args:
            audit: If True, log tool calls. Use audit_log to pass AuditLog; else default path.
            audit_log: AuditLog instance for tool call logging when audit is True.
            guardrails: Optional GuardrailChain for tool input/output validation.
        """
        self.events = Events(self._emit_mcp_event)
        self._audit = bool(audit)
        self._audit_log = audit_log
        self._guardrails = guardrails
        if self._audit and self._audit_log is None:
            from syrin.audit import AuditLog

            self._audit_log = AuditLog(path="./mcp_audit.jsonl")
        self._tool_specs = _collect_mcp_class_tools(self.__class__)
        # Bind instance methods to self
        bound: list[ToolSpec] = []
        import inspect

        for spec in self._tool_specs:
            sig = inspect.signature(spec.func)
            params = list(sig.parameters)
            if params and params[0] == "self":
                bound_func = spec.func.__get__(self, type(self))
                bound.append(
                    ToolSpec(
                        name=spec.name,
                        description=spec.description,
                        parameters_schema=spec.parameters_schema,
                        func=bound_func,
                        requires_approval=spec.requires_approval,
                        inject_run_context=spec.inject_run_context,
                    )
                )
            else:
                bound.append(spec)
        self._tool_specs = bound

    @property
    def audit(self) -> bool:
        """Whether to log tool calls to agent's AuditLog when co-located."""
        return self._audit

    @property
    def audit_log(self) -> object | None:
        """AuditLog for tool call logging; None if not configured."""
        return self._audit_log

    @property
    def guardrails(self) -> object | None:
        """GuardrailChain for tool input/output; None if not configured."""
        return self._guardrails

    def tools(self) -> list[ToolSpec]:
        """Return all tools as ToolSpec list. Use in Agent(tools=[*mcp.tools()])."""
        return list(self._tool_specs)

    def select(self, *names: str) -> list[ToolSpec]:
        """Return only tools with given names. Use in Agent(tools=[mcp.select('a','b')])."""
        name_set = set(names)
        return [t for t in self._tool_specs if t.name in name_set]

    def _emit_mcp_event(self, hook: Hook, ctx: EventContext | dict[str, Any]) -> None:
        """Internal: trigger MCP lifecycle hooks."""
        from syrin.enums import Hook

        if not isinstance(hook, Hook):
            return
        if isinstance(ctx, dict):
            ctx = EventContext(ctx)
        self.events._trigger_before(hook, ctx)
        self.events._trigger(hook, ctx)
        self.events._trigger_after(hook, ctx)

    def serve(
        self,
        *,
        port: int = 3000,
        host: str = "0.0.0.0",
        stdin: TextIO | None = None,
        stdout: TextIO | None = None,
        **kwargs: Any,
    ) -> None:
        """Serve this MCP via HTTP or STDIO. Blocks until stopped.

        Transport is inferred:
        - HTTP: when stdin is None (default). Uses uvicorn.
        - STDIO: when stdin is provided. Reads JSON-RPC from stdin, writes to stdout.

        Requires syrin[serve] (fastapi, uvicorn) for HTTP.

        Args:
            port: Port for HTTP mode. Default 3000.
            host: Host for HTTP mode. Default "0.0.0.0".
            stdin: If provided, use STDIO transport (JSON-RPC over stdin/stdout).
            stdout: Output stream for STDIO. Defaults to sys.stdout.
            **kwargs: Ignored (for API compatibility).

        Example:
            >>> mcp = ProductMCP()
            >>> mcp.serve(port=3000)           # HTTP on localhost:3000
            >>> mcp.serve(stdin=sys.stdin)     # STDIO (subprocess, background)
        """
        if stdin is not None:
            from syrin.mcp.stdio import run_stdio_mcp

            run_stdio_mcp(self, stdin=stdin, stdout=stdout)
        else:
            try:
                import uvicorn
            except ImportError as e:
                raise ImportError(
                    "HTTP serving requires uvicorn. Install with: uv pip install syrin[serve]"
                ) from e
            import sys

            from fastapi import FastAPI

            from syrin.mcp.http import build_mcp_router
            from syrin.mcp.stdio import _syrin_cli_message

            use_color = getattr(sys.stdout, "isatty", lambda: False)()
            print(_syrin_cli_message(use_color=use_color), flush=True)

            app = FastAPI(title=f"MCP: {self.name}", description=self.description or "MCP server")
            app.include_router(build_mcp_router(self), prefix="/mcp")
            uvicorn.run(app, host=host, port=port)
