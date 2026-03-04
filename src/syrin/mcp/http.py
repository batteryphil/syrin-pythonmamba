"""MCP HTTP endpoint — JSON-RPC 2.0 for initialize, tools/list, tools/call."""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING, Any, cast

from starlette.requests import Request

from syrin.mcp.schema import tool_spec_to_mcp, validate_tool_arguments

if TYPE_CHECKING:
    from syrin.mcp.server import MCP

# MCP spec: protocolVersion 2024-11-05 is widely supported
_MCP_PROTOCOL_VERSION = "2024-11-05"


def _init_result(mcp: MCP) -> dict[str, Any]:
    """Build MCP initialize result per spec."""
    try:
        import syrin

        version = getattr(syrin, "__version__", "0.1.0")
    except ImportError:
        version = "0.1.0"
    return {
        "protocolVersion": _MCP_PROTOCOL_VERSION,
        "capabilities": {"tools": {"listChanged": True}},
        "serverInfo": {"name": mcp.name, "version": version},
    }


def _find_mcp_in_tools(tools: list[Any]) -> MCP | None:
    """Return first MCP instance in tools list."""
    for t in tools:
        if isinstance(t, type):
            continue
        if hasattr(t, "_tool_specs") and hasattr(t, "tools"):
            from syrin.mcp.server import MCP as MCPCls

            return cast(MCPCls, t)
    return None


def build_mcp_router(mcp: MCP) -> Any:
    """Build FastAPI router for MCP JSON-RPC endpoint. Add with prefix via include_router."""
    from fastapi import APIRouter
    from fastapi.responses import JSONResponse

    router = APIRouter(include_in_schema=False)

    _empty: dict[str, Any] = {}

    @router.post("")
    @router.post("/")
    async def mcp_handler(request: Request, body: dict[str, Any] | None = None) -> Any:
        """JSON-RPC 2.0: tools/list, tools/call."""
        body = body or _empty
        req_id = body.get("id")
        method = body.get("method", "")
        params = body.get("params") or {}

        # MCP spec: initialize must be first; required for connection test
        if method == "initialize":
            if hasattr(mcp, "_emit_mcp_event"):
                from syrin.enums import Hook

                mcp._emit_mcp_event(Hook.MCP_CONNECTED, {"method": method, "params": params})
            return {"jsonrpc": "2.0", "id": req_id, "result": _init_result(mcp)}
        # initialized is a notification from client; acknowledge with empty result
        if method == "initialized":
            return {"jsonrpc": "2.0", "id": req_id, "result": {}}

        if method == "tools/list":
            specs = mcp.tools()
            tools = [tool_spec_to_mcp(t) for t in specs]
            return {"jsonrpc": "2.0", "id": req_id, "result": {"tools": tools}}
        if method == "tools/call":
            name = params.get("name", "")
            arguments = params.get("arguments") or {}
            emit = getattr(mcp, "_emit_mcp_event", None)
            if emit:
                from syrin.enums import Hook

                emit(Hook.MCP_TOOL_CALL_START, {"tool_name": name, "arguments": arguments})

            # Guardrails: input validation before tool execution
            guardrails = getattr(mcp, "guardrails", None)
            if guardrails is not None:
                from syrin.guardrails.enums import GuardrailStage

                input_check = guardrails.check(str(arguments), GuardrailStage.INPUT)
                if not input_check.passed:
                    if emit:
                        emit(
                            Hook.MCP_TOOL_CALL_END,
                            {
                                "tool_name": name,
                                "arguments": arguments,
                                "error": input_check.reason,
                            },
                        )
                    return JSONResponse(
                        status_code=403,
                        content={
                            "jsonrpc": "2.0",
                            "id": req_id,
                            "error": {
                                "code": -32003,
                                "message": f"Guardrail blocked: {input_check.reason}",
                            },
                        },
                    )

            for spec in mcp.tools():
                if spec.name == name:
                    try:
                        validate_tool_arguments(spec, arguments)
                        result = spec.func(**arguments)

                        # Guardrails: output validation after tool execution
                        if guardrails is not None and result is not None:
                            output_check = guardrails.check(str(result), GuardrailStage.OUTPUT)
                            if not output_check.passed:
                                if emit:
                                    emit(
                                        Hook.MCP_TOOL_CALL_END,
                                        {
                                            "tool_name": name,
                                            "arguments": arguments,
                                            "error": output_check.reason,
                                        },
                                    )
                                return JSONResponse(
                                    status_code=403,
                                    content={
                                        "jsonrpc": "2.0",
                                        "id": req_id,
                                        "error": {
                                            "code": -32003,
                                            "message": f"Guardrail blocked output: {output_check.reason}",
                                        },
                                    },
                                )

                        # Audit: log tool call when audit enabled
                        audit_log = getattr(mcp, "audit_log", None)
                        if audit_log is not None and getattr(mcp, "audit", False):
                            from syrin.audit.models import AuditEntry

                            backend = audit_log.get_backend("./mcp_audit.jsonl")
                            entry = AuditEntry(
                                source=getattr(mcp, "name", "mcp"),
                                event="mcp_tool_call",
                                tool_name=name,
                                duration_ms=0,
                                extra={
                                    "arguments": str(arguments)[:500],
                                    "result_preview": str(result)[:200] if result else None,
                                },
                            )
                            with contextlib.suppress(Exception):
                                backend.write(entry)

                        content = (
                            [{"type": "text", "text": str(result)}] if result is not None else []
                        )
                        if emit:
                            emit(
                                Hook.MCP_TOOL_CALL_END,
                                {"tool_name": name, "arguments": arguments, "result": result},
                            )
                        return {"jsonrpc": "2.0", "id": req_id, "result": {"content": content}}
                    except Exception as e:
                        if emit:
                            emit(
                                Hook.MCP_TOOL_CALL_END,
                                {"tool_name": name, "arguments": arguments, "error": str(e)},
                            )
                        import jsonschema

                        code = -32602 if isinstance(e, jsonschema.ValidationError) else -32603
                        return JSONResponse(
                            status_code=200,
                            content={
                                "jsonrpc": "2.0",
                                "id": req_id,
                                "error": {"code": code, "message": str(e)},
                            },
                        )
            if emit:
                emit(
                    Hook.MCP_TOOL_CALL_END,
                    {"tool_name": name, "arguments": arguments, "error": f"Unknown tool: {name}"},
                )
            return JSONResponse(
                status_code=200,
                content={
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "error": {"code": -32602, "message": f"Unknown tool: {name}"},
                },
            )
        return JSONResponse(
            status_code=200,
            content={
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {"code": -32601, "message": f"Method not found: {method}"},
            },
        )

    return router
