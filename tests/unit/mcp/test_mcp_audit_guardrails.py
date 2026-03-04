"""Tests for MCP audit and guardrails integration."""

from __future__ import annotations

import tempfile
from pathlib import Path

from fastapi import FastAPI
from starlette.testclient import TestClient

from syrin import MCP, AuditLog, GuardrailChain, tool
from syrin.guardrails import ContentFilter
from syrin.mcp.http import build_mcp_router


def test_mcp_audit_logs_tool_call() -> None:
    """MCP with audit=True writes tool call to AuditLog."""

    class ProductMCP(MCP):
        @tool
        def search(self, q: str) -> str:
            return f"found: {q}"

    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
        path = f.name
    try:
        audit = AuditLog(path=path)
        mcp = ProductMCP(audit=True, audit_log=audit)
        router = build_mcp_router(mcp)
        app = FastAPI()
        app.include_router(router, prefix="/mcp")
        client = TestClient(app)

        client.post(
            "/mcp",
            json={
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {"protocolVersion": "2024-11-05", "capabilities": {}, "clientInfo": {}},
            },
        )
        resp = client.post(
            "/mcp",
            json={
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/call",
                "params": {"name": "search", "arguments": {"q": "test"}},
            },
        )
        assert resp.status_code == 200

        content = Path(path).read_text()
        assert "mcp_tool_call" in content
        assert "search" in content
    finally:
        Path(path).unlink(missing_ok=True)


def test_mcp_guardrails_blocks_input() -> None:
    """MCP with guardrails blocks tool call when input fails guardrail."""

    class ProductMCP(MCP):
        @tool
        def search(self, q: str) -> str:
            return f"found: {q}"

    # ContentFilter blocks blocked words
    chain = GuardrailChain([ContentFilter(blocked_words=["blocked"])])
    mcp = ProductMCP(guardrails=chain)
    router = build_mcp_router(mcp)
    app = FastAPI()
    app.include_router(router, prefix="/mcp")
    client = TestClient(app)

    client.post(
        "/mcp",
        json={
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {"protocolVersion": "2024-11-05", "capabilities": {}, "clientInfo": {}},
        },
    )

    resp = client.post(
        "/mcp",
        json={
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {"name": "search", "arguments": {"q": "blocked word here"}},
        },
    )
    assert resp.status_code == 403
    data = resp.json()
    assert "error" in data
    assert "Guardrail blocked" in str(data["error"].get("message", ""))


def test_mcp_guardrails_allows_clean_input() -> None:
    """MCP with guardrails allows tool call when input passes."""

    class ProductMCP(MCP):
        @tool
        def search(self, q: str) -> str:
            return f"found: {q}"

    chain = GuardrailChain([ContentFilter(blocked_words=["blocked"])])
    mcp = ProductMCP(guardrails=chain)
    router = build_mcp_router(mcp)
    app = FastAPI()
    app.include_router(router, prefix="/mcp")
    client = TestClient(app)

    client.post(
        "/mcp",
        json={
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {"protocolVersion": "2024-11-05", "capabilities": {}, "clientInfo": {}},
        },
    )

    resp = client.post(
        "/mcp",
        json={
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {"name": "search", "arguments": {"q": "clean query"}},
        },
    )
    assert resp.status_code == 200
