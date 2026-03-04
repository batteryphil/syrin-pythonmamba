"""MCP Client — consume remote MCP servers as tool source."""

from __future__ import annotations

from typing import Any

import httpx

from syrin.mcp.schema import mcp_tool_to_tool_spec
from syrin.tool import ToolSpec


class MCPClient:
    """Connect to remote MCP server and expose its tools for agents.

    Example:
        >>> mcp = MCPClient("https://mcp.example.com")
        >>> agent = Agent(model=..., tools=[*mcp.tools()])
        >>> # Or select specific tools:
        >>> agent = Agent(model=..., tools=[mcp.select("search", "get")])
    """

    def __init__(
        self,
        url: str,
        *,
        tools: list[str] | None = None,
        timeout: float = 30.0,
        headers: dict[str, str] | None = None,
    ) -> None:
        """Create MCP client. On first tools()/select(), discovers remote tools.

        Args:
            url: MCP server URL (e.g. http://localhost:3000, https://mcp.example.com).
            tools: Whitelist of tool names; None = all tools.
            timeout: HTTP timeout in seconds.
            headers: Optional headers to send with every request.
        """
        self._url = url.rstrip("/")
        self._tool_whitelist = tools
        self._timeout = timeout
        self._headers = dict(headers) if headers else {}
        self._tools_cache: list[ToolSpec] | None = None

    def get_headers(self) -> dict[str, str]:
        """Headers added to every request."""
        return dict(self._headers)

    def _request(self, method: str, params: dict[str, Any] | None = None) -> Any:
        """Send JSON-RPC 2.0 request to MCP server."""
        payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": method,
            "params": params or {},
        }
        headers = self.get_headers()
        with httpx.Client(timeout=self._timeout) as client:
            resp = client.post(self._url, json=payload, headers=headers or None)
            resp.raise_for_status()
            data = resp.json()
        if "error" in data:
            raise RuntimeError(f"MCP error: {data['error']}")
        return data.get("result", {})

    def _discover_tools(self) -> list[ToolSpec]:
        """Fetch tools/list from MCP server and convert to ToolSpec."""
        result = self._request("tools/list")
        raw = result.get("tools", [])
        specs: list[ToolSpec] = []
        for t in raw:
            name = t.get("name", "")
            if self._tool_whitelist is not None and name not in self._tool_whitelist:
                continue

            def make_call(n: str, u: str, to: float) -> Any:
                def call(**kwargs: Any) -> Any:
                    headers = self.get_headers()
                    with httpx.Client(timeout=to) as c:
                        payload = {
                            "jsonrpc": "2.0",
                            "id": 1,
                            "method": "tools/call",
                            "params": {"name": n, "arguments": kwargs},
                        }
                        r = c.post(u, json=payload, headers=headers or None)
                        r.raise_for_status()
                        data = r.json()
                    if "error" in data:
                        raise RuntimeError(f"MCP tools/call error: {data['error']}")
                    res = data.get("result", {})
                    content = res.get("content", [])
                    if content and isinstance(content, list):
                        for c in content:
                            if isinstance(c, dict) and c.get("type") == "text":
                                return c.get("text", "")
                    return str(res)

                return call

            spec = mcp_tool_to_tool_spec(t, make_call(name, self._url, self._timeout))
            specs.append(spec)
        return specs

    def tools(self) -> list[ToolSpec]:
        """Discover and return all (or whitelisted) tools from remote MCP server."""
        if self._tools_cache is None:
            self._tools_cache = self._discover_tools()
        return list(self._tools_cache)

    def select(self, *names: str) -> list[ToolSpec]:
        """Return only tools with given names."""
        all_tools = self.tools()
        name_set = set(names)
        return [t for t in all_tools if t.name in name_set]
