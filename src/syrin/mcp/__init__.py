"""MCP (Model Context Protocol) — syrin.MCP server and syrin.MCPClient."""

from syrin.mcp.client import MCPClient
from syrin.mcp.server import MCP

__all__ = [
    "MCP",
    "MCPClient",
]
