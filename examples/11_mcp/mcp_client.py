"""MCPClient Example — Consume Remote MCP Server.

Demonstrates:
- syrin.MCPClient — Connect to remote MCP server, discover tools
- Agent(tools=[mcp_client]) — Agent uses remote MCP tools
- tools=[mcp.select(...)] — Pick specific tools from remote

Requires: uv pip install syrin[serve]
Visit: http://localhost:8000/playground

Usage:
  1. Start an MCP server in another terminal:
     python -m examples.11_mcp.mcp_standalone_serve
     (Serves at http://localhost:3000/mcp)

  2. Run this example:
     python -m examples.11_mcp.mcp_client
"""

from __future__ import annotations

import sys
from pathlib import Path

_root = Path(__file__).resolve().parents[2]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from dotenv import load_dotenv

from syrin import Agent, MCPClient, Model

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

# Connect to remote MCP server (start mcp_standalone_serve first)
MCP_URL = "http://localhost:3000/mcp"


class ProductAgent(Agent):
    """Agent that uses remote MCP tools via MCPClient."""

    name = "product-agent"
    description = "E-commerce product search (remote MCP tools)"
    model = Model.mock()
    system_prompt = "You help users find products. Use the search and get tools."
    tools = [MCPClient(MCP_URL)]


# To use only specific tools: tools=[MCPClient(MCP_URL).select("search_products")]
# (.select() discovers tools on first use — ensure server is running)


if __name__ == "__main__":
    print("Connecting to MCP server at", MCP_URL)
    print("Start mcp_standalone_serve in another terminal if not running.\n")

    mcp = MCPClient(MCP_URL)
    tools = mcp.tools()
    print("Discovered tools:", [t.name for t in tools])

    agent = ProductAgent()
    result = agent.run("What shoes do you have?")
    preview = result.content[:300] + "..." if len(result.content) > 300 else result.content
    print("\nAgent response:", preview)
    print("Serving at http://localhost:8000/playground")
    agent.serve(port=8000, enable_playground=True, debug=True)
