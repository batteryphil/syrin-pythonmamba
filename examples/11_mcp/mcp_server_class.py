"""MCP Server Class Example.

Demonstrates:
- syrin.MCP base class with @tool
- Same @tool decorator as Agent
- .tools() and .select() for tool access

Requires: uv pip install syrin[serve]
Visit: http://localhost:8000/playground

Run: python -m examples.11_mcp.mcp_server_class
"""

from __future__ import annotations

import sys
from pathlib import Path

_root = Path(__file__).resolve().parents[2]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from syrin import MCP, tool


class ProductMCP(MCP):
    name = "product-mcp"
    description = "Product catalog tools for e-commerce"

    @tool
    def search_products(self, query: str, limit: int = 10) -> str:
        """Search the product catalog by query."""
        return f"Results for '{query}' (limit={limit}): [item1, item2, ...]"

    @tool
    def get_product(self, product_id: str) -> str:
        """Get product details by ID."""
        return f"Product {product_id}: name, price, description"


if __name__ == "__main__":
    from pathlib import Path

    from dotenv import load_dotenv

    from syrin import Agent, Model

    load_dotenv(Path(__file__).resolve().parent.parent / ".env")

    class ProductAgent(Agent):
        name = "product-agent"
        description = "Product catalog agent (MCP tools)"
        model = Model.mock()
        system_prompt = "You help users find products."
        tools = [ProductMCP()]

    mcp = ProductMCP()
    print("Tools:", [t.name for t in mcp.tools()])
    print("Selected:", [t.name for t in mcp.select("get_product")])
    spec = mcp.tools()[0]
    result = spec.func(query="shoes", limit=5)
    print("search_products result:", result)
    agent = ProductAgent()
    print("Serving at http://localhost:8000/playground")
    agent.serve(port=8000, enable_playground=True, debug=True)
