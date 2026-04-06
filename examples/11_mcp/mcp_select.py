"""MCP .select() Example — Choose Subset of Tools.

Demonstrates:
- mcp.select("tool1", "tool2") — Agent gets only selected tools
- Local MCP: ProductMCP().select("search_products")
- Reduces tool surface for simpler agents

Requires: uv pip install syrin[serve]
Visit: http://localhost:8000/playground

Run: python -m examples.11_mcp.mcp_select
"""

from __future__ import annotations

import sys
from pathlib import Path

_root = Path(__file__).resolve().parents[2]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from dotenv import load_dotenv

from syrin import MCP, Agent, Model, tool

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

_MOCK_CATALOG = [
    {"id": "s1", "name": "Nike Air Max 270", "price": 150, "description": "Stylish sneaker."},
    {
        "id": "s2",
        "name": "Adidas Ultraboost",
        "price": 180,
        "description": "Comfortable running shoe.",
    },
]


class ProductMCP(MCP):
    @tool
    def search_products(self, query: str, limit: int = 10) -> str:
        """Search the product catalog."""
        items = _MOCK_CATALOG[:limit]
        return "\n".join(f"- {p['name']}: ${p['price']}" for p in items)

    @tool
    def get_product(self, product_id: str) -> str:
        """Get product by ID."""
        for p in _MOCK_CATALOG:
            if p["id"] == product_id:
                return f"{p['name']}: ${p['price']} — {p['description']}"
        return f"Product {product_id} not found."


# Agent with ALL tools (search + get)
class FullProductAgent(Agent):
    name = "full-agent"
    description = "Search and get products"
    model = Model.mock()
    system_prompt = "You help users find and look up products."
    tools = [ProductMCP()]


# Agent with ONLY search (via .select)
class SearchOnlyAgent(Agent):
    name = "search-agent"
    description = "Search products only"
    model = Model.mock()
    system_prompt = "You help users search products. Use search_products."
    tools = [ProductMCP().select("search_products")]


if __name__ == "__main__":
    mcp = ProductMCP()
    print("All tools:", [t.name for t in mcp.tools()])
    print("Selected:", [t.name for t in mcp.select("search_products")])

    agent = SearchOnlyAgent()
    result = agent.run("Find me shoes under $200")
    print(
        "\nSearchOnlyAgent response:",
        result.content[:200] + "..." if len(result.content) > 200 else result.content,
    )
    print("Serving at http://localhost:8000/playground")
    agent.serve(port=8000, enable_playground=True, debug=True)
