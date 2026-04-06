"""MCP Co-location Example.

Demonstrates:
- MCP instance in Agent tools
- /mcp route auto-mounted alongside /chat
- GET /.well-known/agent-card.json for discovery

Requires: uv pip install syrin[serve]

Run: python -m examples.11_mcp.mcp_colocation
Then: curl http://localhost:8000/.well-known/agent-card.json
      curl -X POST http://localhost:8000/mcp -H "Content-Type: application/json" \\
        -d '{"jsonrpc":"2.0","id":1,"method":"tools/list","params":{}}'
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

# Demo mock catalog — replace with real search in production.
_MOCK_CATALOG = [
    {
        "id": "s1",
        "name": "Nike Air Max 270",
        "price": 150,
        "description": "Stylish sneaker for casual wear.",
    },
    {
        "id": "s2",
        "name": "Adidas Ultraboost",
        "price": 180,
        "description": "Comfortable running shoe.",
    },
    {"id": "s3", "name": "Vans Old Skool", "price": 60, "description": "Iconic skate shoe."},
]


class ProductMCP(MCP):
    @tool
    def search_products(self, query: str, limit: int = 10) -> str:
        """Search the product catalog. Returns product id, name, price, description."""
        items = _MOCK_CATALOG[:limit]
        lines = [f"- {p['name']} (id={p['id']}): ${p['price']} — {p['description']}" for p in items]
        return f"Results for '{query}' (limit={limit}):\n" + "\n".join(lines)

    @tool
    def get_product(self, product_id: str) -> str:
        """Get product by ID."""
        for p in _MOCK_CATALOG:
            if p["id"] == product_id:
                return f"{p['name']}: ${p['price']} — {p['description']}"
        return f"Product {product_id} not found."


class ProductAgent(Agent):
    name = "product-agent"
    description = "E-commerce product search and cart management"
    model = Model.mock()
    system_prompt = "You help users find products."
    tools = [ProductMCP()]


if __name__ == "__main__":
    agent = ProductAgent()
    agent.serve(port=8000)
