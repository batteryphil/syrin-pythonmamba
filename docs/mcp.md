# MCP — Model Context Protocol

Syrin integrates MCP (Model Context Protocol) so you can **group tools in MCP** and **use MCP inside your agent's tools**. Define an MCP server with `@tool` methods (same decorator as Agent), then add the MCP instance to `Agent(tools=[...])`. The agent uses MCP tools as if they were regular tools.

**Recommended path:** Add the MCP instance to `tools=[...]`:
```python
mcp = ProductMCP()
agent = Agent(tools=[mcp], model=..., system_prompt=...)
```
Use `mcp.select("tool1", "tool2")` to whitelist specific tools.

**Two ways to use MCP:**
1. **Group tools** — Create an MCP class with multiple `@tool` methods. Keeps related tools organized.
2. **Use MCP in agents** — Put the MCP instance in `tools=[ProductMCP()]`. Agent can call all MCP tools. When serving, `/mcp` is auto-mounted alongside `/chat`.

**Note:** `httpx` is included with syrin.

## syrin.MCP — Declarative MCP Server (group your tools)

Define tools with `@tool` inside the MCP class, same as Agent. Group related tools (e.g. product catalog) in one MCP.

### Standalone serving

Serve MCP independently (without an Agent):

```python
mcp = ProductMCP()
mcp.serve(port=3000)           # HTTP at http://localhost:3000/mcp
mcp.serve(stdin=sys.stdin)     # STDIO (JSON-RPC over stdin/stdout)
```

Transport is inferred: pass `stdin` for STDIO, otherwise HTTP. Requires `syrin[serve]` for HTTP.

```python
from syrin import Agent, MCP, tool
from syrin.model import Model

class ProductMCP(MCP):
    name = "product-mcp"
    description = "Product catalog tools for e-commerce"

    @tool
    def search_products(self, query: str, limit: int = 10) -> str:
        """Search the product catalog by query."""
        return catalog.search(query, limit)

    @tool
    def get_product(self, product_id: str) -> str:
        """Get product details by ID."""
        return catalog.get(product_id)

mcp = ProductMCP()
mcp.tools()      # [ToolSpec(search_products), ToolSpec(get_product)]
mcp.select("search_products")  # [ToolSpec(search_products)]
```

## syrin.MCPClient — Consume Remote MCP Servers

Connect to a remote MCP server and expose its tools for agents:

```python
from syrin import Agent, MCPClient
from syrin.model import Model

# All tools from remote MCP
shopify_mcp = MCPClient("https://mcp.shopify.com")

# Or whitelist at connection time
shopify_mcp = MCPClient("https://mcp.shopify.com", tools=["search_products", "get_product"])

class ShopAgent(Agent):
    model = Model.Almock()
    tools = [
        shopify_mcp,                                    # All tools
        shopify_mcp.select("search_products", "get_product"),  # Or pick specific tools
    ]
```

Optional `headers=` — pass headers (e.g. custom or API key) for every request:

```python
mcp = MCPClient("https://mcp.example.com", headers={"X-Custom-Header": "value"})
```

## Agent Consuming MCP — Patterns

**1. Direct inclusion** — Agent gets all MCP tools:
```python
tools = [product_mcp]
```

**2. `.select(toolName1, toolName2)`** — Agent gets only selected tools:
```python
tools = [product_mcp.select("search_products", "get_product")]
```

**3. `.tools()`** — Spread into agent's tools:
```python
tools = [*product_mcp.tools()]
```

## Co-location — MCP with Agent on Same Port

When an MCP instance is in `Agent(tools=[...])`, the serve layer auto-mounts `/mcp` alongside `/chat`:

```python
product_mcp = ProductMCP()

class ProductAgent(Agent):
    name = "product-agent"
    description = "E-commerce product search"
    model = Model.Almock()
    tools = [product_mcp, my_add_to_cart_tool]

ProductAgent().serve(port=8000)
# POST /chat          → Agent endpoint
# POST /stream        → SSE streaming
# POST /mcp           → MCP server (tools/list, tools/call)
# GET  /.well-known/agent-card.json  → A2A Agent Card
```

All served from one process, one port.

## MCP JSON-RPC

The `/mcp` endpoint speaks JSON-RPC 2.0:

- `tools/list` — Returns list of tools with name, description, inputSchema
- `tools/call` — Executes a tool: `{"name": "search_products", "arguments": {"query": "shoes"}}`

**Input validation** — Tool arguments are validated against the tool's JSON schema before execution. Invalid arguments (missing required fields, wrong types) return JSON-RPC error `-32602` (Invalid params).

## MCP lifecycle events

MCP emits lifecycle events you can subscribe to:

```python
from syrin import MCP
from syrin.enums import Hook

mcp = ProductMCP()
mcp.events.on(Hook.MCP_CONNECTED, lambda ctx: print(f"Client connected: {ctx.params}"))
mcp.events.on(Hook.MCP_TOOL_CALL_START, lambda ctx: print(f"Tool: {ctx.tool_name}"))
mcp.events.on(Hook.MCP_TOOL_CALL_END, lambda ctx: print(f"Done: {ctx.get('result') or ctx.get('error')}"))
mcp.events.on(Hook.MCP_DISCONNECTED, lambda ctx: print("Client disconnected"))
```

| Hook | When | Context |
|------|------|---------|
| `MCP_CONNECTED` | Client sends `initialize` | `method`, `params` |
| `MCP_TOOL_CALL_START` | Before `tools/call` execution | `tool_name`, `arguments` |
| `MCP_TOOL_CALL_END` | After `tools/call` | `tool_name`, `arguments`, `result` or `error` |
| `MCP_DISCONNECTED` | STDIO EOF (STDIO only) | `transport` |

**Audit logging** — Log MCP tool calls when `audit=True`:

```python
from syrin import AuditLog

mcp = ProductMCP(audit=True, audit_log=AuditLog(path="./mcp_audit.jsonl"))
```

**Guardrails** — Validate tool input and output with `GuardrailChain`:

```python
from syrin import ContentFilter, GuardrailChain

chain = GuardrailChain([ContentFilter(blocked_words=["spam", "forbidden"])])
mcp = ProductMCP(guardrails=chain)
```

## Examples

- `examples/11_mcp/mcp_server_class.py` — syrin.MCP with @tool, .tools(), .select()
- `examples/11_mcp/mcp_standalone_serve.py` — Standalone serve (HTTP/STDIO) + MCP lifecycle events
- `examples/11_mcp/mcp_colocation.py` — MCP co-located with agent
- `examples/11_mcp/mcp_client.py` — MCPClient consuming remote MCP server (start mcp_standalone_serve first)
- `examples/11_mcp/mcp_select.py` — .select() to give agent only a subset of tools
