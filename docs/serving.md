# Serving Agents — HTTP, CLI, STDIO

Serving is built-in — one line to run your agent. No extra wiring.

**Serve via HTTP or CLI:**

| Protocol | When to Use | How |
|----------|-------------|-----|
| **HTTP** | Production API, webhooks, chatbots | `agent.serve(port=8000)` — POST `/chat`, `/stream`, etc. |
| **CLI** | Local dev, interactive testing | `agent.serve(protocol=ServeProtocol.CLI)` — terminal REPL |
| **STDIO** | Background tasks, subprocess | `agent.serve(protocol=ServeProtocol.STDIO)` — JSON lines on stdin/stdout |

- **Web playground** — Add `enable_playground=True` and visit http://localhost:8000/playground. Chat, see cost, budget, and traces.
- **CLI REPL** — Use `protocol=ServeProtocol.CLI` for interactive terminal testing.

**Tip:** Use `debug=True` when serving, or run scripts with `--trace`, to see LLM calls, tool calls, and costs in the console.



## Quick Start

```python
from syrin import Agent
from syrin.model import Model

class Assistant(Agent):
    name = "assistant"
    description = "Helpful assistant"
    model = Model.Almock()
    system_prompt = "You are a helpful assistant."

agent = Assistant()
agent.serve(port=8000)  # HTTP on localhost:8000
```

Visit `http://localhost:8000/health`, POST to `/chat` with `{"message": "Hi"}`, or open `http://localhost:8000/playground` (when `enable_playground=True`) for a chat UI. See [docs/playground.md](playground.md) for details.

## Protocol Comparison

| Protocol | When to Use | Interface |
|----------|-------------|-----------|
| `ServeProtocol.HTTP` | Production API, webhooks, chatbots | FastAPI server with `/chat`, `/stream`, etc. |
| `ServeProtocol.CLI` | Local dev, interactive testing | Terminal REPL (prompt, response, cost/budget) |
| `ServeProtocol.STDIO` | Background tasks, subprocess, MCP host | stdin/stdout JSON lines |

## HTTP Routes

When using `agent.serve()` or `agent.as_router()`:

| Route | Method | Description |
|-------|--------|-------------|
| `/chat` | POST | Run agent, return full response |
| `/stream` | POST | SSE streaming |
| `/health` | GET | Liveness probe |
| `/ready` | GET | Readiness probe |
| `/budget` | GET | Budget state (404 if not configured) |
| `/describe` | GET | Agent introspection (name, tools, budget) |
| `/playground` | GET | Web playground (when `enable_playground=True`) |
| `/mcp` | POST | MCP JSON-RPC (when MCP in agent tools) |
| `/.well-known/agent-card.json` | GET | A2A Agent Card (when discovery enabled) |

**Request body for `/chat` and `/stream`:** `{"message": "..."}` or `{"input": "..."}`

## Agent Discovery — A2A Agent Card

When `enable_discovery` is on (auto when agent has `name`), the serve layer exposes an A2A Agent Card at `GET /.well-known/agent-card.json`. Other agents, frontends, and infrastructure can discover capabilities without manual configuration.

```python
class ProductAgent(Agent):
    name = "product-agent"
    description = "E-commerce product search and cart management"
    model = Model.Almock()
    tools = [search_products, get_product]

ProductAgent().serve(port=8000)
# GET /.well-known/agent-card.json returns: name, description, url, skills (tools), etc.
```

**Override Agent Card** — Set `agent_card = syrin.AgentCard(...)` on your Agent class to override auto-generated fields (provider, authentication, capabilities, name, description, etc.):

```python
from syrin import Agent, AgentCard, AgentCardAuth, AgentCardProvider

class ProductAgent(Agent):
    name = "product-agent"
    description = "E-commerce assistant"
    model = Model.Almock()
    agent_card = AgentCard(
        provider=AgentCardProvider(organization="MyCompany", url="https://mycompany.com"),
        authentication=AgentCardAuth(schemes=["oauth2"], oauth_url="https://auth.mycompany.com/token"),
    )
```

**Discovery hook** — `Hook.DISCOVERY_REQUEST` is emitted when `/.well-known/agent-card.json` is requested. Context: `agent_name`, `path`.

**Disable discovery:** `ServeConfig(enable_discovery=False)`

## Mount on Existing FastAPI App

```python
from fastapi import FastAPI
from syrin import Agent
from syrin.model import Model

class Assistant(Agent):
    name = "assistant"
    model = Model.Almock()
    system_prompt = "You are a helpful assistant."

app = FastAPI(title="My API")
agent = Assistant()
app.include_router(agent.as_router(), prefix="/agent")
```

Then run: `uvicorn my_app:app --reload`

Visit `/agent/health`, POST to `/agent/chat`.

## Multi-Agent Router

Serve multiple agents on one server with routes per agent:

```python
from syrin import Agent
from syrin.model import Model
from syrin.serve import AgentRouter

class Researcher(Agent):
    name = "researcher"
    model = Model.Almock()
    system_prompt = "You are a researcher."

class Writer(Agent):
    name = "writer"
    model = Model.Almock()
    system_prompt = "You are a writer."

router = AgentRouter(agents=[Researcher(), Writer()])
router.serve(port=8000)
```

Routes: `/agent/researcher/chat`, `/agent/writer/chat`, etc.

Or mount on existing app:

```python
app = FastAPI()
app.include_router(router.fastapi_router(), prefix="/api/v1")
```

## ServeConfig

Configure host, port, route prefix:

```python
from syrin.serve import ServeConfig

config = ServeConfig(
    host="0.0.0.0",
    port=8000,
    route_prefix="/api/v1",
)
agent.serve(config=config)
```

## CORS and Auth — Use Your Own Middlewares

Syrin does not handle CORS or auth. Mount agent routes on your own FastAPI app and add middlewares from Starlette or other libraries:

```python
from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware

from syrin import Agent
from syrin.model import Model

class Assistant(Agent):
    name = "assistant"
    model = Model.Almock()
    system_prompt = "You are helpful."

agent = Assistant()
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    origins=["https://myapp.com"],
    allow_credentials=True,
)
# Add your auth middleware (OAuth2, JWT, etc.) here
app.include_router(agent.as_router(), prefix="/agent")
```

Then run: `uvicorn my_app:app` — your CORS and auth apply to all routes including agent routes.

## CLI REPL

```python
agent.serve(protocol=ServeProtocol.CLI)
```

Interactive REPL: prompt (`> `), run agent, show cost/budget per turn. Ctrl+C to exit.

## STDIO (JSON Lines)

```python
agent.serve(protocol=ServeProtocol.STDIO)
```

Reads one JSON per line from stdin, writes one JSON per line to stdout.

**Input (stdin):** `{"input": "Hello", "conversation_id": "optional"}`

**Output (stdout):** `{"content": "...", "cost": 0.0, "tokens": N, "conversation_id": "optional"}`

Use for background tasks, subprocess, MCP host calling your agent.

```bash
echo '{"input": "Hi"}' | python -m examples.serving.stdio_serve
```

## Examples

- `examples/serving/http_serve.py` — Single agent HTTP
- `examples/serving/multi_agent_router.py` — Multiple agents
- `examples/serving/mount_on_existing_app.py` — Mount on FastAPI
- `examples/serving/cli_serve.py` — CLI REPL
- `examples/serving/stdio_serve.py` — STDIO JSON lines
- `examples/11_mcp/` — MCP server, client, co-location (see `docs/mcp.md`)
