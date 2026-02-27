# Serving Examples

Serve Syrin agents over HTTP.

**Requires:** `uv pip install syrin[serve]` (fastapi, uvicorn)

## Examples

- **http_serve.py** — Single agent: `agent.serve(port=8000)`
- **multi_agent_router.py** — Multiple agents: `AgentRouter(agents=[...]).serve(port=8000)`
- **mount_on_existing_app.py** — Mount on your FastAPI app: `app.include_router(agent.as_router(), prefix="/agent")`

## Routes

- `POST /chat` — Run agent, get full response
- `POST /stream` — SSE streaming
- `GET /health` — Liveness probe
- `GET /ready` — Readiness probe
- `GET /budget` — Budget state (if configured)
- `GET /describe` — Agent introspection

For multi-agent: `/agent/{name}/chat`, `/agent/{name}/health`, etc.
