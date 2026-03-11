# Web Playground

A built-in web UI for testing and debugging Syrin agents locally. Chat with your agent, see cost and budget in real time, and observe hook events when debug mode is on.



## Enable the Playground

Set `enable_playground=True` when serving:

```python
from syrin import Agent
from syrin.model import Model

class Assistant(Agent):
    name = "assistant"
    description = "Helpful assistant"
    model = Model.Almock()
    system_prompt = "You are a helpful assistant."

agent = Assistant()
agent.serve(port=8000, enable_playground=True)
```

Visit **http://localhost:8000/playground** to open the chat interface.

## What You Get

| Feature | Description |
|---------|-------------|
| **Chat** | Send messages, see responses streamed in real time |
| **Cost per message** | Cost and token count after each assistant reply |
| **Budget gauge** | Visual indicator of budget remaining (when agent has a budget) |
| **Observability (debug)** | When `debug=True`, a collapsible panel shows hook events (LLM calls, tool calls, etc.) from the last response |

## Debug Mode — Observability

When `debug=True` and `enable_playground=True`, the playground displays an **Observability** panel after each response. It shows the hook stream (lifecycle events) from the last run:

- `agent.run.start` / `agent.run.end`
- `llm.request.start` / `llm.request.end` (model, tokens, duration)
- `tool.call.start` / `tool.call.end` (tool name, args, result)
- Budget checks, validation, etc.

```python
agent.serve(port=8000, enable_playground=True, debug=True)
```

The panel is collapsible and auto-expands when new events arrive. Use it to debug agent behavior, inspect tool calls, and verify cost/token data.

## Multi-Agent

When using `AgentRouter` with multiple agents, the playground shows an **agent selector** dropdown. Pick an agent, then chat. Budget and observability are per agent.

```python
from syrin.serve import AgentRouter
from syrin.serve.config import ServeConfig

config = ServeConfig(enable_playground=True, debug=True)
router = AgentRouter(agents=[researcher, writer], config=config)
router.serve(port=8000)
```

Visit **http://localhost:8000/playground**, select an agent, and chat.

## Custom App

If you include the router in your own FastAPI app (instead of using `agent.serve()` or `router.serve()`), call `add_playground_static_mount()` after `include_router()` so static assets are served:

```python
from fastapi import FastAPI
from syrin.serve import ServeConfig, add_playground_static_mount

app = FastAPI()
router = agent.as_router(ServeConfig(enable_playground=True))
app.include_router(router, prefix="/api")
add_playground_static_mount(app, "/api/playground")
```

The mount path must match the playground route (e.g. `/api/playground` when the router has prefix `/api`).

## Production

The playground is **dev-only**. In production, set `enable_playground=False` (the default). The `/playground` route will not be registered, and requests to `/playground` return 404.

## Syrin Branding

The playground displays a subtle "Powered by Syrin" link in the footer. It is visible but non-intrusive.
