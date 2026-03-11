# Remote Config

Use **syrin.init()** to enable remote config so a backend (Syrin Cloud or your own) can push configuration overrides to running agents without code deploys. When you serve an agent, config routes are available for dashboards or `curl`.



## Examples

- **init_and_serve.py** — Enable remote config (optional), create an agent with budget, and serve. Config routes at `/config`, `/config/stream`. Use `curl` or Postman to GET schema and PATCH overrides.
- **POSTMAN_FLOW.md** — Full API flow for Postman: every call (GET/PATCH /config, /config/stream, health, chat, etc.), request/response shapes, and maintenance notes.
- **postman_collection.json** — Import into Postman (File → Import) for a ready-made collection with `baseUrl` and `agent_id` variables.

## Quick start

```bash
# From project root
PYTHONPATH=. python examples/12_remote_config/init_and_serve.py
```

Then in another terminal:

```bash
# Get schema and current values
curl -s http://localhost:8000/config | jq .

# Apply an override (use agent_id from GET /config)
curl -s -X PATCH http://localhost:8000/config \
  -H "Content-Type: application/json" \
  -d '{"agent_id":"my_agent:Agent","version":1,"overrides":[{"path":"budget.run","value":2.0}]}'
```

## With Syrin Cloud

Set `SYRIN_API_KEY=sk-...` and call `syrin.init()` (or `syrin.init(api_key="sk-...")`). Agents then register with the backend and receive overrides via SSE. No code change needed for config routes; they still work when serving.

## Custom transport

```python
from syrin.remote import init, PollingTransport

init(transport=PollingTransport(
    base_url="https://my-config-server/v1",
    api_key="sk-...",
    poll_interval=15,
))
```
