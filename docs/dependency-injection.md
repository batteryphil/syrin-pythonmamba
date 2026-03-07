# Dependency Injection

Syrin supports dependency injection so tools can receive runtime dependencies (DB, search client, user prefs) from the agent instead of hardcoding. This makes agents **testable** (inject mocks), **multi-tenant** (different deps per user), and **environment-aware** (dev vs prod).

---

## Why dependency injection?

Without DI, tools often hardcode their backends:

```python
# Without DI — hardcoded, untestable
@tool
def get_weather(city: str) -> str:
    return requests.get(f"https://api.weather.com/{city}").text  # Can't mock!

@tool
def get_user_meetings(user_id: str) -> str:
    db = PostgresDB("postgresql://prod...")  # Always production!
    return db.query("SELECT * FROM meetings WHERE user_id = ?", user_id)
```

**Problems:**
- **Untestable** — Tests hit real APIs or DBs
- **Rigid** — Same backend for every user and environment
- **Brittle** — API keys, URLs baked in

With DI, tools receive backends from the agent:

```python
# With DI — injectable, testable, flexible
@tool
def get_weather(ctx: RunContext[WeatherDeps], city: str) -> str:
    return ctx.deps.weather_api.get(city)  # Inject MockWeatherAPI in tests

@tool
def get_user_meetings(ctx: RunContext[AppDeps], user_id: str) -> str:
    return ctx.deps.db.query("SELECT * FROM meetings WHERE user_id = ?", user_id)
```

**Benefits:**
- **Testable** — Inject mocks; no real API or DB calls
- **Multi-tenant** — Different DB or API per user
- **Environment-aware** — Dev vs staging vs prod backends

### Flow: Without DI (hardcoded)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ Agent                                                                        │
│   tools = [get_weather, get_user_meetings]                                   │
└─────────────────────────────────────────────────────────────────────────────┘
         │
         │ LLM calls tool("get_weather", {"city": "Paris"})
         ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ Tool: get_weather(city)                                                      │
│   → requests.get("https://api.weather.com/Paris")  ← HARDCODED               │
│   → Real API, real network, real credentials                                 │
└─────────────────────────────────────────────────────────────────────────────┘
         │
         ▼
   Weather API (production only — can't swap for tests!)
```

**Problem:** Tool is tightly coupled to a single backend. Tests must hit the real API or use fragile patching.

### Flow: With DI (injected)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ Agent(deps=MyDeps(db=..., weather_api=...))                                  │
│   deps created ONCE at agent creation                                        │
└─────────────────────────────────────────────────────────────────────────────┘
         │
         │ LLM calls tool("get_weather", {"city": "Paris"})
         ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ Agent builds RunContext                                                      │
│   ctx = RunContext(deps=agent._deps, agent_name="MyAgent", ...)              │
└─────────────────────────────────────────────────────────────────────────────┘
         │
         │ Tool receives ctx + arguments
         ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ Tool: get_weather(ctx, city)                                                 │
│   → ctx.deps.weather_api.get(city)  ← INJECTED                               │
│   → Uses whatever backend was passed to Agent(deps=...)                      │
└─────────────────────────────────────────────────────────────────────────────┘
         │
         ▼
   Weather API (or MockWeatherAPI in tests — you choose!)
```

**Benefit:** Tool is decoupled from backends. Swap `deps` at agent creation — production, tests, or per-tenant.

---

## How it works

End-to-end flow from agent creation to tool execution:

```
┌──────────────────────────────────────────────────────────────────────────────┐
│ 1. Agent creation                                                             │
│    agent = MyAgent(deps=MyDeps(db=PostgresDB(...), user_id="alice"))          │
│    Agent stores deps internally                                               │
└──────────────────────────────────────────────────────────────────────────────┘
                                         │
                                         ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│ 2. User request                                                               │
│    agent.response("What meetings do I have today?")                           │
│    Agent builds messages, calls LLM                                           │
└──────────────────────────────────────────────────────────────────────────────┘
                                         │
                                         ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│ 3. LLM returns tool call                                                      │
│    {"name": "get_user_meetings", "arguments": {"user_id": "alice"}}           │
│    (Tool schema excludes ctx — LLM never sees it)                             │
└──────────────────────────────────────────────────────────────────────────────┘
                                         │
                                         ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│ 4. Agent executes tool                                                        │
│    • Detects tool has ctx param (inject_run_context=True)                     │
│    • Builds RunContext(deps=agent._deps, agent_name=..., budget_state=...)    │
│    • Calls tool(ctx=run_context, user_id="alice")                             │
└──────────────────────────────────────────────────────────────────────────────┘
                                         │
                                         ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│ 5. Tool runs                                                                  │
│    def get_user_meetings(ctx, user_id):                                       │
│        return ctx.deps.db.query(...)  # Uses injected db                      │
│    Returns result to LLM                                                      │
└──────────────────────────────────────────────────────────────────────────────┘
```

### Decision: Does this tool need ctx?

```
                    Tool function defined
                              │
                              ▼
                    First param named "ctx"?
                     ┌────────┴────────┐
                     │                 │
                    Yes               No
                     │                 │
                     ▼                 ▼
            inject_run_context     inject_run_context
                  = True                = False
                     │                 │
                     ▼                 ▼
            Schema EXCLUDES ctx    Schema includes all
            (LLM only sees other   params (current
             params)                behavior)
                     │                 │
                     ▼                 ▼
            Agent must have       No deps required
            deps=... or raises
```

---

## When to use it

| Situation | Use DI? | Why |
|-----------|---------|-----|
| Simple tools (e.g. `add(a, b)`) | No | No external deps; DI adds nothing |
| Tools that call APIs or DBs | Yes | Test with mocks; switch backends per env/user |
| Multi-tenant SaaS | Yes | Each tenant gets its own DB, API keys, prefs |
| Production agents | Yes | Inject real backends; easy to swap for testing |
| CI / unit tests | Yes | Inject mocks; fast, deterministic tests |

**Rule of thumb:** If your tool talks to something outside the agent (DB, API, file system, user session), use DI.

---

## Use cases

### 1. Testing — fast, deterministic unit tests

**Problem:** Tools that call external APIs or DBs slow down tests and require real credentials.

**Solution:** Inject mock deps in tests. No network, no DB, no flakiness.

```python
@dataclass
class SupportDeps:
    ticket_db: TicketDB
    email_client: EmailClient

@tool
def create_ticket(ctx: RunContext[SupportDeps], title: str, body: str) -> str:
    """Create a support ticket and send confirmation email."""
    ticket = ctx.deps.ticket_db.create(title=title, body=body)
    ctx.deps.email_client.send(ticket.user_email, f"Ticket #{ticket.id} created")
    return f"Created ticket #{ticket.id}"

# Production
agent = SupportAgent(
    deps=SupportDeps(
        ticket_db=PostgresTicketDB(conn_string),
        email_client=SendGridClient(api_key),
    )
)

# Unit test — no Postgres, no SendGrid
class MockTicketDB:
    def create(self, title, body):
        return Ticket(id="T-001", user_email="test@example.com")

class MockEmailClient:
    def __init__(self):
        self.sent: list[tuple[str, str]] = []
    def send(self, to, body):
        self.sent.append((to, body))

email_client = MockEmailClient()
agent = SupportAgent(
    deps=SupportDeps(
        ticket_db=MockTicketDB(),
        email_client=email_client,
    )
)
result = agent._execute_tool("create_ticket", {"title": "Bug", "body": "..."})
assert "T-001" in result
assert len(email_client.sent) == 1
```

---

### 2. Multi-tenant — different data per user or tenant

**Problem:** One agent serves many users; each user has their own data, API keys, or settings.

**Solution:** Create one agent per request/session and inject user-specific deps.

```python
@dataclass
class TenantDeps:
    tenant_id: str
    db: Database           # Tenant's isolated DB or schema
    search_api_key: str    # Tenant's API key
    locale: str

@tool
def search_docs(ctx: RunContext[TenantDeps], query: str) -> str:
    """Search tenant's documentation."""
    client = SearchClient(api_key=ctx.deps.search_api_key)
    return client.search(
        query,
        filter={"tenant_id": ctx.deps.tenant_id},
        locale=ctx.deps.locale,
    )

# Per-request: create agent with tenant-specific deps
def handle_request(tenant_id: str, user_message: str) -> str:
    tenant = get_tenant(tenant_id)
    deps = TenantDeps(
        tenant_id=tenant_id,
        db=tenant.get_db(),
        search_api_key=tenant.search_api_key,
        locale=tenant.default_locale,
    )
    agent = DocsAgent(deps=deps)
    return agent.response(user_message).content
```

---

### 3. Personalization — user prefs, language, context

**Problem:** Tools should respect user language, timezone, or preferences without the LLM needing to pass them every time.

**Solution:** Inject user prefs into deps; tools read from `ctx.deps`.

```python
@dataclass
class UserDeps:
    user_id: str
    language: str          # "en", "es", "fr"
    timezone: str          # "America/New_York"
    preferences: dict[str, Any]  # dark_mode, compact_view, etc.

@tool
def get_schedule(ctx: RunContext[UserDeps], date: str) -> str:
    """Get user's schedule in their language and timezone."""
    events = fetch_events(ctx.deps.user_id, date, ctx.deps.timezone)
    return format_events(events, locale=ctx.deps.language)

@tool
def recommend_content(ctx: RunContext[UserDeps], topic: str) -> str:
    """Recommend content based on user preferences."""
    prefs = ctx.deps.preferences
    return content_engine.recommend(
        topic,
        language=ctx.deps.language,
        limit=prefs.get("recommendation_limit", 5),
    )
```

---

### 4. Environment switching — dev, staging, production

**Problem:** Use different backends in dev (local DB, mock APIs) vs production.

**Solution:** Inject deps based on environment; same agent code everywhere.

```python
import os

def make_deps() -> AppDeps:
    env = os.getenv("ENV", "development")
    if env == "production":
        return AppDeps(
            db=PostgresDB(os.getenv("DATABASE_URL")),
            cache=RedisCache(os.getenv("REDIS_URL")),
            search=ElasticsearchClient(os.getenv("ES_URL")),
        )
    elif env == "staging":
        return AppDeps(
            db=PostgresDB(os.getenv("STAGING_DATABASE_URL")),
            cache=RedisCache(os.getenv("STAGING_REDIS_URL")),
            search=ElasticsearchClient(os.getenv("STAGING_ES_URL")),
        )
    else:
        return AppDeps(
            db=SQLiteDB(":memory:"),
            cache=InMemoryCache(),
            search=MockSearchClient(),
        )

agent = MyAgent(deps=make_deps())
```

---

### 5. Feature flags and A/B testing

**Problem:** Roll out new backends gradually or test different implementations.

**Solution:** Inject the appropriate backend based on user segment or flag.

```python
@dataclass
class SearchDeps:
    search_client: SearchClient

def make_search_deps(user_id: str) -> SearchDeps:
    if feature_flags.is_enabled("new_search", user_id):
        return SearchDeps(search_client=NewSearchClient())
    return SearchDeps(search_client=LegacySearchClient())

agent = SearchAgent(deps=make_search_deps(user_id))
```

---

## Quick start

1. Define a deps dataclass.
2. Create tools with first param `ctx: RunContext[YourDeps]`.
3. Pass `deps=YourDeps(...)` to the agent.

```python
from dataclasses import dataclass
from syrin import Agent, RunContext, tool

@dataclass
class MyDeps:
    db: Database
    search_client: SearchClient
    user_id: str

@tool
def get_data(ctx: RunContext[MyDeps], query: str) -> str:
    """Fetch user-specific data."""
    return ctx.deps.db.query(user_id=ctx.deps.user_id, q=query)

@tool
def search(ctx: RunContext[MyDeps], query: str) -> str:
    """Search with user context."""
    return ctx.deps.search_client.search(query)

class PersonalAgent(Agent):
    tools = [get_data, search]

# Production
agent = PersonalAgent(deps=MyDeps(
    db=PostgresDB("postgresql://..."),
    search_client=BraveSearch(api_key="..."),
    user_id="alice",
))
result = agent.response("What meetings do I have today?")

# Testing — inject mocks
agent = PersonalAgent(deps=MyDeps(
    db=MockDB(data={"alice": [...]}),
    search_client=MockSearch(results=["result 1"]),
    user_id="alice",
))
result = agent.response("What meetings do I have today?")
assert "meeting" in result.content.lower()
```

---

## RunContext

When a tool declares a first parameter named `ctx` (typed as `RunContext[YourDeps]`), the agent injects a `RunContext` at execution time.

| Attribute | Type | Description |
|-----------|------|-------------|
| `deps` | YourDeps | The injected dependencies (Agent.deps). |
| `agent_name` | str | Current agent class name (e.g. `"Researcher"`). |
| `conversation_id` | str \| None | Current conversation ID for state isolation (per-user or per-session). |
| `budget_state` | BudgetState \| None | Current budget state (limit, remaining, spent, percent_used). |
| `retry_count` | int | Current retry attempt (for output validation). |

---

## Tool signature

Tools that need DI use `ctx: RunContext[YourDeps]` as the **first parameter** (after `self` for methods):

```python
@tool
def my_tool(ctx: RunContext[MyDeps], query: str) -> str:
    return ctx.deps.search_client.search(query)
```

The `ctx` parameter is excluded from the LLM tool schema — the model only sees `query`. The agent injects `ctx` at runtime.

---

## Agent without deps

Tools that **do not** declare `ctx` work as before — they receive only the arguments from the LLM:

```python
@tool
def search(query: str) -> str:
    """Search the web."""
    return f"Results for: {query}"

agent = Agent(model=..., tools=[search])  # No deps needed
```

---

## Error when tool expects ctx but agent has no deps

If a tool has `ctx: RunContext[MyDeps]` but the agent was created without `deps`, execution raises:

```
ToolExecutionError: Tool 'get_data' expects ctx: RunContext but Agent has no deps.
Pass deps=MyDeps(...) to Agent.
```

---

## See also

- [Agent API](../agent/README.md) — `deps` constructor param
- [Tools](../tools.md) — `@syrin.tool` decorator
- [Example: dependency_injection](../examples/15_advanced/dependency_injection.py)
