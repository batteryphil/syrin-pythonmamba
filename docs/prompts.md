# Dynamic System Prompts

Syrin supports three ways to define system prompts:

1. **Static string** — `system_prompt="You are helpful."`
2. **@prompt with prompt_vars** — Parameterized prompts with runtime injection
3. **@system_prompt in class** — Encapsulate prompt inside the agent class

---

## Static Prompt

```python
from syrin import Agent, Model

agent = Agent(
    model=Model.OpenAI("gpt-4o-mini"),
    system_prompt="You are a helpful assistant.",
)
```

---

## @prompt with prompt_vars

Use `@prompt` for parameterized prompts. Inject variables via `prompt_vars` (class, instance, or per-call).

```python
from syrin import Agent, Model, prompt

@prompt
def persona_prompt(user_name: str, tone: str = "professional") -> str:
    return f"You assist {user_name or 'user'}. Be {tone}."

agent = Agent(
    model=Model.OpenAI("gpt-4o-mini"),
    system_prompt=persona_prompt,
    prompt_vars={"user_name": "Alice", "tone": "friendly"},
)

agent.response("Hi")  # Resolves persona_prompt at run time
```

**Merge order:** class `prompt_vars` → instance `prompt_vars` → per-call `prompt_vars` (each layer overrides previous).

**Per-call override:**

```python
agent.response("Hi", prompt_vars={"user_name": "Bob"})
```

**Built-ins:** When `inject_builtins=True` (default), `date`, `agent_id`, and `conversation_id` are added to `prompt_vars` unless you already provide them.

---

## @system_prompt in Class

Encapsulate the system prompt inside the agent class. One per agent.

```python
from syrin import Agent, Model, system_prompt

class PersonaAgent(Agent):
    model = Model.OpenAI("gpt-4o-mini")

    @system_prompt
    def my_prompt(self, user_name: str = "") -> str:
        return f"You assist {user_name or 'user'}."

agent = PersonaAgent(prompt_vars={"user_name": "Carol"})
agent.response("Hi")
```

Supported method signatures:

- `(self) -> str` — no params
- `(self, ctx: PromptContext) -> str` — full context (agent, memory, budget, date)
- `(self, **kwargs) -> str` — receives merged prompt_vars

---

## Callable with PromptContext

For full control (DB, memory, custom logic), use a callable that receives `PromptContext`:

```python
from syrin import Agent, Model
from syrin.prompt import PromptContext

def build_prompt(ctx: PromptContext) -> str:
    user = db.get_user(ctx.agent_id)
    memories = ctx.memory.recall("preferences", limit=5) if ctx.memory else []
    return f"You assist {user.name}. Memories: {memories}"

agent = Agent(model=Model.OpenAI("gpt-4o-mini"), system_prompt=build_prompt)
```

`PromptContext` fields: `agent`, `agent_id`, `conversation_id`, `memory`, `budget_state`, `date`, `builtins`.

---

## Introspection

- **effective_prompt_vars(call_vars=None)** — merged prompt_vars (class + instance + call + builtins)
- **get_prompt_builtins()** — `{date, agent_id, conversation_id}` that would be injected
- **inject_builtins=False** — disable built-in injection

---

## Hooks

- `Hook.SYSTEM_PROMPT_BEFORE_RESOLVE` — ctx: prompt_vars, source
- `Hook.SYSTEM_PROMPT_AFTER_RESOLVE` — ctx: resolved (string)

Override `_resolve_system_prompt(prompt_vars, ctx) -> str` for custom resolution.
