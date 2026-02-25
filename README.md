<p align="center">
  <img src="https://raw.githubusercontent.com/Syrin-Labs/cli/main/assets/syrin-logo-dark-bg.png" alt="Syrin" width="200">
</p>

<h1 align="center">Syrin</h1>

<p align="center">
  <b>The most developer-friendly Python library for AI agents</b>
</p>
<p align="center">
  <i>Budget control · Lifecycle hooks · Declarative thresholds · Type-safe APIs</i>
</p>

<p align="center">
  <a href="https://pypi.org/project/syrin/"><img src="https://img.shields.io/pypi/v/syrin.svg" alt="PyPI"></a>
  <a href="https://codecov.io/gh/syrin-labs/syrin-python"><img src="https://codecov.io/gh/syrin-labs/syrin-python/branch/main/graph/badge.svg" alt="Coverage"></a>
  <a href="https://github.com/syrin-labs/syrin-python/blob/main/LICENSE"><img src="https://img.shields.io/github/license/syrin-labs/syrin-python.svg" alt="License"></a>
  <a href="https://pypi.org/project/syrin/"><img src="https://img.shields.io/pypi/pyversions/syrin.svg" alt="Python"></a>
</p>

<p align="center">
  <a href="https://syrin.ai">Website</a> ·
  <a href="https://syrin.ai/docs">Documentation</a> ·
  <a href="https://discord.gg/p4jnKxYKpB">Discord</a> ·
  <a href="https://x.com/syrin_dev">Twitter (X)</a>
</p>

---

## The problem

**You must have felt this:**

1. **How am I gonna monitor the budget for this agent?** — No built-in cap, no “stop at $X,” no way to see spend until the bill arrives.
2. **What is actually happening in the flow?** — LLM calls, tool calls, retries—all invisible. You don’t know where time and tokens went.
3. **Where is the bug?** — Something failed or looped. Logs are scattered. No single place to see every step.
4. **Why do I have no control over my agent?** — Can’t act at 80% budget, can’t switch model when limits hit, can’t enforce guardrails in the pipeline.
5. **How do I know what each run cost?** — No cost or token count on the response. You’re guessing or building it yourself.

Syrin gives you budgeting, thresholds, hooks, observability, context, memory, guardrails, and checkpoints in one lightweight library—so you can answer these questions instead of wondering.

---

## Comparison: Syrin vs. LangChain, LangGraph, AutoGen

| Capability | Syrin | LangChain | LangGraph | AutoGen |
|------------|:-----:|:---------:|:---------:|:-------:|
| **Built-in budget per run** | ✅ First-class | ❌ Missing | ❌ Missing | ❌ Missing |
| **Budget thresholds → auto actions** (warn, switch model, stop) | ✅ Declarative | ❌ You build it | ❌ You build it | ❌ You build it |
| **Rate-limited budgets** (per hour/day/month) | ✅ Built-in | ❌ Missing | ❌ Missing | ❌ Missing |
| **72+ lifecycle hooks, single event path** | ✅ Every step visible | Callbacks + LangSmith | Graph nodes only | Limited |
| **First-party cost/tokens per response** | ✅ Every response | Add-on / LangSmith | Add-on | Manual |
| **Context window control + compaction** | ✅ Config + thresholds | DIY | DIY | DIY |
| **Memory (remember/recall/forget)** | ✅ Built-in, multiple backends | Patterns only | Patterns only | Limited |
| **Guardrails in pipeline** | ✅ Input/output, reports | DIY | DIY | DIY |
| **Checkpoints (save/restore state)** | ✅ Triggers + API | DIY | DIY | DIY |
| **StrEnum everywhere, mypy strict** | ✅ Type-safe | Partial | Partial | Partial |
| **Lightweight core** | ✅ Minimal deps | Heavy | Heavy | Moderate |

Use Syrin when you need **production-grade control and observability in one place**, without gluing together multiple products or building it yourself.

---

## Install and first run

```bash
pip install syrin
```

```python
import os
from syrin import Agent, Model, Budget, stop_on_exceeded

class Researcher(Agent):
    model = Model.OpenAI("gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"))
    budget = Budget(run=0.50, on_exceeded=stop_on_exceeded)

result = Researcher().response("Summarize quantum computing in 3 sentences")
print(result.content)
# Quantum computing uses qubits to represent multiple states simultaneously...
print(f"Cost: ${result.cost:.4f}  |  Budget used: ${result.budget_used:.4f}")
# Cost: $0.0012  |  Budget used: $0.0012
```

Pass your API key explicitly. The run is capped at $0.50; when the budget is exceeded, the agent stops.

---

## Basic agent (no budget)

**When to use:** Simple Q&A, demos, or when cost is not a concern yet.

```python
import os
from syrin import Agent, Model

class Greeter(Agent):
    model = Model.OpenAI("gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"))
    system_prompt = "You are a helpful assistant."

result = Greeter().response("Say hello in one sentence.")
print(result.content)
# Hello! How can I help you today?
print(result.cost, result.tokens.total_tokens)
# 0.0004 42
```

---

## Run budget with hard stop

**When to use:** You want “this single run must never exceed $X.” No thresholds—just stop when exceeded.

```python
import os
from syrin import Agent, Model, Budget, stop_on_exceeded

class SafeResearcher(Agent):
    model = Model.OpenAI("gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"))
    budget = Budget(run=0.50, on_exceeded=stop_on_exceeded)

result = SafeResearcher().response("Explain photosynthesis briefly.")
print(result.content)
# Photosynthesis is the process by which plants...
print(result.cost, result.budget_used)
# 0.0015 0.0015
# If the run had exceeded $0.50, it would stop and you'd get a clear error.
```

---

## Budget thresholds (warn, then switch model)

**When to use:** At 70% budget you want to log; at 90% you want to switch to a cheaper model; at 100% you stop. No custom glue—declare it.

```python
import os
from syrin import Agent, Model, Budget, BudgetThreshold

class AdaptiveResearcher(Agent):
    model = Model.OpenAI("gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"))
    budget = Budget(
        run=0.50,
        thresholds=[
            BudgetThreshold(at=70, action=lambda ctx: print(f"Budget at {ctx.percentage}%")),
            BudgetThreshold(at=90, action=lambda ctx: ctx.parent.switch_model(Model("openai/gpt-4o-mini"))),
        ],
    )

result = AdaptiveResearcher().response("Research AI frameworks in depth.")
# (If budget crosses 70% you'll see: Budget at 70%)
# (If it crosses 90%, the agent switches to the cheaper model automatically)
print(result.cost, result.budget_used)
# e.g. 0.0234 0.0234
```

---

## Real-time cost, tokens, and duration

**When to use:** You need to show or log what every call cost and how long it took. Other libs often don’t attach this to the response.

```python
result = agent.response("Your prompt here")
print(f"Cost: ${result.cost:.4f}")
# Cost: $0.0234
print(f"Tokens: {result.tokens.total_tokens} (in: {result.tokens.input_tokens}, out: {result.tokens.output_tokens})")
# Tokens: 1247 (in: 890, out: 357)
print(f"Budget used: ${result.budget_used:.4f}")
# Budget used: $0.0234
print(f"Duration: {result.duration}s")
# Duration: 1.23
```

---

## Hooks—observe every step

**When to use:** You want one event path for logging, metrics, or debugging. No hidden prompt rewrites; you see every LLM request and every threshold.

```python
import os
from syrin import Agent, Model, Budget
from syrin.enums import Hook

class ObservableAgent(Agent):
    model = Model.OpenAI("gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"))
    system_prompt = "You are a research assistant."
    budget = Budget(run=0.50)

agent = ObservableAgent()
agent.events.on(Hook.LLM_REQUEST_START, lambda ctx: print(f"LLM call #{ctx.iteration}"))
agent.events.on(Hook.BUDGET_THRESHOLD, lambda ctx: print(f"Budget at {ctx.threshold_percent}%"))

result = agent.response("Research AI frameworks.")
# LLM call #1
# (if threshold hits: Budget at 70%)
# ...
```

**In plain English:** Every time the agent calls the LLM, your handler runs. Every time a budget threshold fires, your handler runs. Same for tool calls, guardrails, memory, and 60+ other hooks. One place to plug in observability.

---

## Persistent memory (remember, recall, forget)

**When to use:** The agent should remember facts across turns (e.g. user name, preferences) and you want semantic recall and optional forgetting.

```python
import os
from syrin import Agent, Model, Memory
from syrin.enums import MemoryType

agent = Agent(
    model=Model.OpenAI("gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY")),
    memory=Memory(types=[MemoryType.CORE, MemoryType.EPISODIC]),
)

agent.remember("The user's name is Alice and she prefers short answers.", memory_type=MemoryType.CORE)
# Stored in memory.

entries = agent.recall("user name")
# Semantic search over memories; returns list of MemoryEntry.
# e.g. [MemoryEntry(id='...', content="The user's name is Alice...", ...)]

agent.forget(memory_id=entries[0].id)
# That memory is removed.
```

**In plain English:** You store facts with a type (e.g. CORE for long-term facts). Recall runs a search over stored content. Forget removes by ID. You get this without wiring a vector DB yourself; multiple backends (in-memory, SQLite, Chroma, etc.) are supported.

---

## Guardrails (input/output validation)

**When to use:** You want to block or warn on length, banned words, or custom rules before/after the LLM. Results show up in the response report.

```python
import os
from syrin import Agent, Model, GuardrailChain
from syrin.guardrails import LengthGuardrail, ContentFilter

class SafeAgent(Agent):
    model = Model.OpenAI("gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"))
    guardrails = GuardrailChain([
        LengthGuardrail(max_length=4000),
        ContentFilter(blocked_words=["spam", "malicious"]),
    ])

result = SafeAgent().response("Hello!")
print(result.report.guardrail.passed)
# True
print(result.report.guardrail.blocked)
# False
# If input or output had been blocked, passed would be False and blocked True.
```

**In plain English:** Guardrails run in the pipeline. Input is checked before the LLM; output after. Every response carries a guardrail report so you can alert or log.

---

## Context window control

**When to use:** You need a cap on context size (e.g. 8K tokens) so long conversations don’t blow the model’s window. Syrin lets you set `max_tokens` and optional compaction.

```python
import os
from syrin import Agent, Model
from syrin.context import Context

agent = Agent(
    model=Model.OpenAI("gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY")),
    context=Context(max_tokens=8000),
)

result = agent.response("Summarize the previous conversation.")
# Context is capped at 8000 tokens; you can add thresholds and compactors for auto-compaction.
if result.context_stats:
    print(result.context_stats.total_tokens, result.context_stats.max_tokens)
# e.g. 1200 8000
```

---

## Checkpoints (save and restore state)

**When to use:** Long or expensive runs where you want to save state (e.g. after each step or on error) and resume later.

```python
import os
from syrin import Agent, Model, CheckpointConfig, CheckpointTrigger

agent = Agent(
    model=Model.OpenAI("gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY")),
    checkpoint=CheckpointConfig(storage="memory", trigger=CheckpointTrigger.STEP),
)

agent.response("First step.")
cid = agent.save_checkpoint(reason="after_step_1")
# Returns checkpoint ID.

# Later: restore and continue.
agent.load_checkpoint(cid)
agent.response("Continue from where we left off.")
```

**In plain English:** You enable checkpoints with a trigger (step, tool, error, budget). You can save manually or let the library auto-save. Load by ID to restore messages and state.

---

## Tools (agent can call functions)

**When to use:** The agent should use tools (search, calculator, API calls). You decorate functions with `@tool` and pass them to the agent.

```python
import os
from syrin import Agent, Model, tool, Budget

@tool
def search_web(query: str) -> str:
    """Search the web for information."""
    return f"Results for: {query}"  # Replace with real search in production

class Assistant(Agent):
    model = Model.OpenAI("gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"))
    tools = [search_web]
    budget = Budget(run=0.25)

result = Assistant().response("What is the weather in Tokyo?")
print(result.content)
# e.g. Based on the search results, the weather in Tokyo is...
print(result.cost, result.tool_calls)
# 0.008 [] or list of tool calls if any were made
```

---

## Streaming (async, token-by-token)

**When to use:** You want to stream the reply to the user and show running cost (e.g. in a UI).

```python
import asyncio
import os
from syrin import Agent, Model

class Writer(Agent):
    model = Model.OpenAI("gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"))

async def main():
    agent = Writer()
    async for chunk in agent.astream("Write a short story in 2 sentences."):
        print(chunk.text, end="")
        print(f" [${chunk.cost_so_far:.4f}]", end="\r")
    # Once upon a time... [$0.0023]

asyncio.run(main())
```

---

## Per-response reports (guardrail, tokens, budget, memory, checkpoints)

**When to use:** You need a single object that describes what happened on that run: guardrail result, token usage, budget status, memory ops, checkpoints. No scraping logs.

```python
result = agent.response("Your prompt")

# Guardrail: did input/output pass? Was it blocked?
result.report.guardrail.passed   # True/False
result.report.guardrail.blocked   # True if blocked

# Tokens and cost for this run
result.report.tokens.input_tokens
result.report.tokens.output_tokens
result.report.tokens.total_tokens
result.report.tokens.cost_usd

# Budget status (also on result.budget)
result.budget.remaining
result.budget.used
result.budget.cost

# Memory: how many stores/recalls/forgets this run
result.report.memory.stores
result.report.memory.recalls
result.report.memory.forgets

# Context: compressions this run; optional context_stats has compacted/utilization
result.report.context.compressions
result.context_stats.compacted  # if context_stats is set

# Checkpoints: saves/loads this run
result.report.checkpoints.saves
result.report.checkpoints.loads

# Rate limits
result.report.ratelimits.checks
result.report.ratelimits.exceeded
```

**In plain English:** Every response is a single object. You get cost, tokens, and a full report (guardrail, memory, context, checkpoints, rate limits) without wiring another system.

---

## Why Syrin: one stack, full control

- **Budget and thresholds** — Per-run and per-period limits with declarative actions. Other libs don’t ship this; you’d build it yourself.
- **Hooks and observability** — One event path, 72+ hooks. You see every LLM call, tool call, threshold, and guardrail. No black box.
- **Context, memory, guardrails, checkpoints** — All in the same library. Control context size, remember/recall/forget, validate input/output, save/restore state. No scattered integrations.
- **Reports per response** — Cost, tokens, budget, guardrail, memory, checkpoints on every `Response`. First-party, no extra product.
- **Lightweight** — Minimal dependencies. One coherent API instead of a patchwork of callbacks and third-party dashboards.
- **Type-safe** — StrEnum everywhere, mypy strict. Fewer runtime surprises and better IDE support.

For deep dives: [Budget & thresholds](docs/budget-control.md), [Memory](docs/memory.md), [Observability](docs/observability.md), [Guardrails](docs/guardrails.md), [Context](docs/context.md), [Checkpoints](docs/checkpoint.md), [Architecture](docs/ARCHITECTURE.md). Full [docs index](docs/README.md).

---

## Community — Connect here

- 🌐 [Website](https://syrin.ai) — syrin.ai
- 💬 [Discord](https://discord.gg/p4jnKxYKpB) — Chat and support
- 🐦 [Twitter (X)](https://x.com/syrin_dev) — Updates and tips
- 📧 [Email](mailto:hello@syrin.ai) — Questions and feedback
- 🐛 [Issues](https://github.com/syrin-labs/syrin-python/issues) — Bug reports
- 💡 [Discussions](https://github.com/syrin-labs/syrin-python/discussions) — Feature requests

---

## Contributing

We welcome contributions. See [CONTRIBUTING.md](CONTRIBUTING.md) for setup and guidelines.

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

<p align="center">
  <b>Declare agents. Control costs. See every step. Ship to production.</b>
</p>

<p align="center">
  <a href="https://syrin.ai">syrin.ai</a>
</p>
