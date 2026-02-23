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
  <a href="https://github.com/syrin-labs/syrin-python/actions"><img src="https://github.com/syrin-labs/syrin-python/workflows/Tests/badge.svg" alt="Tests"></a>
  <a href="https://codecov.io/gh/syrin-labs/syrin-python"><img src="https://codecov.io/gh/syrin-labs/syrin-python/branch/main/graph/badge.svg" alt="Coverage"></a>
  <a href="https://github.com/syrin-labs/syrin-python/blob/main/LICENSE"><img src="https://img.shields.io/github/license/syrin-labs/syrin-python.svg" alt="License"></a>
  <a href="https://pypi.org/project/syrin/"><img src="https://img.shields.io/pypi/pyversions/syrin.svg" alt="Python"></a>
</p>

<p align="center">
  <a href="https://syrin.ai">Website</a> ·
  <a href="https://syrin.ai/docs">Documentation</a> ·
  <a href="https://discord.gg/syrin">Discord</a> ·
  <a href="https://twitter.com/syrin_ai">Twitter</a>
</p>

---

> **TL;DR** — A Python library for building AI agents with built-in observability & cost control. Set per-run budgets, get real-time cost tracking, and automatic limits. You can observe every step with lifecycle hooks and get type-safe, IDE-friendly APIs. `pip install syrin`

---

## What is an AI agent?

An AI agent is a program that uses an LLM (like GPT-4) to reason and can call tools (search, calculate, look up data) in a loop until it completes a task. The catch: without limits, agents can make thousands of API calls and burn through your budget.

---

## Try it now

```bash
pip install syrin
```

```python
from syrin import Agent, Model, Budget, OnExceeded

class Researcher(Agent):
    model = Model.OpenAI("gpt-4o-mini")
    budget = Budget(run=0.50, on_exceeded=OnExceeded.STOP)

result = Researcher().response("Summarize quantum computing in 3 sentences")
print(result.content)
print(f"Cost: ${result.cost:.4f}  |  Budget used: ${result.budget_used:.4f}")
```

Requires `OPENAI_API_KEY`. Run it.

---

## The problem

I spent **$1,410** when an AI agent got stuck in a recursive loop.

47,000 API calls in 6 hours. No circuit breaker. No budget limit. No way to stop it.

Current frameworks give you nice abstractions and easy demos—but they don't give you **per-agent budgets**, **real-time cost tracking**, or **automatic circuit breakers**.

Budget overruns on AI agent deployments are common. I was one of them.

---

## The solution

**Syrin** gives you **budget control**, **lifecycle visibility**, **declarative thresholds**, and **type-safe APIs** — designed to stay simple as you move from prototype to production.

```python
from syrin import Agent, Model, Budget, BudgetThreshold

class Researcher(Agent):
    model = Model.OpenAI("gpt-4o-mini")
    budget = Budget(
        run=0.50,
        thresholds=[
            BudgetThreshold(at=70, action=lambda ctx: print(f"Budget at {ctx.percentage}%")),
            BudgetThreshold(at=90, action=lambda ctx: ctx.parent.switch_model(Model("openai/gpt-4o-mini"))),
        ]
    )

result = Researcher().response("Research AI frameworks")
print(result.cost)        # ~$0.02
print(result.budget_used) # cost in USD
```

**That's it.** Budget control with threshold actions (each `action` is a function that receives `ctx` with `percentage`, `parent` agent, etc.).

---

## Why Syrin?

**Budget** · **Lifecycle** · **Thresholds** · **Developer experience.** Four pillars.

### 1. Budget-First Design

Every agent declares its budget upfront. Automatic actions when thresholds hit.

```python
from syrin import Agent, Budget, BudgetThreshold, Model, RateLimit, OnExceeded
from syrin.exceptions import BudgetExceededError

def stop_on_limit(ctx):
    raise BudgetExceededError("Budget limit reached")

budget = Budget(
    run=0.50,
    per=RateLimit(hour=10.00, day=100.00),
    on_exceeded=OnExceeded.ERROR,
    thresholds=[
        BudgetThreshold(at=70, action=lambda ctx: print(f"Budget at {ctx.percentage}%")),
        BudgetThreshold(at=90, action=lambda ctx: ctx.parent.switch_model(Model("openai/gpt-4o-mini"))),
        BudgetThreshold(at=100, action=stop_on_limit),
    ]
)
# Each action is a callable receiving ThresholdContext (ctx.percentage, ctx.parent, etc.)
```

### 2. Real-Time Cost Tracking

See exactly what every operation costs:

```python
result = agent.response("Your prompt here")
print(f"Cost: ${result.cost:.4f}")        # $0.0234
print(f"Tokens: {result.tokens.total_tokens}")    # 1247
print(f"Budget used: ${result.budget_used:.4f}")  # cost in USD
print(f"Duration: {result.duration}s")      # 1.23s
```

### 3. Persistent Memory

Four memory types with Ebbinghaus-inspired decay curves:

```python
from syrin import Agent, Memory
from syrin.enums import MemoryType

agent = Agent(
    model=Model.OpenAI("gpt-4o-mini"),
    memory=Memory(types=[MemoryType.CORE, MemoryType.EPISODIC])
)

agent.remember("User's name is John", memory_type=MemoryType.CORE)
memories = agent.recall("name")  # Semantic search
agent.forget(memory_id="...")    # Selective forgetting
```

### 4. Guardrails

Input/output validation with hooks and reporting:

```python
from syrin import Agent, GuardrailChain
from syrin.guardrails import LengthGuardrail, BlockedWordsGuardrail

class SafeAgent(Agent):
    guardrails = GuardrailChain([
        LengthGuardrail(max_length=4000),
        BlockedWordsGuardrail(blocked=["spam", "malicious"])
    ])

result = SafeAgent().response("Hello!")
print(result.report.guardrail.passed)    # True/False
print(result.report.guardrail.blocked)   # True/False
```

### 5. Comprehensive Reporting

Every response includes detailed reports:

```python
result = agent.response("Your prompt")

# Access any report
result.report.guardrail    # GuardrailReport
result.report.tokens       # TokenReport (input, output, total, cost)
result.report.budget       # BudgetStatus (remaining, used, total)
result.report.memory       # MemoryReport (stores, recalls, forgets)
result.report.context      # ContextReport (tokens, compressions)
result.report.output       # OutputReport (validation results)
result.report.ratelimits   # RateLimitReport
result.report.checkpoints  # CheckpointReport
```

### 6. Extensible by Design

Add any LLM in 30 lines:

```python
from syrin import Model
from syrin.providers.base import ProviderResponse

class DeepSeekModel(Model):
    provider = "deepseek"
    
    def complete(self, messages, **kwargs):
        # Your implementation
        return ProviderResponse(...)
    
    def get_pricing(self):
        return ModelPricing(input=0.14, output=0.28)

model = Model("deepseek/deepseek-chat")
```

### 7. Lifecycle & Events (72+ Hooks)

Every step is observable. No black boxes, no hidden prompt rewrites:

```python
from syrin import Agent, Model, Budget
from syrin.enums import Hook

class Researcher(Agent):
    model = Model.OpenAI("gpt-4o-mini")
    system_prompt = "You are a research assistant."
    budget = Budget(run=0.50)

agent = Researcher()

agent.events.on(Hook.LLM_REQUEST_START, lambda ctx: print(f"Call #{ctx.iteration}"))
agent.events.on(Hook.BUDGET_THRESHOLD, lambda ctx: print(f"Budget {ctx.threshold_percent}% used"))

result = agent.response("Research AI frameworks")
```

Other hooks: `TOOL_CALL_START`, `GUARDRAIL_BLOCKED`, `CONTEXT_COMPRESSED`, `MEMORY_STORE`, and 60+ more. See [docs](https://syrin.ai/docs) for the full list.

---

## Quick Start

### Installation

```bash
pip install syrin
```

### Basic Usage

```python
from syrin import Agent, Model

class Greeter(Agent):
    model = Model.OpenAI("gpt-4o-mini")
    system_prompt = "You are a helpful assistant."

result = Greeter().response("Say hello!")
print(result.content)  # "Hello!"
print(result.cost)     # ~$0.0005
```

### With Budget

```python
from syrin import Agent, Model, Budget, OnExceeded

class Researcher(Agent):
    model = Model.OpenAI("gpt-4o-mini")
    budget = Budget(run=0.50, on_exceeded=OnExceeded.ERROR)

result = Researcher().response("Research quantum computing")
print(result.cost)       # ~$0.04
print(result.budget_used)  # cost in USD
```

### With Tools

```python
from syrin import Agent, Model, tool, Budget

@tool
def search_web(query: str) -> str:
    """Search the web for information."""
    return requests.get(f"https://api.search.com?q={query}").text

class Assistant(Agent):
    model = Model.OpenAI("gpt-4o-mini")
    tools = [search_web]
    budget = Budget(run=0.25)

result = Assistant().response("What is the weather in Tokyo?")
```

### Streaming

```python
for chunk in agent.astream("Write a story"):
    print(chunk.text, end="")
    print(f" [${chunk.cost_so_far:.4f}]", end="\r")
```

---

## Features

### Core
- **Budget & Thresholds** - Per-run, hourly, daily budgets; declare actions at 70%, 90%, 100%
- **Cost Tracking** - Real-time cost and token tracking on every operation
- **Model Switching** - Automatic fallback to cheaper models on budget thresholds
- **Extensible Models** - Add any LLM provider in 30 lines
- **Type Safety** - Full mypy strict compliance, StrEnum everywhere

### Memory
- **4 Memory Types** - Core, Episodic, Semantic, Procedural
- **Forgetting Curves** - Ebbinghaus-inspired decay
- **Budget-Aware** - Memory operations adapt to remaining budget
- **Multiple Backends** - In-memory, SQLite, Qdrant, Chroma, Redis, PostgreSQL

### Guardrails
- **Input/Output Validation** - Block harmful content, enforce length limits
- **Built-in Guardrails** - PII detection, content filtering, budget limits
- **Custom Guardrails** - Extend with your own logic
- **Hooks** - GUARDRAIL_INPUT, GUARDRAIL_OUTPUT, GUARDRAIL_BLOCKED

### Lifecycle & Observability
- **72+ Hooks** - Every step observable (LLM, tools, budget, memory, guardrails)
- **Tracing** - Built-in span-based tracing with OTLP export
- **Audit Logging** - Immutable logs for compliance (SOC 2, GDPR)
- **Reports** - Every response includes detailed operation reports

### Context Management
- **Auto-Compaction** - Smart context compression at 75%
- **Token Counting** - Accurate tiktoken-based counting
- **Threshold Actions** - Warn, summarize, compress, or stop

### Multi-Agent
- **Sequential** - Agent A → Agent B → Agent C
- **Parallel** - Run multiple agents simultaneously
- **Router** - Intelligent handoffs between specialists

---

## Syrin vs. LangChain, LangGraph, AutoGen

Syrin is built for developers who want **simplicity**, **visibility**, and **production control** — without the problems reported across existing frameworks.

### What Syrin has that others don't

| Feature | Syrin | LangChain | LangGraph | AutoGen |
|---------|:-----:|:---------:|:---------:|:-------:|
| **Built-in budget per run** | ✅ | ❌ | ❌ | ❌ |
| **Budget thresholds → auto actions** (warn, switch model, stop) | ✅ | ❌ | ❌ | ❌ |
| **Rate-limited budgets** (per hour/day/month) | ✅ | ❌ | ❌ | ❌ |
| **72+ lifecycle hooks** (every step observable) | ✅ | Callbacks + LangSmith | Graph nodes | Limited |
| **StrEnum everywhere** (no magic strings) | ✅ | Partial | Partial | Partial |
| **mypy strict** (full type safety) | ✅ | Partial | Partial | Partial |
| **TOON tool schemas** (compact format, ~40% fewer chars than JSON) | ✅ | JSON | JSON | JSON |
| **Agent inheritance** (Python class inheritance) | ✅ | Composition-based | Graph-based | Composition-based |
| **Per-response reports** (tokens, cost, budget, guardrails) | ✅ | LangSmith / add-on | Add-on | Manual |
| **Extensible Model** (add any LLM in ~30 lines) | ✅ | Plugins | Plugins | Complex |
| **First-party cost/budget** (no separate service) | ✅ | ❌ | ❌ | ❌ |

### Reported problems ([from research](plan/RESEARCH.md))

LangChain, LangGraph, CrewAI, and AutoGen each have strengths (LangChain's ecosystem, LangGraph's graph model). The issues below are commonly reported by developers:

| Framework | Reported issues |
|-----------|-----------------|
| **LangChain** | Over-abstraction, dependency bloat, breaking changes, inconsistent docs, debugging difficulty, hidden prompt rewrites, token/cost metadata not built-in |
| **LangGraph** | Experimental features, sparse docs, steep DAG learning curve for simple pipelines |
| **CrewAI** | Manager-worker architecture fails in practice, telemetry/privacy concerns (GDPR), high setup complexity |
| **AutoGen** | Complete rewrite v0.2→v0.4 broke existing code; now in maintenance mode (Microsoft Agent Framework) |

### When to use each

- **LangChain** — 300+ integrations, complex RAG, LangSmith observability. Use it for that.
- **CrewAI** — Quick role-based multi-agent prototyping. Use it for that.
- **Syrin** — Production deployments where you need first-party budget control, lifecycle visibility, and type-safe APIs.

---

## Built for developers

- **72+ lifecycle hooks** — Observe every step: `agent.events.on(Hook.LLM_REQUEST_START, handler)`, `Hook.BUDGET_THRESHOLD`, `Hook.TOOL_CALL_START`, etc. No black boxes.
- **Declarative thresholds** — Define actions at 70%, 90%, 100% budget. Warn, switch model, or stop. No custom glue code.
- **StrEnum everywhere** — `OnExceeded.ERROR`, `MemoryType.CORE` — no magic strings, full IDE autocomplete.
- **TOON schemas** — Compact tool format (~40% fewer characters than JSON; see [examples/core/toon_format.py](examples/core/toon_format.py)).
- **mypy strict** — Full type safety across the API.
- **Extensible Model** — Add any LLM in ~30 lines by inheriting `syrin.Model`.

---

## Research-backed

Built on insights from 50+ academic papers:

- **A-MEM** (NeurIPS 2025): Agentic self-organizing memory
- **MIRIX** (2025): 6 memory types, 85.4% accuracy
- **Memory-R1**: RL-trained memory management
- **Google BATS**: Budget-aware tool use

---

## Community

- 💬 [Discord](https://discord.gg/syrin) - Chat with the community
- 🐦 [Twitter](https://twitter.com/syrin_ai) - Updates and tips
- 📧 [Email](mailto:hello@syrin.ai) - Questions and feedback
- 🐛 [Issues](https://github.com/syrin-labs/syrin-python/issues) - Bug reports
- 💡 [Discussions](https://github.com/syrin-labs/syrin-python/discussions) - Feature requests

---

## Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

## License

MIT License - see [LICENSE](LICENSE) for details.

---

<p align="center">
  <b>Declare agents. Control costs. See every step. Ship to production.</b>
</p>

<p align="center">
  <a href="https://syrin.ai">syrin.ai</a>
</p>
