<p align="center">
  <img src="https://raw.githubusercontent.com/Syrin-Labs/cli/main/assets/syrin-logo-dark-bg.png" alt="Syrin" width="180">
</p>

<h2 align="center">The Python Library for Production AI Agents</h2>

<p align="center">
  Budget control · Persistent memory · Multi-agent orchestration · Built-in observability
</p>

<p align="center">
  <a href="https://pypi.org/project/syrin/"><img src="https://img.shields.io/pypi/v/syrin.svg" alt="PyPI"></a>
  <a href="https://pypi.org/project/syrin/"><img src="https://img.shields.io/pypi/pyversions/syrin.svg" alt="Python"></a>
  <a href="https://github.com/syrin-labs/syrin-python/blob/main/LICENSE"><img src="https://img.shields.io/github/license/syrin-labs/syrin-python.svg" alt="License"></a>
  <a href="https://github.com/syrin-labs/syrin-python/stargazers"><img src="https://img.shields.io/github/stars/syrin-labs/syrin-python.svg?style=social" alt="Stars"></a>
  <a href="https://discord.gg/BJUdNBh4TC"><img src="https://img.shields.io/badge/Discord-Join%20Community-5865F2?logo=discord&logoColor=white" alt="Discord"></a>
</p>

<p align="center">
  <a href="https://syrin.ai">Website</a> ·
  <a href="https://docs.syrin.dev/agent-kit/">Docs</a> ·
  <a href="https://discord.gg/BJUdNBh4TC">Discord</a> ·
  <a href="https://reddit.com/r/syrin_ai">Reddit</a> ·
  <a href="https://www.youtube.com/@syrin_dev">YouTube</a>
</p>

---

## The Problem with AI Agents in Production

Most AI agent frameworks hand you primitives and wish you luck. You prototype in a weekend, then spend weeks bolting on cost tracking, memory, observability, and guardrails before anything goes near production.

Syrin is built differently. Budget control, memory, observability, multi-agent orchestration, and safety guardrails are **first-class features** — not afterthoughts.

---

## 30-Second Quickstart

```python
from syrin import Agent, Budget, Model
from syrin.enums import ExceedPolicy

class Analyst(Agent):
    model  = Model.OpenAI("gpt-4o-mini", api_key="...")
    budget = Budget(max_cost=0.10, exceed_policy=ExceedPolicy.STOP)
    system_prompt = "You are a precise financial analyst."

result = Analyst().run("Summarise Q3 revenue trends from the attached report")

print(result.content)
print(f"Cost: ${result.cost:.6f}  |  Tokens: {result.tokens}  |  Remaining: ${result.budget_remaining:.4f}")
```

That's it. The agent hard-stops at $0.10. No surprise invoices, no extra code.

---

## What You Can Build

| Use Case | Key Capabilities |
|---|---|
| **Research pipelines** | Parallel agents gather sources; consensus topology validates findings |
| **Customer support bots** | Session memory, PII guardrails, handoff to human when confidence is low |
| **Document intelligence** | RAG over PDFs/repos, structured output, budget per document |
| **Financial analysis** | Hard cost caps, audit logs, type-safe structured results |
| **Voice AI assistants** | ElevenLabs/Deepgram integration, memory across calls |
| **Compliance review** | PII scanning, output validation, full provenance records |
| **Code review agents** | Knowledge base over your codebase, tool use, structured findings |
| **Autonomous schedulers** | Cron/webhook/queue triggers, checkpoints for crash recovery |

---

## What Makes Syrin Different

### Budget Control — The Industry Gap

Every other library treats cost as a monitoring concern. Syrin treats it as a **runtime constraint**. Agents check their budget before every LLM call and enforce limits with the policy you choose.

```python
from syrin import Agent, Budget, Model, RateLimit
from syrin.enums import ExceedPolicy
from syrin.budget import BudgetThreshold

class ProductionAgent(Agent):
    model  = Model.OpenAI("gpt-4o", api_key="...")
    budget = Budget(
        max_cost=1.00,                        # Hard cap per run
        reserve=0.10,                         # Hold back for final reply
        exceed_policy=ExceedPolicy.STOP,      # STOP | WARN | IGNORE | SWITCH
        rate_limits=RateLimit(
            hour=10.00,                       # $10/hour across all runs
            day=100.00,                       # $100/day
            month=2000.00,                    # $2,000/month
        ),
        thresholds=[
            BudgetThreshold(at=80, action=lambda ctx: alert_ops_team(ctx)),
        ],
    )
```

Pre-call estimation, post-call actuals, threshold callbacks, rate-window enforcement — all declarative, zero boilerplate.

---

### Multi-Agent Orchestration — 5 Topologies

```python
from syrin import Agent, Budget, Model
from syrin.swarm import Swarm, SwarmConfig, BudgetPool
from syrin.enums import SwarmTopology

pool = BudgetPool(total=5.00)  # $5 shared; no agent can exceed its slice

class Researcher(Agent):
    model = Model.OpenAI("gpt-4o", api_key="...")

class FactChecker(Agent):
    model = Model.Anthropic("claude-sonnet-4-5", api_key="...")

class Writer(Agent):
    model = Model.OpenAI("gpt-4o-mini", api_key="...")

swarm = Swarm(
    agents=[Researcher, FactChecker, Writer],
    config=SwarmConfig(
        topology=SwarmTopology.ORCHESTRATOR,  # LLM routes work dynamically
        budget_pool=pool,
    ),
)

result = swarm.run("Research and write a report on battery technology trends")
print(result.content)
print(result.cost_breakdown)   # Per-agent cost breakdown
print(result.budget_report)    # Pool utilisation
```

**Five topologies — one `Swarm` class:**

| Topology | What it does |
|---|---|
| `ORCHESTRATOR` | First agent routes tasks to the rest dynamically |
| `PARALLEL` | All agents run concurrently; results merged |
| `CONSENSUS` | Multiple agents vote; winner selected by strategy |
| `REFLECTION` | Producer–critic loop until quality threshold is met |
| `WORKFLOW` | Sequential, parallel, branch, and dynamic fan-out steps |

---

### Persistent Memory — 4 Types, Extensible Backends

```python
from syrin import Agent, Model
from syrin.enums import MemoryType

agent = Agent(model=Model.OpenAI("gpt-4o-mini", api_key="..."))

# Store — persisted across sessions
agent.remember("User is a TypeScript engineer at a fintech startup", memory_type=MemoryType.CORE)
agent.remember("Previous session: reviewed authentication architecture", memory_type=MemoryType.EPISODIC)

# Recall — semantic search over stored memories
memories = agent.recall("user background", limit=5)

# Forget — when facts are outdated
agent.forget("previous role title")
```

| Memory Type | Stores |
|---|---|
| `CORE` | Long-term facts — user profile, preferences, domain knowledge |
| `EPISODIC` | Conversation history and past events |
| `SEMANTIC` | Embedding-indexed knowledge for RAG |
| `PROCEDURAL` | Skills, workflows, how-to instructions |

**Backends:** SQLite (zero config), PostgreSQL, Redis, Qdrant, ChromaDB — swap with one line.

---

### Observability — 72+ Lifecycle Hooks

Every LLM call, tool invocation, budget event, memory operation, and handoff emits a typed hook. Attach handlers at the agent level — no patching, no monkey-patching.

```python
from syrin.enums import Hook

agent.events.on(Hook.AGENT_RUN_END,      lambda ctx: metrics.record(ctx.cost, ctx.tokens))
agent.events.on(Hook.BUDGET_THRESHOLD,   lambda ctx: pagerduty.alert(f"Budget at {ctx.percentage}%"))
agent.events.on(Hook.TOOL_CALL_END,      lambda ctx: logger.info(f"Tool {ctx.name} → {ctx.duration_ms}ms"))
agent.events.on(Hook.MEMORY_RECALL,      lambda ctx: trace.span("recall", memories=ctx.count))
```

Or enable full tracing from the CLI — no code changes:

```bash
python my_agent.py --trace
```

---

### Live Terminal Debugger — Pry

A Rich-based TUI that lets you step through agent execution, inspect state, and set breakpoints — directly in your terminal.

[![Syrin Pry Debugger](https://img.youtube.com/vi/Wofz35A5a60/0.jpg)](https://youtu.be/Wofz35A5a60)

```python
from syrin.debug import Pry

pry = Pry()
pry.attach(agent)
agent.run("Analyse this dataset")
# [e] events  [t] tools  [m] memory  [g] guardrails  [p] pause  [n] step  [q] quit
```

---

### Guardrails & Safety

```python
from syrin import Agent, Model
from syrin.guardrails import PIIGuardrail, LengthGuardrail, ToolOutputValidator
from pydantic import BaseModel

class SafeAgent(Agent):
    model      = Model.OpenAI("gpt-4o-mini", api_key="...")
    guardrails = [
        PIIGuardrail(redact=True),          # Redact emails, phones, SSNs, card numbers
        LengthGuardrail(max_length=4000),   # Enforce output length
    ]

result = SafeAgent().run("Process: please call me at 555-123-4567")
# result.content → "please call me at ***-***-****"
# result.report.guardrail.passed → False
```

---

### Production-Ready from Day One

**One-line HTTP serving:**
```python
agent.serve(port=8000, enable_playground=True)
# → POST /chat   POST /stream   GET /playground
```

**Crash-proof checkpoints:**
```python
from syrin.checkpoint import CheckpointConfig

agent = Agent(
    model=model,
    checkpoint_config=CheckpointConfig(dir="/tmp/checkpoints", auto_save=True),
)
result = agent.run("Begin long analysis...")
# Crash? Resume exactly where it left off:
agent.load_checkpoint("analysis-run-1")
```

**Event-driven triggers:**
```python
from syrin.watch import CronProtocol, WebhookProtocol

agent.watch(CronProtocol(cron="0 9 * * *"), task="Send morning briefing")
agent.watch(WebhookProtocol(path="/events"), task="Process incoming event")
```

---

## Real-World Examples

### Research Pipeline — Parallel Agents with Consensus

```python
from syrin import Agent, Budget, Model
from syrin.swarm import Swarm, SwarmConfig, ConsensusConfig
from syrin.enums import SwarmTopology, ConsensusStrategy

class ResearchAgent(Agent):
    model = Model.OpenAI("gpt-4o", api_key="...")
    system_prompt = "Research the given topic with citations."

swarm = Swarm(
    agents=[ResearchAgent, ResearchAgent, ResearchAgent],  # 3 independent researchers
    config=SwarmConfig(
        topology=SwarmTopology.CONSENSUS,
        consensus=ConsensusConfig(
            min_agreement=0.67,
            voting_strategy=ConsensusStrategy.MAJORITY,
        ),
        budget_pool=BudgetPool(total=3.00),
    ),
)

result = swarm.run("What are the main risks of lithium-ion batteries at scale?")
print(result.content)           # Consensus answer
print(result.cost_breakdown)    # Cost per agent
```

---

### Customer Support — Memory + Handoffs + Guardrails

```python
from syrin import Agent, Budget, Model
from syrin.guardrails import PIIGuardrail
from syrin.enums import MemoryType

class SupportAgent(Agent):
    model      = Model.OpenAI("gpt-4o-mini", api_key="...")
    budget     = Budget(max_cost=0.05)    # $0.05 per conversation turn
    guardrails = [PIIGuardrail(redact=True)]
    system_prompt = "You are a helpful customer support agent."

class EscalationAgent(Agent):
    model = Model.OpenAI("gpt-4o", api_key="...")
    system_prompt = "You handle escalated support cases requiring senior judgment."

agent = SupportAgent()
agent.remember(f"Customer {user_id}: premium plan, joined 2023", memory_type=MemoryType.CORE)

result = agent.run(user_message)

if result.confidence < 0.6:
    result = agent.handoff(EscalationAgent, context=result.content)
```

---

### Document Intelligence — RAG + Structured Output

```python
from syrin import Agent, Model
from syrin.knowledge import Knowledge
from syrin.enums import KnowledgeBackend
from pydantic import BaseModel

class ContractRisk(BaseModel):
    risk_level: str           # low | medium | high | critical
    key_clauses: list[str]
    recommended_action: str

kb = Knowledge(
    sources=["path/to/contracts/"],
    backend=KnowledgeBackend.QDRANT,
    embedding_provider="openai",
)

class ContractReviewer(Agent):
    model       = Model.OpenAI("gpt-4o", api_key="...")
    budget      = Budget(max_cost=0.25)
    knowledge   = kb
    output_type = ContractRisk

result = ContractReviewer().run("Review the indemnification clause in contract-2024-07.pdf")
risk: ContractRisk = result.output   # Guaranteed typed output
print(f"Risk: {risk.risk_level} — {risk.recommended_action}")
```

---

## Installation

```bash
# Core library
pip install syrin

# With OpenAI support
pip install syrin[openai]

# With Anthropic support
pip install syrin[anthropic]

# Multi-modal — voice, documents, vector stores
pip install syrin[voice,pdf,vector]

# Full install
pip install syrin[openai,anthropic,serve,vector,postgres,pdf,voice]
```

---

## Why Syrin

| Capability | Syrin | DIY / Other Libraries |
|---|---|---|
| **Budget enforcement** | Declarative, pre-call + post-call | Not available or manual |
| **Rate windows** | hour / day / month built-in | Build and persist yourself |
| **Threshold callbacks** | `BudgetThreshold(at=80, ...)` | Write from scratch |
| **Shared budget pools** | Thread-safe `BudgetPool` | Implement locking |
| **Memory (4 types)** | Built-in, auto-managed, backend-agnostic | Manual setup |
| **Multi-agent (5 topologies)** | Single `Swarm` class | Complex orchestration code |
| **Lifecycle hooks** | 72+ typed events | Logging + parsing |
| **Live debugger** | Rich TUI (Pry) | Parse log files |
| **Guardrails** | PII, length, content, output validation | Per-project code |
| **Checkpoints** | Auto-save, crash recovery | DIY |
| **RAG / Knowledge** | GitHub, docs, PDFs, websites | Manual indexing pipeline |
| **Structured output** | Guaranteed Pydantic / JSON | Parse + validate manually |
| **Serving** | One-line HTTP + playground | Build Flask/FastAPI wrapper |
| **Type safety** | StrEnum everywhere, mypy strict | String literals |

---

## Documentation

| Guide | Description |
|---|---|
| [Quick Start](https://docs.syrin.dev/agent-kit/) | Your first agent in 5 minutes |
| [Budget Control](https://docs.syrin.dev/agent-kit/) | Caps, rate limits, thresholds, shared pools |
| [Memory](https://docs.syrin.dev/agent-kit/) | 4 types, backends, decay curves |
| [Multi-Agent Swarms](https://docs.syrin.dev/agent-kit/) | 5 topologies, A2A messaging, budget delegation |
| [Observability & Hooks](https://docs.syrin.dev/agent-kit/) | 72+ events, tracing, Pry debugger |
| [Guardrails](https://docs.syrin.dev/agent-kit/) | PII, length, content filtering, output validation |
| [Knowledge / RAG](https://docs.syrin.dev/agent-kit/) | Ingestion, chunking, retrieval |
| [Serving](https://docs.syrin.dev/agent-kit/) | HTTP API, streaming, playground |
| [Checkpoints](https://docs.syrin.dev/agent-kit/) | State persistence, crash recovery |
| [Examples](examples/README.md) | Runnable code for every use case |

---

## Join the Community

We are building a community of engineers shipping AI agents to production. Ask questions, share what you have built, and help shape the roadmap.

<p align="center">
  <a href="https://discord.gg/BJUdNBh4TC">
    <img src="https://img.shields.io/badge/Discord-Join%20the%20server-5865F2?style=for-the-badge&logo=discord&logoColor=white" alt="Discord">
  </a>
  &nbsp;&nbsp;
  <a href="https://www.youtube.com/@syrin_dev">
    <img src="https://img.shields.io/badge/YouTube-Watch%20tutorials-FF0000?style=for-the-badge&logo=youtube&logoColor=white" alt="YouTube">
  </a>
  &nbsp;&nbsp;
  <a href="https://reddit.com/r/syrin_ai">
    <img src="https://img.shields.io/badge/Reddit-r%2Fsyrin__ai-FF4500?style=for-the-badge&logo=reddit&logoColor=white" alt="Reddit">
  </a>
</p>

| Channel | What's there |
|---|---|
| [Discord](https://discord.gg/BJUdNBh4TC) | Real-time help, showcase your agents, roadmap discussion |
| [Reddit — r/syrin_ai](https://reddit.com/r/syrin_ai) | Longer posts, tutorials, use-case deep dives |
| [YouTube — @syrin_dev](https://www.youtube.com/@syrin_dev) | Walkthroughs, feature demos, production patterns |
| [GitHub Discussions](https://github.com/syrin-labs/syrin-python/discussions) | RFCs, architecture questions, feature requests |
| [Website](https://syrin.ai) | Product overview, roadmap, changelog |

---

## Contributing

Contributions are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on setting up the dev environment, running tests, and submitting pull requests.

---

## License

MIT — see [LICENSE](LICENSE) for details.

---

<p align="center">
  Built for engineers who ship AI to production.
</p>
