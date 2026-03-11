<p align="center">
  <img src="https://raw.githubusercontent.com/Syrin-Labs/cli/main/assets/syrin-logo-dark-bg.png" alt="Syrin" width="200">
</p>

<h1 align="center">Syrin</h1>

<p align="center">
  <b>Agents that ship. Not surprise you with bills.</b>
</p>
<p align="center">
  <i>Python library for AI agents with budget control, memory, and observability built-in.</i>
</p>

<p align="center">
  <a href="https://pypi.org/project/syrin/"><img src="https://img.shields.io/pypi/v/syrin.svg" alt="PyPI"></a>
  <a href="https://github.com/syrin-labs/syrin-python/blob/main/LICENSE"><img src="https://img.shields.io/github/license/syrin-labs/syrin-python.svg" alt="License"></a>
  <a href="https://pypi.org/project/syrin/"><img src="https://img.shields.io/pypi/pyversions/syrin.svg" alt="Python"></a>
  <a href="https://github.com/syrin-labs/syrin-python/stargazers"><img src="https://img.shields.io/github/stars/syrin-labs/syrin-python.svg?style=social" alt="Stars"></a>
</p>

<p align="center">
  <a href="https://syrin.ai">Website</a> ·
  <a href="https://github.com/syrin-labs/syrin-python/blob/main/docs/getting-started.md">Docs</a> ·
  <a href="https://discord.gg/p4jnKxYKpB">Discord</a> ·
  <a href="https://x.com/syrin_dev">Twitter</a>
</p>

## 🚀 Installation

```bash
# Basic install
pip install syrin

# With serving (playground, HTTP API)
pip install syrin[serve]

# With all features
pip install syrin[serve,anthropic,voice]
```

---

## 🎯 The Problem: "Why Did My AI Agent Cost $10,000 Last Month?"

You built an AI agent. It worked perfectly in testing. Then came the bill — a surprise invoice for thousands of dollars with **zero warning**.

This is the #1 reason AI agents never make it to production. Not because they don't work — because they're financially reckless.

**What developers tell us:**

> "I had no idea when my agent hit the budget."
> "My logs don't show where tokens went."
> "I spent 3 weeks building memory from scratch."
> "My agent crashed after 2 hours — no way to resume."
> "I needed 8 libraries just to make one agent."

**Syrin solves this.** One library. Zero surprises. Production-ready from day one.

---

## 🚀 60-Second Quickstart

```bash
pip install syrin
```

```python
from syrin import Agent, Model, Budget, stop_on_exceeded

class Assistant(Agent):
    model = Model.Almock()  # No API key needed
    budget = Budget(run=0.50, on_exceeded=stop_on_exceeded)

result = Assistant().response("Explain quantum computing simply")
print(result.content)
# Cost: $0.0012  |  Budget used: $0.0012
```

**You now have:**
- ✅ Budget cap at $0.50 (stops automatically)
- ✅ Cost tracking per response
- ✅ Token usage breakdown
- ✅ Full observability built-in

---

## 🎯 Syrin Use Cases

Syrin is built to solve the hard parts of building production AI agents. Here’s how it handles specific challenges:

### 1. Context Creation & Management
**The Problem:** Agents run out of context window or feed irrelevant history into the LLM.
**Syrin's Solution:** Automatic token counting, window management, and dynamic context injection.

```python
from syrin import Agent, Context
from syrin.threshold import ContextThreshold

agent = Agent(
    model=Model.Almock(),
    context=Context(
        max_tokens=80000,
        # Automatically compact when context is 75% full
        thresholds=[
            ContextThreshold(at=75, action=lambda ctx: ctx.compact()),
        ],
        # Or proactively compact at 60% to prevent rot
        auto_compact_at=0.6,
    ),
)
```

**Features:**
- **Token counting** with model-specific encodings
- **Compaction strategies** (middle-out truncation, summarization)
- **Dynamic injection** for RAG or runtime data
- **Snapshot view** to debug exactly what the LLM sees

---

### 2. Memory & Knowledge Pool
**The Problem:** Agents forget everything between sessions.
**Syrin's Solution:** First-class persistent memory with 4 specialized types and decay curves.

```python
from syrin import Agent
from syrin.memory import Memory
from syrin.enums import MemoryType

agent = Agent(
    model=Model.Almock(),
    memory=Memory(
        types=[MemoryType.CORE, MemoryType.EPISODIC, MemoryType.SEMANTIC],
        top_k=10,  # Retrieve top 10 relevant memories
    ),
)

# Remember facts (persisted across sessions)
agent.remember("User prefers TypeScript", memory_type=MemoryType.CORE)

# Recall later (semantic search)
memories = agent.recall("user preferences")
```

**Memory Types:**
- **Core** — Long-term facts (user profile, preferences)
- **Episodic** — Conversation history and events
- **Semantic** — Knowledge chunks with embeddings (RAG)
- **Procedural** — Skills and instructions

**Backends:** SQLite (default), Qdrant (vector search), Redis (cache), PostgreSQL (production).

---

### 3. Observability Built In
**The Problem:** "What happened?" — no visibility into agent decisions.
**Syrin's Solution:** Two ways to see everything: programmatic hooks and CLI tracing.

#### Method 1: Programmatic Hooks (debug=True)
```python
agent = Agent(
    model=Model.Almock(),
    debug=True,  # Console output for every lifecycle event
)

# Or subscribe to specific events
agent.events.on("llm.request_start", lambda ctx: print(f"LLM call #{ctx.iteration}"))
agent.events.on("budget.threshold", lambda ctx: print(f"Budget at {ctx.percentage}%"))
```

#### Method 2: CLI Tracing (--trace)
Run your agent script with the `--trace` flag for full observability without code changes:

```bash
# Enable full tracing
python my_agent.py --trace

# Or use the Syrin CLI
syrin trace my_agent.py
```

**What you get:**
- LLM request/response logs
- Tool execution traces
- Budget usage per call
- Memory operations (store/recall)
- Token counts and context utilization

---

## 🔧 Syrin's Power

### 🎛️ **Budget & Cost Control** (Your #1 Problem Solved)

**The Problem:** Agents run wild, you get surprise bills
**Syrin's Solution:** Built-in budget control with automatic stops

```python
# Per-run budget cap
agent = Agent(
    model=Model.OpenAI("gpt-4o-mini", api_key="..."),
    budget=Budget(run=0.50, on_exceeded=stop_on_exceeded),
)

# Budget thresholds (warn at 70%, switch model at 90%)
agent = Agent(
    budget=Budget(
        run=1.00,
        thresholds=[
            BudgetThreshold(at=70, action=lambda ctx: print("⚠️ 70% budget")),
            BudgetThreshold(at=90, action=lambda ctx: ctx.parent.switch_model("gpt-4o-mini")),
        ],
    ),
)

# Rate limiting
agent = Agent(
    budget=Budget(rate_limit=RateLimit(requests=10, window=60)),  # 10 req/min
)
```

**Result:** No surprise bills. Ever.

---

### 🤖 **Multi-Agent Orchestration** (Teams of Agents)

**The Problem:** Building multi-agent systems is complex
**Syrin's Solution:** Simple primitives for powerful orchestration

```python
from syrin import Agent, Model, DynamicPipeline

class Researcher(Agent):
    model = Model.Almock()
    system_prompt = "You research topics."

class Writer(Agent):
    model = Model.Almock()
    system_prompt = "You write reports."

# LLM decides which agents to spawn
pipeline = DynamicPipeline(agents=[Researcher, Writer], model=Model.Almock())
result = pipeline.run("Research AI trends and write a summary")
print(result.content, f"${result.cost:.4f}")

# Or manually:
researcher = Researcher()
result = researcher.handoff(Writer, "Write article from research", transfer_context=True)
```

**Multi-Agent Patterns:**
- **Handoff** — Route to specialist agents
- **Spawn** — Create sub-agents for subtasks
- **DynamicPipeline** — LLM orchestrates agent selection
- **Parallel execution** — Run multiple agents simultaneously

---

### 🛡️ **Guardrails & Safety** (Input/Output Validation)

**The Problem:** Agents produce harmful or incorrect output
**Syrin's Solution:** Built-in guardrails with automatic blocking

```python
from syrin import Agent, Model, GuardrailChain
from syrin.guardrails import LengthGuardrail, ContentFilter

class SafeAgent(Agent):
    model = Model.Almock()
    guardrails = GuardrailChain([
        LengthGuardrail(max_length=4000),
        ContentFilter(blocked_words=["spam", "malicious"]),
    ])

result = SafeAgent().response("User input")
print(result.report.guardrail.passed)   # True/False
print(result.report.guardrail.blocked)  # True if blocked
```

**Guardrail Types:**
- **Length** — Max input/output length
- **ContentFilter** — Block harmful words
- **PII Detection** — Detect personal information
- **Custom** — Your validation logic

---

### 🔌 **Production API & Serving** (Ship to Production)

**The Problem:** "How do I serve this to users?"
**Syrin's Solution:** One-line HTTP API + built-in playground

```python
agent = Assistant()
agent.serve(port=8000, enable_playground=True, debug=True)
# Visit http://localhost:8000/playground
```

**Features:**
- ✅ HTTP API (`POST /chat`, `POST /stream`)
- ✅ Web playground (chat UI with cost display)
- ✅ Real-time observability panel
- ✅ Multi-agent support (agent selector)
- ✅ MCP server integration

---

### 🔄 **Lifecycle & Hooks** (Full Control)

**The Problem:** Need to run custom logic at specific points
**Syrin's Solution:** 72+ hooks for every lifecycle event

| Event | When It Fires |
|-------|---------------|
| `LLM_REQUEST_START` | Before LLM call |
| `TOOL_CALL_START` | Before tool execution |
| `BUDGET_THRESHOLD` | Budget threshold reached |
| `CHECKPOINT_SAVED` | State saved |
| `CIRCUIT_TRIP` | Circuit breaker opens |
| `HANDOFF_START` | Agent hands off work |
| `SPAWN_START` | Sub-agent created |
| ... | 60+ more events |

---

### 🔌 **Remote Configuration** (Control From Anywhere)

**The Problem:** "I need to change agent config without redeploying"
**Syrin's Solution:** Built-in remote configuration server

```python
from syrin import Agent, configure

# Configure agent remotely
configure(
    agent_id="my-agent",
    endpoint="https://config.syrin.ai",
    polling_interval=60,  # Check for updates every 60 seconds
)

agent = Agent(model=Model.OpenAI("gpt-4o-mini"))
agent.serve(port=8000)
```

**Features:**
- ✅ Change config without redeploying
- ✅ A/B testing support
- ✅ Feature flags
- ✅ Dynamic model switching

---

### 🎯 **Why Developers Choose Syrin**

| Feature | Syrin | "Others" |
|---------|-------|----------|
| **Budget control** | ✅ Built-in, declarative | ❌ DIY or missing |
| **Cost tracking** | ✅ Every response | ❌ Guesswork |
| **Agent memory** | ✅ 4 types, auto-managed | ❌ Manual setup |
| **Observability** | ✅ 72+ hooks, full traces | ❌ Add-on tools |
| **Multi-agent** | ✅ Handoff, spawn, pipeline | ❌ Complex orchestration |
| **Type-safe** | ✅ StrEnum, mypy strict | ❌ String hell |
| **Production API** | ✅ One-line serve | ❌ Build Flask wrapper |
| **Remote config** | ✅ Built-in | ❌ DIY |
| **Circuit breaking** | ✅ Built-in | ❌ External library |
| **Checkpoints** | ✅ State persistence | ❌ DIY |

---

## 🎯 Real Projects Built with Syrin

### 🎙️ Voice AI Recruiter ([examples/resume_agent](examples/resume_agent))
A voice agent that handles recruiter calls using Syrin + Pipecat.

**Features:**
- Per-call budget limits ($0.50/call)
- Memory across conversations
- Real-time observability
- Cost tracking per call

**Try it:**
```bash
cd examples/resume_agent
python voice_server.py
```

### 📊 Financial Analysis Agent
Processes financial reports with tool calling, memory, and budget constraints.

### 🔍 Research Assistant
Multi-agent system that researches topics and writes reports with full cost control.

---

## 📚 Documentation

| Resource | Description |
|----------|-------------|
| **[Getting Started](docs/getting-started.md)** | 5-minute guide to your first agent |
| **[Examples](examples/README.md)** | Runnable code for every use case |
| **[API Reference](docs/reference.md)** | Complete API documentation |
| **[Architecture](docs/ARCHITECTURE.md)** | How Syrin works under the hood |
| **[Budget Control](docs/budget-control.md)** | Deep dive into budget features |
| **[Memory](docs/memory.md)** | Memory systems and backends |
| **[Multi-Agent](docs/multi-agent.md)** | Handoff, spawn, DynamicPipeline |



## ⭐ Why Star This Repo?

We're building the agent library we wish existed: **production-ready, financially safe, and actually observable.**

Every star tells us this matters. It helps us prioritize features and shows the community that agents don't have to be black boxes.

**Star Syrin if you want:**
- ✅ Agents that don't surprise you with bills
- ✅ One library instead of 10 glued together
- ✅ Built-in observability (no more log scraping)
- ✅ Memory that actually works
- ✅ Multi-agent orchestration that's simple

<p align="center">
  <a href="https://github.com/syrin-labs/syrin-python">
    <img src="https://img.shields.io/github/stars/syrin-labs/syrin-python?style=social" alt="Star Syrin on GitHub">
  </a>
</p>

---

## 🌐 Community

- 🌐 [Website](https://syrin.ai)
- 💬 [Discord](https://discord.gg/p4jnKxYKpB)
- 🐦 [Twitter](https://x.com/syrin_dev)
- 📧 [Email](mailto:hello@syrin.ai)
- 🐛 [Issues](https://github.com/syrin-labs/syrin-python/issues)
- 💡 [Discussions](https://github.com/syrin-labs/syrin-python/discussions)

---

## 🤝 Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

<p align="center">
  <b>Agents that ship. Not surprise you with bills.</b>
</p>
