# Feature Reference Guide

Complete reference for all Syrin features. Use this as a lookup when building agents.

> **Agent vs standalone:** See [Architecture](ARCHITECTURE.md) to understand which components require an Agent and which work independently.

## Core Components

### Agent

The main class for creating AI agents.

```python
import os
from syrin import Agent, Model

class MyAgent(Agent):
    # model = Model.OpenAI("gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"))
    model = Model.Almock()  # No API Key needed
    system_prompt = "You are helpful"
    tools = []  # Optional tools
    
    def __init__(self):
        super().__init__()
        # Optional: add memory, budget, etc.
```

**Key Methods:**
- `agent.response(prompt)` - Get a response
- `agent.astream(prompt)` - Stream response piece by piece
- `agent.aresponse(prompt)` - Async get response

---

## Models

See **[Models Guide](models.md)** for the complete documentation:
- Built-in models (OpenAI, Anthropic, Google, Ollama, LiteLLM)
- Model.Custom for third-party OpenAI-compatible APIs
- Custom models via inheritance and `make_model()`
- Tweakable properties (temperature, max_tokens, context_window, etc.)
- Fallbacks, structured output, centralized definitions

Quick reference:

```python
import os
from syrin import Model

Model.OpenAI("gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"))
Model.Anthropic("claude-sonnet", api_key=os.getenv("ANTHROPIC_API_KEY"))
Model.Custom("deepseek-chat", api_base="https://api.deepseek.com/v1", api_key="...")
```

**Routing:** Pass `model=[M1, M2, M3]` + `model_router=RoutingConfig(routing_mode=...)` for per-request model selection. See [Routing](routing.md).

---

## Tools

Give agents the ability to do things.

```python
from syrin import tool

@tool
def my_tool(param1: str, param2: int = 10) -> dict:
    """Tool description for AI."""
    return {"result": "value"}

class MyAgent(Agent):
    tools = [my_tool]
```

**Group tools in MCP, use MCP in agents:** Define an MCP with `@tool`, add to `tools=[ProductMCP()]`. See [MCP](mcp.md).

**Supported Types:**
- `str` - Text
- `int` - Whole numbers
- `float` - Decimals
- `bool` - True/False
- `list` - Arrays
- `dict` - Objects
- `Optional[Type]` - May be None

---

## Budget Management

Control costs (USD) and prevent overspending. Budget is **spend only**; for token usage caps use **TokenLimits** separately — see [Budget Control](budget-control.md).

```python
import os
from syrin import Agent, Budget, RateLimit, Model, raise_on_exceeded
from syrin.threshold import BudgetThreshold

class BudgetAgent(Agent):
    def __init__(self):
        super().__init__()
        self.budget = Budget(
            run=0.10,                              # Max $0.10 per request
            per=RateLimit(hour=5.00, day=50.00, month=500.00),
            on_exceeded=raise_on_exceeded,        # or warn_on_exceeded
            thresholds=[
                BudgetThreshold(at=80, action=lambda ctx: print(f"Budget at {ctx.percentage}%")),
                BudgetThreshold(at=95, action=lambda ctx: ctx.parent.switch_model(Model.OpenAI("gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY")))),
            ]
        )
```

**on_exceeded:** Pass a callback. Use `raise_on_exceeded` to stop and raise, or `warn_on_exceeded` to log and continue.

---

## Memory

Make agents remember things.

```python
from syrin import Agent, Memory

class MemoryAgent(Agent):
    def __init__(self):
        super().__init__()
        self.memory = Memory()
```

**Memory Types:**
- `CORE` - User identity/preferences
- `EPISODIC` - Specific events
- `SEMANTIC` - General facts
- `PROCEDURAL` - Learned behaviors

**Decay Strategies:**
- `EXPONENTIAL` - Old memories fade fast
- `LINEAR` - Fade at constant rate
- `LOGARITHMIC` - Fade slowly
- `STEP` - Disappear suddenly
- `NONE` - Never fade

---

## Knowledge (RAG)

Load documents from files, URLs, or raw text and attach them to an agent as a searchable knowledge base. Documents are normalized to a common `Document` type with `content`, `source`, `source_type`, and optional `metadata`.

**Document model:** Immutable. Required: `content`, `source`, `source_type`. Optional: `metadata` (e.g. page, section).

**Source constructors (via `Knowledge.*`):**

- **Files:** `Knowledge.PDF(path)`, `Knowledge.Markdown(path)`, `Knowledge.TextFile(path)`, `Knowledge.YAML(path)`, `Knowledge.JSON(path, jq_path=...)`, `Knowledge.Python(path)`
- **Directories:** `Knowledge.Directory(path, glob="**/*.md", pattern=..., recursive=True)`
- **Raw text:** `Knowledge.Text("inline fact")`, `Knowledge.Texts(["fact1", "fact2"])`
- **Remote:** `Knowledge.URL(url)`, `Knowledge.GitHub(username, repos=["repo1", "repo2"])` (or `repos=None` for all public repos)

**Knowledge class (orchestrator):** Full pipeline: load → chunk → embed → store → search. Requires `embedding` (e.g. `Embedding.OpenAI`). Lazy ingest on first `search()`.

```python
from syrin import Knowledge, KnowledgeBackend
from syrin.embedding import Embedding
from syrin.knowledge import Document

# Document: immutable, with source and source_type
doc = Document(content="Hello", source="user_provided", source_type="text")

# Build sources (loaders)
sources = [
    Knowledge.PDF("./resume.pdf"),
    Knowledge.Markdown("./about.md"),
    Knowledge.Text("I have 5 years of Python experience."),
    Knowledge.GitHub("myorg", repos=["project-a", "project-b"]),
]

# Knowledge orchestrator (embedding required)
knowledge = Knowledge(
    sources=sources,
    embedding=Embedding.OpenAI("text-embedding-3-small"),
    backend=KnowledgeBackend.MEMORY,  # or POSTGRES, SQLITE, etc.
)

# Attach to Agent for auto search_knowledge tool
# agent = Agent(model=..., knowledge=knowledge)
# await knowledge.ingest()  # or lazy on first search
# results = await knowledge.search("Python experience")
```

**Agentic RAG:** Set `agentic=True` to add `search_knowledge_deep` (multi-step retrieval) and `verify_knowledge` (claim verification) tools. The agent gets control over query decomposition, result grading, and iterative refinement.

```python
from syrin.embedding import Embedding
from syrin.knowledge import AgenticRAGConfig, Knowledge

knowledge = Knowledge(
    sources=[Knowledge.PDF("./resume.pdf"), Knowledge.Text("Fact.")],
    embedding=Embedding.OpenAI("text-embedding-3-small"),
    backend=KnowledgeBackend.MEMORY,
    agentic=True,
    agentic_config=AgenticRAGConfig(
        max_search_iterations=3,
        decompose_complex=True,
        grade_results=True,
        relevance_threshold=0.5,
        search_model=None,  # None = use agent's model
    ),
)
# Agent gets: search_knowledge, search_knowledge_deep, verify_knowledge
```

**Hooks:** `KNOWLEDGE_AGENTIC_DECOMPOSE`, `KNOWLEDGE_AGENTIC_GRADE`, `KNOWLEDGE_AGENTIC_REFINE`, `KNOWLEDGE_AGENTIC_VERIFY`.

**Knowledge Store (vector backends):** Store chunks with embeddings for semantic search. Use `get_knowledge_store(backend, ...)` or instantiate directly.

- **Backends:** `KnowledgeBackend.MEMORY` (testing, no deps), `KnowledgeBackend.POSTGRES` (pgvector, production), `KnowledgeBackend.QDRANT`, `KnowledgeBackend.CHROMA`, `KnowledgeBackend.SQLITE` (sqlite-vec)
- **Protocol:** `KnowledgeStore` has `upsert(chunks, embeddings)`, `search(query_embedding, top_k, filter, score_threshold)`, `delete(source=..., document_id=...)`, `count()`
- **Optional deps:** `syrin[knowledge-postgres]` (asyncpg, pgvector), `syrin[knowledge-sqlite]` (sqlite-vec). Qdrant/Chroma use existing `syrin[qdrant]` / `syrin[chroma]`.

```python
from syrin import get_knowledge_store, KnowledgeBackend
from syrin.knowledge import Chunk
from syrin.knowledge.stores import InMemoryKnowledgeStore

# In-memory (no deps)
store = InMemoryKnowledgeStore(embedding_dimensions=1536)

# Or via factory
store = get_knowledge_store(KnowledgeBackend.MEMORY, embedding_dimensions=1536)
# Postgres: get_knowledge_store(KnowledgeBackend.POSTGRES, connection_url="...", embedding_dimensions=1536)
```

**Loaders:** Each source has `.load()` (sync) and `.aload()` (async). `GitHubLoader` and `URLLoader` require async (`.aload()`). All return `list[Document]`.

**Chunking:** Split documents into retrieval-optimized chunks with `ChunkConfig`, `ChunkStrategy`, and `get_chunker()`. Requires `syrin[knowledge]` (chonkie).

- **Strategies:** `RECURSIVE` (default for general text), `MARKDOWN` (header-aware), `PAGE` (one chunk per page), `CODE` (AST-aware), `SENTENCE`, `TOKEN`, `SEMANTIC` (needs `config.embedding`), `AUTO` (selects by `source_type`: code → CODE, markdown → MARKDOWN, pdf+has_pages → PAGE, else RECURSIVE).
- **Config:** `ChunkConfig(strategy=..., chunk_size=512, chunk_overlap=0, min_chunk_size=50, ...)`. Use `get_chunker(config)` to get a `Chunker`; call `chunker.chunk(documents)` or `await chunker.achunk(documents)` for async (e.g. SemanticChunker).
- **Chunk:** Each chunk has `content`, `metadata`, `document_id` (= document source), `chunk_index`, `token_count`. Metadata includes `chunk_strategy` and, for markdown, optional `heading_hierarchy`.

```python
from syrin.knowledge import Document, ChunkConfig, ChunkStrategy, get_chunker

docs = [Document(content="Long text...", source="doc.txt", source_type="text")]
config = ChunkConfig(strategy=ChunkStrategy.RECURSIVE, chunk_size=256, min_chunk_size=0)
chunker = get_chunker(config)
chunks = chunker.chunk(docs)
# chunks[i].content, .document_id, .chunk_index, .token_count, .metadata
```

---

## Response Object

What you get back from `agent.response()`:

```python
response = agent.response("Hello")

response.content          # The answer text
response.cost             # $ spent
response.tokens           # Tokens used
response.model            # Which model
response.duration         # Seconds taken
response.stop_reason      # Why it stopped
response.tool_calls       # Tools used
response.budget_remaining # Budget left
response.budget_used      # Budget used
response.raw             # Raw API response
```

---

## Serving — HTTP, CLI, STDIO

Serve your agent via HTTP, CLI, or STDIO:

```python
from syrin.enums import ServeProtocol

agent.serve(port=8000)  # HTTP: POST /chat, /stream, GET /health, etc.
agent.serve(protocol=ServeProtocol.CLI)  # CLI: terminal REPL
agent.serve(protocol=ServeProtocol.STDIO)  # STDIO: JSON lines on stdin/stdout
agent.serve(port=8000, enable_playground=True)  # Web playground at /playground
```

**Requires:** `uv pip install syrin[serve]`. See [Serving](serving.md).

---

## MCP — Group Tools, Use in Agents

Group related tools in an MCP and add MCP to your agent's tools:

```python
from syrin import MCP, Agent, tool

class ProductMCP(MCP):
    @tool
    def search_products(self, query: str) -> str:
        """Search products."""
        return f"Results: {query}"

class ProductAgent(Agent):
    tools = [ProductMCP()]  # MCP tools become agent tools
```

When serving, `/mcp` is auto-mounted alongside `/chat`. See [MCP](mcp.md).

---

## Enums (Options)

Use these constants instead of strings:

```python
from syrin import LoopStrategy

# For budgets
warn_on_exceeded
raise_on_exceeded

# For loops
LoopStrategy.REACT  # Think-Act-Observe (default)
LoopStrategy.SINGLE_SHOT  # Single response
LoopStrategy.PLAN_EXECUTE  # Plan then execute
LoopStrategy.CODE_ACTION  # Generate and execute code
```

---

## Examples by Task

### Simple Q&A
See [Use Case 1: Simple Q&A Agent](simple-qa-agent.md)

### With Tools
See [Use Case 2: Research Agent with Tools](research-agent-with-tools.md)

### With Memory
See [Use Case 3: Agent with Memory](agent-with-memory.md)

### Budget Control
See [Use Case 4: Budget Control](budget-control.md)

### Multi-Agent Teams
See [Use Case 5: Multi-Agent Orchestration](multi-agent.md)

### Streaming
See [Use Case 6: Streaming](streaming.md)

---

## Quick Reference: Common Patterns

### Pattern 1: Basic Agent

```python
import os
from syrin import Agent, Model

class SimpleAgent(Agent):
    # model = Model.OpenAI("gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"))
    model = Model.Almock()  # No API Key needed
    system_prompt = "Help users"

agent = SimpleAgent()
response = agent.response("Your question")
print(response.content)
```

### Pattern 2: Agent with Tools

```python
import os
from syrin import Agent, Model, tool

@tool
def do_something(param: str) -> dict:
    """Do something."""
    return {"result": param.upper()}

class ToolAgent(Agent):
    # model = Model.OpenAI("gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"))
    model = Model.Almock()  # No API Key needed
    tools = [do_something]

agent = ToolAgent()
response = agent.response("Use your tool")
```

### Pattern 3: Agent with Memory

```python
import os
from syrin import Agent, Memory, Model

class MemoryAgent(Agent):
    # model = Model.OpenAI("gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"))
    model = Model.Almock()  # No API Key needed
    
    def __init__(self):
        super().__init__()
        self.memory = Memory()

agent = MemoryAgent()
agent.response("Remember: my name is Alice")
agent.response("What's my name?")  # It remembers!
```

### Pattern 4: Agent with Budget

```python
import os
from syrin import Agent, Budget, Model, raise_on_exceeded

class BudgetAgent(Agent):
    # model = Model.OpenAI("gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"))
    model = Model.Almock()  # No API Key needed
    
    def __init__(self):
        super().__init__()
        self.budget = Budget(
            run=0.10,
            on_exceeded=raise_on_exceeded
        )

agent = BudgetAgent()
response = agent.response("Do something")
print(f"Cost: ${response.cost:.4f}")
```

### Pattern 5: Multiple Agents

```python
import os
from syrin import Agent, Model

class Agent1(Agent):
    # model = Model.OpenAI("gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"))
    model = Model.Almock()  # No API Key needed
    system_prompt = "You are Agent 1"

class Agent2(Agent):
    # model = Model.OpenAI("gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"))
    model = Model.Almock()  # No API Key needed
    system_prompt = "You are Agent 2"

a1 = Agent1()
a2 = Agent2()

r1 = a1.response("Question 1")
r2 = a2.response("Question 2")
```

### Pattern 6: Streaming

```python
import os
from syrin import Agent, Model

class StreamAgent(Agent):
    # model = Model.OpenAI("gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"))
    model = Model.Almock()  # No API Key needed

agent = StreamAgent()

for chunk in agent.astream("Your prompt"):
    print(chunk.text, end="", flush=True)
```

---

## Troubleshooting

**"API key not found"**
- Pass `api_key` explicitly: `Model.OpenAI("gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"))`
- The library does not auto-read API keys from environment variables

**"Model not found"**
- Check model name spelling
- Make sure API credentials are valid
- Verify you have API credits

**"Budget exceeded"**
- Check your budget settings
- Use cheaper models (gpt-4o-mini, claude-3-haiku)
- Reduce the scope of requests

**"Tool not working"**
- Make sure tool returns a dict
- Check parameter types match usage
- Verify tool is in agent.tools list

---

## Performance Tips

1. **Use right model**
   - Testing: Use gpt-4o-mini
   - Production: Use gpt-4o or claude-3-sonnet
   - Advanced: Use gpt-4 or claude-3-opus

2. **Set budgets**
   - Always set per-request budget
   - Add daily/monthly limits

3. **Use tools wisely**
   - Keep tools simple
   - Don't overload with tools
   - Document tool purpose clearly

4. **Stream for large responses**
   - Use `astream()` for essays/stories
   - Use `run()` for short responses

5. **Cache prompts**
   - Reuse agent instances
   - Don't recreate agents per request

---

## Remote Config

Configuration overrides from a backend (Syrin Cloud or self-hosted) without code deploys. Call **`syrin.init(api_key=...)`** (or set `SYRIN_API_KEY`) to enable; agents then register and receive overrides. When serving, **GET/PATCH /config** and **GET /config/stream** are available. Types and wire format: [Remote Config](remote-config.md).

---

## Useful Links

- [OpenAI Models](https://platform.openai.com/docs/models)
- [Anthropic Claude](https://console.anthropic.com/)
- [Google Gemini](https://ai.google.dev/)
- [Local Models with Ollama](https://ollama.ai/)

---

## Examples Repository

All examples are in `/examples/` directory:
- `examples/phase1_basic_agent.py` - Basic agent
- `examples/phase2_memory_system.py` - Memory
- `examples/phase3_multi_agent.py` - Multi-agent
- `examples/phase4_budget_testing.py` - Budget
- `examples/phase5_async_streaming.py` - Streaming
- `examples/advanced/` - Advanced patterns

---

## Next Steps

- Pick a [Use Case Guide](#examples-by-task)
- Try building your own agent
- Check examples in `/examples/`
- Read [Getting Started](getting-started.md) again if stuck

---

**Need help?** Check the [FAQ](getting-started.md#common-questions)
