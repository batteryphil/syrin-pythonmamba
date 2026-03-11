# Frequently Asked Questions

Common questions about Syrin, organized by topic.

---

## Getting Started

### How do I install Syrin?

```bash
pip install syrin
```

You'll also need an API key:
```bash
export OPENAI_API_KEY="sk-..."  # or ANTHROPIC_API_KEY for Claude
```

### What's the difference between `run()` and `response()`?

Both do the same thing! `run()` is shorter:
```python
result = agent.response("Hello")      # Same as
result = agent.response("Hello")
```

### How do I know which model to use?

Start with these:
- **gpt-4o-mini** - Cheap, fast, good for simple tasks
- **gpt-4o** - More capable, more expensive
- **claude-sonnet** - Great for complex reasoning

```python
from syrin.model import Model

agent = Agent(
    # model=Model.OpenAI("gpt-4o-mini"),  # Cheap
    model=Model.Almock(),  # No API Key needed
)
# Or Model.Anthropic("claude-sonnet-4-20250514") when you have a key
```

### Can I use Ollama or local models?

Yes! Syrin supports Ollama, LiteLLM, and third-party APIs:

```python
from syrin import Agent, Model

agent = Agent(model=Model.Ollama("llama3"))  # Local
agent = Agent(model=Model.LiteLLM("ollama/llama3"))  # Via LiteLLM

# Third-party OpenAI-compatible APIs (DeepSeek, KIMI, Grok, etc.)
agent = Agent(model=Model.Custom(
    "deepseek-chat",
    api_base="https://api.deepseek.com/v1",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
))
```

---

## Tools

### How do I create a tool?

Use the `@tool` decorator:

```python
from syrin import tool

@tool
def search(query: str) -> str:
    """Search the web for information."""
    # Your code here
    return f"Results for: {query}"

# Use it
agent = Agent(tools=[search])
result = agent.response("Search for Python tutorials")
```

### Can tools return complex data?

Yes! Tools can return strings, dicts, or objects. The agent receives the result as a string.

### How do tools work with memory?

Tools execute and their results are added to conversation history. The agent can then use those results in subsequent calls.

### How do I group tools in MCP and use them in my agent?

Create an MCP class with `@tool` methods (same as Agent), then add the MCP to your agent's `tools`:

```python
from syrin import MCP, Agent, tool

class ProductMCP(MCP):
    @tool
    def search_products(self, query: str) -> str:
        """Search products."""
        return f"Results: {query}"

    @tool
    def get_product(self, id: str) -> str:
        """Get product by ID."""
        return f"Product {id}"

class ProductAgent(Agent):
    model = almock
    tools = [ProductMCP()]  # MCP tools become agent tools
```

When serving, `/mcp` is auto-mounted alongside `/chat`. See [MCP](mcp.md).

---

## Serving

### How do I serve my agent via HTTP or CLI?

Serving is built-in. Use HTTP (default), CLI REPL, or STDIO:

```python
from syrin.enums import ServeProtocol

agent.serve(port=8000)  # HTTP: POST /chat, /stream, etc.
agent.serve(protocol=ServeProtocol.CLI)  # CLI: terminal REPL in your terminal
agent.serve(protocol=ServeProtocol.STDIO)  # STDIO: JSON lines on stdin/stdout
```

Add `enable_playground=True` for the web playground at http://localhost:8000/playground. See [Serving](serving.md).

---

## Memory

### What are the different memory types?

| Type | Purpose | Use Case |
|------|---------|----------|
| `CORE` | Essential facts about user | Name, preferences, facts |
| `EPISODIC` | Past experiences | Conversations, events |
| `SEMANTIC` | General knowledge | Learned information |
| `PROCEDURAL` | How-to knowledge | Instructions, patterns |

### How do I use memory?

```python
from syrin import Agent, Memory
from syrin.enums import MemoryType

agent = Agent(
    # model=Model.OpenAI("gpt-4o-mini"),
    model=Model.Almock(),  # No API Key needed
    memory=Memory(types=[MemoryType.CORE, MemoryType.EPISODIC]),
)

# Store something
agent.remember("User's name is John", memory_type=MemoryType.CORE)

# Later - agent auto-recalls
result = agent.response("What's my name?")  # "Your name is John"
```

### Can I control what gets remembered?

Yes! Use decay strategies:

```python
from syrin.memory import Decay
from syrin.enums import DecayStrategy

memory = Memory(
    types=[MemoryType.CORE],
    decay=Decay(
        strategy=DecayStrategy.TIME_BASED,
        half_life_hours=24,  # Forget after 24 hours
    ),
)
```

---

## Budget & Cost

### How do I set a budget?

```python
from syrin import Budget
from syrin.enums import RateWindow

budget = Budget(
    total_limit=10.0,      # $10 max
    rate_limit=RateWindow(
        minutes=60,        # Per hour
        max_requests=100,
    ),
)

agent = Agent(
    # model=Model.OpenAI("gpt-4o-mini"),
    model=Model.Almock(),  # No API Key needed
    budget=budget,
)
```

### What happens when budget runs out?

Configure with `on_exceeded` (a callback):

```python
from syrin import raise_on_exceeded, warn_on_exceeded

budget = Budget(
    run=10.0,
    on_exceeded=raise_on_exceeded,  # Raise exception; or warn_on_exceeded to log and continue
)
```

### How do I track costs?

```python
result = agent.response("Hello")
print(f"Cost: ${result.cost:.4f}")
print(f"Tokens: {result.tokens}")
```

Or use hooks:

```python
def log_cost(ctx):
    print(f"Cost: ${ctx.cost:.4f}")

agent.events.on_complete(log_cost)
```

---

## Multi-Agent

### When should I use multiple agents?

Use agent teams when:
- Tasks have distinct phases (research → write → edit)
- Different expertise needed
- Parallel processing possible

### How do agents communicate?

```python
from syrin.agent import AgentTeam, sequential, parallel

# Sequential: output of one feeds into next
team = AgentTeam(
    agents=[researcher, writer, editor],
    strategy=sequential,
)

# Parallel: all run at once, results merged
team = AgentTeam(
    agents=[agent_a, agent_b, agent_c],
    strategy=parallel,
)

result = team.run_task("Write about AI")
```

---

## Async & Streaming

### Does Syrin support async?

Yes! Use `arun()` or `response_async()`:

```python
import asyncio
from syrin import Agent

async def main():
    agent = Agent(
    # model=Model.OpenAI("gpt-4o-mini"),
    model=Model.Almock(),  # No API Key needed
)
    result = await agent.arun("Hello!")
    print(result.content)

asyncio.run(main())
```

### How do I stream responses?

```python
for chunk in agent.stream("Tell me a story"):
    print(chunk, end="", flush=True)
```

### Can I combine async with streaming?

```python
async for chunk in agent.stream_async("Tell me a story"):
    print(chunk, end="", flush=True)
```

---

## Context Management

### What is context management?

It prevents "context rot" - when LLM quality drops because too much history accumulates. Syrin automatically:
- Counts tokens
- Compacts context when needed
- Tracks statistics

### How do I configure context?

```python
from syrin import Context
from syrin.threshold import ContextThreshold

agent = Agent(
    # model=Model.OpenAI("gpt-4o"),
    model=Model.Almock(),  # No API Key needed
    context=Context(
        max_tokens=80000,
        thresholds=[
            ContextThreshold(at=75, action=lambda ctx: ctx.compact() if ctx.compact else None),
        ],
    ),
)
```

---

## Observability & Debugging

### How do I debug my agent?

Enable debug mode:

```python
agent = Agent(
    # model=Model.OpenAI("gpt-4o-mini"),
    model=Model.Almock(),  # No API Key needed
    debug=True,  # Prints detailed logs
)
```

### How do I trace execution?

```python
from syrin.observability import trace, span

@trace
def my_function():
    with span("step1"):
        # Do something
        pass
    
    result = agent.response("Hello")
    return result
```

### How do I see what tools were called?

```python
result = agent.response("Search for X")

# Access tool calls
for call in result.tool_calls:
    print(f"Tool: {call.name}")
    print(f"Args: {call.arguments}")
```

---

## Production Deployment

### How do I handle errors gracefully?

```python
from syrin import BudgetExceededError, RateLimitError

try:
    result = agent.response("Hello")
except BudgetExceededError:
    print("Budget exceeded!")
except RateLimitError:
    print("Rate limited, retry later")
except Exception as e:
    print(f"Error: {e}")
```

### Can I persist agent state?

Yes! Use checkpointing:

```python
from syrin.checkpoint import Checkpointer

checkpointer = Checkpointer()

# Save state
checkpoint_id = checkpointer.save(
    agent_name="my_agent",
    state={"messages": agent.messages},
)

# Restore later
checkpoint = checkpointer.load(checkpoint_id)
agent.messages = checkpoint.messages
```

### How do I monitor costs in production?

1. Use budget hooks:
```python
def log_budget(ctx):
    log.info(f"Budget: ${ctx.remaining:.2f} remaining")

agent.events.on_budget(log_budget)
```

2. Export traces:
```python
from syrin.observability import get_tracer, JSONLExporter
tracer = get_tracer()
tracer.add_exporter(JSONLExporter("traces.jsonl"))
```

---

## Advanced

### Can I create custom hooks?

Yes! Use `before()` and `after()`:

```python
def modify_request(ctx):
    ctx["custom_data"] = "value"

def log_response(ctx):
    print(f"Response: {ctx.content[:50]}")

agent.events.before(Hook.LLM_REQUEST_START, modify_request)
agent.events.after(Hook.LLM_REQUEST_END, log_response)
```

### How do I create a guardrail?

```python
from syrin.guardrails import Guardrail, GuardrailResult
from syrin.enums import GuardrailStage

class MyGuardrail(Guardrail):
    def check(self, text: str, stage: GuardrailStage) -> GuardrailResult:
        if "bad" in text.lower():
            return GuardrailResult(passed=False, reason="Contains bad word")
        return GuardrailResult(passed=True)

agent = Agent(
    # model=Model.OpenAI("gpt-4o-mini"),
    model=Model.Almock(),  # No API Key needed
    guardrails=GuardrailChain([MyGuardrail()]),
)
```

### Can I customize the loop strategy?

Yes! Choose from:
- `SingleShotLoop` - One call, no iteration
- `ReactLoop` - Think → Act → Observe (default)
- `HumanInTheLoop` - Require approval for tools
- `PlanExecuteLoop` - Plan all steps, then execute each
- `CodeActionLoop` - Generate and execute Python code

```python
from syrin.loop import PlanExecuteLoop, CodeActionLoop

# Plan then execute - good for complex multi-step tasks
agent = Agent(
    # model=Model.OpenAI("gpt-4o-mini"),
    model=Model.Almock(),  # No API Key needed
    loop=PlanExecuteLoop(max_plan_iterations=3, max_execution_iterations=10),
)

# Code action - generate and run Python code
agent = Agent(
    # model=Model.OpenAI("gpt-4o-mini"),
    model=Model.Almock(),  # No API Key needed
    loop=CodeActionLoop(max_iterations=5),
)
```

### How do I integrate with Langfuse/Arize?

Syrin supports observability integrations:

```python
# Check docs for specific integration
from syrin.observability import LangfuseExporter  # If available
```

---

## Troubleshooting

### Agent isn't using tools

1. Check tools are passed correctly:
```python
agent = Agent(tools=[my_tool])  # Must be passed to constructor
```

2. Tools need descriptions:
```python
@tool
def search(query: str) -> str:
    """SEARCH THE WEB FOR INFORMATION."""  # Important!
    ...
```

### Costs are higher than expected

1. Check model being used:
```python
print(agent.model)  # Which model?
```

2. Enable budget alerts:
```python
from syrin import warn_on_exceeded
budget = Budget(run=10.0, on_exceeded=warn_on_exceeded)
```

3. Check token counts:
```python
result = agent.response("Hello")
print(result.token_usage)
```

### Memory isn't working

1. Ensure memory is enabled:
```python
agent = Agent(
    # model=Model.OpenAI("gpt-4o-mini"),
    model=Model.Almock(),  # No API Key needed
    memory=Memory(types=[MemoryType.CORE]),  # Must specify types
)
```

2. Check memory is being stored:
```python
agent.remember("Test", memory_type=MemoryType.CORE)
print(agent.memory.store)  # Check what's stored
```

### Context is getting too long

1. Trigger compaction earlier via a threshold action:
```python
context = Context(
    max_tokens=80000,
    thresholds=[
        ContextThreshold(at=50, action=lambda ctx: ctx.compact() if ctx.compact else None),
    ],
)
```

2. Add hard limits (e.g. raise when full):
```python
def raise_on_full(ctx):
    raise ValueError("Context full")

context = Context(
    max_tokens=80000,
    thresholds=[
        ContextThreshold(at=100, action=raise_on_full),
    ],
)
```

---

## Contributing & Support

### Where can I get help?

- GitHub Issues: Report bugs
- Documentation: See all guides
- Examples: Check `/examples` folder

### How do I contribute?

1. Fork the repo
2. Create a feature branch
3. Add tests
4. Submit PR

### What's the license?

MIT License - free for commercial use.

---

## Related Guides

- [Getting Started](getting-started.md) - Basics
- [Simple QA Agent](simple-qa-agent.md) - Basic agent
- [Tools](research-agent-with-tools.md) - Tool system
- [Memory](agent-with-memory.md) - Memory types
- [Budget Control](budget-control.md) - Cost management
- [Advanced Topics](advanced-topics.md) - Deep dives
