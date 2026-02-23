# Feature Reference Guide

Complete reference for all Syrin features. Use this as a lookup when building agents.

## Core Components

### Agent

The main class for creating AI agents.

```python
from Syrin import Agent
from Syrin.model import Model

class MyAgent(Agent):
    model = Model.OpenAI("gpt-4o-mini")
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

Different AI models to choose from:

```python
from Syrin.model import Model

# OpenAI
Model.OpenAI("gpt-4o")        # Most advanced
Model.OpenAI("gpt-4o-mini")   # Cheap & fast
Model.OpenAI("gpt-4")         # Previous version

# Anthropic (Claude)
Model.Anthropic("claude-3-opus")     # Most advanced
Model.Anthropic("claude-3-sonnet")   # Balanced
Model.Anthropic("claude-3-haiku")    # Cheap & fast

# Google
Model.Google("gemini-pro")

# Ollama (Local, free)
Model.Ollama("llama2")
Model.Ollama("mistral")

# LiteLLM (50+ providers)
Model.LiteLLM("provider/model")
```

---

## Tools

Give agents the ability to do things.

```python
from Syrin.tool import tool

@tool
def my_tool(param1: str, param2: int = 10) -> dict:
    """Tool description for AI."""
    return {"result": "value"}

class MyAgent(Agent):
    tools = [my_tool]
```

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

Control costs and prevent overspending.

```python
from Syrin import Agent, Budget, OnExceeded
from Syrin.threshold import BudgetThreshold

class BudgetAgent(Agent):
    def __init__(self):
        super().__init__()
        self.budget = Budget(
            run=0.10,           # Max $0.10 per request
            hourly=5.00,        # Max $5 per hour
            daily=50.00,        # Max $50 per day
            monthly=500.00,     # Max $500 per month
            on_exceeded=OnExceeded.ERROR,  # or WARN
            thresholds=[
                BudgetThreshold(at=80, action=lambda ctx: print(f"Budget at {ctx.percentage}%")),
                BudgetThreshold(at=95, action=lambda ctx: ctx.parent.switch_model(Model("openai/gpt-4o-mini"))),
            ]
        )
```

**OnExceeded Options:**
- `OnExceeded.WARN` - Show warning but continue
- `OnExceeded.ERROR` - Stop and raise error

---

## Memory

Make agents remember things.

```python
from Syrin import Agent, Memory

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

## Response Object

What you get back from `agent.response()`:

```python
response = agent.response("Hello")

response.content          # The answer text
response.cost_usd         # $ spent
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

## Enums (Options)

Use these constants instead of strings:

```python
from Syrin import OnExceeded, LoopStrategy

# For budgets
OnExceeded.WARN
OnExceeded.ERROR

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
from Syrin import Agent
from Syrin.model import Model

class SimpleAgent(Agent):
    model = Model.OpenAI("gpt-4o-mini")
    system_prompt = "Help users"

agent = SimpleAgent()
response = agent.response("Your question")
print(response.content)
```

### Pattern 2: Agent with Tools

```python
from Syrin import Agent
from Syrin.model import Model
from Syrin.tool import tool

@tool
def do_something(param: str) -> dict:
    """Do something."""
    return {"result": param.upper()}

class ToolAgent(Agent):
    model = Model.OpenAI("gpt-4o-mini")
    tools = [do_something]

agent = ToolAgent()
response = agent.response("Use your tool")
```

### Pattern 3: Agent with Memory

```python
from Syrin import Agent, Memory
from Syrin.model import Model

class MemoryAgent(Agent):
    model = Model.OpenAI("gpt-4o-mini")
    
    def __init__(self):
        super().__init__()
        self.memory = Memory()

agent = MemoryAgent()
agent.response("Remember: my name is Alice")
agent.response("What's my name?")  # It remembers!
```

### Pattern 4: Agent with Budget

```python
from Syrin import Agent, Budget, OnExceeded
from Syrin.model import Model

class BudgetAgent(Agent):
    model = Model.OpenAI("gpt-4o-mini")
    
    def __init__(self):
        super().__init__()
        self.budget = Budget(
            run=0.10,
            on_exceeded=OnExceeded.ERROR
        )

agent = BudgetAgent()
response = agent.response("Do something")
print(f"Cost: ${response.cost_usd:.4f}")
```

### Pattern 5: Multiple Agents

```python
from Syrin import Agent
from Syrin.model import Model

class Agent1(Agent):
    model = Model.OpenAI("gpt-4o-mini")
    system_prompt = "You are Agent 1"

class Agent2(Agent):
    model = Model.OpenAI("gpt-4o-mini")
    system_prompt = "You are Agent 2"

a1 = Agent1()
a2 = Agent2()

r1 = a1.response("Question 1")
r2 = a2.response("Question 2")
```

### Pattern 6: Streaming

```python
from Syrin import Agent
from Syrin.model import Model

class StreamAgent(Agent):
    model = Model.OpenAI("gpt-4o-mini")

agent = StreamAgent()

for chunk in agent.astream("Your prompt"):
    print(chunk.text, end="", flush=True)
```

---

## Troubleshooting

**"API key not found"**
- Set `OPENAI_API_KEY` or `ANTHROPIC_API_KEY` environment variable

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
