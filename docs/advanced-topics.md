# Advanced Topics

This guide covers advanced Syrin features for building production-ready, observable, and robust AI agents.

---

## Core stability

Syrin guarantees that **Agent**, **Model**, **Loop**, **Budget**, **Memory**, and **Response** behave in a predictable way under all documented configurations:

- **Agent lifecycle:** `response()` and `arun()` follow the same sequence: build messages → budget/rate checks → LLM → tools → record cost → build response. No silent path differences between sync and async.
- **Loop strategies:** REACT, SINGLE_SHOT, PLAN_EXECUTE, and CODE_ACTION all return the same `LoopResult` shape (content, stop_reason, iterations, cost_usd, token_usage, tool_calls). Exceptions from the provider or tools are typed (e.g. `ProviderError`, `ToolExecutionError`).
- **Model resolution:** `Model.OpenAI`, `Model.Anthropic`, etc., and the provider registry always resolve to a valid Provider when the name is known. Invalid `provider=` when using `ModelConfig` raises `ProviderNotFoundError` with a clear message. See [Model — Provider resolution and errors](agent/model.md#provider-resolution-and-errors).
- **Budget:** Per-run and per-period limits are enforced; threshold actions (warn, switch model, stop) are triggered when configured. See [Budget control](budget-control.md).
- **Rate limit thresholds:** When a rate limit threshold is hit, the threshold’s `action(ctx)` callback is run; implement desired behavior there (e.g. raise, wait, switch model). See [Rate limiting](ratelimit.md).
- **Memory:** Default persistent memory (Memory with InMemoryBackend) and conversation memory (e.g. BufferMemory) are stable; `remember`/`recall`/`forget` do not corrupt state. See [Memory](memory.md).
- **Response:** Every `Response` has `cost`, `tokens`, `tool_calls`, and `stop_reason` set correctly; guardrail or budget-exit paths still return a valid Response with the appropriate `stop_reason`.
- **Structured output:** When `output=Output(MyModel)` is set, validation runs and retries are applied; see [Structured output](agent/structured-output.md).

The test suite in `tests/unit/` and `tests/integration/` encodes these guarantees (lifecycle, loop strategies, model resolution, budget, memory, response contract, structured output). See `tests/README.md` for layout.

---

## Loop strategy comparison

| Strategy        | When to use it | Tool use | Typical flow |
|----------------|----------------|----------|---------------|
| **REACT**      | General agents that may call tools; one step per turn. | Yes | User → LLM → (tool calls or answer) → optional tool loop → final answer. |
| **SINGLE_SHOT**| No tools; one prompt, one completion. | No | User → LLM → answer. |
| **PLAN_EXECUTE**| Multi-step tasks: plan first, then execute steps. | Yes | User → plan → execute steps (each step can use tools). |
| **CODE_ACTION**| Agent emits code or actions to run (e.g. sandbox). | Yes | User → LLM → code/actions → execute → optional loop. |

All strategies return the same `LoopResult` shape (content, stop_reason, iterations, cost_usd, token_usage, tool_calls). Choose REACT for most agents with tools; SINGLE_SHOT when you have no tools; PLAN_EXECUTE for explicit planning; CODE_ACTION when the agent outputs code or structured actions.

---

## Lifecycle Hooks

Hooks let you execute code at specific moments during an agent's execution. This is essential for logging, metrics, authentication, and custom behaviors.

### Understanding Hooks

Syrin emits events at key moments:
- `agent.run.start` - Agent starts processing
- `llm.request.start` - About to call the LLM
- `llm.request.end` - LLM responded
- `tool.call.start` - Tool execution begins
- `tool.call.end` - Tool execution finished
- `agent.run.end` - Agent finished
- `budget.check` - Budget is being checked
- `tool.error` - An error occurred
- `handoff.start` / `handoff.end` / `handoff.blocked` - Agent handoff (see [Handoff & Spawn](agent/handoff-spawn.md))
- `spawn.start` / `spawn.end` - Sub-agent spawn
- `circuit.trip` / `circuit.reset` - Circuit breaker (see [Circuit Breaker](circuit-breaker.md))
- `hitl.pending` / `hitl.approved` / `hitl.rejected` - Human-in-the-loop (see [HITL](hitl.md))

### Basic Hook Usage

```python
from syrin import Agent, Model
from syrin.enums import Hook

# Create agent
agent = Agent(
    # model=Model.OpenAI("gpt-4o-mini"),
    model=Model.Almock(),  # No API Key needed
    system_prompt="You are a helpful assistant.",
)

# Define handler function
def on_complete(ctx):
    print(f"Agent finished!")
    print(f"  Cost: ${ctx.get('cost', 0):.4f}")
    print(f"  Iterations: {ctx.get('iterations', 1)}")
    print(f"  Content: {ctx.get('content', '')[:50]}...")

# Register handler for when the agent finishes
agent.events.on(Hook.AGENT_RUN_END, on_complete)

result = agent.response("What is Python?")
```

### Before and After Hooks

You can modify data *before* an event or take action *after*:

```python
from syrin import Agent, Model
from syrin.enums import Hook

agent = Agent(
    # model=Model.OpenAI("gpt-4o-mini"),
    model=Model.Almock(),  # No API Key needed
    system_prompt="You are a helpful assistant.",
)

# Define handler to modify request BEFORE LLM call
def modify_request(ctx):
    ctx["custom_flag"] = True
    ctx["request_id"] = "req_12345"
    # Can also modify temperature, etc.
    ctx["temperature"] = 0.7

# Register BEFORE handler
agent.events.before(Hook.LLM_REQUEST_START, modify_request)

# Define handler to log AFTER LLM responds
def log_response(ctx):
    print(f"Response received!")
    print(f"  Tokens: {ctx.get('token_usage', {}).get('total', 0)}")
    print(f"  Cost: ${ctx.get('cost', 0):.6f}")

# Register AFTER handler
agent.events.after(Hook.LLM_REQUEST_END, log_response)

result = agent.response("Hello!")
```

### Shortcut Methods

For common hooks, use shortcuts:

```python
agent.events.on_start(lambda ctx: print(f"Starting: {ctx.input}"))
agent.events.on_request(lambda ctx: print("LLM request"))
agent.events.on_response(lambda ctx: print(f"Response: {ctx.content[:50]}"))
agent.events.on_tool(lambda ctx: print(f"Tool: {ctx.name}"))
agent.events.on_error(lambda ctx: print(f"Error: {ctx.error}"))
agent.events.on_complete(lambda ctx: print(f"Done! Cost: ${ctx.cost}"))
agent.events.on_budget(lambda ctx: print(f"Budget: {ctx.remaining}"))
```

### Listening to All Events

```python
from syrin.enums import Hook

def handle_all(hook: Hook, ctx):
    print(f"Event: {hook.name}")
    # Do something with ctx

agent.events.on_all(handle_all)
```

### Handoff & Spawn Hooks

When using `handoff()` or `spawn()`, additional hooks are emitted for observability and interception:

- **HANDOFF_START** — Before transfer. Context: `source_agent`, `target_agent`, `task`, `mem_count`. Use `before(Hook.HANDOFF_START, fn)` to validate; raise `HandoffBlockedError` to block.
- **HANDOFF_END** — After target completes. Context: `cost`, `duration`, `response_preview` (first ~200 chars for debugging).
- **HANDOFF_BLOCKED** — When a before-handler blocks handoff. Context: `reason`, `task`, etc.
- **SPAWN_START** / **SPAWN_END** — Before/after child runs (when `task` given).

Example: observe what is passed on handoff:

```python
from syrin import Agent, Hook

source.events.on(Hook.HANDOFF_START, lambda ctx: print(f"Handoff: {ctx.source_agent} → {ctx.target_agent}, task={ctx.task[:50]}..."))
source.events.on(Hook.HANDOFF_END, lambda ctx: print(f"Cost: ${ctx.cost:.4f}, preview: {ctx.response_preview[:60]}..."))
```

See [Handoff & Spawn](agent/handoff-spawn.md) for full documentation, blocking, retry, and `examples/07_multi_agent/handoff_intercept.py`.

---

## Observability

Syrin provides built-in tracing to understand exactly what your agent is doing.

### Spans and Traces

```python
from syrin.observability import get_tracer, span, current_span

# Get the tracer
tracer = get_tracer()

# Create spans - they are nested automatically
def my_agent_function():
    with span("prepare_data"):
        # Do preparation
        pass
    
    with span("call_agent") as s:
        s.set_attribute("key", "value")
        result = agent.response("Hello")
        s.set_attribute("result_length", len(result.content))
    
    return result

# Access current span
with span("outer"):
    inner = current_span()
    if inner:
        inner.set_attribute("key", "value")
```

### Session

Sessions group related operations:

```python
from syrin.observability import get_tracer, session, current_session

# Get the tracer
tracer = get_tracer()

# Create a session
with tracer.session("user_conversation") as sess:
    result1 = agent.response("I need help")
    result2 = agent.response("Actually, I have another question")
    print(f"Session ID: {sess.id}")
    print(f"Span count: {sess.span_count}")
    
# Get current session
current = current_session()
if current:
    print(f"Session: {current.id}")
```

### Exporters

Export traces to different destinations. First, get the tracer and add exporters:

```python
from syrin.observability import get_tracer, ConsoleExporter, JSONLExporter, InMemoryExporter

# Get the tracer
tracer = get_tracer()

# Console - print to stdout
tracer.add_exporter(ConsoleExporter())

# JSONL - write to file
tracer.add_exporter(JSONLExporter("traces.jsonl"))

# InMemory - collect for later analysis
memory_exporter = InMemoryExporter()
tracer.add_exporter(memory_exporter)

# After running agent
for span in memory_exporter.spans:
    print(f"Span: {span.name}, Duration: {span.duration_ms}ms")

# Or enable debug mode on agent for automatic console output
agent = Agent(
    # model=Model.OpenAI("gpt-4o-mini"),
    model=Model.Almock(),  # No API Key needed
    debug=True,  # Enables trace output
)
```

### Hook + Observability Integration

Automatically connect hooks to observability:

```python
from syrin.observability.hooks import observe_hooks

# Connect events to spans
observe_hooks(agent)

# Now every hook event adds to the current span
result = agent.response("Hello")
# Spans automatically include event data!
```

---

## Checkpointing

Save and restore agent state for long-running tasks, recovery, or resumption.

### Basic Checkpointing

```python
from syrin import Agent, Model
from syrin.checkpoint import Checkpointer, CheckpointState

# Create checkpointer
checkpointer = Checkpointer()

agent = Agent(
    # model=Model.OpenAI("gpt-4o-mini"),
    model=Model.Almock(),  # No API Key needed
    system_prompt="You are a helpful assistant.",
)

# Save checkpoint
result = agent.response("Step 1: What's Python?")
checkpoint_id = checkpointer.save(
    agent_name="my_agent",
    state={
        "messages": agent.messages,
        "iteration": 1,
    }
)
print(f"Saved checkpoint: {checkpoint_id}")

# ... later ...

# Restore checkpoint
checkpoint = checkpointer.load(checkpoint_id)
if checkpoint:
    agent.messages = checkpoint.messages
    print(f"Restored from iteration {checkpoint.metadata['iteration']}")
```

### List and Manage Checkpoints

```python
# List all checkpoints for an agent
checkpoints = checkpointer.list_checkpoints("my_agent")
print(f"Found {len(checkpoints)} checkpoints: {checkpoints}")

# Delete old checkpoints
checkpointer.delete("my_agent_1")
checkpointer.delete("my_agent_2")
```

### Custom Backend

Implement your own storage:

```python
from syrin.checkpoint import CheckpointBackendProtocol, CheckpointState

class FileCheckpointBackend(CheckpointBackendProtocol):
    """Save checkpoints to files."""
    
    def __init__(self, directory: str = "./checkpoints"):
        self.directory = directory
        import os
        os.makedirs(directory, exist_ok=True)
    
    def save(self, state: CheckpointState) -> None:
        import json
        path = f"{self.directory}/{state.checkpoint_id}.json"
        with open(path, 'w') as f:
            f.write(state.model_dump_json())
    
    def load(self, checkpoint_id: str) -> CheckpointState | None:
        import json
        path = f"{self.directory}/{checkpoint_id}.json"
        try:
            with open(path, 'r') as f:
                return CheckpointState.model_validate_json(f.read())
        except FileNotFoundError:
            return None
    
    def list(self, agent_name: str) -> list[str]:
        import os
        return [f.replace('.json', '') 
                for f in os.listdir(self.directory) 
                if f.startswith(agent_name)]
    
    def delete(self, checkpoint_id: str) -> None:
        import os
        path = f"{self.directory}/{checkpoint_id}.json"
        if os.path.exists(path):
            os.remove(path)

# Use custom backend
checkpointer = Checkpointer(backend=FileCheckpointBackend())
```

---

## Guardrails

Guardrails validate input/output/action to enforce policies, block unwanted content, ensure quality, and provide structural authority control. Syrin Guardrails provides a comprehensive safety system with three layers:

1. **Foundation Layer**: Content filtering, PII detection, parallel evaluation
2. **Authority Layer**: Permission-based authorization, budget enforcement, human approval
3. **Intelligence Layer**: Context-aware tracking, escalation detection, adaptive thresholds, attack simulation

📚 **Full Documentation**: See [Guardrails Documentation](guardrails.md) for complete API reference and examples.

### Quick Start

```python
from syrin import Agent, Model
from syrin.guardrails import ContentFilter, PIIScanner

# Simple content filtering
agent = Agent(
    # model=Model.OpenAI("gpt-4o"),
    model=Model.Almock(),  # No API Key needed
    guardrails=[
        ContentFilter(blocked_words=["password", "secret"]),
        PIIScanner(redact=True),
    ]
)
```

### Modern Guardrail

```python
import asyncio
from syrin.guardrails import Guardrail, GuardrailContext, GuardrailDecision

class EmailGuardrail(Guardrail):
    """Async guardrail with rich context and decisions."""
    
    async def evaluate(self, context: GuardrailContext) -> GuardrailDecision:
        import re
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        
        if re.search(email_pattern, context.text):
            return GuardrailDecision(
                passed=False,
                rule="email_detected",
                reason="Contains email address",
                confidence=1.0,
                alternatives=["Use 'email' instead of actual address"],
            )
        
        return GuardrailDecision(passed=True)

# Use with async/await
async def main():
    guardrail = EmailGuardrail()
    context = GuardrailContext(text="Contact john@example.com")
    result = await guardrail.evaluate(context)
    print(f"Passed: {result.passed}")  # False
    print(f"Reason: {result.reason}")

asyncio.run(main())
```

### Authority Layer Example

```python
from syrin.guardrails import AuthorityCheck, BudgetEnforcer, ThresholdApproval

# Combine authority guardrails
guardrails = [
    AuthorityCheck(requires="finance:transfer"),
    BudgetEnforcer(max_amount=10000),
    ThresholdApproval(k=2, n=3),  # Require 2 of 3 approvers
]
```

### Intelligence Layer Example

```python
from syrin.guardrails.intelligence import (
    ContextAwareGuardrail,
    EscalationDetector,
    AttackSimulator,
)

# Context-aware protection with escalation detection
intelligence_guardrails = [
    ContextAwareGuardrail(max_history=5),
    EscalationDetector(violation_threshold=3),
]

# Test your guardrails with simulated attacks
simulator = AttackSimulator()
attacks = simulator.generate_jailbreak_attempts(target="sensitive info", count=10)
```

## Loop Strategies

Control how your agent iterates when handling tasks.

### Single Shot

One LLM call only - no tool iteration:

```python
from syrin import Agent, Model
from syrin.loop import SingleShotLoop

agent = Agent(
    # model=Model.OpenAI("gpt-4o-mini"),
    model=Model.Almock(),  # No API Key needed
    system_prompt="You are a helpful assistant.",
    loop=SingleShotLoop(),
)

# Single call, no matter what
result = agent.response("What is 2+2?")
```

### REACT Loop

Think → Act → Observe (default for tools):

```python
from syrin import Agent, Model
from syrin.loop import ReactLoop

agent = Agent(
    # model=Model.OpenAI("gpt-4o-mini"),
    model=Model.Almock(),  # No API Key needed
    system_prompt="You are a helpful assistant.",
    loop=ReactLoop(),  # Default when tools are present
    tools=[my_tool],
)
```

### Human in the Loop

Require human approval for tool calls:

```python
from syrin import Agent, Model
from syrin.loop import HumanInTheLoop

# Define approval function
async def approve_tool(tool_name: str, arguments: dict) -> bool:
    print(f"Tool '{tool_name}' wants to run with {arguments}")
    response = input("Approve? (y/n): ")
    return response.lower() == 'y'

agent = Agent(
    # model=Model.OpenAI("gpt-4o-mini"),
    model=Model.Almock(),  # No API Key needed
    system_prompt="You are a helpful assistant.",
    loop=HumanInTheLoop(approve_tool),
    tools=[dangerous_tool],
)

result = agent.response("Do the sensitive thing")
```

### Plan → Execute Loop

The `PlanExecuteLoop` is a 3-phase loop that first generates a plan, then executes each step, and finally reviews the results. Best for complex multi-step tasks.

```python
from syrin.loop import PlanExecuteLoop

agent = Agent(
    # model=Model.OpenAI("gpt-4o-mini"),
    model=Model.Almock(),  # No API Key needed
    loop=PlanExecuteLoop(
        max_plan_iterations=3,      # Max iterations for planning
        max_execution_iterations=20,  # Max iterations for execution
    ),
)

result = agent.response("Research AI trends and summarize findings")
print(result.content)
print(f"Iterations: {result.iterations}")  # Total iterations used
```

### Code Action Loop

The `CodeActionLoop` generates Python code, executes it, and interprets the results. Best for mathematical computations and data processing.

```python
from syrin.loop import CodeActionLoop

agent = Agent(
    # model=Model.OpenAI("gpt-4o-mini"),
    model=Model.Almock(),  # No API Key needed
    loop=CodeActionLoop(
        max_iterations=5,       # Max code generation attempts
        timeout_seconds=30,     # Code execution timeout
    ),
)

result = agent.response("What is the sum of even numbers from 1 to 100?")
print(result.content)  # "The sum is 2550"
```

### Custom Loop

Create your own iteration strategy:

```python
from syrin.loop import Loop, LoopResult

class MyCustomLoop(Loop):
    name = "my_custom"
    
    async def run(self, agent, user_input: str) -> LoopResult:
        # Custom logic
        messages = agent._build_messages(user_input)
        
        # Call LLM
        response = await agent.complete(messages)
        
        return LoopResult(
            content=response.content,
            stop_reason="custom",
            iterations=1,
        )

agent = Agent(
    # model=Model.OpenAI("gpt-4o-mini"),
    model=Model.Almock(),  # No API Key needed
    loop=MyCustomLoop(),
)
```

---

## Tracing

Detailed execution tracing for debugging and analysis.

### Basic Tracing

```python
from syrin.tracing import ConsoleTracer, TraceStep

tracer = ConsoleTracer(level="verbose")

# Manually add steps
tracer.add_llm_step(
    model="gpt-4o-mini",
    input_tokens=100,
    output_tokens=50,
    cost=0.001,
    duration=500.0,
)

tracer.add_tool_step(
    tool_name="search",
    tool_input={"query": "Python"},
    tool_output="Results...",
    duration=200.0,
)

# Export to console
from syrin.tracing import ConsoleExporter
exporter = ConsoleExporter()
exporter.export(tracer.steps)
```

---

## Complete Example: Production Agent

Putting it all together:

```python
from syrin import Agent, Model, Budget
from syrin.checkpoint import Checkpointer
from syrin.guardrails import GuardrailChain, ContentFilter
from syrin.loop import ReactLoop
from syrin.observability import trace, ConsoleExporter
from syrin.enums import Hook, RateWindow

# 1. Configure budget
from syrin import raise_on_exceeded
budget = Budget(
    run=10.0,  # $10 max per run
    on_exceeded=raise_on_exceeded,
)

# 2. Configure guardrails
guardrails = GuardrailChain([
    ContentFilter(blocked_words=["illegal", "harmful"]),
])

# 3. Configure observability - add exporters to tracer
from syrin.observability import get_tracer
tracer = get_tracer()
tracer.add_exporter(ConsoleExporter())

agent = Agent(
    # model=Model.OpenAI("gpt-4o"),
    model=Model.Almock(),  # No API Key needed
    system_prompt="You are a professional customer support agent.",
    budget=budget,
    guardrails=guardrails,
    loop=ReactLoop(),
)

# 4. Add hooks for monitoring
def on_start(ctx):
    print(f"Starting request: {ctx.input[:50]}...")

def on_end(ctx):
    print(f"Request complete! Cost: ${ctx.cost:.4f}")

def on_budget(ctx):
    print(f"Budget remaining: ${ctx.remaining:.2f}")

agent.events.on(Hook.AGENT_RUN_START, on_start)
agent.events.on(Hook.AGENT_RUN_END, on_end)
agent.events.on(Hook.BUDGET_CHECK, on_budget)

# 5. Checkpoint for recovery
checkpointer = Checkpointer()

# 6. Run with tracing
@trace
def handle_customer(query: str):
    # Save checkpoint before processing
    checkpoint_id = checkpointer.save(
        agent_name="support",
        state={"messages": agent.messages},
    )
    
    try:
        result = agent.response(query)
        return result.content
    except Exception as e:
        # Restore on error
        checkpoint = checkpointer.load(checkpoint_id)
        if checkpoint:
            agent.messages = checkpoint.messages
        raise

# Handle customer
response = handle_customer("I need help with my order #12345")
print(response)
```

---

## Reliability

### Circuit Breaker

Prevent cascading failures when the LLM provider is down. After N consecutive failures, the circuit trips and uses a fallback model or raises `CircuitBreakerOpenError`.

```python
from syrin import Agent, AgentConfig, CircuitBreaker, Model

cb = CircuitBreaker(failure_threshold=5, recovery_timeout=60, fallback=Model.Ollama("llama3.1"))
agent = Agent(model=model, config=AgentConfig(circuit_breaker=cb))
```

📚 **Full Documentation**: [Circuit Breaker](circuit-breaker.md)

### Human-in-the-Loop (HITL)

Gate tool execution behind human approval. Use `@syrin.tool(requires_approval=True)` for per-tool approval, or `HumanInTheLoop(approve=fn)` for all-tools approval.

```python
from syrin import Agent, AgentConfig, ApprovalGate, tool

@tool(requires_approval=True)
def delete_record(id: str) -> str: ...

gate = ApprovalGate(callback=lambda msg, t, ctx: input("Approve? [y/n]: ") == "y")
agent = Agent(
    model=model,
    tools=[delete_record],
    config=AgentConfig(approval_gate=gate),
    human_approval_timeout=300,
)
```

📚 **Full Documentation**: [HITL](hitl.md)

---

## Related Topics

- [Budget Control](budget-control.md) - Cost management
- [Multi-Agent](multi-agent.md) - Agent teams
- [Memory](agent-with-memory.md) - Persistent memory
- [Tools](research-agent-with-tools.md) - Tool system
- [Observability](observability.md) - Detailed tracing guide
