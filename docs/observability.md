# Observability

Syrin provides a comprehensive observability system that gives you deep visibility into how your agents operate. From a single LLM call to complex multi-agent workflows, every operation can be traced, measured, and analyzed.

## Why Observability Matters

When building AI agents, traditional debugging doesn't work. Your agent might:
- Make 12 LLM calls in a single response
- Use 8 different tools
- Retrieve memories from a vector store
- Fail on the last step

Without observability, you have no idea what happened. With Syrin's observability, you can see the complete execution path, understand costs, identify bottlenecks, and debug failures.

---

## Quick Start

The fastest way to get visibility into your agent:

```python
from syrin import Agent, Model

class MyAgent(Agent):
    model = Model("openai/gpt-4o-mini")
    system_prompt = "You are helpful."

# Just add debug=True
agent = MyAgent(debug=True)
result = agent.response("Hello!")

# You'll see a hierarchical trace in your console!
```

**Output:**
```
agent: MyAgent.response
  trace_id=a1b2c3d4 span_id=e5f6g7h8
  duration=1250.50ms status=ok
  attributes:
    agent.name=MyAgent
    input=Hello!
  llm: llm.iteration_1
    trace_id=a1b2c3d4 span_id=i9j0k1l2
    duration=1245.30ms status=ok
    attributes:
      llm.model=gpt-4o-mini
      llm.tokens.input=25
      llm.tokens.output=45
```

That's it! With one parameter, you get complete visibility.

---

## Core Concepts

### Spans

A **span** represents a single operation in your agent's execution. Spans form a tree hierarchy - an agent span might contain LLM spans, which contain tool spans.

```python
from syrin.observability import Span, SpanKind, SpanStatus

# Spans have:
span.name        # Name of the operation
span.kind        # Type: AGENT, LLM, TOOL, MEMORY, etc.
span.duration_ms # How long it took
span.status      # OK, ERROR, CANCELLED
span.attributes  # Key-value metadata
span.events      # Timestamped events within the span
span.children    # Child spans
```

### Span Kinds

Every span has a **kind** that identifies what type of operation it represents:

| Kind | Description |
|------|-------------|
| `SpanKind.AGENT` | Agent execution (the entire response flow) |
| `SpanKind.LLM` | LLM completion call |
| `SpanKind.TOOL` | Tool execution |
| `SpanKind.MEMORY` | Memory operation (remember/recall/forget) |
| `SpanKind.BUDGET` | Budget check/operation |
| `SpanKind.GUARDRAIL` | Guardrail validation |
| `SpanKind.HANDOFF` | Agent handoff to another agent |
| `SpanKind.WORKFLOW` | Custom workflow (user-defined) |
| `SpanKind.INTERNAL` | Internal framework operation (including `routing.decision`) |

### Fallback and Provider Error Traces

When a primary model fails and fallback is used, or when provider errors occur, `debug=True` records span events:

**Model fallback** (`Model.with_fallback()`): Events `llm.provider_error` and `llm.fallback` record the failed model, the model tried next, and the error.

**Circuit breaker fallback**: When the circuit breaker is open and a fallback model is used, `Hook.LLM_FALLBACK` is emitted with `from_model`, `to_model`, and `reason=circuit_breaker_open`.

**Provider errors**: Any provider failure (API error, rate limit, auth failure) is recorded on the span as `llm.provider_error` with `llm.provider_error.model`, `error.type`, and `error.message`.

Example trace with fallback:

```
llm: llm.iteration_1
  attributes:
    llm.model=anthropic/claude-sonnet
  events:
    - llm.provider_error: llm.provider_error.model=anthropic/claude-sonnet, error.type=ProviderError, error.message=...
    - llm.fallback: llm.fallback.from_model=anthropic/claude-sonnet, llm.fallback.to_model=openai/gpt-4o-mini, error.type=..., error.message=...
```

Use `Hook.LLM_FALLBACK` and `Hook.LLM_RETRY` for custom logging. See [Events & Hooks](agent/events-hooks.md).

### Routing Traces

When using Agent with a model list and routing, `debug=True` (or `--trace`) emits a `routing.decision` span showing which model was chosen and why:

```
internal: routing.decision
  trace_id=... span_id=...
  duration=...ms status=ok
  attributes:
    routing.model=code-model
    routing.model_id=almock/...
    routing.reason=COST_FIRST: selected cheapest capable model
    routing.task_type=CODE
    routing.cost_estimate=$0.000021
    routing.confidence=0.85
    routing.alternatives=general-model
```

Use `Hook.ROUTING_DECISION` for custom logging. See [Events & Hooks](agent/events-hooks.md) and [Routing](routing.md).

### Sessions

A **session** groups related spans together - typically a multi-turn conversation:

```python
from syrin.observability import session

# All spans within this block share the same session_id
with session("conversation_001", user="alice"):
    agent.response("Hi, my name is Bob")
    agent.response("What's my name?")  # Session knows context
    agent.response("Thanks!")           # Continues the conversation
# session.span_count is updated as spans are exported within the context
```

### Semantic Attributes

Semantic attributes are standardized keys that make your traces queryable and analyzable:

```python
from syrin.observability import SemanticAttributes

# Instead of arbitrary keys, use standardized ones:
span.set_attribute(SemanticAttributes.LLM_MODEL, "gpt-4o")
span.set_attribute(SemanticAttributes.LLM_TOKENS_TOTAL, 150)
span.set_attribute(SemanticAttributes.LLM_COST, 0.0045)
span.set_attribute(SemanticAttributes.TOOL_NAME, "calculator")
span.set_attribute(SemanticAttributes.BUDGET_REMAINING, 0.50)
```

**Available Semantic Attributes:**

| Category | Attributes |
|---------|------------|
| **Agent** | `AGENT_NAME`, `AGENT_CLASS`, `AGENT_ITERATION` |
| **LLM** | `LLM_MODEL`, `LLM_PROVIDER`, `LLM_PROMPT`, `LLM_COMPLETION`, `LLM_TOKENS_INPUT`, `LLM_TOKENS_OUTPUT`, `LLM_TOKENS_TOTAL`, `LLM_COST`, `LLM_TEMPERATURE`, `LLM_STOP_REASON` |
| **Tool** | `TOOL_NAME`, `TOOL_INPUT`, `TOOL_OUTPUT`, `TOOL_ERROR`, `TOOL_DURATION_MS` |
| **Memory** | `MEMORY_OPERATION`, `MEMORY_TYPE`, `MEMORY_QUERY`, `MEMORY_RESULTS_COUNT` |
| **Budget** | `BUDGET_LIMIT`, `BUDGET_USED`, `BUDGET_REMAINING`, `BUDGET_PERCENTAGE` |
| **Guardrail** | `GUARDRAIL_NAME`, `GUARDRAIL_STAGE`, `GUARDRAIL_PASSED`, `GUARDRAIL_VIOLATION` |
| **Handoff** | `HANDOFF_SOURCE`, `HANDOFF_TARGET`, `HANDOFF_MEMORIES_TRANSFERRED` |

---

## Manual Span Creation

For custom workflows, create your own spans:

```python
from syrin.observability import span, SpanKind

# Basic span
with span("my_workflow") as s:
    result = do_something()
    s.set_attribute("result", result)

# With kind and attributes
with span(
    "data_processing",
    kind=SpanKind.WORKFLOW,
    user_id="user_123",
    feature="batch_processing"
) as s:
    # Nested spans automatically form parent-child relationship
    with span("fetch_data", kind=SpanKind.INTERNAL) as fetch:
        data = fetch_from_db()
        fetch.set_attribute("rows", len(data))
    
    with span("process_data", kind=SpanKind.INTERNAL) as process:
        result = process_data(data)
        process.set_attribute("processed", len(result))
    
    s.set_attribute("total_rows", len(data))
```

**Output:**
```
workflow: data_processing
  trace_id=abc123 span_id=def456
  duration=1500.00ms status=ok
  attributes:
    user_id=user_123
    feature=batch_processing
    total_rows=1000
  internal: fetch_data
    trace_id=abc123 span_id=ghi789
    duration=500.00ms status=ok
    attributes:
      rows=1000
  internal: process_data
    trace_id=abc123 span_id=jkl012
    duration=1000.00ms status=ok
    attributes:
      processed=1000
```

---

## Convenience Decorators

Syrin provides quick ways to create spans for common operations:

### `llm_span()` - LLM Operations

```python
from syrin.observability import llm_span

with llm_span("gpt-4o", prompt="Hello world") as s:
    response = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": "Hello world"}]
    )
    s.set_attribute("completion", response.choices[0].message.content)
```

### `tool_span()` - Tool Operations

```python
from syrin.observability import tool_span

with tool_span("calculator", input={"expression": "2+2"}) as s:
    result = eval("2+2")
    s.set_attribute("output", result)
# Output: {"expression": "2+2"} -> "4"
```

### `memory_span()` - Memory Operations

```python
from syrin.observability import memory_span

with memory_span("recall", memory_type="episodic", query="user preferences") as s:
    results = memory.retrieve("user preferences")
    s.set_attribute(SemanticAttributes.MEMORY_RESULTS_COUNT, len(results))
```

### `budget_span()` - Budget Operations

```python
from syrin.observability import budget_span

with budget_span("check", limit=10.0, used=5.0) as s:
    remaining = limit - used
    s.set_attribute(SemanticAttributes.BUDGET_REMAINING, remaining)
```

### `guardrail_span()` - Guardrail Operations

```python
from syrin.observability import guardrail_span

with guardrail_span("content_filter", stage="input") as s:
    result = guardrail.check(user_input)
    s.set_attribute(SemanticAttributes.GUARDRAIL_PASSED, result.passed)
```

### `handoff_span()` - Agent Handoffs

```python
from syrin.observability import handoff_span

with handoff_span("triage_agent", "specialist_agent") as s:
    s.set_attribute(SemanticAttributes.HANDOFF_MEMORIES_TRANSFERRED, 5)
```

### `agent_span()` - Agent Operations

```python
from syrin.observability import agent_span

with agent_span("research_agent", user_id="user_123") as s:
    result = agent.response("Research AI")
    s.set_attribute("iterations", 3)
```

---

## Guardrails with Observability

Syrin automatically instruments guardrails when you add them to your agent:

```python
from syrin import Agent, Model
from syrin.guardrails import ContentFilter

# Create guardrail
blocked = ContentFilter(
    blocked_words=["badword", "forbidden"],
    name="content_filter"
)

class GuardedAgent(Agent):
    model = Model("openai/gpt-4o-mini")
    system_prompt = "You are helpful."
    guardrails = [blocked]

agent = GuardedAgent(debug=True)
result = agent.response("Hello, say badword please")
```

**Output shows the guardrail span:**
```
guardrail: guardrails.input
  trace_id=abc123 span_id=def456
  duration=5.20ms status=error
  attributes:
    guardrail.stage=input
    guardrail.passed=False
    guardrail.violation=Blocked word found: badword
```

The guardrail blocks the request and the span shows:
- What was validated (`guardrail.stage`)
- Whether it passed (`guardrail.passed`)
- What violation occurred (`guardrail.violation`)

---

## Memory Tracing

Memory operations (when using persistent memory) are automatically instrumented:

```python
from syrin import Agent, Model
from syrin.memory import Memory
from syrin.enums import MemoryType

class MemoryAgent(Agent):
    model = Model("openai/gpt-4o-mini")
    system_prompt = "You are helpful."
    persistent_memory = Memory(
        backend="memory",
    )

agent = MemoryAgent(debug=True)

# These operations are automatically traced:
agent.remember("User prefers dark mode", memory_type=MemoryType.SEMANTIC)
results = agent.recall(query="preferences")
agent.forget(query="old data")
```

**Output shows memory spans:**
```
memory: memory.store
  trace_id=abc123 span_id=def456
  duration=12.50ms status=ok
  attributes:
    memory.operation=store
    memory.type=semantic
    memory.id=mem_abc123

memory: memory.recall
  trace_id=abc123 span_id=ghi789
  duration=15.30ms status=ok
  attributes:
    memory.operation=recall
    memory.type=episodic
    memory.query=preferences
    memory.results.count=5

memory: memory.forget
  trace_id=abc123 span_id=jkl012
  duration=8.20ms status=ok
  attributes:
    memory.operation=forget
    memory.type=all
    memory.query=old data
    memory.deleted_count=3
```

---

## Metrics Aggregation

Syrin automatically aggregates metrics from spans. Get insights into cost, latency, and performance:

```python
from syrin.observability.metrics import get_metrics
from syrin import Agent, Model

# Clear previous metrics
metrics = get_metrics()
metrics.clear()

agent = Agent(
    model=Model("openai/gpt-4o-mini"),
    system_prompt="You are helpful."
)

# Run some requests
agent.response("Hello!")
agent.response("How are you?")
agent.response("What's 2+2?")

# Get metrics summary
summary = metrics.get_summary()

print(f"Total Cost: ${summary['llm']['cost']:.4f}")
print(f"Total Tokens: {summary['llm']['tokens_total']}")
print(f"Avg Latency: {summary['llm']['latency_avg']:.2f}ms")
print(f"p95 Latency: {summary['llm']['latency_p95']:.2f}ms")
print(f"Agent Runs: {summary['agent']['runs']}")
print(f"Errors: {summary['agent']['errors']}")
```

**Output:**
```
Total Cost: $0.0012
Total Tokens: 150
Avg Latency: 850.00ms
p95 Latency: 1200.00ms
Agent Runs: 3
Errors: 0
```

### Metrics schema

`get_metrics().get_summary(window)` returns a dict with:

- **counters** — All counter names and current values.
- **gauges** — All gauge names and current values.
- **llm** — `requests`, `tokens_input`, `tokens_output`, `tokens_total`, `cost`, `latency_avg`, `latency_p95`.
- **agent** — `runs`, `errors`, `cost`, `latency_avg`, `latency_p95`.
- **tool** — `calls` (and per-tool timing when present).
- **guardrail** — `passed`, `blocked` (counts).
- **memory** — `recalls`, `stores`, `forgets` (counts).

Pass a `timedelta` as `window` to restrict aggregation to recent data.

### Manual Metric Recording

You can also manually record metrics:

```python
from syrin.observability.metrics import get_metrics

metrics = get_metrics()

# Counters
metrics.increment("requests.total")
metrics.increment("errors.count", tags={"type": "timeout"})

# Gauges
metrics.gauge("queue.size", 42)
metrics.gauge("active.agents", 5)

# Timing
metrics.timing("db.query", 150.0)  # 150ms
metrics.timing("api.call", 320.0)    # 320ms

# Get aggregated values
avg_db_time = metrics.get("db.query", aggregate="avg")
p95_api_time = metrics.get("api.call", aggregate="p95")
```

---

## Sampling Strategies

For high-volume agents, you don't want to record every trace. Sampling lets you control what's recorded:

### Probabilistic Sampler

Sample a percentage of traces:

```python
from syrin.observability.sampling import (
    SamplingPolicy, ProbabilisticSampler, get_tracer
)

# Sample 10% of traces
policy = SamplingPolicy(rate=0.1)
sampler = ProbabilisticSampler(policy)

tracer = get_tracer()
tracer.set_sampler(sampler)
```

### Always Sample Errors

Configure to always capture errors even at low sampling rates:

```python
policy = SamplingPolicy(
    rate=0.1,                    # Sample 10%
    sample_errors=True,           # Always sample errors
    sample_slow_traces=True,      # Always sample slow traces (>5s)
    slow_threshold_ms=5000,
    sample_expensive=True,        # Always sample expensive traces (>$1)
    expensive_threshold_usd=1.0,
)
```

### Deterministic Sampler

Same trace ID always gets the same sampling decision:

```python
from syrin.observability.sampling import DeterministicSampler

policy = SamplingPolicy(rate=0.5)  # 50%
sampler = DeterministicSampler(policy)
```

### Rate Limiting Sampler

Limit samples per second:

```python
from syrin.observability.sampling import RateLimitingSampler

# Max 10 samples per second
sampler = RateLimitingSampler(
    SamplingPolicy(),
    max_samples_per_second=10.0
)
```

### Adaptive Sampler

Automatically adjusts sampling based on error rate:

```python
from syrin.observability.sampling import AdaptiveSampler

sampler = AdaptiveSampler(
    SamplingPolicy(target_error_rate=0.1),
    target_error_rate=0.1  # Want 10% error rate
)

# If errors increase, sampling increases to capture more data
# If errors decrease, sampling decreases to save resources
```

### Factory Function

Use the factory for quick creation:

```python
from syrin.observability.sampling import create_sampler

sampler = create_sampler(
    "probabilistic",
    rate=0.1,
    sample_errors=True
)
```

---

## Exporters

Spans can be sent to multiple destinations. Syrin provides several built-in exporters:

### Console Exporter

The default exporter for debug mode - prints human-readable traces:

```python
from syrin.observability import ConsoleExporter, get_tracer

tracer = get_tracer()
tracer.add_exporter(ConsoleExporter(colors=True))
```

**Output:**
```
agent: MyAgent.response
  trace_id=abc123 span_id=def456
  duration=1250.50ms status=ok
  attributes:
    agent.name=MyAgent
    input=Hello!
```

### In-Memory Exporter

Store spans in memory for testing or analysis:

```python
from syrin.observability import InMemoryExporter, get_tracer

exporter = InMemoryExporter()
tracer = get_tracer()
tracer.add_exporter(exporter)

# After running agent
for span in exporter.spans:
    print(span.name, span.duration_ms)

exporter.clear()  # Clear for next run
```

### JSONL Exporter

Append spans to a JSONL file:

```python
from syrin.observability import JSONLExporter, get_tracer

tracer = get_tracer()
tracer.add_exporter(JSONLExporter("traces.jsonl"))
```

Each line is a JSON object representing a span.

### Custom Exporter

Create your own exporter for any backend:

```python
from syrin.observability import Span, SpanExporter

class MyExporter(SpanExporter):
    def __init__(self):
        self.spans = []
    
    def export(self, span: Span) -> None:
        # Only store root spans (not children)
        if span.parent_span_id is None:
            self.spans.append(span)
            # Send to your backend...
            # send_to_datadog(span)
            # send_to_custom_backend(span)
```

---

## OpenTelemetry (OTLP) Export

Send spans to any OpenTelemetry-compatible backend:

```python
# Requires: pip install opentelemetry-exporter-otlp
from syrin.observability.otlp import OTLPExporter
from syrin.observability import get_tracer

exporter = OTLPExporter(
    endpoint="http://localhost:4318/v1/traces",
    headers={"Authorization": "Bearer token"},
    service_name="my-agent"
)

tracer = get_tracer()
tracer.add_exporter(exporter)
```

This works with:
- Jaeger
- Grafana Tempo
- Datadog
- Honeycomb
- Any OTLP-compatible backend

---

## Langfuse Integration

Export to Langfuse for AI-native observability:

```python
# Requires: pip install langfuse
from syrin.observability.langfuse import LangfuseExporter
from syrin.observability import get_tracer

exporter = LangfuseExporter(
    public_key="pk-...",
    secret_key="sk-...",
)

tracer = get_tracer()
tracer.add_exporter(exporter)
```

Langfuse provides:
- Trace visualization
- Prompt management
- Cost analytics
- Evaluation tools

---

## Phoenix (Arize) Integration

Local debugging with Phoenix:

```python
# Requires: pip install arize-phoenix
from syrin.observability.phoenix import PhoenixExporter
from syrin.observability import get_tracer

exporter = PhoenixExporter(
    project_name="my-agent",
    endpoint="http://localhost:6006"
)

tracer = get_tracer()
tracer.add_exporter(exporter)
```

Then open http://localhost:6006 to see your traces in the Phoenix UI.

---

## Hook Integration

Bridge the Events system with spans:

```python
from syrin.observability.hooks import observe_hooks
from syrin import Agent, Model

class MyAgent(Agent):
    model = Model("openai/gpt-4o-mini")

agent = MyAgent()

# Attach observability to events
observer = observe_hooks(agent)

# Register event handlers
agent.events.on("start", lambda ctx: print(f"Starting: {ctx.input}"))
agent.events.on("complete", lambda ctx: print(f"Done! Cost: ${ctx.cost}"))

result = agent.response("Hello!")
```

Events are automatically added to spans as events, giving you unified observability.

---

## Production Setup Example

Here's a complete production-ready setup:

```python
from syrin import Agent, Model
from syrin.guardrails import ContentFilter
from syrin.observability import (
    ConsoleExporter,
    InMemoryExporter,
    session,
    get_tracer,
)
from syrin.observability.metrics import get_metrics
from syrin.observability.sampling import create_sampler

# 1. Configure tracer
tracer = get_tracer()
tracer.clear()

# 2. Add exporters (console + in-memory)
tracer.add_exporter(ConsoleExporter(colors=True))
memory_exporter = InMemoryExporter()
tracer.add_exporter(memory_exporter)

# 3. Set up sampling
tracer.set_sampler(create_sampler(
    "probabilistic",
    rate=0.1,           # Sample 10%
    sample_errors=True,  # Always capture errors
))

# 4. Enable metrics
tracer.set_collect_metrics(True)
metrics = get_metrics()

# 5. Create agent with guardrails
blocked = ContentFilter(
    blocked_words=["spam", "forbidden"],
    name="content_filter"
)

class ProductionAgent(Agent):
    model = Model("openai/gpt-4o-mini")
    system_prompt = "You are a helpful assistant."
    guardrails = [blocked]

# 6. Run with session tracking
with session("prod_session_001", environment="prod", version="1.0"):
    agent = ProductionAgent()
    
    response = agent.response("Hello! Help me with coding.")
    print(f"Response: {response.content}")
    print(f"Cost: ${response.cost}")

# 7. Get metrics
summary = metrics.get_summary()
print(f"\nProduction Metrics:")
print(f"  Total Cost: ${summary['llm']['cost']:.4f}")
print(f"  Total Tokens: {summary['llm']['tokens_total']}")
print(f"  Latency p95: {summary['llm']['latency_p95']:.0f}ms")
print(f"  Errors: {summary['agent']['errors']}")
```

---

## Configuration Options

### Agent Options

```python
agent = Agent(
    model=Model("openai/gpt-4o-mini"),
    debug=True,                    # Enable debug mode
    tracer=get_tracer(),           # Custom tracer instance
)
```

### Global Configuration

```python
from syrin import configure, get_config

configure(
    trace=True,              # Enable tracing globally
    debug=True,              # Enable debug mode (prints traces to console)
)

# Or set debug mode globally
from syrin.observability import set_debug
set_debug(True)

# Read current config
assert get_config().trace is True
assert get_config().debug is True
```

---

## API Reference

### Core Classes

| Class | Description |
|-------|-------------|
| `Span` | A single operation in the trace |
| `SpanKind` | Enum for span types (AGENT, LLM, TOOL, etc.) |
| `SpanStatus` | Enum for span status (OK, ERROR, CANCELLED) |
| `SpanContext` | Propagation context (trace_id, span_id) |
| `Session` | Groups related spans |
| `Tracer` | Main class for creating spans |
| `SpanExporter` | Base class for exporters |

### Functions

| Function | Description |
|---------|-------------|
| `configure()` | Set global options: `trace`, `debug`, `default_model`, `default_api_key`. |
| `get_config()` | Return `GlobalConfig` singleton; `.trace`, `.debug`, `.default_model`, `.default_api_key`. |
| `span()` | Create a span context manager |
| `session()` | Create a session context manager |
| `current_span()` | Get the current span |
| `current_session()` | Get the current session |
| `set_debug()` | Enable/disable debug mode |
| `get_tracer()` | Get the global tracer |
| `llm_span()` | Convenience decorator for LLM spans |
| `tool_span()` | Convenience decorator for tool spans |
| `memory_span()` | Convenience decorator for memory spans |
| `budget_span()` | Convenience decorator for budget spans |
| `guardrail_span()` | Convenience decorator for guardrail spans |
| `handoff_span()` | Convenience decorator for handoff spans |
| `agent_span()` | Convenience decorator for agent spans |
| `get_metrics()` | Get the metrics collector |

### Exporters

| Exporter | Description |
|----------|-------------|
| `ConsoleExporter` | Print to console (human-readable) |
| `JSONLExporter` | Append to JSONL file |
| `InMemoryExporter` | Store in memory |
| `OTLPExporter` | Send to OpenTelemetry backend |
| `LangfuseExporter` | Send to Langfuse |
| `PhoenixExporter` | Send to Arize Phoenix |

---

## What's Traced by Default

When you create an agent with `debug=True` (or attach a tracer with exporters), Syrin automatically creates spans for:

1. **Agent execution** — One root span per `response()` / `arun()` with `SpanKind.AGENT`.
2. **LLM calls** — A child span per completion (SingleShot, REACT, PlanExecute, CodeAction, HumanInTheLoop) with `SpanKind.LLM`, tokens and model attributes.
3. **Tool executions** — A child span per tool call with `SpanKind.TOOL`, name, input, and output.
4. **Memory operations** — When using persistent memory: `remember`, `recall`, `forget` each create a `SpanKind.MEMORY` span.
5. **Guardrail checks** — Input/output guardrails create `SpanKind.GUARDRAIL` spans with stage and passed/violation.
6. **Budget checks** — (Optional) Budget threshold/exceeded can be wrapped in spans; cost is recorded on agent/LLM spans.

Spans form a tree: agent → LLM → tool (and agent → guardrail, agent → memory). Every span includes:

- Start/end timestamps and duration
- Status (ok / error / cancelled)
- Semantic attributes (e.g. `llm.model`, `tool.name`, `memory.operation`)
- Session ID when used inside `session()`

---

## Best Practices

1. **Start with debug mode** - It's the easiest way to see what's happening

2. **Use semantic attributes** - They make traces queryable:
   ```python
   span.set_attribute(SemanticAttributes.LLM_MODEL, "gpt-4o")
   ```

3. **Group with sessions** - For multi-turn conversations:
   ```python
   with session("conversation_001"):
       # All spans linked together
   ```

4. **Set up sampling in production** - Don't record everything:
   ```python
   sampler = create_sampler("probabilistic", rate=0.1)
   ```

5. **Export to external services** - For production monitoring:
   ```python
   tracer.add_exporter(OTLPExporter(endpoint="..."))
   ```

6. **Monitor metrics** - Track cost and latency:
   ```python
   summary = get_metrics().get_summary()
   ```

---

## Troubleshooting

### No spans being captured?

1. Check if debug mode is enabled:
   ```python
   agent = Agent(..., debug=True)
   ```

2. Check if exporters are added:
   ```python
   tracer = get_tracer()
   print(tracer._exporters)  # Should have exporters
   ```

### Spans not showing in console?

1. Make sure ConsoleExporter is added:
   ```python
   tracer.add_exporter(ConsoleExporter())
   ```

### Metrics showing zero?

1. Ensure metrics collection is enabled:
   ```python
   tracer.set_collect_metrics(True)
   ```

2. Check if spans are being exported (metrics come from spans)

### Performance concerns?

1. Use sampling:
   ```python
   sampler = create_sampler("probabilistic", rate=0.1)
   tracer.set_sampler(sampler)
   ```

2. Disable console output in production:
   ```python
   # Don't add ConsoleExporter in production
   ```

---

## Examples

See `examples/advanced/observability_comprehensive.py` for complete working examples of:

1. Debug Mode
2. Manual Spans
3. Session Tracking
4. Convenience Decorators
5. Guardrail Tracing
6. Memory Tracing
7. Metrics Aggregation
8. Sampling Strategies
9. Custom Exporters
10. Hook Integration
11. Semantic Attributes
12. Production Setup

---

## Auto-tracing with --trace flag

When you run any Syrin script with `--trace`, observability is automatically
enabled and you'll see detailed execution logs in your terminal.

```bash
# Run your script normally
python my_agent.py

# Run with full observability
python my_agent.py --trace
```

# Run - all events will be captured and printed
result = agent.response("Hello")

# Print summary at the end
debugger.print_summary()
```

**Example with an Agent:**

```python
from syrin import Agent, Model
from syrin.cli import WorkflowDebugger
from syrin.tool import tool

@tool(name="calculator", description="Perform calculations")
def calculator(expression: str) -> str:
    try:
        result = eval(expression)
        return f"Result: {result}"
    except Exception as e:
        return f"Error: {e}"

# Create debugger
debugger = WorkflowDebugger(verbose=True)

# Create agent and attach debugger
agent = Agent(
    # model=Model.LiteLLM("gpt-4o-mini"),
    model=Model.Almock(),  # No API Key needed
    system_prompt="You are a helpful assistant with access to tools.",
    tools=[calculator],
)

debugger.attach(agent)

# Run the agent
result = agent.response("Calculate 123 * 456")

# Print summary
debugger.print_summary()
```

**Example with DynamicPipeline:**

```python
from syrin.agent.multi_agent import DynamicPipeline
from syrin.cli import WorkflowDebugger

# Create debugger
debugger = WorkflowDebugger(verbose=True)

# Create pipeline
pipeline = DynamicPipeline(
    agents=[ResearchAgent, AnalysisAgent, WriterAgent],
    # model=Model.OpenAI("gpt-4o"),
    model=Model.Almock(),  # No API Key needed
)

# Attach debugger
debugger.attach(pipeline)

# Run with full observability
result = pipeline.run("Research and write about AI", mode="parallel")

# Print summary
debugger.print_summary()
```

### Silent Mode

If you want to capture events without printing to terminal:

```python
# Silent mode - capture only, no output
debugger = WorkflowDebugger(verbose=False)
debugger.attach(agent)

result = agent.response("Hello")

# Analyze events programmatically
for event in debugger.events:
    if "TOOL_CALL" in event.hook.value:
        print(f"Tool called: {event.data.get('tool_name')}")

# Print summary at the end
debugger.print_summary()
```

### Export to JSONL

Export captured events to a JSONL file for further analysis:

```python
debugger = WorkflowDebugger()
debugger.attach(agent)

result = agent.response("Hello")

# Export to file
debugger.export_jsonl("/tmp/debug_trace.jsonl")
```

### Running Scripts with Tracing

```bash
# Run normally
python my_agent.py

# Run with tracing
python my_agent.py --trace

# Pass arguments to script
python my_agent.py --trace -- --arg1 --arg2
```

### What Gets Captured

The debugger captures all lifecycle events:

- **Agent events:** AGENT_RUN_START, AGENT_RUN_END
- **LLM events:** LLM_REQUEST_START, LLM_REQUEST_END
- **Tool events:** TOOL_CALL_START, TOOL_CALL_END, TOOL_ERROR
- **Pipeline events:** DYNAMIC_PIPELINE_START, AGENT_SPAWN, AGENT_COMPLETE, etc.
- **Budget events:** BUDGET_CHECK, BUDGET_THRESHOLD, BUDGET_EXCEEDED
- **Error events:** All error hooks

### Output Format

Events are printed with color coding:

- 🟢 **Green (▶):** Start/init events
- 🔵 **Blue (✓):** End/complete events
- 🔷 **Cyan (→):** Spawn/handoff events
- 🟡 **Yellow (🔧):** Tool calls
- ⚪ **White (💬):** LLM calls
- 🔴 **Red (✗):** Errors
- 🟣 **Magenta (◉):** Plan/check events

Example output:

```
▶ 14:32:10.123 AGENT_RUN_START
     Agent: my_agent

💬 14:32:10.234 LLM_REQUEST_START
     Model: gpt-4o-mini

💬 14:32:12.456 LLM_REQUEST_END
     Duration: 2222.00ms

🔧 14:32:12.567 TOOL_CALL_START
     Tool: calculator

🔧 14:32:12.678 TOOL_CALL_END
     Duration: 111.00ms

✓ 14:32:12.789 AGENT_RUN_END
     Duration: 2666.00ms

======================================================================
WORKFLOW EXECUTION SUMMARY
======================================================================

Events captured:
  AGENT_RUN_START: 1
  AGENT_RUN_END: 1
  LLM_REQUEST_START: 1
  LLM_REQUEST_END: 1
  TOOL_CALL_START: 1
  TOOL_CALL_END: 1

Statistics:
  Agent runs: 1 started, 1 completed
  Tool calls: 2
  LLM calls: 1
  Errors: 0
  Total cost: $0.0012
  Total events: 6
```

### Comparison: Before vs After

**Before (manual hook registration):**

```python
class PipelineDebugger:
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.events_log = []

    def log(self, hook: Hook, ctx: dict):
        # ... 20+ lines of manual logging code ...

    def _print_event(self, timestamp, hook, ctx):
        # ... 30+ lines of manual formatting ...

    def print_summary(self):
        # ... 40+ lines of manual summary ...

# Usage - requires manual hook registration
debugger = PipelineDebugger()
pipeline.events.on(Hook.DYNAMIC_PIPELINE_START, lambda ctx: debugger.log(...))
pipeline.events.on(Hook.DYNAMIC_PIPELINE_PLAN, lambda ctx: debugger.log(...))
# ... register all 7 hooks manually ...
```

**After (with built-in WorkflowDebugger):**

```python
from syrin.cli import WorkflowDebugger

# Usage - just attach and go!
debugger = WorkflowDebugger()
debugger.attach(pipeline)  # All hooks registered automatically
```

### Benefits

1. **No custom code needed:** Use `WorkflowDebugger` instead of writing your own
2. **Auto-discovery:** Automatically attaches to all available hooks
3. **Beautiful output:** Color-coded terminal output with clear formatting
4. **CLI integration:** Run with `--trace` or `syrin trace script.py`
5. **Export support:** Save traces to JSONL for analysis
6. **Silent mode:** Capture events without printing for programmatic analysis
