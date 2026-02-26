# DynamicPipeline

DynamicPipeline is a powerful multi-agent orchestration system where an LLM decides how many agents to spawn and what they should do. This enables truly autonomous agentic workflows where the AI itself determines the execution strategy.

## Overview

Unlike static pipelines where you define the execution order upfront, DynamicPipeline uses a **two-step approach**:

1. **Planning Phase**: LLM analyzes the task and generates a JSON plan specifying which agents to spawn and what they should do
2. **Execution Phase**: Pipeline executes the planned agents in parallel or sequential mode

This approach is more deterministic than tool-based agent spawning because it doesn't rely on the LLM to use a tool - it simply returns a structured plan.

## Key Features

- **LLM-driven orchestration**: The LLM decides which agents are needed and what they should do
- **Parallel execution**: Spawn multiple agents simultaneously (up to `max_parallel` limit)
- **Sequential execution**: Pass context from one agent to the next
- **Full observability**: Hooks at every stage for debugging and monitoring
- **Budget control**: Optional budget shared across all spawned agents
- **TOON format support**: Compact schema format for internal communication (~40% fewer tokens)

## Basic Usage

```python
from Syrin import Agent, Model
from Syrin.agent.multi_agent import DynamicPipeline
from Syrin.enums import Hook

# Define specialized agents
class ResearcherAgent(Agent):
    _syrin_name = "researcher"  # Custom name for the LLM to reference
    model = Model(provider="openai", model_id="gpt-4o-mini")
    system_prompt = "You research and gather information."

class WriterAgent(Agent):
    _syrin_name = "writer"
    model = Model(provider="openai", model_id="gpt-4o-mini")
    system_prompt = "You write reports."

# Create pipeline
pipeline = DynamicPipeline(
    agents=[ResearcherAgent, WriterAgent],
    model=Model(provider="openai", model_id="gpt-4o-mini"),
    max_parallel=5,
)

# Run with a task - LLM decides which agents to spawn
result = pipeline.run("Research AI trends and write a summary")
print(result.content)
```

## Agent Naming

Agents can be referenced by:
- **Class name** (lowercase): `ResearcherAgent` → `researcheragent`
- **`_syrin_name` attribute**: `class Agent: _syrin_name = "researcher"` → `researcher`

The LLM will use these names in the plan:

```json
[
  {"type": "researcher", "task": "Research AI in healthcare"},
  {"type": "writer", "task": "Write a summary based on the research"}
]
```

## Observability & Hooks

DynamicPipeline emits hooks at every lifecycle stage:

### Available Hooks

| Hook | Description | Context |
|------|-------------|---------|
| `DYNAMIC_PIPELINE_START` | Pipeline execution begins | `task`, `mode`, `model`, `available_agents`, `budget_remaining` |
| `DYNAMIC_PIPELINE_PLAN` | LLM has generated the plan | `plan`, `plan_count` |
| `DYNAMIC_PIPELINE_EXECUTE` | Starting agent execution | `plan`, `plan_count`, `mode` |
| `DYNAMIC_PIPELINE_AGENT_SPAWN` | About to spawn an agent | `agent_type`, `task`, `spawn_time`, `execution_mode` |
| `DYNAMIC_PIPELINE_AGENT_COMPLETE` | Agent finished execution | `agent_type`, `task`, `cost`, `tokens`, `duration` |
| `DYNAMIC_PIPELINE_END` | Pipeline completed | `agents_spawned`, `total_cost`, `total_tokens`, `duration` |
| `DYNAMIC_PIPELINE_ERROR` | Error occurred | `error`, `error_type`, `agents_spawned`, `total_cost` |

### Registering Hooks

```python
from Syrin.enums import Hook

# Simple handler
pipeline.events.on(Hook.DYNAMIC_PIPELINE_AGENT_SPAWN, 
    lambda ctx: print(f"Spawning: {ctx.agent_type}"))

# Before handler (can modify context)
pipeline.events.before(Hook.DYNAMIC_PIPELINE_AGENT_SPAWN,
    lambda ctx: ctx.update({"modified": True}))

# After handler
pipeline.events.after(Hook.DYNAMIC_PIPELINE_AGENT_COMPLETE,
    lambda ctx: print(f"Cost so far: ${ctx.cost:.4f}"))

# All events
pipeline.events.on_all(lambda hook, ctx: print(f"{hook}: {ctx.task}"))
```

### Developer Debugging Example

```python
class PipelineDebugger:
    def __init__(self):
        self.events = []
    
    def log(self, hook, ctx):
        self.events.append({
            "hook": hook.value,
            "timestamp": time.time(),
            "data": dict(ctx)
        })
        print(f"[{hook.value}] {ctx.get('agent_type', 'N/A')}")

debugger = PipelineDebugger()
pipeline.events.on(Hook.DYNAMIC_PIPELINE_START, debugger.log)
pipeline.events.on(Hook.DYNAMIC_PIPELINE_AGENT_SPAWN, debugger.log)
pipeline.events.on(Hook.DYNAMIC_PIPELINE_END, debugger.log)
```

## Execution Modes

### Parallel Mode (Default)

All agents run simultaneously. Results are combined at the end.

```python
result = pipeline.run("Complex task", mode="parallel")
# Agents run at the same time
# Result.content contains combined output from all agents
```

### Sequential Mode

Agents run one after another. Each agent receives the previous agent's output as context.

```python
result = pipeline.run("Complex task", mode="sequential")
# Agent 1 runs → output passed to Agent 2
# Agent 2 runs with context → output passed to Agent 3
# ...and so on
```

## Configuration

### Required Parameters

- **`agents`**: List of Agent classes available for spawning
- **`model`**: Model instance for the orchestrator agent (REQUIRED)

### Optional Parameters

- **`budget`**: Budget instance shared across all spawned agents
- **`format`**: Communication format (`"toon"`, `"json"`, `"yaml"`)
- **`max_parallel`**: Maximum agents to spawn in parallel (default: 10)

```python
from Syrin import Budget
from Syrin.enums import DocFormat

pipeline = DynamicPipeline(
    agents=[ResearcherAgent, WriterAgent],
    model=Model(provider="openai", model_id="gpt-4o"),
    budget=Budget(run=5.0),  # Max $5 total
    format=DocFormat.TOON,   # Compact format
    max_parallel=3,          # Max 3 agents at once
)
```

## Complete Example

```python
"""Market research with full observability."""
from Syrin import Agent, Model
from Syrin.agent.multi_agent import DynamicPipeline
from Syrin.enums import Hook

# Define agents
class TechResearchAgent(Agent):
    _syrin_name = "tech"
    model = Model(provider="openai", model_id="gpt-4o-mini")
    system_prompt = "Research technology trends."

class FinanceResearchAgent(Agent):
    _syrin_name = "finance"
    model = Model(provider="openai", model_id="gpt-4o-mini")
    system_prompt = "Research financial metrics."

# Create pipeline
pipeline = DynamicPipeline(
    agents=[TechResearchAgent, FinanceResearchAgent],
    model=Model(provider="openai", model_id="gpt-4o-mini"),
    max_parallel=5,
)

# Add observability
pipeline.events.on(Hook.DYNAMIC_PIPELINE_START, 
    lambda ctx: print(f"Starting: {ctx.task[:50]}..."))
pipeline.events.on(Hook.DYNAMIC_PIPELINE_PLAN, 
    lambda ctx: print(f"Plan: {ctx.plan_count} agents"))
pipeline.events.on(Hook.DYNAMIC_PIPELINE_AGENT_SPAWN, 
    lambda ctx: print(f"→ Spawning: {ctx.agent_type}"))
pipeline.events.on(Hook.DYNAMIC_PIPELINE_AGENT_COMPLETE, 
    lambda ctx: print(f"✓ Completed: {ctx.agent_type} (${ctx.cost:.4f})"))
pipeline.events.on(Hook.DYNAMIC_PIPELINE_END, 
    lambda ctx: print(f"Done! Total: ${ctx.total_cost:.4f}"))

# Run
result = pipeline.run("Research AI in healthcare")
print(result.content)
```

## Error Handling

The pipeline handles errors gracefully:

- **Unknown agent types**: Skipped with warning
- **Empty plan**: Returns "No agents to spawn"
- **Agent failures**: Logged but don't stop other agents
- **Exceptions**: Emits `DYNAMIC_PIPELINE_ERROR` hook before raising

## Best Practices

1. **Use `_syrin_name`**: Give agents descriptive, easy-to-reference names
2. **Set `max_parallel`**: Don't exceed your rate limits
3. **Add hooks**: Always add observability for production use
4. **Use `before()` handlers**: For validation, logging, or context modification
5. **Test with small tasks first**: Verify your agents work as expected

## Comparison with Static Pipeline

| Feature | Static Pipeline | DynamicPipeline |
|---------|----------------|-----------------|
| Execution order | Fixed upfront | LLM decides |
| Flexibility | Low | High |
| Observability | Hooks | Hooks |
| Complexity | Simple | More complex |
| Use case | Known workflows | Exploratory tasks |

## Architecture

```
┌─────────────────────────────────────────┐
│           User Task                     │
└────────────────┬────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────┐
│  Step 1: Planning Phase                 │
│  - LLM analyzes task                    │
│  - Generates JSON plan                  │
│  - Hook: DYNAMIC_PIPELINE_PLAN          │
└────────────────┬────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────┐
│  Step 2: Execution Phase                │
│  - Parse plan                           │
│  - Spawn agents                         │
│  - Hook: DYNAMIC_PIPELINE_AGENT_SPAWN   │
│  - Collect results                      │
│  - Hook: DYNAMIC_PIPELINE_AGENT_COMPLETE│
└────────────────┬────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────┐
│  Step 3: Completion                     │
│  - Consolidate results                  │
│  - Hook: DYNAMIC_PIPELINE_END           │
└─────────────────────────────────────────┘
```

## Performance Considerations

- **Parallel execution**: Faster but uses more API calls simultaneously
- **Sequential execution**: Slower but uses fewer concurrent API calls
- **TOON format**: ~40% fewer tokens than JSON for plan communication
- **Hook overhead**: Minimal, hooks are synchronous and fast

## Advanced: Custom Hook Handlers

```python
# Modify context before agent spawns
def validate_task(ctx):
    if len(ctx.get("task", "")) > 1000:
        ctx["task"] = ctx["task"][:1000] + "... [truncated]"

pipeline.events.before(Hook.DYNAMIC_PIPELINE_AGENT_SPAWN, validate_task)

# Track cumulative costs
cumulative_cost = 0.0

def track_cost(ctx):
    global cumulative_cost
    cumulative_cost += ctx.get("cost", 0)
    print(f"Running total: ${cumulative_cost:.4f}")

pipeline.events.after(Hook.DYNAMIC_PIPELINE_AGENT_COMPLETE, track_cost)
```

## See Also

- `examples/07_multi_agent/dynamic_pipeline_basic.py` - Minimal dynamic pipeline
- `examples/07_multi_agent/dynamic_pipeline_full.py` - Full demo with multiple agents
- `tests/unit/agent/test_multi_agent.py` - Comprehensive test suite
- `Syrin.enums.Hook` - All available hooks
