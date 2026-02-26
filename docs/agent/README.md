# Agent Documentation

Complete documentation for creating and configuring AI agents in Syrin. This directory covers every feature available to end users.

> **Agent vs standalone:** Some components (Model, Guardrails, BudgetStore) work without an agent. See [Architecture](../ARCHITECTURE.md) for the full mapping.

## Quick Start

```python
from syrin import Agent, Model

agent = Agent(
    # model=Model.OpenAI("gpt-4o-mini"),
    model=Model.Almock(),  # No API Key needed
    system_prompt="You are a helpful assistant.",
)
response = agent.response("Hello!")
print(response.content)
```

## Documentation Index

### Core Concepts

- **[Overview](overview.md)** — What is an agent, architecture, key concepts
- **[Creating Agents](creating-agents.md)** — Class vs instance, inheritance, MRO merge rules
- **[Constructor Reference](constructor.md)** — Every parameter explained in full
- **[Model](model.md)** — Model param, `switch_model()`, budget/rate-limit integration

### Execution

- **[Running Agents](running.md)** — `response()`, `arun()`, `stream()`, `astream()`
- **[Response Object](response.md)** — Fields, properties, structured output, reports
- **[Loop Strategies](loop-strategies.md)** — REACT, SingleShot, HumanInTheLoop, PlanExecute, CodeAction, custom loops

### Capabilities

- **[Tools](tools.md)** — Adding tools, tool execution, tool specs
- **[Memory](memory.md)** — Persistent memory, `remember()`, `recall()`, `forget()`, Memory
- **[Structured Output](structured-output.md)** — Output config, validation, Pydantic models

### Observability & Control

- **[Events & Hooks](events-hooks.md)** — Lifecycle events, `agent.events`, EventContext, all hooks
- **[Checkpointing](checkpointing.md)** — Save/load state, triggers, configuration
- **[Budget](budget.md)** — Cost limits, thresholds, budget store
- **[Guardrails](guardrails.md)** — Input/output validation
- **[Rate Limiting](rate-limiting.md)** — API rate limits, thresholds

### Multi-Agent

- **[Handoff & Spawn](handoff-spawn.md)** — `handoff()`, `spawn()`, `spawn_parallel()`
- **[Multi-Agent Patterns](multi-agent-patterns.md)** — Pipeline, AgentTeam, DynamicPipeline, `parallel()`, `sequential()`

### Reference

- **[Properties & State](properties.md)** — All agent properties (`budget_state`, `tools`, `model_config`, `memory`, `report`, etc.)
- **[Complete API Reference](api-reference.md)** — Full method and parameter reference

## Documentation by Use Case

| I want to... | See |
|--------------|-----|
| Create a basic agent | [Creating Agents](creating-agents.md), [Constructor](constructor.md) |
| Choose or change model | [Model](model.md) |
| Run an agent (sync/async/stream) | [Running Agents](running.md) |
| Add tools | [Tools](tools.md) |
| Use memory | [Memory](memory.md) |
| Control costs | [Budget](budget.md) |
| Validate input/output | [Guardrails](guardrails.md) |
| Get structured (Pydantic) output | [Structured Output](structured-output.md) |
| Hook into lifecycle | [Events & Hooks](events-hooks.md) |
| Save/restore state | [Checkpointing](checkpointing.md) |
| Limit API usage | [Rate Limiting](rate-limiting.md) |
| Delegate to another agent | [Handoff & Spawn](handoff-spawn.md) |
| Orchestrate multiple agents | [Multi-Agent Patterns](multi-agent-patterns.md) |
| Change loop behavior | [Loop Strategies](loop-strategies.md) |
| Inspect run metrics | [Response](response.md), [Properties](properties.md) |
