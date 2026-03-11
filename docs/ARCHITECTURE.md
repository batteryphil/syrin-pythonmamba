# Syrin architecture

High-level package layout, dependency direction, and code-quality conventions.

## 1. Package layout

```
src/syrin/
├── agent/           # Agent, run loop orchestration, context builder
├── budget/          # Budget, limits, thresholds
├── budget_store/    # Budget persistence (in-memory, file)
├── checkpoint/     # Checkpoint config, backends, checkpointer
├── cli/             # CLI, workflow debugger, --trace
├── config.py        # GlobalConfig, configure(), get_config()
├── context/         # Context, ContextManager, compactors, token counter
├── cost/            # Pricing, token counting, cost calculation
├── domain_events.py # DomainEvent, EventBus
├── enums.py         # All StrEnum options (Hook, LoopStrategy, etc.)
├── events.py        # EventContext, Events
├── exceptions.py    # SyrinError hierarchy
├── guardrails/      # Guardrail, chain, built-ins, intelligence
├── loop.py          # Loop, REACT, SINGLE_SHOT, etc.
├── memory/          # Memory, backends, store, conversation
├── model/           # Model, providers, structured output
├── observability/   # Tracer, Span, Session, exporters
├── output.py        # Output type wrapper
├── prompt/          # Prompt, prompt(), validated()
├── providers/       # Provider protocol, registry, OpenAI/Anthropic/LiteLLM
├── ratelimit/       # Rate limit manager, backends
├── response/        # Response, reports
├── task.py          # task()
├── threshold.py     # ThresholdContext, BudgetThreshold, etc.
├── tool/            # tool(), ToolSpec (schema_to_toon, to_format)
├── types/           # ModelConfig, Message, TaskSpec, etc.
└── validation.py    # ValidationPipeline, validate_output
```

## 2. Dependency direction (layering)

- **Bottom:** `types`, `enums`, `exceptions`, `cost`
- **Mid:** `providers`, `memory` (backends), `budget`, `context`, `observability` (core)
- **Upper:** `model`, `loop`, `guardrails`, `checkpoint`, `ratelimit`, `threshold`
- **Top:** `agent`, `cli`, public `syrin` API

Dependencies flow bottom → mid → upper → top. No reverse dependencies. See `tests/architecture/test_layering.py`.

## 3. Event flow

- One event emission path: e.g. `_emit_event` in agent; hooks and domain events (EventBus) are documented in `docs/observability.md` and `docs/event-bus.md`.
- Hooks use the `Hook` enum only; no ad-hoc string events in public API.

## 4. Extension points (how to plug)

| Extension        | Interface              | Location / pattern                    |
|-----------------|------------------------|----------------------------------------|
| Provider        | `Provider` (Protocol)  | `syrin/providers/base.py`, registry   |
| Loop            | `Loop` (Protocol)       | `syrin/loop.py`, use `AgentRunContext` |
| BudgetStore     | `BudgetStore`          | `syrin/budget_store/`                 |
| Memory backend  | `MemoryBackend` (Protocol) | `syrin/memory/backends/`         |
| Checkpoint      | `CheckpointBackendProtocol` | `syrin/checkpoint/`              |
| SpanExporter    | `SpanExporter` (Protocol)   | `syrin/observability/`            |
| Guardrail       | `Guardrail` (Protocol)  | `syrin/guardrails/`                   |
| ContextCompactor| `ContextCompactorProtocol` | `syrin/context/compactors.py`     |
| RateLimit       | `RateLimitBackend`     | `syrin/ratelimit/backends.py`         |

See `docs/extension-points.md` for implementation examples.

## 5. Model vs ModelConfig vs Provider

- **Model:** User-facing. Use `Model.OpenAI(...)`, `Model.Anthropic(...)`, or subclass.
- **ModelConfig:** Internal config (provider id, model_id, etc.). Prefer `Model.to_config()` rather than constructing `ModelConfig` directly when wiring agents.
- **Provider:** Execution backend. Resolution: `Model` → `get_provider()` or registry. Single path; no duplicate resolution logic.

## 6. Consistent patterns (code quality)

### Naming

- **`*Protocol`** — Typing `Protocol` for extension interfaces (e.g. `SpanExporter`, `CheckpointBackendProtocol`).
- **`*Backend`** — Concrete or abstract backend implementations (e.g. `InMemoryBackend`, `RateLimitBackend`).
- **`*Config`** — Pydantic or frozen config objects (e.g. `CheckpointConfig`, `Context` in context/config).
- **`*Store`** / **`*Manager`** — Runtime state or service (e.g. `BudgetStore`, `RateLimitManager`, `ContextManager`).

### Error handling

- Use **exceptions** for control flow. All public exceptions extend `SyrinError` (see `syrin/exceptions.py` and API_DESIGN §26).
- Do not use Result types in public API for error handling; reserve Result-like patterns for internal optional outcomes if needed.

### Protocol vs ABC

- Prefer **`typing.Protocol`** for extension points (structural subtyping; no inheritance required).
- Use **ABC** only when you need shared implementation or `@abstractmethod` with inheritance.

### Config vs runtime

- **Config:** Immutable where possible (e.g. `CheckpointConfig` frozen). Pydantic for validation.
- **Runtime:** Trackers, buffers, caches live in Store/Manager classes. Keep config and runtime state clearly separated.

### Public API typing

- All public functions and methods have full type annotations.
- No `Any` in public API where a concrete type is possible (e.g. use `Tracer` instead of `Any` for tracer parameters where the type is known).
- Tools use clear descriptions in the tool spec; other public APIs use docstrings with at least one example where helpful.

## 7. Config vs runtime (summary)

- **Config objects:** Pydantic or frozen dataclasses; built at agent/setup time.
- **Runtime state:** Stored in Store/Manager instances; may be mutable; not part of the public “config” surface.

## 8. Thread safety and event loop

- Sync `response()` runs the async loop via `asyncio.run()` (or equivalent); document that nested or multi-threaded use may require an explicit loop/runner.
- Observability and budget stores may use locks where shared across threads; see docstrings in the relevant modules.
