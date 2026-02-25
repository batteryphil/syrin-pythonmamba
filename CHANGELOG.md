# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **Core stability (Step 4):** New test suite `tests/core_stability/` with TDD tests for agent lifecycle (sync/async parity), loop strategies (LoopResult shape, exception handling), model resolution (valid provider, strict mode), budget enforcement (per-run, threshold actions), memory (remember/recall/forget, conversation path), Response contract (cost/tokens/tool_calls/stop_reason), and structured output validation.
- **Model resolution:** `ProviderNotFoundError` exception and `get_provider(name, strict=True)`; when `strict=True`, unknown provider names raise with a clear message listing known providers. Agent uses strict resolution when resolving from `ModelConfig` so invalid `provider=` fails fast.
- **Observability:** Span coverage for agent runs: root agent span plus automatic child spans for each LLM call and tool execution when the loop uses a tracer. `AgentRunContext` now exposes an optional `tracer` property so loops can create LLM/tool spans.
- **Observability:** `_llm_span_context` and `_tool_span_context` helpers in `loop` so SingleShot and React loops create `SpanKind.LLM` and `SpanKind.TOOL` spans with semantic attributes (tokens, model, tool name, input/output).
- **Observability:** OTLP exporter implementation in `syrin.observability.otlp`: `OTLPExporter` converts Syrin spans to OpenTelemetry and exports to OTLP HTTP endpoint when optional dependency is installed; no-op when not.
- **Tests:** `tests/observability/test_observability_integration.py` — TDD tests for span coverage (agent/LLM/tool), session propagation, metrics schema, sampling parent-child consistency, debug mode, OTLP exporter, hook coverage, and export format.

### Changed

- **Model resolution:** `get_provider(provider_name, *, strict=False)` — new `strict` parameter; Agent construction with `ModelConfig(provider="typo")` now raises `ProviderNotFoundError` (breaking for callers relying on fallback to LiteLLM).
- **Observability:** Session ID from `trace.session()` now propagates to all spans created during agent runs (spans already had `session_id`; context was already used by tracer).
- **Docs:** `docs/observability.md` — Updated "What's Traced by Default" with exact span kinds and tree shape; added Metrics schema subsection; fixed Production Setup imports (create_sampler from sampling).
- **Plan:** Observability checklist in `plan/v0.2.0.md` marked complete.

### Fixed

- **Step 5 (Bug fixes):** Response data regression tests for sync/async parity and tool_calls from loop. recall() contract tests (Agent/Memory/MemoryStore return shape). spawn() return type (Response vs Agent) and budget inheritance tests. Hook emission audit (agent run emits only Hook enum). Rate limit: threshold behavior is fully controlled by the user’s `action(ctx)` callback on `RateLimitThreshold` (no separate `rate_limit_action`; user implements stop/wait/switch in the callback). Guardrail block tests (input block skips LLM; output block returns GUARDRAIL response). Checkpoint trigger tests (STEP fires, MANUAL does not; save/load). Session `span_count` updated when spans are exported in session context. Edge-case tests (empty tools, no budget, no persistent memory, unknown provider) and typed errors.

### Deprecated

### Removed

### Security

---

## [0.1.1] - 2026-02-25

**syrin** v0.1.1 — Python library for building AI agents with budget management, declarative definitions, and production-ready observability.

Initial Release

---

**Install:** `pip install syrin==0.1.1`
