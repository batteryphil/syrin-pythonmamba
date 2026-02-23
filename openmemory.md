# Syrin Python — OpenMemory Guide

## Overview

**Syrin** is a Python library for building AI agents with budget management, declarative agent definitions, and DSL-to-Python code generation. Companion to the Syrin DSL.

- **Package:** `Syrin` (src layout: `src/syrin/`)
- **Tooling:** uv (project/venv), ruff (format + lint), mypy (strict), pytest + pytest-asyncio
- **Python:** >= 3.10

## Architecture

- **Foundation (Day 1–2):** `types.py`, `exceptions.py`, `model.py`, providers, agent, tool/task decorators
- **Budget (Day 3–4):** `budget.py`, `cost.py`, budget engine
- **Day 4:** `budget_store.py` (BudgetStore, InMemoryBudgetStore, FileBudgetStore), `prompt.py` (PromptTemplate, @prompt), `response.py` (Response, TraceStep); Agent returns Response, optional budget_store; tracing deferred
- **Phase 3 Day 5:** memory.py (Memory, BufferMemory, WindowMemory), pipe.py (pipe(), Pipe.then/result/result_async, async steps), Agent inheritance (__init_subclass__, MRO merge tools / override prompt-model-budget), Agent memory param and response() appends to memory
- **Transport (CLI):** cli.py — CLI class (user_tag, agent_tag, style), serve(agent) REPL; commands /quit, /cost, /trace, /clear, /switch <model_id>; argparse: Syrin serve -a MODULE:CLASS, run SCRIPT, cost, version (no codegen)

## User Defined Namespaces

- (Leave blank — user populates)

## Components

- **types.py** — Pydantic models: `ModelConfig`, `ToolSpec`, `TaskSpec`, `Message`, `ToolCall`, `TokenUsage`, `CostInfo`, `ProviderResponse`, `AgentConfig`
- **exceptions.py** — `SyrinError`, `BudgetExceededError`, `BudgetThresholdError`, `ModelNotFoundError`, `ToolExecutionError`, `TaskError`, `ProviderError`, `CodegenError`
- **model.py** — `Model` (provider auto-detect, env var resolution), `ModelRegistry` (singleton)
- **providers/base.py** — Abstract `Provider` with `complete()` and `complete_sync()`; `ProviderResponse` from types
- **providers/anthropic.py** — `AnthropicProvider` (optional dep: anthropic)
- **providers/openai.py** — `OpenAIProvider` (optional dep: openai)
- **providers/litellm.py** — `LiteLLMProvider` (optional dep: litellm)
- **tool.py** — `@tool` decorator: builds `ToolSpec` from name, docstring, type-hint JSON schema (str/int/float/bool/list/dict, Optional)
- **task.py** — `@task` decorator: returns `TaskSpec` with name, parameters, return_type, func
- **agent.py** — `Agent(model, system_prompt, tools)`, `response(user_input)` with tool-call loop; `_get_provider(provider_name)` resolves to provider instance
- **prompt.py** — `PromptTemplate(template, name)`, `render(**kwargs)` / `__call__`; `@prompt` on no-arg function returning template string
- **response.py** — `Response(content, token_usage, cost, model, trace, tool_calls, latency_ms)`, `TraceStep`; `str(response)` → content
- **budget_store.py** — `BudgetStore` ABC (get/save), `InMemoryBudgetStore`, `FileBudgetStore(path, single_file=True)`
- **cost.py** — `MODEL_PRICING` (USD per 1M tokens), `Pricing`, `calculate_cost()`, `count_tokens()` (tiktoken or estimate)
- **budget.py** — `Budget` (run, per RateLimit, on_exceeded, thresholds), `RateLimit`, `Threshold`, `SwitchModelAction`/`StopAction`/`WarnAction`/`CustomAction`, `SwitchModel()`/`Stop()`/`Warn()` constructors, `BudgetTracker` (record, check_budget, check_thresholds, reset_run, get_summary, rolling windows), `BudgetStatus`, `BudgetSummary`, `CostEntry`; re-exports `Pricing`
- **agent.py** — Optional `budget`, `budget_store`, `budget_store_key`, `memory`; `__init_subclass__` sets _Syrin_default_* from MRO (tools merged, prompt/model/budget first-defined); `_build_messages` includes memory.get_messages(); after response, memory.add(user + assistant); `response()` returns `Response`; `switch_model(model)`, `budget_summary`
- **memory.py** — `Memory` ABC (add, get_messages, clear), `BufferMemory`, `WindowMemory(k)` (last k message pairs)
- **pipe.py** — `Pipe(value).then(fn).result()` / `.result_async()`, `pipe(value, f1, f2, ...)`; async steps run via asyncio.run in result()
- **cli.py** — `CLI(user_tag=..., agent_tag=..., style=...)`, `serve(agent)` (agent instance or class); REPL commands /quit, /cost, /trace, /clear, /switch <model_id>; on exit prints budget summary. Entry point: `Syrin serve -a module:attr`, `Syrin run script.py`, `Syrin cost`, `Syrin version`. `_resolve_agent(spec)` loads module:attr.

## Launch / GTM

- **plan/GTM_PRODUCTHUNT_HN.md** — Full playbook for ProductHunt and Hacker News launches (30-day prep, tagline formats, maker comment structure, timing)

## Patterns

- Virtual env: always use `.venv` and `uv`; install with `uv pip install -e ".[dev]"` from project root
- Tests: `tests/` with pytest; conftest clears `ModelRegistry` between tests that mutate it
- Examples: `examples/` for minimal usage (e.g. `hello_model.py`)
- Lint/format: `ruff check . && ruff format .`; typecheck: `mypy src/Syrin`
- No version pinning when adding deps — use latest; pyproject lists minimums (e.g. pydantic>=2.0)
