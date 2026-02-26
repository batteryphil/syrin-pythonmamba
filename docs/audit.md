# Audit Logging

Audit logging records lifecycle events (LLM calls, tool calls, handoffs, spawns) for compliance, debugging, and cost attribution.

## Quick Start

```python
from syrin import Agent, AuditLog, Model

audit = AuditLog(path="./audit.jsonl")
agent = Agent(
    model=Model.OpenAI("gpt-4o-mini"),
    system_prompt="You are helpful.",
    audit=audit,
)
agent.response("Hello")
# Writes AGENT_RUN_START, LLM_CALL, AGENT_RUN_END to audit.jsonl
```

## Configuration

| Option | Default | Description |
|--------|---------|-------------|
| `path` | `./audit.jsonl` | File path for JSONL backend |
| `include_llm_calls` | True | Log LLM request start/end |
| `include_tool_calls` | True | Log tool call start/end/error |
| `include_handoff_spawn` | True | Log handoff and spawn events |
| `include_budget` | False | Log budget check/threshold/exceeded |
| `include_user_input` | False | Include input in extra (PII risk) |
| `include_model_output` | True | Include content in extra |
| `custom_backend` | None | Use custom AuditBackendProtocol |

## Pipeline & DynamicPipeline

```python
from syrin import Pipeline, AuditLog, DynamicPipeline

audit = AuditLog(path="./pipeline_audit.jsonl")
pipeline = Pipeline(audit=audit)
pipeline.run([(Researcher, "Research X"), (Writer, "Write report")])

dyn = DynamicPipeline(agents=[...], model=model, audit=audit)
dyn.run("Create a report")
```

## Custom Backend

Implement `AuditBackendProtocol`:

```python
from syrin.audit import AuditBackendProtocol, AuditEntry, AuditFilters

class S3AuditBackend(AuditBackendProtocol):
    def write(self, entry: AuditEntry) -> None:
        # Upload to S3, etc.
        ...

    def query(self, filters: AuditFilters) -> list[AuditEntry]:
        # Optional
        ...

agent = Agent(model=model, audit=AuditLog(custom_backend=S3AuditBackend(...)))
```

## Event Types

| Audit Event | Source |
|-------------|--------|
| `agent_run_start`, `agent_run_end` | Agent lifecycle |
| `llm_call`, `llm_retry`, `llm_fallback` | LLM |
| `tool_call`, `tool_error` | Tools |
| `handoff_start`, `handoff_end`, `handoff_blocked` | Handoff |
| `spawn_start`, `spawn_end` | Spawn |
| `pipeline_start`, `pipeline_end` | Static Pipeline |
| `dynamic_pipeline_start`, `dynamic_pipeline_end` | Dynamic Pipeline |

## Concurrency & Sharing

### Sharing AuditLog across parallel agents

Do **not** share the same `AuditLog` object across agents that run in parallel (e.g. `Pipeline.run(agents).parallel()` or multiple threads/processes). The built-in `JsonlAuditBackend` uses append-only file writes; concurrent writes from different threads or processes can interleave and corrupt log lines.

**Safe:**
- Single async event loop (e.g. `asyncio.gather`) — execution is cooperative, writes are sequential.
- Sequential execution — one agent at a time.

**Unsafe:**
- Multiple threads writing to the same audit file.
- Multiple processes writing to the same audit file.

**Recommendation:** Give each agent or pipeline its own `AuditLog` (and thus its own file) when running in parallel, or use a custom backend that serializes writes (e.g. with a lock).

### DynamicPipeline with audit

When `audit` is attached to `DynamicPipeline`, it logs **pipeline-level events only**:

- `dynamic_pipeline_start`, `dynamic_pipeline_plan`, `dynamic_pipeline_execute`
- `dynamic_pipeline_agent_spawn`, `dynamic_pipeline_agent_complete`
- `dynamic_pipeline_end`, `dynamic_pipeline_error`

Sub-agents created by the pipeline (e.g. Researcher, Writer) **do not** receive the pipeline’s audit. Their LLM calls, tool calls, and handoffs are **not** written to the pipeline’s audit backend. The pipeline audit records which agents were spawned, when they completed, and cost/token aggregates — not per-agent internals.

**Parallel execution:** DynamicPipeline runs parallel agents via `asyncio.gather`. `AGENT_COMPLETE` hooks are emitted sequentially after all agents finish, so audit writes remain sequential and safe even when the pipeline uses parallel mode.
