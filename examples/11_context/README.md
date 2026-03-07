# Context management examples

Syrin gives you **full visibility and control** over the context window: what goes in, why, and how much.

## Run the examples

```bash
# Tour: basics, snapshot, manual compaction, thresholds, custom manager
python -m examples.11_context.context_management

# Best example: full snapshot & breakdown (capacity, components, provenance, rot risk, export)
python -m examples.11_context.context_snapshot_demo

# Best example: thresholds and compaction (react when context fills up)
python -m examples.11_context.context_thresholds_compaction_demo

# Custom compaction prompt (override summarization prompts + optional LLM)
python -m examples.11_context.context_custom_compaction_prompt_demo

# Compaction methods: none, middle_out_truncate, summarize (when each runs)
python -m examples.11_context.context_compaction_methods_demo

# Proactive compaction: auto_compact_at (e.g. 0.6 = 60% utilization)
python -m examples.11_context.context_proactive_compaction_demo

# Runtime context injection: RAG, per-call inject
python -m examples.11_context.context_runtime_injection_demo
```

## What each example shows

| Example | What you see |
|--------|----------------|
| **context_management** | Short tour: `Context(max_tokens=, thresholds=)`, stats after `response()`, snapshot breakdown, manual `MiddleOutTruncator` / `ContextCompactor`, threshold actions, custom `ContextManager`. |
| **context_snapshot_demo** | Full **context snapshot**: capacity (tokens used/max/utilization), **breakdown** (system, tools, memory, messages), **why_included**, **message_preview** (role, snippet, tokens, source), **provenance**, **context_rot_risk**, and **to_dict()** for dashboards. |
| **context_thresholds_compaction_demo** | Small context window + threshold at 50% that runs **compaction**; **context.threshold** and **context.compact** events; **stats.compacted** and **compact_method** in snapshot. |
| **context_proactive_compaction_demo** | **Context(auto_compact_at=0.6)** — compact once per prepare when utilization ≥ 60%; no threshold needed; same **context.compact** event. |
| **context_custom_compaction_prompt_demo** | **Context(compaction_prompt=..., compaction_system_prompt=..., compaction_model=...)** to override summarization prompts and use an LLM when compaction runs. |
| **context_compaction_methods_demo** | **CompactionMethod**: when you get **none**, **middle_out_truncate**, or **summarize** — uses ContextCompactor with different budgets/message counts to trigger each method. |
| **context_runtime_injection_demo** | **Runtime injection**: **Context.runtime_inject** (RAG callable) and **response(inject=...)** per-call; snapshot provenance and **injected_tokens** in breakdown. |

## Key APIs

- **`agent.context_stats`** — Total tokens, utilization, **breakdown** (after prepare), compacted, compact_method.
- **`agent.context.snapshot()`** — Full view: breakdown, message_preview, provenance, why_included, context_rot_risk; **`snapshot.to_dict()`** for export.
- **`Context(max_tokens=, reserve=, thresholds=[...])`** — Window size and actions at utilization %.
- **`Context(auto_compact_at=0.6)`** — Proactively compact when utilization ≥ 60% (one knob; no threshold needed).
- **`ContextThreshold(at=N, action=lambda evt: evt.compact())`** — Run compaction when usage hits N%.
