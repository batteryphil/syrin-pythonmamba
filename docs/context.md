# Context Management

Syrin’s context system manages **token limits**, **on-demand compaction**, and **token usage caps** for agent conversations. It runs on every request when the agent uses the default context manager: messages are counted, thresholds evaluated, and compaction runs only when you request it (e.g. from a threshold action).

---

## Table of contents

- [Overview](#overview)
- [Quick start](#quick-start)
- [Concepts](#concepts)
- [Configuration](#configuration)
- [ContextThreshold and ThresholdContext](#contextthreshold-and-thresholdcontext)
- [Compaction](#compaction)
- [TokenLimits (token caps)](#tokenlimits-token-caps)
- [Agent API](#agent-api)
- [Context snapshot (full view)](#context-snapshot-full-view)
- [Events](#events)
- [Token counting](#token-counting)
- [Custom context manager](#custom-context-manager)
- [Compactors](#compactors)
- [Observability](#observability)
- [Integration with memory and budget](#integration-with-memory-and-budget)
- [API reference](#api-reference)
- [Troubleshooting](#troubleshooting)
- [Limitations](#limitations)
  - [More limitations (with examples)](#more-limitations-with-examples)
- [Complete examples](#complete-examples)

---

## Overview

**What the context system does:**

- **Counts tokens** for messages, system prompt, and tools on each request.
- **Applies a context budget** from `Context` (max tokens, optional reserve for response). Utilization is computed as used tokens vs. available budget.
- **Evaluates thresholds** (e.g. at 50%, 75%, 100% utilization) and runs your **action** callables. No automatic compaction; you trigger it by calling **`ctx.compact()`** (or **`agent.context.compact()`**) from an action.
- **Optional token caps** via **`Context.token_limits`** (run and/or per-hour/day/week/month). Enforced by the budget tracker; separate from cost (Budget).
- **Emits events** (`context.compact`, `context.threshold`) and exposes **stats** after each call.

**When it runs:** On every `agent.response()` (and streaming) when the agent uses the default context manager. You can pass a **`Context`** config or a custom **`ContextManager`** to the agent.

**Default behavior:** If you don’t pass `context`, the agent gets `Context()` with `max_tokens=None` (resolved from the model or 128k), no thresholds, and no budget. Compaction never runs unless a threshold action calls **`ctx.compact()`**.

### What to use when

| Goal | Use | Notes |
|------|-----|--------|
| Limit **cost** (USD) per run or per period | **Budget** | `Agent(budget=Budget(run=10.0, per=RateLimit(day=50)))` |
| Limit **tokens** per run or per period | **Context.token_limits** (TokenLimits) | Same field names as Budget: run, per, on_exceeded |
| Set **context window** size and reserve | **Context** (max_tokens, reserve) | reserve = tokens reserved for model output |
| React at utilization (e.g. compact at 75%) | **Context.thresholds** + **ContextThreshold** + **compact_if_available** | Action receives **ThresholdContext**; call **evt.compact()** to compact |
| Proactively compact at a fraction (e.g. 60%) | **Context.auto_compact_at** (e.g. `0.6`) | One knob; compaction runs once per prepare when utilization ≥ value; no threshold needed |
| Per-call stats (tokens, utilization, compact_count) | **result.context_stats** or **agent.context_stats** | **result.context** = Context used for that call (overrides agent's when passed) |
| **Full context view** (what, why, where, rot risk) | **agent.context.snapshot()** | **ContextSnapshot**: message_preview, provenance, why_included, breakdown, context_rot_risk; export via **to_dict()** for viz tools. |
| **Inject RAG/dynamic context** at prepare time | **Context.runtime_inject** or **response(..., inject=...)** | See [Runtime context injection](#runtime-context-injection). |

**Budget vs token caps:** **Budget** = cost limits in USD. **Context.token_limits** (TokenLimits) = token caps (run and/or per period). Same field names (run, per, on_exceeded) for consistency.

---

## Quick start

```python
from syrin import Agent, Model

# Default context (no thresholds, no compaction)
agent = Agent(
    # model=Model("openai/gpt-4o"),
    model=Model.Almock(),  # No API Key needed
    system_prompt="You are a helpful assistant.",
)
result = agent.response("Hello!")
print(agent.context_stats.total_tokens)  # e.g. 45
print(agent.context_stats.utilization)  # e.g. 0.00035
```

With a limit and compaction at 75%:

```python
from syrin import Agent, Context, Model
from syrin.threshold import ContextThreshold

agent = Agent(
    # model=Model("openai/gpt-4o"),
    model=Model.Almock(),  # No API Key needed
    system_prompt="You are helpful.",
    context=Context(
        max_tokens=80000,
        thresholds=[
            ContextThreshold(at=75, action=lambda evt: evt.compact()),
        ],
    ),
)
result = agent.response("Long conversation...")
```

---

## Concepts

- **Context window** – The maximum tokens allowed for the current request’s context. Set by **`Context.max_tokens`** or inferred from the model (or 128k).
- **Context window capacity** – Internal **ContextWindowCapacity**: `available = max_tokens - reserve` (default reserve 2000, or model’s **default_reserve_tokens** when set). Utilization = used tokens / available, capped at 1.0 when at or over capacity. You don’t construct this; it’s built from **Context** during prepare.
- **Thresholds** – **ContextThreshold** instances: at a given utilization (e.g. `at=75` or `at_range=(70, 75)`), an **action** callable is run. The callable receives **ThresholdContext** with `percentage`, `current_value`, `limit_value`, and **`compact()`** (context only).
- **Compaction** – Reducing message list size (e.g. middle-out truncation or summarization). Happens when **`ctx.compact()`** (or **`agent.context.compact()`**) is called during **prepare**, or when **Context.auto_compact_at** is set and utilization reaches that fraction (proactive compaction). With **auto_compact_at=None** (default), only threshold actions trigger compaction.
- **TokenLimits** – Optional token caps on **Context** via **`Context.token_limits`**: **run** (max tokens per run), **per** (hour/day/week/month), **on_exceeded** (callback). Same field names as **Budget** (run, per, on_exceeded). Enforced by the budget tracker; separate from cost (Budget).

---

## Configuration

### Context

**`Context`** is the main configuration object. Pass it to the agent as **`context=Context(...)`**. It controls the context window, reserves, thresholds, token caps, encoding, and compactor.

#### Context fields (why, what, how)

| Field | Why | What | How |
|-------|-----|------|-----|
| **max_tokens** | Cap total context size so you don’t exceed the model’s window or your own limit. | Max tokens for the context window. If `None`, resolved from the model’s **context_window** or 128k. Must be **> 0** when set. | Set explicitly (e.g. `80000`) or leave `None` to use model/default. |
| **reserve** | Leave room for the model’s reply; otherwise utilization is based on input only. | Tokens subtracted from **max_tokens** to get “available” for input. Default 2000. **≥ 0**. Can be overridden by the model’s **default_reserve_tokens**. | Set on **Context** or on the model via **Model(..., default_reserve_tokens=8000)**. |
| **thresholds** | React when utilization hits a percentage (e.g. compact at 75%, raise at 100%). | List of **ContextThreshold** (at/at_range, action). Only context thresholds are allowed. | Add **ContextThreshold(at=75, action=lambda ctx: ctx.compact() if ctx.compact else None)**. |
| **token_limits** | Cap token usage per run and/or per period (separate from USD Budget). | Optional **TokenLimits**: **run**, **per**, **on_exceeded**. Same names as **Budget**. | **Context(token_limits=TokenLimits(run=50_000, on_exceeded=raise_on_exceeded))** or with **per=TokenRateLimit(...)**. |
| **encoding** | Token counting must use the same encoding as the model (e.g. cl100k_base for OpenAI). | TokenCounter encoding string. Default **"cl100k_base"**. | Set if you use a model with a different tokenizer; the default context manager uses it. |
| **compactor** | Use a custom compaction strategy (e.g. summarizer) instead of the default middle-out truncation. | Optional **ContextCompactorProtocol**: **compact(messages, budget) -> CompactionResult**. | **Context(compactor=MyCompactor())**; default manager calls it during prepare when compaction runs. |
| **compaction_prompt** | Override the user prompt template for summarization (e.g. "Summarize: {messages}"). | Optional **str**. **{messages}** is replaced with the conversation text. **None** = default from **syrin.context.prompts**. | Passed to the default **ContextCompactor**'s **Summarizer** when compaction runs. |
| **compaction_system_prompt** | Override the system prompt for the summarization LLM. | Optional **str**. **None** = default from **syrin.context.prompts**. | Use with **compaction_model** for custom summarization behavior. |
| **compaction_model** | Model used for summarization when compaction runs. | Optional **Model**. **None** = placeholder (no LLM; keeps system + last 4 messages and a summary line). | Set to e.g. **Model.Almock()** or **Model.OpenAI("gpt-4o-mini")** for real LLM summarization. |
| **auto_compact_at** | Proactively compact when context utilization reaches a fraction (e.g. 60%) to reduce context rot. | **float \| None** in **[0.0, 1.0]** (e.g. `0.6` = 60%). **None** = no proactive compaction. | **Context(auto_compact_at=0.6)** to compact once per prepare when utilization ≥ 60%; same **context.compact** event and compactor as threshold-triggered compaction. |
| **runtime_inject** | Inject context at prepare time (e.g. RAG results, dynamic blocks). | Callable **PrepareInput → list[dict]** (role, content). Called when no per-call `inject` is provided. | **Context(runtime_inject=my_rag_fn)**; see [Runtime context injection](#runtime-context-injection). |
| **inject_placement** | Where to place injected messages. | **InjectPlacement**: `prepend_to_system`, `before_current_turn` (default), `after_current_turn`. | `Context(inject_placement=InjectPlacement.BEFORE_CURRENT_TURN)` for RAG (docs before the question). |
| **inject_source_detail** | Provenance label for injected content. | **str** (e.g. `"rag"`, `"dynamic_rules"`). | **Context(inject_source_detail="rag")**; appears in snapshot provenance and **why_included**. |

Example:

```python
from syrin import Context, TokenLimits, TokenRateLimit

ctx = Context(
    max_tokens=80000,
    reserve=2000,
    encoding="cl100k_base",
    thresholds=[...],        # see ContextThreshold section
    token_limits=TokenLimits(run=50_000, per=TokenRateLimit(hour=100_000)),
)
```

### Resolving max_tokens

- If **`Context.max_tokens`** is set, that value is used.
- If it is **`None`** and the agent has a **model** with **ModelSettings.context_window**, that is used.
- Otherwise **128000** is used.

### ContextWindowCapacity (internal)

The default context manager builds a **ContextWindowCapacity** from **Context** (and optional model). You don’t construct it in normal use; it’s the internal “window” budget (max tokens, reserve, utilization) used during **prepare**.

- **max_tokens** – From **Context** (or model / default).
- **reserve** – Reserved for the model’s reply. From **Context.reserve** unless the model has **default_reserve_tokens** set (see [reserve and model](#4-reserve-and-model-hint)).
- **available** – `max(0, max_tokens - reserve)`.
- **used_tokens** – Set during **prepare**.
- **utilization** – `used_tokens / available`, capped at **1.0** when at or over capacity (or when **available** ≤ 0 and **used_tokens** > 0).
- **percent** – 0–100.

**Context.get_capacity(model)** returns a **ContextWindowCapacity** for the current config. When **model** is provided, the model’s **default_reserve_tokens** (if set) is used for **reserve**.

---

## ContextThreshold and ThresholdContext

**ContextThreshold** has the same shape as **BudgetThreshold**: **at**, **at_range**, **action**, **metric**, **window**. For context, **metric** is always tokens and **window** is always **ThresholdWindow.MAX_TOKENS** (current context window).

### ContextThreshold parameters

| Parameter   | Type | Description |
|------------|------|-------------|
| **at**     | `int` (0–100) | Trigger when utilization **≥** this percentage. Ignored if **at_range** is set. |
| **at_range** | `tuple[int, int] \| None` | Trigger only when utilization is in **[min, max]** (e.g. `(70, 75)`). |
| **action** | `Callable[[ThresholdContext], None]` | Required. Called when the threshold triggers. |
| **metric** | (fixed) | **ThresholdMetric.TOKENS**. |
| **window** | (fixed) | **ThresholdWindow.MAX_TOKENS**. You don’t set it; other windows are invalid for context. |

Only one of **at** or **at_range** should be used. **action** is required.

### ThresholdContext (passed to your action)

Your **action** receives a **ThresholdContext** with:

| Attribute        | Type   | Description |
|------------------|--------|-------------|
| **percentage**   | `int`  | Utilization percentage (0–100) that triggered the threshold. |
| **metric**       |        | **ThresholdMetric.TOKENS**. |
| **current_value**| `float`| Current token count (same as used_tokens). |
| **limit_value**  | `float`| Max tokens (context window). |
| **budget_run**   | `float`| Alias for limit_value. |
| **parent**       |        | Parent object (e.g. agent). |
| **metadata**     | `dict` | Extra data. |
| **compact**      | `Callable[[], None] \| None` | **Call this to run compaction** during the action. Only set during **prepare**; may be `None` if not in that path. |

**Important:** Call **`ctx.compact()`** inside the action to compact the current message list. You can also call **`agent.context.compact()`** from the action (e.g. via **`ctx.parent.context.compact()`**) when **parent** is the agent.

### Examples

**Warn at 50%, compact at 75%, raise at 100%:**

```python
from syrin import Context
from syrin.threshold import ContextThreshold

def raise_on_full(ctx):
    raise ValueError("Context full")

Context(
    max_tokens=80000,
    thresholds=[
        ContextThreshold(at=50, action=lambda ctx: print(f"Tokens at {ctx.percentage}%")),
        ContextThreshold(at=75, action=lambda ctx: ctx.compact() if ctx.compact else None),
        ContextThreshold(at=100, action=raise_on_full),
    ],
)
```

**Trigger only in a band (e.g. 70–75%):**

```python
ContextThreshold(at_range=(70, 75), action=lambda ctx: ctx.compact() if ctx.compact else None)
```

**Switch model when nearing limit (concept):**

```python
ContextThreshold(
    at=85,
    action=lambda ctx: ctx.parent.switch_model(Model("openai/gpt-4o-mini")) if getattr(ctx.parent, "switch_model", None) else None,
)
```

---

## Compaction

Compaction runs during **prepare** when (1) a threshold action calls **`ctx.compact()`** or **`agent.context.compact()`**, or (2) **Context.auto_compact_at** is set and utilization reaches that fraction (proactive compaction). With **auto_compact_at=None** (default), only threshold actions trigger compaction.

- **When:** Only during **prepare**, when the manager has set the compact callback (e.g. when checking thresholds) or when proactive compaction triggers. Calling **compact()** outside that is a no-op.
- **What it does:** The default **ContextCompactor** first tries middle-out truncation; if overage is large, it may summarize then truncate. The current message list is replaced in place so the rest of **prepare** uses the compacted list.
- **Events:** After compaction, a **context.compact** event is emitted (see [Events](#events)). Same event for both proactive and threshold-triggered compaction.
- **Stats:** **ContextStats.compacted**, **compact_count** (this run only), **compact_method** reflect the last run. Use **result.context_stats** for per-call stats.

### Proactive compaction (auto_compact_at)

Set **Context.auto_compact_at** to a fraction in **[0.0, 1.0]** (e.g. **0.6** for 60%) to compact **once per prepare** when utilization reaches that value, **before** threshold actions run. This reduces context rot without adding a **ContextThreshold**; research suggests keeping utilization under about 60–70% helps quality.

- **None** (default): no proactive compaction; only threshold actions (e.g. **compact_if_available** at 75%) can trigger compact.
- **0.6**: compact when utilization ≥ 60%; same compactor and **context.compact** event as threshold-triggered compaction.
- If both **auto_compact_at** and thresholds are set, proactive compact runs first; then thresholds see the updated utilization.

**Custom compaction prompt:** Set **Context.compaction_prompt** (user template with **{messages}**), **Context.compaction_system_prompt** (optional), and **Context.compaction_model** (optional) to override the default summarization prompts and use an LLM for compaction. When **compaction_model** is **None**, the default compactor uses a placeholder (no LLM). Default prompts live in **syrin.context.prompts** (**DEFAULT_COMPACTION_SYSTEM_PROMPT**, **DEFAULT_COMPACTION_USER_TEMPLATE**).

**Compact from a threshold:**

```python
# Preferred: use the helper so you don't need to guard ctx.compact
from syrin.threshold import ContextThreshold, compact_if_available
ContextThreshold(at=75, action=compact_if_available)

# Or explicitly:
ContextThreshold(at=75, action=lambda ctx: ctx.compact() if ctx.compact else None)
```

**Compact from agent (e.g. inside the same action):**

```python
# Inside an action, ctx.parent is the agent:
ctx.parent.context.compact()
```

**CompactionResult** (internal) has **messages**, **method** (a **CompactionMethod** value), **tokens_before**, **tokens_after**.

### Compaction methods (CompactionMethod)

All possible values are on **CompactionMethod** (StrEnum). Use **`list(CompactionMethod)`** or **`from syrin.context import CompactionMethod`** to see and compare.

| Value | When it runs |
|--------|----------------|
| **none** | Messages already fit in budget; no compaction. |
| **middle_out_truncate** | Over budget and **overage** (tokens_before / budget) **&lt; 1.5** → keep start/end of conversation, drop middle. |
| **summarize** | Overage **≥ 1.5** and **&gt; 4 non-system messages** → older messages summarized (LLM if **compaction_model** set); if result still over budget, **middle_out_truncate** is applied (so you may see that method after a summarize step). |

So **middle_out_truncate** appears when the context is only slightly over budget (under 1.5×). To get **summarize**, use a smaller budget or more/longer messages so overage ≥ 1.5, and ensure you have more than 4 non-system messages so the summarizer runs.

---

## TokenLimits (token caps)

**TokenLimits** is the token-cap configuration you set on **Context.token_limits**. It uses the same field names as **Budget** (USD): **run**, **per**, **on_exceeded**.

**Why:** Cap token usage (input + output) per run and/or per time window without mixing with cost. Budget = USD only; token caps live on Context.

**What:** Optional **run** (max tokens per request run), optional **per** (TokenRateLimit: hour/day/week/month), and **on_exceeded** (callback when a limit is hit). When set, the agent’s budget tracker enforces these after each LLM call.

**How:** Pass **Context(token_limits=TokenLimits(run=..., per=..., on_exceeded=...))** to the agent. Use **raise_on_exceeded**, **warn_on_exceeded**, or a custom callable.

### TokenLimits fields

| Field           | Type | Description |
|-----------------|------|-------------|
| **run**         | `int \| None` | Max tokens per run (one request/response cycle, input + output). Same name as **Budget.run** (which is USD). |
| **per**         | **TokenRateLimit \| None** | Token caps per hour/day/week/month. Same name as **Budget.per** (which is USD rate limits). |
| **per_hour**    | `int \| None` | Convenience: `per.hour` when per is set; None otherwise. |
| **on_exceeded** | `Callable[[BudgetExceededContext], None] \| None` | Called when a token limit is exceeded. Raise to stop the run; return to continue (e.g. warn). Same as **Budget.on_exceeded**. |

### TokenRateLimit fields

| Field             | Type | Description |
|-------------------|------|-------------|
| **hour**          | `int \| None` | Max tokens in the current hour (rolling). |
| **day**           | `int \| None` | Max tokens in the current day (rolling). |
| **week**          | `int \| None` | Max tokens in the current week (rolling). |
| **month**         | `int \| None` | Max tokens in the month window (see **month_days**). |
| **month_days**    | `int` | Number of days for the month window (default 30). Ignored if **calendar_month** is True. |
| **calendar_month**| `bool` | If True, month = current calendar month; else last **month_days** days. |

### Example with output

```python
from syrin import Agent, Context, TokenLimits, Model, TokenRateLimit
from syrin.budget import raise_on_exceeded

agent = Agent(
    # model=Model("openai/gpt-4o"),
    model=Model.Almock(),  # No API Key needed
    context=Context(
        token_limits=TokenLimits(
            run=50_000,
            per=TokenRateLimit(hour=100_000, day=400_000),
            on_exceeded=raise_on_exceeded,
        ),
    ),
)
result = agent.response("Explain quantum computing briefly.")
print(result.context_stats.total_tokens)  # e.g. 412
print(agent.budget_state.spent)  # same run's cost; use get_budget_tracker().get_summary() for token counts
# If run or per limit is exceeded, on_exceeded is called (here: raises).
```

**Output (example):**
```
412
412
```

---

## Agent API

### Per-call context

**response**, **arun**, **stream**, and **astream** accept an optional **context**. When set, it overrides the agent's default context for that call:

- **response(user_input, context=None)**  
- **arun(user_input, context=None)**  
- **stream(user_input, context=None)**  
- **astream(user_input, context=None)**

When **context** is a **Context** instance, that run uses its **max_tokens**, **reserve**, **thresholds**, **budget**, **encoding**, and **compactor** instead of the agent’s default context. Useful for one-off limits or different thresholds without building a new agent.

```python
from syrin import Agent, Context, Model

agent = Agent(
    # model=Model("openai/gpt-4o"),
    model=Model.Almock(),  # No API Key needed
    context=Context(max_tokens=128000),
)
# This call uses a smaller window and no thresholds:
result = agent.response("Summarize this", context=Context(max_tokens=4000))
```

### result.context_stats and result.context

Every **Response** from **response()** or **arun()** includes **context_stats** and **context** for that call:

- **result.context_stats** – **ContextStats** for this run (total_tokens, utilization, compact_count for this prepare, etc.). Use this instead of **agent.context_stats** when you need stats for a specific call (e.g. with concurrent or multiple runs).
- **result.context** – The **Context** used for this call. When you pass **context=** to **response()** / **arun()**, it overrides the agent's default for that call; otherwise the agent's default (the **context** if one was passed, otherwise the agent’s default context).

**Note:** **stream()** and **astream()** yield chunks and do not return a **Response**, so they do not set **context_stats** or **context** on a result. Use **agent.context_stats** after the stream if you need stats for that run.

```python
r = agent.response("Hello", context=Context(max_tokens=4000))
print(r.context.max_tokens)    # 4000
print(r.context_stats.total_tokens) # tokens for this call
```

### agent.context

- **With default context manager:** Returns a facade that exposes **Context** config (e.g. **max_tokens**, **thresholds**, **budget**) and **compact()**. So you can read **agent.context.max_tokens** and call **agent.context.compact()** (valid only during **prepare**, e.g. from a threshold action).
- **compact()** – Requests compaction of the current context. No-op if not inside **prepare**.

### agent.context_stats

**ContextStats** for the last **prepare** (e.g. last **response**):

| Field                    | Type     | Description |
|--------------------------|----------|-------------|
| **total_tokens**         | `int`    | Tokens used in that run. |
| **max_tokens**           | `int`    | Context window used. |
| **utilization**          | `float`  | 0–1. |
| **compacted**            | `bool`   | Whether compaction ran. |
| **compact_count**    | `int`    | Compactions in this run (this prepare) only. |
| **compact_method**   | `str \| None` | e.g. `"middle_out_truncate"`, `"summarize"`, or `None`. |
| **thresholds_triggered**| `list[str]` | Metric names of triggered thresholds. |
| **breakdown**            | **ContextBreakdown \| None** | Token counts by component (system, tools, memory, messages). Set after **prepare()**; **None** before any run. Same data as **snapshot().breakdown**. |

**Why:** Inspect token usage and compaction after each run. **What:** Snapshot from the last **prepare** (or use **result.context_stats** for that call). **How:** Read **agent.context_stats** or **result.context_stats** after **response()** / **arun()**. Use **breakdown** for component-level token counts without building a full snapshot.

Example:

```python
result = agent.response("Hello!")
print(agent.context_stats.total_tokens)
print(agent.context_stats.utilization)
print(agent.context_stats.compacted)
# Per-call stats (same run):
print(result.context_stats.total_tokens, result.context.max_tokens)
```

### Context snapshot (full view)

**`agent.context.snapshot()`** returns a **ContextSnapshot** — a point-in-time view of what was sent to the LLM: which messages, where they came from, why they were included, token breakdown, and **context rot risk** (low / medium / high based on utilization).

Use it to debug context formation, feed visualization tools (e.g. Letta-style UIs), or log what actually went into the model.

**When:** After **response()** or **arun()**; the snapshot reflects the last **prepare**. Before any run, returns an empty snapshot (zeros, **context_rot_risk** = low).

**ContextSnapshot fields:**

| Field | Type | Description |
|-------|------|-------------|
| **timestamp** | `float` | When the snapshot was taken. |
| **total_tokens**, **max_tokens**, **tokens_available** | `int` | Capacity. |
| **utilization_pct** | `float` | Used vs available (0–100). |
| **breakdown** | **ContextBreakdown** | **system_tokens**, **tools_tokens**, **memory_tokens**, **messages_tokens**, **injected_tokens**. |
| **message_preview** | `list[MessagePreview]` | Per-message: **role**, **content_snippet**, **token_count**, **source** (system / memory / conversation / tools / current_prompt / injected). |
| **provenance** | `list[ContextSegmentProvenance]` | Per segment: **segment_id**, **source**, **source_detail**. |
| **why_included** | `list[str]` | Human-readable reasons (e.g. "system prompt", "recalled memory", "conversation history", "current user message"). |
| **context_rot_risk** | `"low" \| "medium" \| "high"` | Derived from utilization: low &lt; 60%, medium 60–70%, high ≥ 70%. |
| **compacted**, **compact_method** | `bool`, `str \| None` | Whether compaction ran and which method. |

**Export for visualization:** **`snapshot.to_dict()`** returns a JSON-serializable dict. Use **`to_dict(include_raw_messages=True)`** only when you need the full message list (can be large).

Example:

```python
result = agent.response("Hello!")
snap = agent.context.snapshot()
print(snap.utilization_pct, snap.context_rot_risk)  # e.g. 0.5, "low"
for p in snap.provenance:
    print(p.source.value, p.source_detail)
print(snap.why_included)
# Export for dashboards / viz tools
data = snap.to_dict()
```

**Hook:** **context.snapshot** is emitted after each **prepare** with payload **snapshot** (the **to_dict()** result) and **utilization_pct**.

### Runtime context injection

Inject additional context at prepare time (RAG results, dynamic rules, etc.) without changing the agent class.

**Two options:**

1. **Context.runtime_inject** — callable configured on Context, invoked each prepare.
2. **Per-call inject** — pass `inject=` to **response()**, **arun()**, **stream()**, or **astream()**.

**PrepareInput** (passed to `runtime_inject`) has: `messages`, `system_prompt`, `tools`, `memory_context`, `user_input`.

**Placement** (InjectPlacement):
- **prepend_to_system** — injected before the first system message.
- **before_current_turn** (default) — injected between conversation history and the current user message (typical for RAG).
- **after_current_turn** — injected after the current user message.

Injected messages appear in the snapshot with `source=injected` and your `inject_source_detail` (e.g. `"rag"`). **ContextBreakdown** includes **injected_tokens**.

**Example: RAG with runtime_inject:**

```python
from syrin import Agent, Context, Model
from syrin.context import InjectPlacement, PrepareInput

def rag_injector(inp: PrepareInput) -> list[dict]:
    docs = vector_store.search(inp.user_input, top_k=5)
    return [{"role": "system", "content": f"[RAG]\n{doc}"} for doc in docs]

agent = Agent(
    model=Model.Almock(),
    context=Context(
        runtime_inject=rag_injector,
        inject_placement=InjectPlacement.BEFORE_CURRENT_TURN,
        inject_source_detail="rag",
    ),
)
result = agent.response("What does the docs say about X?")
```

**Example: Per-call inject (e.g. from async RAG layer):**

```python
docs = await fetch_rag(user_input)
result = agent.response(
    user_input,
    inject=[{"role": "system", "content": f"[RAG]\n{d}"} for d in docs],
    inject_source_detail="rag",
)
```

**Handoff and spawn:** The same context snapshot is exposed in **HANDOFF_START** and **HANDOFF_BLOCKED** as **handoff_context**, and **SPAWN_START** includes **parent_context_tokens** (parent's snapshot token count). See [Handoff & Spawn](agent/handoff-spawn.md) for hook payloads and context visibility.

---

## Events

Subscribe with **agent.events.on(event_name, handler)**. The handler receives a single payload dict.

### context.compact

Emitted when compaction runs (after a **compact()** call during **prepare**).

**Payload:** **method**, **tokens_before**, **tokens_after**, **messages_before**, **messages_after**.

```python
agent.events.on("context.compact", lambda e: print(e["method"], e["tokens_before"], e["tokens_after"]))
```

### context.threshold

Emitted when a context threshold triggers.

**Payload:** **at**, **at_range**, **percent**, **metric**, **tokens**, **max_tokens**.

```python
agent.events.on("context.threshold", lambda e: print(f"Threshold {e['percent']}%"))
```

### context.snapshot

Emitted after each **prepare** with the full context snapshot for that run.

**Payload:** **snapshot** (ContextSnapshot as dict via **to_dict()**), **utilization_pct**.

```python
agent.events.on("context.snapshot", lambda e: print(e["utilization_pct"], e["snapshot"]["context_rot_risk"]))
```

---

## Token counting

**TokenCounter** – Counts tokens for text, message lists, and tool definitions. Uses **tiktoken** when available (encoding **cl100k_base** by default), with an estimation fallback.

**From syrin.context:** **TokenCounter**, **TokenCount**, **get_counter**.

### count(text: str) -> int

Count tokens in a string.

### count_messages(messages, system_prompt="") -> TokenCount

**TokenCount** has **total**, **system**, **messages** (and optionally **tools**, **memory**). Counts role overhead (e.g. 4 tokens per message).

### count_tools(tools: list[dict]) -> int

Approximate tokens for tool definitions.

### count_breakdown(system_prompt, memory_context, tools, tokens_used) -> ContextBreakdown

Returns **ContextBreakdown** with **system_tokens**, **tools_tokens**, **memory_tokens**, and **messages_tokens** (residual so that the sum matches **tokens_used** when possible). Used internally by the context manager to populate **ContextStats.breakdown** and **ContextSnapshot.breakdown**. Call it directly if you need component counts without running a full prepare.

### get_counter() -> TokenCounter

Returns the default **TokenCounter** singleton.

Example:

```python
from syrin.context import TokenCounter, get_counter

counter = get_counter()
n = counter.count("Hello, world!")
counts = counter.count_messages([{"role": "user", "content": "Hi"}])
print(counts.total)
```

---

## Custom context manager

Implement the **ContextManager** protocol and pass an instance as **context=** to the agent.

**Protocol:**

- **prepare(messages, system_prompt, tools, memory_context, budget, context=None) -> ContextPayload**  
  Return a **ContextPayload** with **messages**, **system_prompt**, **tools**, **tokens**. **context** is optional; when the agent passes a **Context** override for the run, it is provided here (custom managers may ignore it). Implementations should accept **context** in the signature for protocol compatibility.
- **on_compact(event: CompactionResult) -> None**  
  Optional; called after compaction if your implementation does it.

**ContextPayload** is a dataclass: **messages**, **system_prompt**, **tools**, **tokens**.

Example (keep only last N messages):

```python
from typing import Any
from syrin.context import ContextManager, ContextPayload, TokenLimits
from syrin.context.counter import get_counter

class RecentOnlyManager:
    def __init__(self, keep: int = 10):
        self._keep = keep
        self._counter = get_counter()

    def prepare(
        self,
        messages: list[dict],
        system_prompt: str,
        tools: list[dict],
        memory_context: str,
        capacity: ContextWindowCapacity,
        context: Any = None,
    ) -> ContextPayload:
        recent = messages[-self._keep:] if len(messages) > self._keep else messages
        tokens = self._counter.count_messages(recent, system_prompt).total
        return ContextPayload(
            messages=recent,
            system_prompt=system_prompt,
            tools=tools,
            tokens=tokens,
        )

    def on_compact(self, event) -> None:
        pass

agent = Agent(model=model, context=RecentOnlyManager(keep=20))
```

**create_context_manager(context=None, emit_fn=None, tracer=None)** returns a **DefaultContextManager** for a given **Context** and optional emit/tracer.

---

## Compactors

The default manager uses **ContextCompactor**, which combines truncation and summarization.

### MiddleOutTruncator

Keeps the start and end of the conversation and truncates the middle (primacy/recency). **compact(messages, budget, counter=None)** returns **CompactionResult** (**messages**, **method**, **tokens_before**, **tokens_after**). If tokens already fit, **method** is **"none"**.

### Summarizer

**Summarizer(system_prompt=None, user_prompt_template=None, model=None)** – Summarizes older messages. When **model** is set, calls the model with the given (or default) system and user prompts; the user template should contain **{messages}**, which is replaced with the conversation text. When **model** is **None**, uses a placeholder: keeps system + last 4 non-system messages and adds a summary line (no LLM). Default prompts are in **syrin.context.prompts**.

**summarize(messages, counter=None)** – Returns a shortened message list (system + summary block + last 4 messages).

### ContextCompactor

**ContextCompactor(compaction_prompt=None, compaction_system_prompt=None, compaction_model=None)** – Builds an internal **Summarizer** with these arguments. When omitted, the default compactor uses **Context**'s **compaction_prompt**, **compaction_system_prompt**, and **compaction_model** when the manager creates it.

**compact(messages, budget)** (no counter arg; uses internal counter):

1. If within budget, return as-is (**method="none"**).
2. If overage &lt; 1.5×, use **MiddleOutTruncator**.
3. Otherwise summarize (via **Summarizer**), then truncate if still over.

You can replace the compactor in a custom manager or extend **DefaultContextManager** with a different compactor. Custom compactors must implement **ContextCompactorProtocol** (``compact(messages, budget) -> CompactionResult``); the default is **ContextCompactor**.

---

## Observability

- **Tracing:** When a tracer is set on the context manager (e.g. via the agent), **context.prepare** spans are created with attributes such as **context.max_tokens**, **context.available**, **context.tokens**, **context.utilization**, **context.compacted**, **context.compact_method**, **context.thresholds_triggered**.
- **Debug:** **Agent(..., debug=True)** enables console trace output so you can see context and compaction in logs.

---

## Integration with memory and budget

- **Memory:** Persistent memory (**Memory**) is recalled and formatted as **memory_context**. The default context manager injects it as a system message (**[Memory]\n...**) when **memory_context** is non-empty. So context stats and compaction apply to the full message list including memory.
- **Budget:** Cost limits are **Budget** (USD). Token usage caps are **Context.token_limits** (**TokenLimits**). You can use both: **Agent(budget=Budget(...), context=Context(token_limits=TokenLimits(...)))**.
- **Long-running sessions:** For agents that run across restarts, combine **checkpoint** (saves messages + context snapshot), **BufferMemory** (restored on load), and **auto_compact_at** (e.g. 0.6). See [Agent: Checkpointing - Long-running sessions](agent/checkpointing.md) and `examples/12_checkpoints/long_running_agent.py`.

---

## API reference

| Class / type | Description |
|--------------|-------------|
| **Context** | Config: max_tokens, reserve, thresholds, token_limits, encoding, compactor, **compaction_prompt**, **compaction_system_prompt**, **compaction_model**. **get_capacity(model)**. |
| **ContextStats** | total_tokens, max_tokens, utilization, compacted, compact_count (this run only), compact_method, thresholds_triggered, **breakdown** (ContextBreakdown \| None; set after prepare). |
| **ContextSnapshot** | Full view from **agent.context.snapshot()**: timestamp, total_tokens, max_tokens, utilization_pct, breakdown, message_preview, provenance, why_included, context_rot_risk, compacted, compact_method. **to_dict(include_raw_messages=False)** for export. |
| **ContextBreakdown** | system_tokens, tools_tokens, memory_tokens, messages_tokens, **injected_tokens**; **total_tokens** property. |
| **InjectPlacement** | StrEnum: **PREPEND_TO_SYSTEM**, **BEFORE_CURRENT_TURN**, **AFTER_CURRENT_TURN**. Use with **Context.inject_placement**. |
| **PrepareInput** | Dataclass passed to **runtime_inject**: messages, system_prompt, tools, memory_context, user_input. |
| **MessagePreview** | role, content_snippet, token_count, **source** (ContextSegmentSource). |
| **ContextSegmentProvenance** | segment_id, source, source_detail. |
| **ContextSegmentSource** | StrEnum: SYSTEM, MEMORY, CONVERSATION, TOOLS, CURRENT_PROMPT, INJECTED. |
| **TokenLimits** | Token caps: **run**, **per** (TokenRateLimit), **on_exceeded**. Use on **Context.token_limits**. |
| **ContextWindowCapacity** | Internal: max_tokens, reserve, available, used_tokens, utilization, percent, reset(). |
| **ContextThreshold** | at, at_range, action, metric=TOKENS, window=MAX_TOKENS. **should_trigger(percent, metric)**. |
| **ThresholdContext** | percentage, metric, current_value, limit_value, budget_run, parent, metadata, **compact**. |
| **ContextPayload** | messages, system_prompt, tools, tokens. |
| **ContextManager** | Protocol: prepare(), on_compact(). |
| **DefaultContextManager** | context, prepare(), **stats**, **snapshot()**, current_tokens, **compact()**, set_emit_fn(), set_tracer(). |
| **TokenCounter** | count(), count_messages(), count_tools(), **count_breakdown()** (system_prompt, memory_context, tools, tokens_used → ContextBreakdown). **TokenCount**: total, system, messages, tools, memory. |
| **get_counter()** | Default TokenCounter. |
| **CompactionMethod** | StrEnum: **NONE**, **MIDDLE_OUT_TRUNCATE**, **SUMMARIZE**. Use **list(CompactionMethod)** to see all; **stats.compact_method** is one of these values. |
| **CompactionResult** | messages, method (CompactionMethod value), tokens_before, tokens_after. |
| **Compactor** | Base: compact(messages, budget, counter). |
| **MiddleOutTruncator** | Compactor: middle-out truncation. |
| **Summarizer** | **Summarizer(system_prompt=None, user_prompt_template=None, model=None)**. **summarize(messages, counter)**. |
| **ContextCompactor** | compact(messages, budget): truncate and/or summarize. |
| **ContextCompactorProtocol** | Protocol for custom compactors: **compact(messages, budget) -> CompactionResult**. Use for **Context.compactor** type. |
| **create_context_manager(context, emit_fn, tracer)** | Build DefaultContextManager. |
| **Response** (from agent) | **context_stats**, **context** – per-call context stats and effective Context for that run. |
| **TokenLimits** | Token caps on Context: **run**, **per** (TokenRateLimit), **on_exceeded**. Same field names as Budget. |
| **ContextWindowCapacity** | Internal: max_tokens, reserve, available, used_tokens, utilization (used during prepare). |

---

## Troubleshooting

**Context never compacts**  
Compaction runs only when **evt.compact()** (or **agent.context.compact()**) is called during **prepare**. Add a **ContextThreshold** (e.g. at 75%) whose **action** calls **evt.compact()** or use **compact_if_available**.

**Utilization always 0 or very low**  
Check **agent.context_stats.total_tokens** and **max_tokens**. If the model’s context window is large and conversations are short, utilization will be low.

**Threshold action not called**  
Ensure utilization actually reaches the threshold (e.g. use many or long messages). For a band, use **at_range=(min, max)**. The action must be a callable that accepts one argument (**ThresholdContext**).

**Want to raise when context is full**  
Use a threshold at 100% with an action that raises, e.g. **ContextThreshold(at=100, action=lambda ctx: (_ for _ in ()).throw(ValueError("Context full")))** or a named function that raises.

**Token caps not enforced**  
Set **Context.token_limits** (**run** and/or **per=TokenRateLimit(...)**). The agent’s budget tracker must be present (it is when the agent has a **Budget** or **Context.token_limits**).

---

## Limitations

- **compact() only during prepare** – **ctx.compact()** / **agent.context.compact()** only take effect when the default context manager is inside **prepare** (e.g. from a threshold action). Calling **compact()** outside that path is a no-op.
- **Stats not persisted** – **ContextStats** (total_tokens, utilization, compact_count, etc.) reflect the last run only; they are not persisted across process restarts. **compact_count** is per run (this prepare only). For per-call stats use **result.context_stats** on the **Response**.
- **Custom manager and budget** – If you pass a custom **ContextManager** as **context=**, it may ignore **Context** (and thus **budget**, **encoding**, **compactor**). The default manager respects all **Context** options.
- **Context validation** – **max_tokens** must be **> 0** when set; **reserve** must be **≥ 0**. **get_capacity()** also validates **max_tokens** when resolved from the model.

### More limitations (with examples)

**1. Tools can consume the whole budget — compaction cannot free space**

If `tools` are so large that `available_for_messages = max_tokens - reserve - tools_tokens <= 0`, there is no room for message compaction. Thresholds **still run** (at 100% utilization) and **stats are updated** for that call. Compaction is a no-op because there is no message budget. You can still use a 100% threshold to e.g. raise or switch model.

```python
# Many/large tools leave no room for messages
agent = Agent(
    # model=Model("openai/gpt-4o"),
    model=Model.Almock(),  # No API Key needed
    context=Context(max_tokens=4000, reserve=500, thresholds=[
        ContextThreshold(at=100, action=lambda ctx: (_ for _ in ()).throw(ValueError("Context full"))),
    ]),
    tools=[huge_tool_spec] * 50,  # e.g. 4000+ tokens
)
# Thresholds run; stats updated; compaction cannot help
agent.response("Hello")
```

**Mitigation:** Use a larger **max_tokens** or fewer/smaller tools so that `available_for_messages > 0` if you want compaction to be able to free space.

---

**2. Compaction only shrinks messages, not tools**

The default compactor only compacts the **message** list. Tool definitions are not trimmed. If system + tools + minimal messages already exceed the window, compaction cannot fix it.

```python
# Compaction won't remove tool tokens
agent = Agent(
    # model=Model("openai/gpt-4o"),
    model=Model.Almock(),  # No API Key needed
    context=Context(max_tokens=8000, thresholds=[
        ContextThreshold(at=80, action=lambda ctx: ctx.compact() if ctx.compact else None),
    ]),
    tools=dozens_of_large_tools,  # e.g. 6000 tokens
)
# Even after compact(), tools still use 6000 tokens; you may still be over budget
```

**Mitigation:** Reduce the number or size of tools, or use a custom context manager that trims/selects tools.

---

**3. agent.context does not reflect per-call context**

**agent.context** still exposes the agent’s default **Context** config. For the config and stats **actually used** on a given call, use **result.context** and **result.context_stats** (set on every **Response** from **response()** / **arun()**).

```python
agent = Agent(
    # model=Model("openai/gpt-4o"),
    model=Model.Almock(),  # No API Key needed
    context=Context(max_tokens=128000),
)
result = agent.response("Summarize this", context=Context(max_tokens=4000))
# Default context (unchanged):
print(agent.context.max_tokens)  # 128000
# What was used for this call:
print(result.context.max_tokens)   # 4000
print(result.context_stats.total_tokens)
```

---

**4. reserve and model hint**

**Context** uses **reserve** (default 2000). You can make it model-aware by setting **default_reserve_tokens** on the model; **Context.get_capacity(model)** then uses that when building the capacity.

```python
# Per-model reserve via Model constructor or with_params
model = Model("openai/gpt-4o", default_reserve_tokens=8000)
agent = Agent(model=model, context=Context(max_tokens=128000))
# Budget for this agent uses reserve 8000 from the model
```

When the model has **default_reserve_tokens** set, that value is used for **reserve** when **get_capacity(model)** is called; otherwise **Context.reserve** is used.

---

**5. Threshold actions are synchronous only**

**ContextThreshold**’s **action** is **Callable[[ThresholdContext], None]**. You cannot await or run async work inside it. If you need to call an API or do async work before deciding to compact, you must do it outside the action (e.g. schedule compact on next prepare or use a sync stub).

```python
# This cannot be async
ContextThreshold(
    at=75,
    action=lambda ctx: ctx.compact() if ctx.compact else None,  # sync only
)
# If you needed: async def my_action(ctx): ... await fetch_something(); ctx.compact()
# you'd need a different design (e.g. sync wrapper that schedules compact)
```

---

**6. Stats overwritten on concurrent use**

**agent.context_stats** is the manager’s last **ContextStats**. With concurrent calls, use **result.context_stats** for per-call stats (each **Response** carries the stats for that run).

```python
import asyncio
async def run():
    t1 = asyncio.create_task(agent.arun("First"))
    t2 = asyncio.create_task(agent.arun("Second"))
    r1, r2 = await t1, await t2
    print(r1.context_stats.total_tokens)  # stats for "First"
    print(r2.context_stats.total_tokens)  # stats for "Second"
```

---

## Complete examples

Runnable scripts in **examples/11_context/** showcase snapshot, breakdown, thresholds, and compaction:

- **context_management.py** — Tour: basics, snapshot, manual compaction, thresholds, custom ContextManager. Run: `python -m examples.11_context.context_management`
- **context_snapshot_demo.py** — Full snapshot and breakdown (capacity, components, provenance, why_included, context_rot_risk, export). Run: `python -m examples.11_context.context_snapshot_demo`
- **context_thresholds_compaction_demo.py** — Threshold at 50% triggers compaction; events and stats. Run: `python -m examples.11_context.context_thresholds_compaction_demo`

See **examples/11_context/README.md** for a short index.

### Minimal (default context)

```python
from syrin import Agent, Model

agent = Agent(
    # model=Model("openai/gpt-4o"),
    model=Model.Almock(),  # No API Key needed
    system_prompt="You are helpful.",
)
result = agent.response("Hello!")
print(agent.context_stats.total_tokens, agent.context_stats.utilization)
```

### With thresholds and compaction

```python
from syrin import Agent, Context, Model
from syrin.threshold import ContextThreshold

def on_full(ctx):
    raise ValueError("Context full")

agent = Agent(
    # model=Model("openai/gpt-4o"),
    model=Model.Almock(),  # No API Key needed
    system_prompt="You are helpful.",
    context=Context(
        max_tokens=80000,
        thresholds=[
            ContextThreshold(at=50, action=lambda ctx: print(f"Tokens at {ctx.percentage}%")),
            ContextThreshold(at=75, action=lambda ctx: ctx.compact() if ctx.compact else None),
            ContextThreshold(at=100, action=on_full),
        ],
    ),
)
agent.events.on("context.compact", lambda e: print("Compacted:", e["method"]))
result = agent.response("A long message...")
```

### With budget (TokenLimits)

```python
from syrin import Agent, Context, TokenLimits, Model, TokenRateLimit
from syrin.budget import warn_on_exceeded

agent = Agent(
    # model=Model("openai/gpt-4o"),
    model=Model.Almock(),  # No API Key needed
    context=Context(
        max_tokens=128000,
        token_limits=TokenLimits(
            run=30_000,
            per=TokenRateLimit(hour=100_000),
            on_exceeded=warn_on_exceeded,
        ),
    ),
)
```

### With memory and full config

```python
from syrin import Agent, Context, Model
from syrin.threshold import ContextThreshold
from syrin.memory import Memory
from syrin.enums import MemoryType

def on_full(ctx):
    raise ValueError("Context full")

agent = Agent(
    # model=Model("openai/gpt-4o"),
    model=Model.Almock(),  # No API Key needed
    system_prompt="You are helpful.",
    memory=Memory(types=[MemoryType.CORE, MemoryType.EPISODIC]),
    context=Context(
        max_tokens=128000,
        thresholds=[
            ContextThreshold(at=50, action=lambda ctx: print(f"Tokens at {ctx.percentage}%")),
            ContextThreshold(at=75, action=lambda ctx: ctx.compact() if ctx.compact else None),
            ContextThreshold(at=100, action=on_full),
        ],
    ),
)
agent.events.on("context.compact", lambda e: print("Compacted:", e["method"]))
agent.events.on("context.threshold", lambda e: print("Threshold:", e["percent"]))
result = agent.response("Hello!")
print(agent.context_stats.total_tokens, agent.context_stats.compacted)
```
