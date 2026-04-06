"""Enterprise Budget — Answers to every question before production deployment.

Covers:
1.  Pre-call estimation vs. post-call actual cost
2.  What happens when a limit is exceeded (STOP / WARN / IGNORE)
3.  Keeping an important task alive when the budget is hit
4.  Getting warned *before* the budget is fully gone (thresholds)
5.  Model switching at a cost threshold
6.  What `safety_margin` does and why you need it
7.  Budget in observability (hooks / events)
8.  budget_summary() and export_costs() — the dashboard API
9.  Remote config — tweak limits without redeployment
10. Parallel multi-agent with a shared budget pool
11. Overriding model pricing (when providers update rates)
12. Custom BudgetStore — bring your own persistence backend

No API key needed — uses Almock (mock LLM).

Run:
    python examples/03_budget/enterprise_budget.py
"""

from __future__ import annotations

import json

from syrin import Agent, Budget, Model, RateLimit
from syrin.budget_store import BudgetStore, BudgetTracker
from syrin.cost import ModelPricing
from syrin.enums import ExceedPolicy, Hook
from syrin.exceptions import BudgetExceededError
from syrin.threshold import BudgetThreshold, ThresholdContext

model = Model.mock()

SEP = "=" * 60


# ============================================================
# 1. Pre-call estimation vs. post-call actual cost
# ============================================================
print(SEP)
print("1. PRE-CALL ESTIMATION vs. POST-CALL ACTUAL COST")
print(SEP)

# Syrin checks budget TWICE:
#   a) BEFORE the LLM call: estimate_cost(messages) → if estimate would exceed, fail fast
#   b) AFTER the LLM call:  actual tokens from provider → final cost recorded
#
# This means you're protected even before you spend a cent.

agent = Agent(model=model, budget=Budget(max_cost=1.00))
result = agent.run("What is machine learning?")

print(f"   Pre-call estimate:  ${result.cost_estimated or 0:.6f}  (computed before the call)")
print(f"   Post-call actual:   ${result.cost:.6f}  (from provider token usage)")
print(f"   Difference:         ${abs((result.cost_estimated or 0) - result.cost):.6f}")
print()
print("   → Budget check fires BEFORE the call so you never overshoot unexpectedly.")
print()

# ============================================================
# 2. What happens when a limit is exceeded
# ============================================================
print(SEP)
print("2. WHAT HAPPENS WHEN THE LIMIT IS EXCEEDED")
print(SEP)

# Four policies — pick one per agent:

# STOP (ExceedPolicy.STOP): raises BudgetExceededError immediately
agent_stop = Agent(
    model=model,
    budget=Budget(max_cost=0.0001, exceed_policy=ExceedPolicy.STOP),
)
try:
    agent_stop.run("This will trip the tiny limit")
except BudgetExceededError as e:
    print(f"   STOP  → BudgetExceededError raised: {str(e)[:70]}")

# WARN (ExceedPolicy.WARN): logs a warning, call continues
agent_warn = Agent(
    model=model,
    budget=Budget(max_cost=0.0001, exceed_policy=ExceedPolicy.WARN),
)
result_warn = agent_warn.run("This exceeds budget but continues")
print(f"   WARN  → logged warning, got response anyway. Cost: ${result_warn.cost:.6f}")

# IGNORE (ExceedPolicy.IGNORE): silently continues, no log
agent_ignore = Agent(
    model=model,
    budget=Budget(max_cost=0.0001, exceed_policy=ExceedPolicy.IGNORE),
)
result_ignore = agent_ignore.run("Budget ignored for this agent")
print(f"   IGNORE→ silent continue. Cost: ${result_ignore.cost:.6f}")


# WARN policy: log and continue (exceed_policy is the canonical way to configure behaviour)
agent_custom = Agent(
    model=model,
    budget=Budget(max_cost=0.0001, exceed_policy=ExceedPolicy.WARN),
)
agent_custom.run("Custom handler fires here")
print()

# ============================================================
# 3. Important task — do NOT stop when budget is hit
# ============================================================
print(SEP)
print("3. KEEPING AN IMPORTANT TASK ALIVE WHEN BUDGET IS HIT")
print(SEP)

# Use ExceedPolicy.WARN or ExceedPolicy.IGNORE so the task completes.
# Pair with `safety_margin` to guarantee budget for the reply.

agent_critical = Agent(
    model=model,
    budget=Budget(
        max_cost=0.0001,  # Tiny budget to trigger the scenario
        safety_margin=0.00005,  # Always hold this back for the final reply
        exceed_policy=ExceedPolicy.WARN,  # Warn but never abort
    ),
)
result = agent_critical.run("Process this critical financial report")
print(f"   Budget was exceeded, but task completed. Content length: {len(result.content)} chars")
print("   reserve ensures the reply has room — no mid-sentence cutoff.")
print()

# ============================================================
# 4. Getting warned BEFORE the budget is fully gone
# ============================================================
print(SEP)
print("4. PROACTIVE WARNINGS VIA THRESHOLDS")
print(SEP)

# Thresholds fire at a percentage of budget — before you hit the hard limit.
# This gives you time to react: alert ops, switch models, or stop new requests.

warnings_received: list[str] = []


def on_80_pct(ctx: ThresholdContext) -> None:
    msg = f"Budget at {ctx.percentage:.0f}% (${ctx.current_value:.4f} / ${ctx.limit_value:.4f})"
    warnings_received.append(msg)
    print(f"   ⚠ THRESHOLD: {msg}")


# Thresholds fire post-call based on actual spend vs max_cost.
# Set max_cost just above a single call's cost so one call crosses 80%.
# Almock costs ~$0.000038/call → max_cost=$0.000046 makes one call = 83% of budget.
agent_alert = Agent(
    model=model,
    budget=Budget(
        max_cost=0.000046,  # One Almock call (~$0.000038) = 83% → fires the 80% threshold
        thresholds=[BudgetThreshold(at=80, action=on_80_pct)],
        exceed_policy=ExceedPolicy.WARN,
    ),
)
agent_alert.run("This call will use ~83% of budget and fire the 80% threshold")
print(f"   Warnings received: {len(warnings_received)}")
print("   → Use thresholds to alert Slack/PagerDuty, log to Datadog, or switch models.")
print()

# ============================================================
# 5. Model switching at a cost threshold
# ============================================================
print(SEP)
print("5. AUTOMATIC MODEL SWITCHING AT A THRESHOLD")
print(SEP)

# When budget is 70% spent, downgrade to a cheaper model automatically.
# This is threshold-driven switching — no human intervention needed.

expensive = Model.mock()
cheap = Model.mock()  # In production: Model.OpenAI("gpt-4o-mini", ...)


def switch_to_cheap(ctx: ThresholdContext) -> None:
    if ctx.parent is not None:
        ctx.parent.switch_model(cheap)
        print(f"   Switched to cheap model at {ctx.percentage:.0f}% budget")


agent_switch = Agent(
    model=expensive,
    budget=Budget(
        max_cost=0.000046,  # One call = 83% → fires the 70% threshold, switches model
        thresholds=[BudgetThreshold(at=70, action=switch_to_cheap)],
        exceed_policy=ExceedPolicy.WARN,
    ),
)
agent_switch.run("This call trips the 70% threshold and switches to cheap model")
print("   → Use BudgetThreshold(at=N, action=switch_fn) to automate model switching.")
print()

# ============================================================
# 6. What safety_margin does and why you need it
# ============================================================
print(SEP)
print("6. WHAT RESERVE DOES")
print(SEP)

# max_cost=1.00, safety_margin=0.20:
#   effective limit for processing = $0.80
#   the remaining $0.20 is held back so the model always has room to write the reply
#
# Without safety_margin: agent might spend $0.99 on tool calls and get a 2-token reply.
# With safety_margin:    agent stops processing at $0.80 and replies with its full budget.

b = Budget(max_cost=1.00, safety_margin=0.20)
# effective_limit = max_cost - safety_margin = $0.80
print("   max_cost: $1.00  safety_margin: $0.20  → effective processing limit: $0.80")
print(f"   Summary: {Agent(model=model, budget=b).budget_summary()}")
print()

# ============================================================
# 7. Budget in observability (hooks / events)
# ============================================================
print(SEP)
print("7. BUDGET IN OBSERVABILITY (HOOKS)")
print(SEP)

budget_events: list[str] = []


def on_budget_event(payload: object) -> None:
    spent = getattr(payload, "budget_spent", 0.0)
    event = getattr(payload, "event", "")
    budget_events.append(f"{event}: cost=${spent:.6f}")


agent_obs = Agent(
    model=model,
    budget=Budget(max_cost=1.00, exceed_policy=ExceedPolicy.WARN),
)
# Subscribe to budget-related hook events
agent_obs.events.on(Hook.BUDGET_EXCEEDED, on_budget_event)
agent_obs.events.on(Hook.BUDGET_THRESHOLD, on_budget_event)

agent_obs.run("Observed call")
print(f"   Budget events captured: {budget_events or ['none (budget not exceeded)']}")
print("   → Every BUDGET_EXCEEDED and BUDGET_THRESHOLD event carries full state.")
print("   → Set debug=True on agent.serve() to see budget in every trace span.")
print()

# ============================================================
# 8. Budget dashboard — budget_summary() and export_costs()
# ============================================================
print(SEP)
print("8. BUDGET DASHBOARD: budget_summary() AND export_costs()")
print(SEP)

agent_dash = Agent(
    model=model,
    budget=Budget(max_cost=10.00, safety_margin=0.50, exceed_policy=ExceedPolicy.WARN),
)
agent_dash.run("First query")
agent_dash.run("Second query")

summary = agent_dash.budget_summary()
print("   budget_summary():")
for k, v in summary.items():
    print(f"     {k}: {v}")

print()
costs_json = agent_dash.export_costs(format="json")
parsed = json.loads(costs_json)
print(f"   export_costs(format='json'): {len(parsed)} entries")
for entry in parsed:
    print(
        f"     model={entry['model']}  cost=${entry['cost_usd']:.6f}  tokens={entry['total_tokens']}"
    )
print()

# ============================================================
# 9. Remote config — tweak budget without redeployment
# ============================================================
print(SEP)
print("9. REMOTE CONFIG — TWEAK BUDGET WITHOUT REDEPLOYMENT")
print(SEP)

# Budget implements RemoteConfigurable. A config server or feature-flag system
# can push new limits to a running agent without touching code or restarting.
#
# Supported fields via remote config: max_cost, safety_margin, shared.
# Rate limits are NOT remote-configurable (require restart to avoid race conditions).
#
# Usage with syrin's remote config system:
#
#   from syrin.remote import RemoteConfig
#
#   rc = RemoteConfig(source=YourConfigSource())
#   rc.attach(agent)
#   # Later, config source pushes: {"budget": {"max_cost": 5.0}}
#   # → agent's budget cap updates without restart
#
# Whitelisted to prevent arbitrary field injection (security by default):
#   GlobalConfig._PUBLIC_KEYS enforces only documented fields can be set remotely.

print("   Budget fields remotely configurable: max_cost, safety_margin, shared")
print("   Attach a RemoteConfig source to push live updates without restart.")
print("   → See docs/production/remote-config.md for a full walkthrough.")
print()

# ============================================================
# 10. Parallel multi-agent with shared budget pool
# ============================================================
print(SEP)
print("10. PARALLEL MULTI-AGENT — SHARED BUDGET POOL")
print(SEP)

# All child agents consume from the same Budget() pool.
# The BudgetTracker uses a thread-safe SQLite store so parallel writes don't corrupt.

shared_budget = Budget(max_cost=5.00, exceed_policy=ExceedPolicy.WARN)
orchestrator = Agent(model=model, budget=shared_budget)


class WorkerAgent(Agent):
    model = model


# Spawn three workers — each call borrows from the shared $5 pool
for i in range(3):
    orchestrator.spawn(WorkerAgent, task=f"Subtask {i + 1}")

print(f"   After 3 workers, orchestrator budget state: {orchestrator.budget_state}")
print("   → Budget.: all spawned children deduct from the same pool.")
print("   → Thread-safe: SQLite WAL mode prevents double-spend in parallel runs.")
print()

# ============================================================
# 11. Custom model pricing (when providers update rates)
# ============================================================
print(SEP)
print("11. CUSTOM MODEL PRICING — WHEN PROVIDERS UPDATE RATES")
print(SEP)

# Syrin ships a pricing table in src/syrin/cost/__init__.py.
# When a provider changes rates, you have two options:
#
# Option A: Override per-model at agent construction (immediate, no code change needed)
# Option B: Update the library pricing table and release a patch

# Option A — ModelPricing override (preferred for negotiated / updated rates)
custom_priced_model = Model(
    model_id="openai/gpt-4o",
    pricing=ModelPricing(
        input_per_1m=2.50,  # Update when OpenAI changes pricing
        output_per_1m=10.00,
    ),
)
print("   Custom pricing: $2.50/1M input, $10.00/1M output")
print("   → Pass pricing=ModelPricing(...) to any Model() to override the built-in table.")
print("   → Built-in table is in src/syrin/cost/__init__.py — update and cut a patch release.")
print()

# ============================================================
# 12. Custom BudgetStore — bring your own persistence backend
# ============================================================
print(SEP)
print("12. CUSTOM BUDGETSTORE — BRING YOUR OWN BACKEND")
print(SEP)

# BudgetStore is an ABC with two methods: load() and save().
# Implement it to persist budget state in PostgreSQL, Redis, DynamoDB, etc.


class PostgresBudgetStore(BudgetStore):
    """Example stub — replace with real DB calls."""

    def __init__(self) -> None:
        self._data: dict[str, BudgetTracker] = {}

    def load(self, key: str) -> BudgetTracker | None:
        # In production: SELECT * FROM budget_trackers WHERE key = %s
        return self._data.get(key)

    def save(self, key: str, tracker: BudgetTracker) -> None:
        # In production: UPSERT INTO budget_trackers (key, data) VALUES (%s, %s)
        self._data[key] = tracker


pg_store = PostgresBudgetStore()
agent_pg = Agent(
    model=model,
    budget=Budget(max_cost=10.00, rate_limits=RateLimit(day=50.00)),
    budget_store=pg_store,
    budget_store_key="user:enterprise_123",
)
agent_pg.run("First call — state saved to PostgresBudgetStore")
agent_pg.run("Second call — state accumulated across calls")
print(f"   Calls tracked in custom store. Budget state: {agent_pg.budget_state}")
print("   → Inherit BudgetStore and implement load()/save() for any backend.")
print("   → Built-in options: InMemoryBudgetStore (default), FileBudgetStore (JSON).")
print()

print(SEP)
print("SUMMARY")
print(SEP)
print("  Budget = per-run USD cap + rate windows + thresholds + callbacks")
print("  Two-stage checking: pre-call estimate + post-call actual")
print("  ExceedPolicy: STOP | WARN | IGNORE")
print("  safety_margin: always hold back budget for the reply")
print("  Thresholds: proactive alerts at any percentage")
print("  budget_summary() / export_costs(): programmatic dashboard")
print("  apply_remote_overrides(): live tuning without restart")
print("  : unified pool across parallel agents")
print("  ModelPricing: override rates when providers change")
print("  BudgetStore ABC: plug in any persistence backend")
