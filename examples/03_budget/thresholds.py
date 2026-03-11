"""Budget Thresholds — Get callbacks when spend hits a percentage.

Demonstrates:
- BudgetThreshold(at=, action=) for callback at spend percentage
- ThresholdMetric.COST and ThresholdMetric.TOKENS
- Threshold fallthrough (multiple thresholds fire in order)
- Class-level thresholds via Agent subclass

No API key needed (uses Almock).

Run:
    python examples/03_budget/thresholds.py
"""

from __future__ import annotations

from syrin import Agent, Budget, Model, warn_on_exceeded
from syrin.enums import ThresholdMetric
from syrin.threshold import BudgetThreshold, ThresholdContext

# Create a mock model — no API key needed
model = Model.Almock()

# ---------------------------------------------------------------------------
# 1. Simple threshold — warn at 50%
# ---------------------------------------------------------------------------
print("=" * 60)
print("1. Single threshold at 50% of budget")
print("=" * 60)

events: list[str] = []


def on_50_pct(ctx: ThresholdContext) -> None:
    events.append(f"50% threshold: {ctx.percentage}%")
    print(f"   Threshold fired: {ctx.percentage}% of budget used")


agent = Agent(
    model=model,
    budget=Budget(
        run=0.10,
        thresholds=[
            BudgetThreshold(at=50, action=on_50_pct, metric=ThresholdMetric.COST),
        ],
        on_exceeded=warn_on_exceeded,
    ),
)
agent.response("Hello!")
print(f"   Events collected: {events}")

# ---------------------------------------------------------------------------
# 2. Multiple thresholds — fallthrough
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("2. Multiple thresholds (25%, 50%, 75%, 100%)")
print("=" * 60)

levels: list[int] = []


def make_handler(pct: int):
    def handler(ctx: ThresholdContext) -> None:
        levels.append(pct)
        print(f"   {pct}% threshold reached")

    return handler


agent = Agent(
    model=model,
    budget=Budget(
        run=0.10,
        thresholds=[
            BudgetThreshold(at=25, action=make_handler(25)),
            BudgetThreshold(at=50, action=make_handler(50)),
            BudgetThreshold(at=75, action=make_handler(75)),
            BudgetThreshold(at=100, action=make_handler(100)),
        ],
        on_exceeded=warn_on_exceeded,
    ),
)
agent.response("Tell me about AI")
print(f"   Thresholds triggered: {levels}")

# ---------------------------------------------------------------------------
# 3. Class-level thresholds (reusable agent definition)
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("3. Class-level: warn at 80% of $1.00 budget")
print("=" * 60)


class MonitoredAgent(Agent):
    """Agent with budget thresholds — warns at 80% spend."""

    _agent_name = "monitored"
    _agent_description = "Agent with budget thresholds (warn at 80%)"
    model = model
    budget = Budget(
        run=1.00,
        thresholds=[
            BudgetThreshold(
                at=80,
                action=lambda ctx: print(f"   80% budget used: ${ctx.current_value:.4f}"),
                metric=ThresholdMetric.COST,
            ),
        ],
        on_exceeded=warn_on_exceeded,
    )


agent = MonitoredAgent()
result = agent.response("Explain gradient descent.")
print(f"   Cost:         ${result.cost:.6f}")
print(f"   Budget state: {agent.budget_state}")

# --- Serve with web playground (uncomment to try) ---
# agent.serve(port=8000, enable_playground=True, debug=True)
# Visit http://localhost:8000/playground
