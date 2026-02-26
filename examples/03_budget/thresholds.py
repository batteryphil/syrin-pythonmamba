"""Budget Thresholds Example.

Demonstrates:
- BudgetThreshold(at=, action=) for callback at spend percentage
- ThresholdMetric.COST and ThresholdMetric.TOKENS
- Threshold fallthrough (multiple thresholds fire in order)
- Class-level thresholds

Run: python -m examples.03_budget.thresholds
"""

from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv

from examples.models.models import almock
from syrin import Agent, Budget, warn_on_exceeded
from syrin.enums import ThresholdMetric
from syrin.threshold import BudgetThreshold, ThresholdContext

load_dotenv(Path(__file__).resolve().parent.parent / ".env")


# 1. Simple threshold — warn at 50%
events: list[str] = []


def on_50_pct(ctx: ThresholdContext) -> None:
    events.append(f"50% threshold: {ctx.percentage}%")
    print(f"  Threshold fired: {ctx.percentage}% of budget used")


agent = Agent(
    model=almock,
    budget=Budget(
        run=0.10,
        thresholds=[
            BudgetThreshold(at=50, action=on_50_pct, metric=ThresholdMetric.COST),
        ],
        on_exceeded=warn_on_exceeded,
    ),
)
agent.response("Hello!")
print(f"Events collected: {events}")

# 2. Multiple thresholds — fallthrough
levels: list[int] = []


def make_handler(pct: int):
    def handler(ctx: ThresholdContext) -> None:
        levels.append(pct)
        print(f"  {pct}% threshold reached")

    return handler


agent = Agent(
    model=almock,
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
print(f"Thresholds triggered: {levels}")


# 3. Class-level thresholds
class MonitoredAgent(Agent):
    model = almock
    budget = Budget(
        run=1.00,
        thresholds=[
            BudgetThreshold(
                at=80,
                action=lambda ctx: print(f"  80% budget used: ${ctx.current_value:.4f}"),
                metric=ThresholdMetric.COST,
            ),
        ],
        on_exceeded=warn_on_exceeded,
    )


agent = MonitoredAgent()
agent.response("Hello!")
print(f"Budget state: {agent.budget_state}")
