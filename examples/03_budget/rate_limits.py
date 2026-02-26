"""Rate Limits Example.

Demonstrates:
- RateLimit(hour/day/week/month) in USD
- month_days for configurable rolling window
- calendar_month=True for current calendar month
- APIRateLimit for RPM, TPM, RPD limits
- RateLimitThreshold with ThresholdMetric

Run: python -m examples.03_budget.rate_limits
"""

from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv

from examples.models.models import almock
from syrin import Agent, Budget, RateLimit, warn_on_exceeded
from syrin.enums import ThresholdMetric
from syrin.ratelimit import APIRateLimit
from syrin.threshold import RateLimitThreshold, ThresholdContext

load_dotenv(Path(__file__).resolve().parent.parent / ".env")


# 1. Budget with rate limits (USD caps per window)
agent = Agent(
    model=almock,
    budget=Budget(
        run=0.05,
        per=RateLimit(hour=2.00, day=10.00, month=100.00, month_days=30),
        on_exceeded=warn_on_exceeded,
    ),
)
result = agent.response("What is AI?")
print(f"Cost: ${result.cost:.6f}")
print(f"Budget state: {agent.budget_state}")

# 2. Configurable month window (month_days)
r7 = RateLimit(month=20.00, month_days=7)
r30 = RateLimit(month=100.0)
print(f"month_days=7: last {r7.month_days} days; default: {r30.month_days}")

# 3. Calendar month
r_cal = RateLimit(month=500.00, calendar_month=True)
print(f"calendar_month={r_cal.calendar_month}")

# 4. APIRateLimit for RPM / TPM / RPD
agent = Agent(model=almock, rate_limit=APIRateLimit(rpm=500, tpm=150_000))
print(f"Rate limit config: {agent.rate_limit}")


# 5. RateLimitThreshold — warn at 80% RPM
def on_warning(ctx: ThresholdContext) -> None:
    print(f"  WARNING: {ctx.metric} at {ctx.percentage}%")


agent = Agent(
    model=almock,
    rate_limit=APIRateLimit(
        rpm=100,
        thresholds=[
            RateLimitThreshold(at=80, action=on_warning, metric=ThresholdMetric.RPM),
        ],
    ),
)

# 6. Multiple thresholds (RPM + TPM)
agent = Agent(
    model=almock,
    rate_limit=APIRateLimit(
        rpm=500,
        tpm=150_000,
        thresholds=[
            RateLimitThreshold(
                at=50,
                action=lambda ctx: print(f"  RPM at {ctx.percentage}%"),
                metric=ThresholdMetric.RPM,
            ),
            RateLimitThreshold(
                at=70,
                action=lambda ctx: print(f"  TPM at {ctx.percentage}%"),
                metric=ThresholdMetric.TPM,
            ),
            RateLimitThreshold(
                at=100,
                action=lambda _: print("  RPM limit reached!"),
                metric=ThresholdMetric.RPM,
            ),
        ],
    ),
)
