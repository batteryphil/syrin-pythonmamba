"""Rate Limits — Control spend rate and API request limits.

Demonstrates:
- RateLimit(hour/day/week/month) in USD
- month_days for configurable rolling window
- calendar_month=True for current calendar month
- APIRateLimit for RPM, TPM, RPD limits
- RateLimitThreshold with ThresholdMetric

No API key needed (uses Almock).

Run:
    python examples/03_budget/rate_limits.py
"""

from __future__ import annotations

from syrin import Agent, AgentConfig, Budget, Model, RateLimit, warn_on_exceeded
from syrin.enums import ThresholdMetric
from syrin.ratelimit import APIRateLimit
from syrin.threshold import RateLimitThreshold, ThresholdContext

# Create a mock model — no API key needed
model = Model.Almock()

# ---------------------------------------------------------------------------
# 1. Budget with rate limits (USD caps per window)
# ---------------------------------------------------------------------------
print("=" * 60)
print("1. Budget with hourly/daily/monthly rate limits")
print("=" * 60)

agent = Agent(
    model=model,
    budget=Budget(
        run=0.05,
        per=RateLimit(hour=2.00, day=10.00, month=100.00, month_days=30),
        on_exceeded=warn_on_exceeded,
    ),
)
result = agent.response("What is AI?")
print(f"   Cost:         ${result.cost:.6f}")
print(f"   Budget state: {agent.budget_state}")

# ---------------------------------------------------------------------------
# 2. Configurable month window (month_days)
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("2. Configurable month window")
print("=" * 60)

r7 = RateLimit(month=20.00, month_days=7)
r30 = RateLimit(month=100.0)
print(f"   month_days=7:  last {r7.month_days} days")
print(f"   default:       last {r30.month_days} days")

# ---------------------------------------------------------------------------
# 3. Calendar month
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("3. Calendar month (resets on the 1st)")
print("=" * 60)

r_cal = RateLimit(month=500.00, calendar_month=True)
print(f"   calendar_month={r_cal.calendar_month}")

# ---------------------------------------------------------------------------
# 4. APIRateLimit for RPM / TPM / RPD
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("4. API rate limits (RPM, TPM)")
print("=" * 60)

agent = Agent(
    model=model,
    config=AgentConfig(rate_limit=APIRateLimit(rpm=500, tpm=150_000)),
)
print(f"   Rate limit config: {agent.rate_limit}")

# ---------------------------------------------------------------------------
# 5. RateLimitThreshold — warn at 80% RPM
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("5. Threshold callback at 80% RPM")
print("=" * 60)


def on_warning(ctx: ThresholdContext) -> None:
    print(f"   WARNING: {ctx.metric} at {ctx.percentage}%")


agent = Agent(
    model=model,
    config=AgentConfig(
        rate_limit=APIRateLimit(
            rpm=100,
            thresholds=[
                RateLimitThreshold(at=80, action=on_warning, metric=ThresholdMetric.RPM),
            ],
        )
    ),
)
print(f"   Agent configured with RPM=100, threshold at 80%")

# ---------------------------------------------------------------------------
# 6. Multiple thresholds (RPM + TPM)
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("6. Multiple thresholds (RPM at 50%, TPM at 70%, RPM at 100%)")
print("=" * 60)

agent = Agent(
    model=model,
    config=AgentConfig(
        rate_limit=APIRateLimit(
            rpm=500,
            tpm=150_000,
            thresholds=[
                RateLimitThreshold(
                    at=50,
                    action=lambda ctx: print(f"   RPM at {ctx.percentage}%"),
                    metric=ThresholdMetric.RPM,
                ),
                RateLimitThreshold(
                    at=70,
                    action=lambda ctx: print(f"   TPM at {ctx.percentage}%"),
                    metric=ThresholdMetric.TPM,
                ),
                RateLimitThreshold(
                    at=100,
                    action=lambda _: print("   RPM limit reached!"),
                    metric=ThresholdMetric.RPM,
                ),
            ],
        ),
    ),
)
print(f"   Agent configured with 3 thresholds on RPM/TPM")

# ---------------------------------------------------------------------------
# 7. Class-level rate limits (reusable agent definition)
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("7. Class-level rate-limited agent")
print("=" * 60)


class RateLimitedAgent(Agent):
    """Agent with rate limits (hour/day/month)."""

    _agent_name = "rate-limited"
    _agent_description = "Agent with rate limits (hour, day, month)"
    model = model
    budget = Budget(
        run=0.05,
        per=RateLimit(hour=2.00, day=10.00, month=100.00),
        on_exceeded=warn_on_exceeded,
    )


agent = RateLimitedAgent()
result = agent.response("Explain neural networks briefly.")
print(f"   Cost:         ${result.cost:.6f}")
print(f"   Budget state: {agent.budget_state}")

# --- Serve with web playground (uncomment to try) ---
# agent.serve(port=8000, enable_playground=True, debug=True)
# Visit http://localhost:8000/playground
