"""Token Limits — Cap token usage per-run and per-window.

Demonstrates:
- TokenLimits for per-run and per-window token caps
- TokenRateLimit for hourly/daily token windows
- Context(token_limits=TokenLimits(...)) to apply token caps
- Combining Budget (USD) with TokenLimits (tokens)
- Class-level token limits via Agent subclass

No API key needed (uses Almock).

Run:
    python examples/03_budget/token_limits.py
"""

from __future__ import annotations

from syrin import (
    Agent,
    AgentConfig,
    Budget,
    Context,
    Model,
    TokenLimits,
    TokenRateLimit,
    warn_on_exceeded,
)

# Create a mock model — no API key needed
model = Model.Almock()

# ---------------------------------------------------------------------------
# 1. TokenLimits — per-run token cap
# ---------------------------------------------------------------------------
print("=" * 60)
print("1. Per-run token cap (15,000 tokens)")
print("=" * 60)

agent = Agent(
    model=model,
    config=AgentConfig(
        context=Context(token_limits=TokenLimits(run=15_000, on_exceeded=warn_on_exceeded))
    ),
)
result = agent.response("What is machine learning?")
print(f"   Tokens used: {result.tokens.total_tokens}")

# ---------------------------------------------------------------------------
# 2. TokenRateLimit — hourly/daily windows
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("2. Per-run cap + hourly/daily windows")
print("=" * 60)

agent = Agent(
    model=model,
    config=AgentConfig(
        context=Context(
            token_limits=TokenLimits(
                run=15_000,
                per=TokenRateLimit(hour=50_000, day=200_000),
                on_exceeded=warn_on_exceeded,
            )
        )
    ),
)
result = agent.response("Tell me about Python.")
print(f"   Tokens used: {result.tokens.total_tokens}")

# ---------------------------------------------------------------------------
# 3. Combining Budget (USD) + TokenLimits (tokens)
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("3. Budget ($0.05/run) + token limits (15k/run, 50k/hour)")
print("=" * 60)

agent = Agent(
    model=model,
    system_prompt="You are concise.",
    budget=Budget(run=0.05, on_exceeded=warn_on_exceeded),
    config=AgentConfig(
        context=Context(
            token_limits=TokenLimits(
                run=15_000,
                per=TokenRateLimit(hour=50_000, day=200_000),
                on_exceeded=warn_on_exceeded,
            )
        )
    ),
)
result = agent.response("What is AI in one paragraph?")
print(f"   Cost:         ${result.cost:.6f}")
print(f"   Tokens:       {result.tokens.total_tokens}")
print(f"   Budget state: {agent.budget_state}")

# ---------------------------------------------------------------------------
# 4. Class-level token limits (reusable agent definition)
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("4. Class-level: Budget + TokenLimits + TokenRateLimit")
print("=" * 60)


class TokenLimitedAgent(Agent):
    """Agent with Budget + TokenLimits + TokenRateLimit."""

    _agent_name = "token-limited"
    _agent_description = "Agent with token limits (per-run, hourly, daily)"
    model = model
    system_prompt = "You are concise."
    budget = Budget(run=0.05, on_exceeded=warn_on_exceeded)
    context = Context(
        token_limits=TokenLimits(
            run=15_000,
            per=TokenRateLimit(hour=50_000, day=200_000),
            on_exceeded=warn_on_exceeded,
        )
    )


agent = TokenLimitedAgent()
result = agent.response("Summarize quantum computing in two sentences.")
print(f"   Cost:         ${result.cost:.6f}")
print(f"   Tokens:       {result.tokens.total_tokens}")
print(f"   Budget state: {agent.budget_state}")

# --- Serve with web playground (uncomment to try) ---
# agent.serve(port=8000, enable_playground=True, debug=True)
# Visit http://localhost:8000/playground
