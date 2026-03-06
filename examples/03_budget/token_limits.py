"""Token Limits Example.

Demonstrates:
- TokenLimits for per-run and per-window token caps
- TokenRateLimit for hourly/daily token windows
- Context(token_limits=TokenLimits(...)) to apply token caps
- Combining Budget (USD) with TokenLimits (tokens)

Run: python -m examples.03_budget.token_limits
Visit: http://localhost:8000/playground

Requires: uv pip install syrin[serve]
"""

from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv

from examples.models.models import almock
from syrin import Agent, Budget, Context, TokenLimits, TokenRateLimit, warn_on_exceeded

load_dotenv(Path(__file__).resolve().parent.parent / ".env")


# 1. TokenLimits — per-run token cap
agent = Agent(
    model=almock,
    context=Context(token_limits=TokenLimits(run=15_000, on_exceeded=warn_on_exceeded)),
)
result = agent.response("What is machine learning?")
print(f"Tokens used: {result.tokens.total_tokens}")

# 2. TokenRateLimit — hourly/daily windows
agent = Agent(
    model=almock,
    context=Context(
        token_limits=TokenLimits(
            run=15_000,
            per=TokenRateLimit(hour=50_000, day=200_000),
            on_exceeded=warn_on_exceeded,
        )
    ),
)
result = agent.response("Tell me about Python.")
print(f"Tokens used: {result.tokens.total_tokens}")

# 3. Combining Budget (USD) + TokenLimits (tokens)
agent = Agent(
    model=almock,
    system_prompt="You are concise.",
    budget=Budget(run=0.05, on_exceeded=warn_on_exceeded),
    context=Context(
        token_limits=TokenLimits(
            run=15_000,
            per=TokenRateLimit(hour=50_000, day=200_000),
            on_exceeded=warn_on_exceeded,
        )
    ),
)
result = agent.response("What is AI in one paragraph?")
print(f"Cost: ${result.cost:.6f}, Tokens: {result.tokens.total_tokens}")
print(f"Budget state: {agent.budget_state}")


class TokenLimitedAgent(Agent):
    """Agent with Budget + TokenLimits + TokenRateLimit."""

    _agent_name = "token-limited"
    _agent_description = "Agent with token limits (per-run, hourly, daily)"
    model = almock
    system_prompt = "You are concise."
    budget = Budget(run=0.05, on_exceeded=warn_on_exceeded)
    context = Context(
        token_limits=TokenLimits(
            run=15_000,
            per=TokenRateLimit(hour=50_000, day=200_000),
            on_exceeded=warn_on_exceeded,
        )
    )


if __name__ == "__main__":
    agent = TokenLimitedAgent()
    print("Serving at http://localhost:8000/playground")
    agent.serve(port=8000, enable_playground=True, debug=True)
