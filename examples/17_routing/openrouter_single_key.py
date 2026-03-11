"""OpenRouter — single API key for multiple providers.

Uses OpenRouterBuilder to create models with one key. Route between
Claude, GPT, Gemini, etc. without separate API keys.

Requires: OPENROUTER_API_KEY env var.
Run: python examples/17_routing/openrouter_single_key.py
"""

from __future__ import annotations

import os
import sys

if not os.getenv("OPENROUTER_API_KEY"):
    print("Skipped: set OPENROUTER_API_KEY to run this example.")
    sys.exit(0)

from syrin import Agent
from syrin.model import OpenRouterBuilder
from syrin.router import RoutingConfig, RoutingMode

API_KEY = os.environ["OPENROUTER_API_KEY"]


def main() -> None:
    builder = OpenRouterBuilder(api_key=API_KEY)
    claude = builder.model("anthropic/claude-sonnet-4-5")
    gpt = builder.model("openai/gpt-4o-mini")
    gemini = builder.model("google/gemini-2.0-flash")

    agent = Agent(
        model=[claude, gpt, gemini],
        model_router=RoutingConfig(routing_mode=RoutingMode.COST_FIRST),
        system_prompt="You are helpful. Be concise.",
    )

    r = agent.response("Say hello in one sentence")
    print(f"Model used: {r.model_used or r.model}")
    print(f"Content: {r.content[:80]}...")
    if r.routing_reason:
        print(f"Routed to: {r.routing_reason.selected_model} | {r.routing_reason.reason}")


if __name__ == "__main__":
    main()
