"""Agent with Budget Example.

Demonstrates:
- Creating an Agent with a Budget
- Budget tracking via response.cost and agent.budget_state
- Budget limits and exceeded handling

Run: python ./examples/01_minimal/agent_with_budget.py
  or: python -m examples.01_minimal.agent_with_budget (from project root)
"""

from __future__ import annotations

import sys
from pathlib import Path

_root = Path(__file__).resolve().parents[2]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from dotenv import load_dotenv

from examples.models.models import almock
from syrin import Agent, Budget, warn_on_exceeded

load_dotenv(Path(__file__).resolve().parent.parent / ".env")


class Assistant(Agent):
    model = almock
    system_prompt = "You are a helpful assistant."
    budget = Budget(run=0.10, on_exceeded=warn_on_exceeded)


assistant = Assistant()
print(f"Budget: ${assistant.budget.run}")
result = assistant.response("Explain quantum computing briefly")
print(f"Response: {result.content[:80]}...")
print(f"Cost: ${result.cost:.6f}")
state = assistant.budget_state
print(
    f"Budget: spent=${state.spent:.4f}, remaining=${state.remaining:.4f}" if state else "No budget"
)
