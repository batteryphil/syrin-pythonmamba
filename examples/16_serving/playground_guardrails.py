"""Playground: Agent with Guardrails Example.

Demonstrates:
- Agent with ContentFilter guardrail
- Blocked words: spam, scam
- Visit http://localhost:8000/playground — try messages with/without blocked words

Run: python -m examples.serving.playground_guardrails
"""

from __future__ import annotations

import sys
from pathlib import Path

_root = Path(__file__).resolve().parents[2]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from dotenv import load_dotenv

from syrin import Agent, Model
from syrin.guardrails import ContentFilter, GuardrailChain

load_dotenv(Path(__file__).resolve().parent.parent / ".env")


chain = GuardrailChain(
    [
        ContentFilter(blocked_words=["spam", "scam"], name="NoSpam"),
    ]
)


class GuardedAssistant(Agent):
    name = "guarded"
    description = "Assistant with content filter (blocks spam, scam)"
    model = Model.mock(latency_min=1, latency_max=3, lorem_length=800, pricing_tier="high")
    system_prompt = "You are a helpful assistant."
    guardrails = chain


if __name__ == "__main__":
    agent = GuardedAssistant()
    agent.serve(port=8000, enable_playground=True, debug=True)
