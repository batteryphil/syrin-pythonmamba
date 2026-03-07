"""Context mode demo: full vs focused.

Shows how Context.context_mode controls conversation history:
- full (default): full history, compaction when over capacity
- focused: keep only last N turns (user+assistant pairs)

Use focused when the user switches topics; older turns are excluded.

Run: python -m examples.11_context.context_context_mode_demo
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

from examples.models.models import almock, gpt4_mini
from syrin import Agent, Context
from syrin.context import ContextMode
from syrin.memory import Memory

load_dotenv(Path(__file__).resolve().parent.parent / ".env")
_model = gpt4_mini if os.environ.get("USE_REAL_MODEL") == "1" else almock


def _main() -> None:
    print("=== Context mode: focused (keep last 3 turns) ===\n")

    agent = Agent(
        model=_model,
        system_prompt="You are helpful. Keep answers brief.",
        memory=Memory(),  # Keep conversation history across turns
        context=Context(
            max_tokens=8000,
            context_mode=ContextMode.FOCUSED,
            focused_keep=3,
        ),
    )

    # Simulate a long conversation, then a topic shift
    for i in range(1, 6):
        agent.response(f"Tell me about topic {i} in one sentence.")
    snap_before = agent.context.snapshot()
    print(
        f"After 5 turns, context_mode={snap_before.context_mode}, dropped={snap_before.context_mode_dropped_count}"
    )

    # Topic shift: user asks something new
    result = agent.response("What is 2+2?")
    snap = agent.context.snapshot()
    print(
        f"After topic shift: context_mode={snap.context_mode}, dropped={snap.context_mode_dropped_count}"
    )
    print(f"Response: {result.content or '(empty)'}\n")

    # Compare: full mode keeps everything (subject to compaction)
    print("=== Context mode: full (default) ===\n")
    agent_full = Agent(
        model=_model,
        system_prompt="You are helpful.",
        context=Context(max_tokens=8000, context_mode=ContextMode.FULL),
    )
    agent_full.response("Hello")
    snap_full = agent_full.context.snapshot()
    print(f"context_mode={snap_full.context_mode}, dropped={snap_full.context_mode_dropped_count}")


if __name__ == "__main__":
    _main()
