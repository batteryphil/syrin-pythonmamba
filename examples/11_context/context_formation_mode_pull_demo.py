"""Formation mode: pull-based context demo.

Shows formation_mode=PULL: conversation segments are stored in Memory and
retrieved by relevance to the current prompt. Use for long conversations
where only relevant past turns should be included.

Run: python -m examples.11_context.context_formation_mode_pull_demo
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

from examples.models.models import gpt4_mini
from syrin import Agent, Context, Model
from syrin.context import FormationMode
from syrin.memory import Memory

load_dotenv(Path(__file__).resolve().parent.parent / ".env")
_model = gpt4_mini if os.environ.get("USE_REAL_MODEL") == "1" else Model.mock()


def _main() -> None:
    print("=== Formation mode: pull (relevance-based retrieval) ===\n")

    agent = Agent(
        model=_model,
        system_prompt="You are helpful. Keep answers brief.",
        memory=Memory(),
        context=Context(
            max_tokens=8000, formation_mode=FormationMode.PULL, pull_top_k=10, pull_threshold=0.0
        ),
    )

    agent.run("Tell me about Python in one sentence.")
    agent.run("Tell me about JavaScript in one sentence.")
    agent.run("Tell me about Rust in one sentence.")

    # Ask about Python — only Python-related segments should be pulled
    result = agent.run("Give me a Python example")
    snap = agent.context.snapshot()
    print(f"Response: {result.content[:80] if result.content else '(empty)'}...")
    print(f"pulled_segments count: {len(snap.pulled_segments)}")
    print(f"pull_scores: {snap.pull_scores}")
    for i, seg in enumerate(snap.pulled_segments[:3]):
        print(
            f"  [{i}] role={seg.get('role')} score={seg.get('score'):.2f} content={seg.get('content', '')[:50]}..."
        )


if __name__ == "__main__":
    _main()
