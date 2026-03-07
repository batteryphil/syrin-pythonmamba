"""Persistent context map demo (Step 12).

Shows map_backend="file", inject_map_summary=True: durable session index
(topics, decisions, summary) survives resets. Map summary is injected at prepare
when non-empty.

Run: python -m examples.11_context.context_map_demo
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

from examples.models.models import almock, gpt4_mini
from syrin import Agent, Context

load_dotenv(Path(__file__).resolve().parent.parent / ".env")
_model = gpt4_mini if os.environ.get("USE_REAL_MODEL") == "1" else almock


def _main() -> None:
    print("=== Persistent context map (session summary across resets) ===\n")

    map_path = Path(__file__).resolve().parent / ".demo_map.json"

    agent = Agent(
        model=_model,
        system_prompt="You are helpful. Stay grounded in any session summary provided.",
        context=Context(
            max_tokens=8000,
            map_backend="file",
            map_path=str(map_path),
            inject_map_summary=True,
        ),
    )

    # Simulate prior session: update map with summary
    agent.context.update_map(
        {
            "topics": ["Python", "type hints", "mypy"],
            "decisions": ["User prefers strict typing."],
            "summary": "User asked about Python type hints and mypy. We discussed Optional[str], Union, and strict mode.",
        }
    )

    # Current turn: model receives [Session summary]\n... before the user message
    result = agent.response("What should I enable in mypy for stricter checking?")
    print(f"Response: {result.content[:200] if result.content else '(empty)'}...\n")

    m = agent.context.get_map()
    print(f"Map topics: {m.topics}")
    print(f"Map summary: {m.summary[:80]}...")
    print(f"Map last_updated: {m.last_updated:.0f}")

    # Cleanup demo file
    if map_path.exists():
        map_path.unlink()


if __name__ == "__main__":
    _main()
