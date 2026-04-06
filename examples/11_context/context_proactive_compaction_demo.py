"""Proactive compaction — compact automatically when utilization hits a fraction (e.g. 60%).

Shows:
- Context.auto_compact_at: one knob to compact once per prepare when utilization >= that fraction
- No need for a ContextThreshold just to compact at 60%; set auto_compact_at=0.6
- Same context.compact event and compactor as threshold-triggered compaction
- Reduces context rot (research suggests keeping utilization under ~60–70%)

Run: python -m examples.11_context.context_proactive_compaction_demo
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

from examples.models.models import gpt4_mini
from syrin import Agent, Model
from syrin.context import Context

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

_model = gpt4_mini if os.environ.get("USE_REAL_MODEL") == "1" else Model.mock()


def main() -> None:
    print("=" * 60)
    print("PROACTIVE COMPACTION — auto_compact_at (e.g. 60% utilization)")
    print("=" * 60)

    compact_events: list[dict] = []

    # Small context so we exceed 60% utilization; proactive compact runs once
    agent = Agent(
        model=_model,
        system_prompt="You are a helpful assistant. Be very brief.",
        context=Context(
            max_tokens=200,
            reserve=20,
            auto_compact_at=0.6,  # Compact when utilization >= 60%
        ),
    )

    agent.events.on("context.compact", lambda e: compact_events.append(e))

    print("\n📐 Setup: max_tokens=200, reserve=20 → 180 available.")
    print("   auto_compact_at=0.6 → compact once when utilization ≥ 60%.\n")

    long_prompt = "Summarize in one sentence: " + " the key idea of each paragraph. " * 12
    print("  Sending a long prompt to push utilization over 60%...\n")
    agent.run(long_prompt)

    snap = agent.context.snapshot()
    stats = agent.context_stats
    print(f"  After response: {snap.total_tokens} tokens, {snap.utilization_pct:.0f}% utilization")
    print(f"  Context rot risk: {snap.context_rot_risk}")
    if stats.compacted:
        print(f"  Proactive compaction ran: {stats.compact_method}")

    if compact_events:
        print("\n📤 context.compact events:")
        for e in compact_events:
            print(
                f"  method={e.get('method')}, tokens_before={e.get('tokens_before')}, "
                f"tokens_after={e.get('tokens_after')}"
            )

    print("\n" + "=" * 60)
    print("Use Context(auto_compact_at=0.6) to proactively compact at 60%.")
    print("None = no proactive compaction (default); only thresholds trigger compact.")
    print("=" * 60)


if __name__ == "__main__":
    main()
