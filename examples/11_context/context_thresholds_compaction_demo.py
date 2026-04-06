"""Context thresholds and compaction — react when the context window fills up.

Shows:
- Context.thresholds: run custom logic at utilization levels (e.g. 50%, 75%)
- Compaction: reduce message list size (middle-out truncation or summarization)
  when a threshold fires and the action calls evt.compact()
- Events: context.threshold and context.compact for observability
- Snapshot and stats after prepare (compacted=True, compact_method set)

Run: python -m examples.11_context.context_thresholds_compaction_demo
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

from examples.models.models import gpt4_mini
from syrin import Agent
from syrin.context import Context
from syrin.model import Model
from syrin.threshold import ContextThreshold

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

# Use real gpt-4o-mini when USE_REAL_MODEL=1
_model: Model = gpt4_mini if os.environ.get("USE_REAL_MODEL") == "1" else Model.mock()


def main() -> None:
    print("=" * 60)
    print("CONTEXT THRESHOLDS & COMPACTION — React when context fills up")
    print("=" * 60)

    # Collect events from hooks
    threshold_events: list[dict] = []
    compact_events: list[dict] = []

    # Very small context window so we hit the 50% threshold and trigger compaction
    # Available = max_tokens - reserve (e.g. 120 - 20 = 100). 50% = 50 tokens.
    agent = Agent(
        model=_model,
        system_prompt="You are a helpful assistant. Be very brief.",
        context=Context(
            max_tokens=120,
            reserve=20,
            thresholds=[
                ContextThreshold(at=50, action=lambda evt: evt.compact()),  # compact at 50%
            ],
        ),
    )

    agent.events.on("context.threshold", lambda e: threshold_events.append(e))
    agent.events.on("context.compact", lambda e: compact_events.append(e))

    print("\n📐 Setup: max_tokens=120, reserve=20 → 100 available.")
    print("   Threshold: at 50% → run compaction.\n")

    # One long user message to push over 50% and trigger compaction
    long_prompt = "Explain in one sentence: " + " what is the meaning of life? " * 8
    print(f"  Sending a long prompt ({len(long_prompt)} chars) to fill context...\n")
    agent.run(long_prompt)

    snap = agent.context.snapshot()
    stats = agent.context_stats
    print(f"  After response: {snap.total_tokens} tokens, {snap.utilization_pct:.0f}% utilization")
    if stats.compacted:
        print(f"  Compaction ran: {stats.compact_method}")

    # --- What happened? ---
    print("\n📊 RESULTS")
    snap = agent.context.snapshot()
    stats = agent.context_stats
    print(f"  Total tokens now:  {snap.total_tokens}")
    print(f"  Utilization:       {snap.utilization_pct:.1f}%")
    print(f"  Context rot risk:  {snap.context_rot_risk}")
    print(f"  Compaction ran:    {stats.compacted}")
    if stats.compacted:
        print(f"  Compact method:    {stats.compact_method}")
        print(f"  Compact count:     {stats.compact_count} time(s) this prepare")

    if compact_events:
        print("\n📤 context.compact events (from hook):")
        for e in compact_events:
            print(
                f"  method={e.get('method')}, tokens_before={e.get('tokens_before')}, tokens_after={e.get('tokens_after')}"
            )

    if threshold_events:
        print("\n📤 context.threshold events:")
        for t in threshold_events:
            print(
                f"  at={t.get('at')}%, percent={t.get('percent')}, tokens={t.get('tokens')}/{t.get('max_tokens')}"
            )

    print("\n" + "=" * 60)
    print("Use ContextThreshold(at=N, action=...) to run code at N% utilization.")
    print("Call evt.compact() inside the action to compact and free space.")
    print("=" * 60)


if __name__ == "__main__":
    main()
