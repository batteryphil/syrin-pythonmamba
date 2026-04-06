"""Custom compaction prompt — override summarization prompts and use an LLM for compaction.

Shows:
- Context.compaction_prompt: user prompt template (use {messages} for conversation text)
- Context.compaction_system_prompt: optional system prompt for the summarization LLM
- Context.compaction_model: model used when compaction runs (None = placeholder, no LLM)
- When compaction runs with a model, the Summarizer calls the model with these prompts

Run: python -m examples.11_context.context_custom_compaction_prompt_demo
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

from examples.models.models import gpt4_mini
from syrin import Agent
from syrin.context import CompactionMethod, Context
from syrin.model import Model
from syrin.threshold import ContextThreshold

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

# Use real gpt-4o-mini when USE_REAL_MODEL=1
_model: Model = gpt4_mini if os.environ.get("USE_REAL_MODEL") == "1" else Model.mock()


def main() -> None:
    print("=" * 60)
    print("CUSTOM COMPACTION PROMPT — Override summarization prompts + optional LLM")
    print("=" * 60)

    compact_events: list[dict] = []

    # Use a small context so we hit threshold and trigger compaction.
    # compaction_model=Model.mock(): when summarization runs (many messages, overage >= 1.5), use Model.mock().
    agent = Agent(
        model=_model,
        system_prompt="You are helpful. Be brief.",
        context=Context(
            max_tokens=120,
            reserve=20,
            compaction_prompt="Summarize the following in one short paragraph. Keep key facts:\n\n{messages}",
            compaction_system_prompt="You are a summarization assistant. Output only the summary.",
            compaction_model=_model,
            thresholds=[
                ContextThreshold(at=50, action=lambda evt: evt.compact() if evt.compact else None),
            ],
        ),
    )

    agent.events.on("context.compact", lambda e: compact_events.append(e))

    print(
        "\n📐 Setup: compaction_prompt, compaction_system_prompt, compaction_model (gpt-4o-mini when USE_REAL_MODEL=1)"
    )
    print(
        "   When compaction runs, the Summarizer uses these prompts (and the model if summarization path runs).\n"
    )

    # One long prompt to exceed 50% and trigger compaction (method may be middle_out or summarize)
    long_prompt = "Explain in one sentence: " + " what is the meaning of life? " * 12
    print("  Sending a long prompt to trigger compaction...\n")
    agent.run(long_prompt)

    stats = agent.context_stats
    print(f"  Compaction ran: {stats.compacted}, method: {stats.compact_method}")

    # Show how to discover all methods and why this method was used
    print(f"\n📋 All compaction methods: {[m.value for m in CompactionMethod]}")
    if stats.compact_method == CompactionMethod.MIDDLE_OUT_TRUNCATE:
        print(
            "  → middle_out_truncate: overage (tokens/budget) was < 1.5. Use smaller budget or more messages to get 'summarize'."
        )

    if compact_events:
        print("\n📤 context.compact events:")
        for e in compact_events:
            print(
                f"  method={e.get('method')}, tokens_before={e.get('tokens_before')}, "
                f"tokens_after={e.get('tokens_after')}"
            )

    print("\n" + "=" * 60)
    print("Set Context(compaction_prompt=..., compaction_system_prompt=..., compaction_model=...)")
    print("to customize summarization. Use {messages} in the prompt for the conversation text.")
    print("=" * 60)


if __name__ == "__main__":
    main()
