"""Context snapshot and breakdown -- see exactly what is in the context window.

This example shows Syrin's full visibility into context formation:
- Token breakdown by component (system, tools, memory, messages)
- Per-message preview with role, snippet, token count, and source
- Provenance (where each segment came from) and why_included (human-readable)
- Context rot risk (low / medium / high from utilization)
- Export via snapshot.to_dict() for dashboards or viz tools

Run:
    python examples/11_context/context_snapshot_demo.py
"""

from __future__ import annotations

from syrin import Agent, tool
from syrin.context import Context
from syrin.context.snapshot import ContextSegmentSource
from syrin.model import Model

_model = Model.mock()


@tool
def add_numbers(a: int, b: int) -> int:
    """Add two integers. Use when the user asks for a sum."""
    return a + b


@tool
def get_word_count(text: str) -> int:
    """Return the number of words in the given text."""
    return len(text.split())


def _source_label(source: ContextSegmentSource) -> str:
    return source.value.replace("_", " ").title()


def main() -> None:
    print("=" * 60)
    print("CONTEXT SNAPSHOT & BREAKDOWN -- What's in the context window?")
    print("=" * 60)

    agent = Agent(
        model=_model,
        system_prompt="You are a concise assistant. Use tools when asked to compute.",
        context=Context(max_tokens=8_000),
        tools=[add_numbers, get_word_count],
    )

    # Run a short conversation; tools are in context so breakdown shows tools_tokens
    agent.run("What is 2 + 2?")
    agent.run("Use the add tool: what is 10 + 20?")
    agent.run("Summarize what we did in one line.")

    snap = agent.context.snapshot()
    stats = agent.context_stats

    # --- Capacity ---
    print("\nCAPACITY")
    print(f"  Total tokens used:    {snap.total_tokens}")
    print(f"  Max tokens (window):  {snap.max_tokens}")
    print(f"  Tokens available:     {snap.tokens_available}")
    print(f"  Utilization:          {snap.utilization_pct:.1f}%")
    print(f"  Context rot risk:     {snap.context_rot_risk}")

    # --- Breakdown by component ---
    print("\nBREAKDOWN (tokens per component)")
    b = snap.breakdown
    print(f"  System prompt:   {b.system_tokens:>6} tokens")
    print(f"  Tool definitions:{b.tools_tokens:>6} tokens")
    print(f"  Memory (recall): {b.memory_tokens:>6} tokens")
    print(f"  Messages:        {b.messages_tokens:>6} tokens")
    print("  -------------------------")
    print(f"  Total:           {b.total_tokens:>6} tokens")

    # Same breakdown is on stats (set after each prepare)
    assert stats.breakdown is not None
    if b.tools_tokens > 0:
        print(f"\n  Tools are in context ({b.tools_tokens} tokens for tool definitions)")
    else:
        print("\n  (No tools on this agent -> tools_tokens = 0)")
    print("  (agent.context_stats.breakdown has the same data for this run)")

    # --- Why each part is included ---
    print("\nWHY INCLUDED (human-readable)")
    for reason in snap.why_included:
        print(f"  - {reason}")

    # --- Message preview (role, snippet, tokens, source) ---
    print("\nMESSAGE PREVIEW (role | snippet | tokens | source)")
    for i, p in enumerate(snap.message_preview[:12], 1):
        snippet = (
            p.content_snippet[:50] + "..." if len(p.content_snippet) > 50 else p.content_snippet
        )
        print(
            f"  {i:2}. [{p.role:8}] {snippet:52} {p.token_count:>4} tok  <- {_source_label(p.source)}"
        )
    if len(snap.message_preview) > 12:
        print(f"  ... and {len(snap.message_preview) - 12} more")

    # --- Provenance (for debugging / tooling) ---
    print("\nPROVENANCE (segment_id | source | detail)")
    for prov in snap.provenance[:8]:
        detail = prov.source_detail or ""
        print(f"  {prov.segment_id:>4}  {prov.source.value:16}  {detail}")
    if len(snap.provenance) > 8:
        print(f"  ... and {len(snap.provenance) - 8} more")

    # --- Export for viz / logging ---
    exported = snap.to_dict()
    print("\nEXPORT (snapshot.to_dict()) -- keys for dashboards / viz:")
    print(f"  {list(exported.keys())}")

    print("\n" + "=" * 60)
    print("Use agent.context.snapshot() after any response() to inspect context.")
    print("Use agent.context_stats.breakdown for component counts without full snapshot.")
    print("=" * 60)


if __name__ == "__main__":
    main()
