"""Handoff and Spawn Context Visibility.

Demonstrates how Syrin exposes context metadata during handoffs and spawns:
- HANDOFF_START / HANDOFF_BLOCKED include handoff_context (ContextSnapshot)
- SPAWN_START includes context_inherited, initial_context_tokens, parent_context_tokens
- Using handoff_context for logging and audit (total_tokens, context_rot_risk, to_dict())

Run: python examples/07_multi_agent/handoff_spawn_context_visibility.py
"""

from __future__ import annotations

from syrin import Agent, Hook, Model

model = Model.Almock()


class Researcher(Agent):
    model = Model.Almock()
    system_prompt = "You are a researcher. Answer briefly."


class Writer(Agent):
    model = Model.Almock()
    system_prompt = "You write clearly."


class ChildAgent(Agent):
    model = Model.Almock()
    system_prompt = "You are a helper."


def main_handoff_context() -> None:
    """Show handoff_context in HANDOFF_START (ContextSnapshot)."""
    print("=== Handoff context visibility ===\n")

    source = Researcher()

    def on_handoff_start(ctx) -> None:
        handoff_context = ctx.get("handoff_context")
        if handoff_context is not None:
            print(f"  handoff_context: total_tokens={handoff_context.total_tokens}")
            print(f"    utilization_pct={handoff_context.utilization_pct:.1f}%")
            print(f"    context_rot_risk={handoff_context.context_rot_risk}")
            print(
                f"    breakdown: system={handoff_context.breakdown.system_tokens}, "
                f"messages={handoff_context.breakdown.messages_tokens}"
            )
            # Export for audit / JSON
            d = handoff_context.to_dict()
            print(f"    (to_dict keys: {list(d.keys())[:6]}...)")
        print(
            f"  source_agent={ctx.source_agent} -> target_agent={ctx.target_agent}, "
            f"task={ctx.task[:40]}...\n"
        )

    source.events.on(Hook.HANDOFF_START, on_handoff_start)

    # 1) Handoff without prior response: snapshot is empty (total_tokens=0)
    print("1) Handoff without prior response (source never ran response()):\n")
    result = source.handoff(
        Writer, "Write one sentence about solar energy.", transfer_context=False
    )
    print(f"Result: {result.content[:80]}...\n")

    # 2) Handoff after source ran response(): snapshot has non-zero tokens
    print("2) Handoff after source ran response() (snapshot reflects last prepare):\n")
    source.response("What are key benefits of wind energy? Answer in one sentence.")
    result2 = source.handoff(Writer, "Summarize that in five words.", transfer_context=False)
    print(f"Result: {result2.content[:80]}...\n")


def main_spawn_context_metadata() -> None:
    """Show context_inherited, initial_context_tokens, parent_context_tokens in SPAWN_START."""
    print("=== Spawn context metadata ===\n")

    parent = Researcher()

    def on_spawn_start(ctx) -> None:
        print(f"  SPAWN_START: {ctx.source_agent} -> child {ctx.child_agent}")
        print(f"    context_inherited={ctx.get('context_inherited')}")
        print(f"    initial_context_tokens={ctx.get('initial_context_tokens')}")
        print(f"    parent_context_tokens={ctx.get('parent_context_tokens')}")
        print()

    parent.events.on(Hook.SPAWN_START, on_spawn_start)

    # 1) Spawn without prior response: parent_context_tokens=0
    print("1) Spawn without prior response (parent never ran response()):\n")
    result = parent.spawn(ChildAgent, task="Say hello in one word.")
    print(f"Result: {result.content}\n")

    # 2) Spawn after parent ran response(): parent_context_tokens > 0
    print("2) Spawn after parent ran response() (parent_context_tokens from last prepare):\n")
    parent.response("What is 2+2? Answer with one number.")
    result2 = parent.spawn(ChildAgent, task="Reply with the number 4.")
    print(f"Result: {result2.content}\n")


def main() -> None:
    main_handoff_context()
    main_spawn_context_metadata()
    print("Done. Handoff and spawn hooks now expose context snapshot and spawn metadata.")


if __name__ == "__main__":
    main()
