"""Context Management -- control what fits inside the model's context window.

Demonstrates:
- Context(max_tokens=) to set the window size
- agent.context_stats and agent.context.snapshot() after a response
- Token breakdown by component (system, tools, messages)
- Manual compaction with MiddleOutTruncator and ContextCompactor
- ContextThreshold to fire actions at utilization percentages
- Custom ContextManager via Protocol

Run:
    python examples/11_context/context_management.py
"""

from __future__ import annotations

from typing import Any

from syrin import Agent, Context, Model
from syrin.context import (
    ContextCompactor,
    ContextManager,
    ContextPayload,
    ContextWindowCapacity,
    MiddleOutTruncator,
)
from syrin.context.counter import TokenCounter
from syrin.threshold import ContextThreshold

model = Model.mock()


def section(title: str) -> None:
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# 1. Context basics -- max_tokens, one response, then stats
# ---------------------------------------------------------------------------
section("1. Context basics (max_tokens, stats after response)")

agent = Agent(
    model=model,
    system_prompt="You are a helpful assistant.",
    context=Context(max_tokens=80_000),
)
agent.run("What is 2+2? Answer in one sentence.")

stats = agent.context_stats
print(f"  Max tokens (window): {agent.context.max_tokens}")
print(f"  Total tokens used:   {stats.total_tokens}")
print(f"  Utilization:         {stats.utilization:.2%}")

# ---------------------------------------------------------------------------
# 2. Snapshot and breakdown -- what is in the window right now?
# ---------------------------------------------------------------------------
section("2. Snapshot & breakdown")

snap = agent.context.snapshot()
print(f"  Utilization:   {snap.utilization_pct:.1f}%")
print(f"  Context rot:   {snap.context_rot_risk}")
print(
    f"  Breakdown:     system={snap.breakdown.system_tokens}, "
    f"tools={snap.breakdown.tools_tokens}, "
    f"messages={snap.breakdown.messages_tokens}"
)
print(f"  Why included:  {snap.why_included}")

# ---------------------------------------------------------------------------
# 3. Manual compaction (MiddleOutTruncator, ContextCompactor)
# ---------------------------------------------------------------------------
section("3. Manual compaction")

counter = TokenCounter()
messages: list[dict[str, str]] = [{"role": "system", "content": "You are helpful."}]
for i in range(15):
    messages.append({"role": "user", "content": f"Message {i}: Tell me about topic {i}."})
    messages.append({"role": "assistant", "content": f"Response about topic {i}. " * 30})

before = counter.count_messages(messages).total
truncator = MiddleOutTruncator()
result = truncator.compact(messages, 2000, counter)
after = counter.count_messages(result.messages).total
print(f"  MiddleOutTruncator: {before} -> {after} tokens (budget 2000)")

messages2: list[dict[str, str]] = [{"role": "system", "content": "You are helpful."}]
for i in range(8):
    messages2.append({"role": "user", "content": f"User {i} " + "x" * 80})
    messages2.append({"role": "assistant", "content": f"Response {i} " + "y" * 80})
before2 = counter.count_messages(messages2).total
compact_result = ContextCompactor().compact(messages2, 3000)
after2 = counter.count_messages(compact_result.messages).total
print(f"  ContextCompactor:   {before2} -> {after2} tokens (budget 3000)")

# ---------------------------------------------------------------------------
# 4. Thresholds -- fire actions at utilization percentages
# ---------------------------------------------------------------------------
section("4. Thresholds (actions at 50%, 70%, 100%)")

fired: list[int] = []
agent2 = Agent(
    model=model,
    system_prompt="You are helpful.",
    context=Context(
        max_tokens=5000,
        thresholds=[
            ContextThreshold(at=50, action=lambda _: fired.append(50)),
            ContextThreshold(at=70, action=lambda _: fired.append(70)),
            ContextThreshold(at=100, action=lambda _: fired.append(100)),
        ],
    ),
)
agent2.run("Hello!")
print(f"  Thresholds fired: {fired or 'none (utilization below 50%)'}")

# ---------------------------------------------------------------------------
# 5. Custom ContextManager (Protocol)
# ---------------------------------------------------------------------------
section("5. Custom ContextManager")


class PassThroughContextManager(ContextManager):
    """A no-op manager that passes messages through unchanged."""

    def prepare(
        self,
        messages: list[dict[str, Any]],
        system_prompt: str,
        tools: list[dict[str, Any]],
        memory_context: str = "",
        capacity: ContextWindowCapacity | None = None,
        context: Context | None = None,
        **kwargs: Any,
    ) -> ContextPayload:
        tok = TokenCounter()
        n = tok.count_messages(messages).total
        if system_prompt:
            n += tok.count(system_prompt) + tok._role_overhead("system")
        n += tok.count_tools(tools)
        return ContextPayload(
            messages=messages,
            system_prompt=system_prompt,
            tools=tools,
            tokens=n,
        )

    def on_compact(self, _event: Any) -> None:
        pass


agent3 = Agent(
    model=model,
    context=PassThroughContextManager(),
)
agent3.run("Hi, custom manager!")
print(f"  Pass-through manager: {agent3.context_stats.total_tokens} tokens, no compaction")

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
section("Summary")
print("  - Context(max_tokens=, reserve=, thresholds=) controls the window.")
print("  - After response(): agent.context_stats and agent.context.snapshot().")
print("  - MiddleOutTruncator / ContextCompactor for manual compaction.")
print("  - ContextThreshold(at=N, action=...) for utilization-based actions.")
print("  - Implement ContextManager Protocol for fully custom logic.")
print()

# ---------------------------------------------------------------------------
# Optional: serve with playground UI (requires syrin[serve])
# ---------------------------------------------------------------------------
# agent.serve(port=8000, enable_playground=True, debug=True)
