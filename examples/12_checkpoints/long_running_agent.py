"""Long-running agent: checkpoint + memory + auto_compact_at.

Demonstrates:
- Session continuity across restarts (checkpoint stores messages + context snapshot)
- save_checkpoint / load_checkpoint restores conversation and iteration
- Context.auto_compact_at for proactive compaction
- Memory for conversation history (restored on load)

Run: python -m examples.12_checkpoints.long_running_agent
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from syrin import Agent, CheckpointConfig, CheckpointTrigger, Context, Model
from syrin.memory import Memory


def main() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "long_running.db"

        mem = Memory()
        agent = Agent(
            model=Model.mock(),
            system_prompt="You are a helpful assistant. Keep replies concise.",
            memory=mem,
            context=Context(
                max_tokens=8000,
                auto_compact_at=0.6,
            ),
            checkpoint=CheckpointConfig(
                storage="sqlite",
                path=str(db_path),
                trigger=CheckpointTrigger.STEP,
                max_checkpoints=5,
            ),
        )

        # Simulate multi-turn conversation
        agent.run("What is 2+2?")
        agent.run("And 3+3?")
        print(f"After 2 turns: {len(agent.messages)} messages")

        # Save checkpoint
        cid = agent.save_checkpoint(reason="step")
        print(f"Saved checkpoint: {cid}")
        ids = agent.list_checkpoints()
        print(f"Available checkpoints: {ids}")

        # Simulate restart: new agent, load from same storage
        mem2 = Memory()
        agent2 = Agent(
            model=Model.mock(),
            system_prompt="You are a helpful assistant. Keep replies concise.",
            memory=mem2,
            context=Context(max_tokens=8000, auto_compact_at=0.6),
            checkpoint=CheckpointConfig(
                storage="sqlite",
                path=str(db_path),
                trigger=CheckpointTrigger.STEP,
            ),
        )
        ok = agent2.load_checkpoint(cid)
        assert ok, "Load should succeed"
        print(f"After restore: {len(agent2.messages)} messages, iteration={agent2.iteration}")

        # Continue conversation
        r = agent2.run("What did we discuss?")
        print(f"Follow-up reply: {r.content[:80]}...")
        print("Long-running agent demo complete.")


if __name__ == "__main__":
    main()
