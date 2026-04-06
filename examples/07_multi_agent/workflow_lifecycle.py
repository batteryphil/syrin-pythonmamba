"""Workflow lifecycle — play(), pause, inspect RunHandle, resume, cancel.

Demonstrates the full lifecycle control API for a Workflow:

- wf.play(task)           — start in background, return RunHandle immediately
- handle.status           — check current WorkflowStatus
- await wf.pause(mode)    — request pause after current step or drain
- await wf.resume()       — resume a paused workflow
- await wf.cancel()       — cancel; resume() afterwards raises WorkflowCancelledError
- await handle.wait()     — block until completion, return Response

Also demonstrates:
- Checkpoint backend for durable state (cross-process resume)
- PauseMode.DRAIN — waits for current step to finish before pausing
- PauseMode.AFTER_CURRENT_STEP — default, pause after current step
- WorkflowCancelledError when resuming after cancel

Run:
    OPENAI_API_KEY=sk-... uv run python examples/07_multi_agent/workflow_lifecycle.py
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

from dotenv import load_dotenv

from syrin import Agent, Model
from syrin.checkpoint._core import MemoryCheckpointBackend, SQLiteCheckpointBackend
from syrin.enums import PauseMode, WorkflowStatus
from syrin.workflow import Workflow
from syrin.workflow.exceptions import WorkflowCancelledError

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

_MODEL = Model.OpenAI("gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"))

# ── Agent definitions ─────────────────────────────────────────────────────────


class ResearchAgent(Agent):
    """Researches market trends and produces a structured summary."""

    model = _MODEL
    system_prompt = (
        "You are a research analyst. Summarise the key market trends "
        "for the given topic in 2-3 bullet points."
    )


class AnalysisAgent(Agent):
    """Analyses the research findings and identifies strategic implications."""

    model = _MODEL
    system_prompt = (
        "You are a strategic analyst. Given research findings, identify "
        "the 2 most important strategic implications for investment decisions."
    )


class WriterAgent(Agent):
    """Drafts a concise executive report from the analysis."""

    model = _MODEL
    system_prompt = (
        "You write concise executive reports. Given strategic analysis, "
        "produce a one-paragraph recommendation for senior leadership."
    )


# ── Example 1: play() + wait() ────────────────────────────────────────────────
#
# play() is non-blocking — it starts the workflow in a background task and
# returns a RunHandle.  Use handle.wait() to await the final result.


async def example_play_and_wait() -> None:
    print("\n── Example 1: play() + handle.wait() ────────────────────────────")

    wf = Workflow("play-demo").step(ResearchAgent).step(AnalysisAgent).step(WriterAgent)

    handle = wf.play("AI market trends in enterprise software")
    print(f"  Status immediately after play(): {handle.status}")

    result = await handle.wait()
    print(f"  Status after wait():             {handle.status}")
    print(f"  Result preview:                  {result.content[:120]}...")
    print(f"  Cost:                            ${result.cost:.6f}")


# ── Example 2: pause() + resume() ────────────────────────────────────────────
#
# Call pause() after play() to suspend execution after the current step
# finishes.  Call resume() to continue from where execution stopped.


async def example_pause_resume() -> None:
    print("\n── Example 2: pause() + resume() ────────────────────────────────")

    wf = Workflow("pause-resume-demo").step(ResearchAgent).step(AnalysisAgent).step(WriterAgent)

    handle = wf.play("Cloud infrastructure investment trends")

    # Let the first step start, then request a pause
    await asyncio.sleep(0.1)
    await wf.pause(PauseMode.AFTER_CURRENT_STEP)
    print(f"  Requested pause. Status: {handle.status}")

    # Simulate a human-in-the-loop review window
    await asyncio.sleep(0.05)
    print("  (Paused — simulating human review...)")

    await wf.resume()
    print(f"  Resumed. Status: {handle.status}")

    result = await handle.wait()
    print(f"  Final status: {handle.status}")
    print(f"  Result: {result.content[:100]}...")


# ── Example 3: PauseMode.DRAIN ────────────────────────────────────────────────
#
# DRAIN waits for all pending tool calls within the current step to complete
# before pausing.  Use when you need a clean checkpoint boundary.


async def example_pause_drain() -> None:
    print("\n── Example 3: PauseMode.DRAIN ────────────────────────────────────")

    wf = Workflow("drain-demo").step(ResearchAgent).step(AnalysisAgent).step(WriterAgent)

    handle = wf.play("Renewable energy market opportunities")
    await asyncio.sleep(0.1)

    await wf.pause(PauseMode.DRAIN)
    print(f"  Paused (DRAIN). Status: {handle.status}")

    await wf.resume()
    result = await handle.wait()
    print(f"  Completed. Cost: ${result.cost:.6f}")


# ── Example 4: cancel() ───────────────────────────────────────────────────────
#
# cancel() stops the workflow.  Subsequent resume() raises WorkflowCancelledError.


async def example_cancel() -> None:
    print("\n── Example 4: cancel() ───────────────────────────────────────────")

    wf = Workflow("cancel-demo").step(ResearchAgent).step(AnalysisAgent).step(WriterAgent)

    handle = wf.play("Semiconductor supply chain trends")
    await asyncio.sleep(0.1)

    await wf.cancel()
    print(f"  Status after cancel(): {handle.status}")

    try:
        await wf.resume()
    except WorkflowCancelledError as exc:
        print(f"  resume() raised WorkflowCancelledError (expected): {exc}")

    try:
        await handle.wait()
    except WorkflowCancelledError:
        print("  handle.wait() raised WorkflowCancelledError (expected)")
    except Exception as exc:
        print(f"  handle.wait() raised {type(exc).__name__}: {exc}")


# ── Example 5: In-memory checkpoint backend ──────────────────────────────────
#
# Pass a CheckpointBackendProtocol to Workflow() to persist step state after
# each step completes.  MemoryCheckpointBackend stores state in-process.


async def example_checkpoint_in_memory() -> None:
    print("\n── Example 5: In-memory checkpoint backend ──────────────────────")

    backend = MemoryCheckpointBackend()

    wf = (
        Workflow("checkpoint-demo", checkpoint_backend=backend)
        .step(ResearchAgent)
        .step(AnalysisAgent)
        .step(WriterAgent)
    )

    result = await wf.run("Biotech investment landscape 2025")
    run_id = result.run_id if hasattr(result, "run_id") else "run-0"
    print(f"  Completed. Cost: ${result.cost:.6f}")
    print(f"  Run ID: {run_id}")
    print(f"  Checkpoint entries: {len(backend._checkpoints)}")


async def example_checkpoint_sqlite() -> None:
    print("\n── Example 6: SQLite checkpoint backend (persistent) ────────────")

    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    try:
        backend = SQLiteCheckpointBackend(path=db_path)

        wf = (
            Workflow("sqlite-checkpoint-demo", checkpoint_backend=backend)
            .step(ResearchAgent)
            .step(AnalysisAgent)
            .step(WriterAgent)
        )

        result = await wf.run("FinTech regulatory changes 2025")
        print(f"  Completed. Cost: ${result.cost:.6f}")
        print(f"  SQLite DB: {db_path}")
        print("  (In production, save result.run_id and pass resume_run_id= to restart.)")
    finally:
        os.unlink(db_path)


# ── Example 7: Inspect RunHandle status mid-run ───────────────────────────────
#
# RunHandle.status is a WorkflowStatus enum you can poll at any time.


async def example_inspect_handle() -> None:
    print("\n── Example 7: Inspect RunHandle status mid-run ──────────────────")

    wf = Workflow("inspect-handle-demo").step(ResearchAgent).step(AnalysisAgent).step(WriterAgent)

    handle = wf.play("Healthcare AI disruption trends")

    statuses_seen: list[WorkflowStatus] = []
    for _ in range(5):
        statuses_seen.append(handle.status)
        await asyncio.sleep(0.05)

    await handle.wait()
    statuses_seen.append(handle.status)

    seen: list[str] = []
    for s in statuses_seen:
        if not seen or seen[-1] != str(s):
            seen.append(str(s))
    print(f"  Status transitions observed: {' → '.join(seen)}")


# ── Main ──────────────────────────────────────────────────────────────────────


async def main() -> None:
    await example_play_and_wait()
    await example_pause_resume()
    await example_pause_drain()
    await example_cancel()
    await example_checkpoint_in_memory()
    await example_checkpoint_sqlite()
    await example_inspect_handle()
    print("\nAll workflow lifecycle examples completed.")


if __name__ == "__main__":
    asyncio.run(main())
