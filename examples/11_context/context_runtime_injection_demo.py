"""Runtime context injection demo.

Shows how to inject RAG results or dynamic context at prepare time using:
- Context.runtime_inject (callable)
- Per-call inject via agent.run(inject=...)

Run: python -m examples.11_context.context_runtime_injection_demo
Use real gpt-4o-mini: USE_REAL_MODEL=1 python -m examples.11_context.context_runtime_injection_demo
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

from examples.models.models import gpt4_mini
from syrin import Agent, Context, Model
from syrin.context import InjectPlacement, PrepareInput

load_dotenv(Path(__file__).resolve().parent.parent / ".env")
_model = gpt4_mini if os.environ.get("USE_REAL_MODEL") == "1" else Model.mock()


def _main() -> None:
    # Simulated RAG: return docs based on user_input
    fake_docs = {
        "syrin": "Syrin is a Python library for building AI agents with budget management.",
        "context": "Context management controls token limits, compaction, and the context window.",
    }

    def rag_injector(inp: PrepareInput) -> list[dict]:
        """Inject RAG results before the current user message."""
        keyword = inp.user_input.split()[0].lower() if inp.user_input else ""
        doc = fake_docs.get(keyword, "No matching document.")
        return [{"role": "system", "content": f"[RAG]\n{doc}"}]

    agent = Agent(
        model=_model,
        system_prompt="You are helpful. Use the [RAG] block when relevant.",
        context=Context(
            max_tokens=8000,
            runtime_inject=rag_injector,
            inject_placement=InjectPlacement.BEFORE_CURRENT_TURN,
            inject_source_detail="rag",
        ),
    )

    print("=== Runtime injection via Context.runtime_inject ===\n")
    result = agent.run("What is syrin?")
    snap = agent.context.snapshot()

    # Show injected content in snapshot
    injected = [p for p in snap.provenance if p.source.value == "injected"]
    print(f"Injected segments: {len(injected)}")
    for p in injected:
        print(f"  - source_detail: {p.source_detail}")

    why_injected = [w for w in snap.why_included if "injected" in w.lower()]
    print(f"Why included (injected): {why_injected}")

    content = result.content or "(empty)"
    preview = content[:250] + "..." if len(content) > 250 else content
    print(f"Model response (used RAG):\n  {preview}\n")

    if snap.breakdown:
        print(f"Breakdown injected_tokens: {snap.breakdown.injected_tokens}")

    print("\n=== Per-call inject (overrides runtime_inject) ===\n")
    agent_no_runtime = Agent(
        model=_model, system_prompt="You are helpful.", context=Context(max_tokens=8000)
    )

    agent_no_runtime.run(
        "Summarize this",
        inject=[{"role": "system", "content": "[Injected] Custom per-call context here."}],
        inject_source_detail="custom",
    )

    snap2 = agent_no_runtime.context.snapshot()
    injected2 = [p for p in snap2.provenance if p.source.value == "injected"]
    print(f"Per-call injected segments: {len(injected2)}")
    print(f"source_detail: {injected2[0].source_detail if injected2 else 'N/A'}")


if __name__ == "__main__":
    _main()
