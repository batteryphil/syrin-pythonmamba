"""Lifecycle and Flow Diagrams — Visualize agent structure.

Demonstrates:
- agent.lifecycle_diagram(): Mermaid state diagram of the full agent lifecycle
- agent.flow_diagram(): Mermaid flowchart with tools, memory, budget checkpoints
- Export to .md file
- Render inline (paste into https://mermaid.live)

Run:
    python examples/10_observability/lifecycle_diagrams.py

Output: prints Mermaid diagrams to stdout, exports to /tmp/lifecycle.md and /tmp/flow.md
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from syrin import Agent, Budget, Memory, Model, tool  # noqa: E402


@tool
def search_web(query: str) -> str:
    """Search the web for information."""
    return f"[mock] Results for: {query}"


@tool
def summarize_document(text: str) -> str:
    """Summarize a document."""
    return f"[mock] Summary of: {text[:50]}..."


class ResearchAgent(Agent):
    name = "research_agent"
    description = "Research agent with tools and memory"
    model = Model.mock(latency_min=1, latency_max=3, lorem_length=800, pricing_tier="high")
    system_prompt = "You are a research assistant."
    tools = [search_web, summarize_document]
    budget = Budget(max_cost=1.00)
    memory = Memory()


def main() -> None:
    agent = ResearchAgent()

    # --- Lifecycle diagram ---
    lifecycle = agent.lifecycle_diagram()

    print("=" * 60)
    print(" LIFECYCLE DIAGRAM (Mermaid stateDiagram-v2)")
    print("=" * 60)
    print(lifecycle)
    print()

    # Export to file
    lifecycle_path = Path("/tmp/lifecycle.md")
    agent.lifecycle_diagram(str(lifecycle_path))
    print(f"Exported to: {lifecycle_path}")
    print()

    # --- Flow diagram ---
    flow = agent.flow_diagram()

    print("=" * 60)
    print(" FLOW DIAGRAM (Mermaid flowchart)")
    print("=" * 60)
    print(flow)
    print()

    flow_path = Path("/tmp/flow.md")
    agent.flow_diagram(str(flow_path))
    print(f"Exported to: {flow_path}")

    print()
    print("To render: paste the diagram source at https://mermaid.live")
    print("Or: use agent.serve(enable_playground=True) to see it interactively.")


if __name__ == "__main__":
    main()
