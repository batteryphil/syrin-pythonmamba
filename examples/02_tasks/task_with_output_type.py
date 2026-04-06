"""Task with Output Type -- Returning structured data from an agent method.

Demonstrates:
- Output(Model): agent returns a typed Pydantic instance via result.output
- A triage method that unwraps result.output instead of manually parsing text
- Type-safe access — no dict keys, no string splitting

Run: python examples/02_tasks/task_with_output_type.py
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

from syrin import Agent, Model, Output


class TriageResult(BaseModel):
    """Structured triage result."""

    priority: Literal["high", "medium", "low"]
    category: str
    summary: str


_triage_json = (
    '{"priority": "high", "category": "infrastructure", '
    '"summary": "CPU has been at 98% for 10 minutes — likely runaway process or traffic spike."}'
)


class TriageAgent(Agent):
    """Classifies incidents by priority, category, and summary."""

    name = "triage"
    model = Model.mock(response_mode="custom", custom_response=_triage_json)
    system_prompt = (
        "You are a triage assistant. Return a JSON object with: "
        "priority (high/medium/low), category, and summary."
    )
    output = Output(TriageResult)

    def triage(self, item: str) -> TriageResult:
        """Triage an item and return a typed TriageResult."""
        result = self.run(f"Triage: {item}")
        return result.output  # type: ignore[return-value]


if __name__ == "__main__":
    agent = TriageAgent()

    result = agent.triage("Server CPU at 98% for the last 10 minutes")
    print("Triage result:")
    print(f"  priority: {result.priority}")
    print(f"  category: {result.category}")
    print(f"  summary:  {result.summary}")
