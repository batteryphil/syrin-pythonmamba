"""Task with Output Type -- Returning structured data from a @task.

Demonstrates:
- A @task that returns a dict with structured fields
- TriageAgent that classifies items by priority, category, and summary
- Parsing agent output into a structured result

Run: python examples/02_tasks/task_with_output_type.py
"""

from __future__ import annotations

from syrin import Agent, Model, task


# --- Define the triage agent ---

class TriageAgent(Agent):
    """Agent that triages items with structured output."""

    _agent_name = "triage"
    _agent_description = "Triage agent returning priority, category, summary"
    model = Model.Almock()
    system_prompt = (
        "You are a triage assistant. For each item, return priority (high/medium/low), "
        "category, and a brief summary. Be concise."
    )

    @task
    def triage(self, item: str) -> dict:
        """Triage an item. Returns dict with priority, category, summary."""
        response = self.response(
            f"Triage this item: {item}. "
            "Respond with: priority (high/medium/low), category, and summary."
        )
        # Parse structure from response (Almock returns lorem; a real model returns JSON)
        content = response.content or ""
        return {
            "priority": "medium",
            "category": "general",
            "summary": content[:100] if content else "No summary",
        }


# --- Run it ---

if __name__ == "__main__":
    agent = TriageAgent()

    result = agent.triage("Server CPU at 98% for the last 10 minutes")
    print("Triage result:")
    for key, value in result.items():
        print(f"  {key}: {value}")

    # Optional: serve with playground UI
    # agent.serve(port=8000, enable_playground=True, debug=True)
