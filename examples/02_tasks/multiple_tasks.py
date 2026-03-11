"""Multiple Tasks Example -- Agent with several @task methods.

Demonstrates:
- Defining multiple tasks on a single agent
- Chaining tasks: research first, then write
- Each task is an independent named entry point

Run: python examples/02_tasks/multiple_tasks.py
"""

from __future__ import annotations

from syrin import Agent, Model, task

# --- Define the agent with two tasks ---


class Writer(Agent):
    """Agent with research and write tasks."""

    _agent_name = "writer"
    _agent_description = "Writer with research(topic) and write(topic, style) tasks"
    model = Model.Almock()
    system_prompt = "You are a professional writer. Research thoroughly and write clearly."

    @task
    def research(self, topic: str) -> str:
        """Research a topic and return key points."""
        r = self.response(f"Research {topic}. List 3-5 key points.")
        return r.content or ""

    @task
    def write(self, topic: str, style: str = "professional") -> str:
        """Write about a topic in the given style."""
        r = self.response(f"Write a short paragraph about {topic} in a {style} style.")
        return r.content or ""


# --- Run it ---

if __name__ == "__main__":
    writer = Writer()

    # Call each task independently
    print("--- Research Task ---")
    research_result = writer.research("artificial intelligence")
    print(research_result)

    print("\n--- Write Task ---")
    write_result = writer.write("artificial intelligence", style="casual")
    print(write_result)

    # Optional: serve with playground UI
    # writer.serve(port=8000, enable_playground=True, debug=True)
