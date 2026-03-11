"""Single Task Example -- Using @task to define named entry points.

Demonstrates:
- Defining a named task with the @task decorator
- A Researcher agent with a research(topic) task method
- Invoking the task and printing the result

Run: python examples/02_tasks/single_task.py
"""

from __future__ import annotations

from syrin import Agent, Model, task

# --- Define the agent with a single task ---


class Researcher(Agent):
    """Agent that researches topics. Uses @task for a named API."""

    _agent_name = "researcher"
    _agent_description = "Research assistant with research(topic) task"
    model = Model.Almock()
    system_prompt = "You are a research assistant. Provide concise, factual summaries."

    @task
    def research(self, topic: str) -> str:
        """Research a topic and return a summary."""
        response = self.response(f"Research the following topic and summarize: {topic}")
        return response.content or ""


# --- Run it ---

if __name__ == "__main__":
    researcher = Researcher()

    # Call the task directly and print output
    result = researcher.research("quantum computing")
    print("Research result:")
    print(result)

    # Optional: serve with playground UI
    # researcher.serve(port=8000, enable_playground=True, debug=True)
