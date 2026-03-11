"""AgentTeam Example.

Demonstrates:
- Creating an AgentTeam with multiple specialized agents
- Shared budget across team members
- Automatic agent selection for tasks (team.select_agent)
- Running tasks via team.run_task()

Run: python examples/07_multi_agent/team.py
"""

from __future__ import annotations

from syrin import Agent, Budget, Model, prompt
from syrin.agent.multi_agent import AgentTeam

model = Model.Almock()


@prompt
def researcher_prompt(domain: str) -> str:
    return f"You are a researcher specializing in {domain}."


@prompt
def writer_prompt(style: str) -> str:
    return f"You are a writer with a {style} style."


# --- Specialized team agents ---

class Researcher(Agent):
    _agent_name = "researcher"
    _agent_description = "Researches topics (technology)"
    model = Model.Almock()
    system_prompt = researcher_prompt(domain="technology")


class Writer(Agent):
    _agent_name = "writer"
    _agent_description = "Writes content in engaging style"
    model = Model.Almock()
    system_prompt = writer_prompt(style="engaging")


# --- General-purpose agents for selection demo ---

class GeneralResearcher(Agent):
    _agent_name = "general-researcher"
    _agent_description = "General researcher"
    model = Model.Almock()
    system_prompt = researcher_prompt(domain="general")


class GeneralWriter(Agent):
    _agent_name = "general-writer"
    _agent_description = "General writer"
    model = Model.Almock()
    system_prompt = writer_prompt(style="general")


if __name__ == "__main__":
    # 1. Team with shared budget — run_task routes to the best agent
    print("=== Team with shared budget ===\n")
    team = AgentTeam(
        agents=[Researcher(), Writer()],
        budget=Budget(run=0.50, shared=True),
    )
    result = team.run_task("Research AI trends")
    print(f"Result: {result.content[:80]}...")
    print(f"Cost: ${result.cost:.6f}\n")

    # 2. Agent selection — team picks the best agent for a given task
    print("=== Agent selection ===\n")
    team = AgentTeam(agents=[GeneralResearcher(), GeneralWriter()])
    selected = team.select_agent("research machine learning")
    print(f"Task 'research ML' -> {selected.__class__.__name__}")
    selected = team.select_agent("write an article about AI")
    print(f"Task 'write article' -> {selected.__class__.__name__}")

    # Optional: serve both agents via AgentRouter
    # from syrin.serve import AgentRouter
    # router = AgentRouter(agents=[Researcher(), Writer()])
    # router.serve(port=8000, enable_playground=True, debug=True)
