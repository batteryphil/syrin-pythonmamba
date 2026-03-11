"""Prompt Decorator Example.

Demonstrates:
- @prompt decorator for parameterized system prompts
- Creating specialized agents from a single template
- Dynamic prompt resolution at runtime via template_variables

Run: python examples/14_prompts/prompt_decorator.py
"""

from __future__ import annotations

from syrin import Agent, Model, prompt

mock = Model.Almock(latency_seconds=0.01, lorem_length=50)


@prompt
def expert_prompt(domain: str, tone: str = "professional") -> str:
    """Generate a system prompt for expert agents."""
    return f"You are an expert in {domain}. Provide accurate, detailed, and {tone} responses."


@prompt
def role_prompt(role: str, specialization: str = "") -> str:
    """Generate a role-based system prompt."""
    base = f"You are a {role}."
    if specialization:
        base += f" You specialize in {specialization}."
    return base


# --- 1. Parameterized prompts assigned at class level ---


class ScienceExpert(Agent):
    model = mock
    system_prompt = expert_prompt(domain="quantum physics", tone="academic")


class BusinessExpert(Agent):
    model = mock
    system_prompt = expert_prompt(domain="business strategy", tone="practical")


# --- 2. Role-based prompts ---


class Researcher(Agent):
    model = mock
    system_prompt = role_prompt(role="researcher", specialization="machine learning")


class Writer(Agent):
    model = mock
    system_prompt = role_prompt(role="technical writer")


if __name__ == "__main__":
    print("--- Prompt Decorator Example ---\n")

    # Different agents from the same template
    science = ScienceExpert()
    business = BusinessExpert()
    question = "What is innovation?"

    r1 = science.response(question)
    r2 = business.response(question)
    print(f"Science expert:  {r1.content[:80]}")
    print(f"Business expert: {r2.content[:80]}")

    # Role-based prompts
    researcher = Researcher()
    writer = Writer()
    print(f"\nResearcher prompt: {researcher._system_prompt}")
    print(f"Writer prompt:     {writer._system_prompt}")

    # Dynamic prompt at runtime using template_variables
    print("\n--- Dynamic prompts per domain ---")
    for domain in ["Python", "JavaScript", "Rust"]:
        agent = Agent(
            model=mock,
            system_prompt=expert_prompt,
            template_variables={"domain": domain, "tone": "concise"},
        )
        result = agent.response(f"What is {domain} best for?")
        print(f"  {domain}: {result.content[:60]}")

    print("\nDone.")

    # Optional: serve the agent with playground UI
    # agent.serve(port=8000, enable_playground=True, debug=True)
