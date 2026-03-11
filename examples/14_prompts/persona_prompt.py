"""Persona Prompt with @system_prompt In-Class Example.

Demonstrates:
- @system_prompt decorator: encapsulate the prompt inside the agent class
- One system prompt per agent, generated from a method
- Method receives template_variables as keyword arguments

Run: python examples/14_prompts/persona_prompt.py
"""

from __future__ import annotations

from syrin import Agent, Model, system_prompt


class PersonaAgent(Agent):
    _agent_name = "persona-agent"
    _agent_description = "Agent with @system_prompt in-class"
    model = Model.Almock(latency_seconds=0.01, lorem_length=60)

    @system_prompt
    def my_prompt(self, user_name: str = "", tone: str = "professional") -> str:
        """In-class system prompt. Receives template_variables."""
        return f"You assist {user_name or 'the user'}. Be {tone}."


if __name__ == "__main__":
    # Create agent with template variables that feed into @system_prompt
    agent = PersonaAgent(template_variables={"user_name": "Carol", "tone": "witty"})

    print("--- Persona Prompt Example ---")
    print(f"System prompt resolves to: {agent._system_prompt}")

    r = agent.response("What's your personality?")
    print(f"Response: {r.content[:120]}")
    print(f"Cost: ${r.cost:.6f}")
    print("Done.")

    # Optional: serve the agent with playground UI
    # agent.serve(port=8000, enable_playground=True, debug=True)
