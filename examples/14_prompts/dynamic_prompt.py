"""Dynamic Prompts -- inject runtime variables into system prompts.

Demonstrates:
- @prompt decorator with template_variables for runtime injection
- Class-level template_variables on an Agent
- Instance-level overrides (constructor)
- Per-call overrides via response(template_variables=...)
- effective_template_variables() for introspection

Run:
    python examples/14_prompts/dynamic_prompt.py
"""

from syrin import Agent, Model, prompt

# ---------------------------------------------------------------------------
# 1. Define a dynamic prompt with @prompt
# ---------------------------------------------------------------------------


@prompt
def persona_prompt(
    user_name: str,
    tone: str = "professional",
) -> str:
    """Persona system prompt with runtime variables."""
    return f"You assist {user_name or 'the user'}. Be {tone}."


# ---------------------------------------------------------------------------
# 2. Agent with class-level template_variables
# ---------------------------------------------------------------------------


class PersonaAgent(Agent):
    _agent_name = "persona-agent"
    _agent_description = "Agent with dynamic template_variables"
    model = Model.Almock()
    system_prompt = persona_prompt
    template_variables = {"tone": "friendly"}


# ---------------------------------------------------------------------------
# 3. Instance override -- constructor template_variables replace class ones
# ---------------------------------------------------------------------------
print("-- 1. Instance override --")

alice = PersonaAgent(template_variables={"user_name": "Alice", "tone": "casual"})
vars_ = alice.effective_template_variables()
print(f"  Effective vars: user_name={vars_['user_name']}, tone={vars_['tone']}")

r1 = alice.response("What can you help me with?")
print(f"  Alice response: {r1.content[:80]}...")

# ---------------------------------------------------------------------------
# 4. Per-call override -- same agent, different user
# ---------------------------------------------------------------------------
print("\n-- 2. Per-call override --")

r2 = alice.response("Hi", template_variables={"user_name": "Bob"})
print(f"  Bob response (per-call): {r2.content[:80]}...")

# ---------------------------------------------------------------------------
# 5. Fresh agent with different defaults
# ---------------------------------------------------------------------------
print("\n-- 3. Fresh agent --")

demo = PersonaAgent(template_variables={"user_name": "Demo", "tone": "concise"})
r3 = demo.response("Summarize your role.")
print(f"  Demo response: {r3.content[:80]}...")

# ---------------------------------------------------------------------------
# Optional: serve with playground UI (requires syrin[serve])
# ---------------------------------------------------------------------------
# agent = PersonaAgent(template_variables={"user_name": "Demo", "tone": "concise"})
# agent.serve(port=8000, enable_playground=True, debug=True)
