"""Input/Output Guardrails -- block unwanted content before it reaches the model.

Demonstrates:
- ContentFilter: block messages containing specific words
- GuardrailChain: combine multiple guardrails into one pipeline
- Attaching guardrails to an Agent via the guardrails= parameter
- Testing the chain directly with GuardrailStage.INPUT

Run:
    python examples/09_guardrails/input_output_guardrails.py
"""

from syrin import Agent, ContentFilter, GuardrailChain, GuardrailStage, Model

# ---------------------------------------------------------------------------
# 1. Build a content filter guardrail
# ---------------------------------------------------------------------------

spam_filter = ContentFilter(blocked_words=["spam", "scam"], name="NoSpam")

# ---------------------------------------------------------------------------
# 2. Test the chain directly (no agent needed)
# ---------------------------------------------------------------------------

chain = GuardrailChain([spam_filter])

clean = chain.check("Hello, legitimate message", GuardrailStage.INPUT)
print(f"Clean text  -> passed={clean.passed}")

blocked = chain.check("This is spam", GuardrailStage.INPUT)
print(f"Blocked text -> passed={blocked.passed}, reason={blocked.reason}")

# ---------------------------------------------------------------------------
# 3. Attach guardrails to an Agent (pass a list, not a chain)
# ---------------------------------------------------------------------------


class GuardedAgent(Agent):
    _agent_name = "guarded-agent"
    _agent_description = "Agent with ContentFilter guardrail"
    model = Model.Almock()
    system_prompt = "You are helpful."
    guardrails = [spam_filter]


agent = GuardedAgent()

response = agent.response("Hello, how are you?")
print(f"\nAgent response: {response.content[:80]}...")

# ---------------------------------------------------------------------------
# Optional: serve with playground UI (requires syrin[serve])
# ---------------------------------------------------------------------------
# agent.serve(port=8000, enable_playground=True, debug=True)
