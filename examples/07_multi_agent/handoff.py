"""Handoff — Delegate tasks between agents.

One agent analyzes, then hands off to another agent to present the results.

Run:
    python examples/07_multi_agent/handoff.py
"""

from syrin import Agent, Model

model = Model.Almock()


class Analyzer(Agent):
    model = model
    system_prompt = "You are an analyzer agent. Analyze information and provide key findings."


class Presenter(Agent):
    model = model
    system_prompt = "You are a presenter agent. Present information clearly and concisely."


# Step 1: Analyzer processes the request
analyzer = Analyzer()
result = analyzer.response("Analyze the benefits of renewable energy")
print("=== Analyzer ===")
print(f"{result.content[:120]}...")
print(f"Cost: ${result.cost:.6f}")
print()

# Step 2: Hand off to Presenter
handoff_result = analyzer.handoff(Presenter, "Present the analysis")
print("=== Presenter (via handoff) ===")
print(f"{handoff_result.content[:120]}...")
print(f"Cost: ${handoff_result.cost:.6f}")

# --- Serve (uncomment to try playground) ---
# analyzer.serve(port=8000, enable_playground=True)
