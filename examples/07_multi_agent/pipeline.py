"""Pipeline -- run multiple agents in sequence, passing output forward.

Demonstrates:
- Pipeline() for sequential multi-agent execution
- @prompt for per-agent dynamic system prompts
- Pipeline result with content and cost

Run:
    python examples/07_multi_agent/pipeline.py
"""

from syrin import Agent, Model, Pipeline, prompt

model = Model.Almock()

# ---------------------------------------------------------------------------
# 1. Define agents with dynamic prompts
# ---------------------------------------------------------------------------


@prompt
def researcher_prompt(domain: str) -> str:
    return f"You are a researcher specializing in {domain}."


@prompt
def writer_prompt(style: str) -> str:
    return f"You are a writer with a {style} style."


class Researcher(Agent):
    _agent_name = "researcher"
    _agent_description = "Researches topics and gathers information"
    model = model
    system_prompt = researcher_prompt(domain="technology")


class Writer(Agent):
    _agent_name = "writer"
    _agent_description = "Writes content in professional style"
    model = model
    system_prompt = writer_prompt(style="professional")


# ---------------------------------------------------------------------------
# 2. Run the pipeline
# ---------------------------------------------------------------------------
print("-- Pipeline: Researcher -> Writer --")

pipeline = Pipeline()
result = pipeline.run(
    [
        (Researcher, "Find information about renewable energy"),
        (Writer, "Write about renewable energy"),
    ]
)

print(f"  Result: {result.content[:100]}...")
print(f"  Cost:   ${result.cost:.6f}")

# ---------------------------------------------------------------------------
# Optional: serve with playground UI (requires syrin[serve])
# ---------------------------------------------------------------------------
# pipeline.serve(port=8000, enable_playground=True, debug=True)
