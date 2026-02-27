"""Pipeline Example.

Demonstrates:
- Sequential pipeline execution
- Parallel pipeline execution
- Pipeline with budget

Run: python -m examples.07_multi_agent.pipeline
"""

from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv

from examples.models.models import almock
from syrin import Agent, Pipeline, prompt

load_dotenv(Path(__file__).resolve().parent.parent / ".env")


@prompt
def researcher_prompt(domain: str) -> str:
    return f"You are a researcher specializing in {domain}."


@prompt
def writer_prompt(style: str) -> str:
    return f"You are a writer with a {style} style."


class Researcher(Agent):
    model = almock
    system_prompt = researcher_prompt(domain="technology")


class Writer(Agent):
    model = almock
    system_prompt = writer_prompt(style="professional")


pipeline = Pipeline()
result = pipeline.run(
    [
        (Researcher, "Find information about renewable energy"),
        (Writer, "Write about renewable energy"),
    ]
)
print(f"Pipeline result: {result.content[:100]}...")
print(f"Cost: ${result.cost:.6f}")
