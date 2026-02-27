"""Dynamic Pipeline Basic — Minimal setup (~80 lines).

Demonstrates:
- DynamicPipeline with a few agents
- run(task, mode="parallel")
- Basic orchestration without full debugging

Run: python -m examples.07_multi_agent.dynamic_pipeline_basic
"""

from pathlib import Path

from dotenv import load_dotenv

from examples.models.models import almock
from syrin import Agent
from syrin.agent.multi_agent import DynamicPipeline

load_dotenv(Path(__file__).resolve().parent.parent / ".env")


class TechAgent(Agent):
    model = almock
    _syrin_name = "tech"
    system_prompt = "You research technology trends."


class FinanceAgent(Agent):
    model = almock
    _syrin_name = "finance"
    system_prompt = "You research financial markets."


class SummarizerAgent(Agent):
    model = almock
    _syrin_name = "summarizer"
    system_prompt = "You synthesize research into a concise report."


pipeline = DynamicPipeline(
    agents=[TechAgent, FinanceAgent, SummarizerAgent],
    model=almock,
    max_parallel=3,
)
result = pipeline.run(
    "Conduct market research on AI in healthcare. Provide a brief summary.",
    mode="parallel",
)
print(f"Result: {result.content[:300]}...")
print(f"Cost: ${result.cost:.4f}")
