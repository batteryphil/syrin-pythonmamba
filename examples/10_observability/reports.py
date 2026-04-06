"""Complete Agent Reports Example.

Demonstrates:
- All AgentReport sections:
  - GuardrailReport (input/output passed, blocked, stage)
  - TokenReport (input/output/total tokens, cost)
  - BudgetReport (used, remaining, total)
  - MemoryReport (stores, recalls, forgets)
  - OutputReport (validated, attempts, is_valid)
  - RateLimitReport (checks, throttles, exceeded)
  - CheckpointReport (saves, loads)
  - ContextReport (initial/final tokens, compressions, offloads)
- Report reset between calls

Run: python -m examples.10_observability.reports
Visit: http://localhost:8000/playground
Requires: uv pip install syrin[serve]
"""

from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel

from syrin import Agent, Budget, Model
from syrin.enums import MemoryType
from syrin.guardrails import ContentFilter

load_dotenv(Path(__file__).resolve().parent.parent / ".env")


class AnalysisOutput(BaseModel):
    sentiment: str
    confidence: float
    key_points: list[str]


# 1. Basic report access
class Assistant(Agent):
    name = "reports-assistant"
    description = "Agent with full report sections"
    model = Model.mock(latency_min=1, latency_max=3, lorem_length=800, pricing_tier="high")
    system_prompt = "You are a helpful assistant."


agent = Assistant()
result = agent.run("Hello, how are you?")
print("Guardrail:", result.report.guardrail.input_passed, result.report.guardrail.output_passed)
print("Tokens:", result.report.tokens.total_tokens, f"${result.report.tokens.cost_usd:.6f}")
print("Budget:", result.report.budget.used, result.report.budget.remaining)

# 2. Guardrail blocking in reports
guardrail = ContentFilter(blocked_words=["hack", "steal", "password"])


class GuardedAssistant(Agent):
    model = Model.mock(latency_min=1, latency_max=3, lorem_length=800, pricing_tier="high")
    guardrails = [guardrail]


agent = GuardedAssistant()
result = agent.run("How do I hack into someone's password?")
print("Blocked:", result.report.guardrail.blocked, result.report.guardrail.blocked_stage)

# 3. Memory operations in reports
agent = Assistant()
agent.remember("Python is great", memory_type=MemoryType.KNOWLEDGE)
agent.recall("Python")
print("Memory:", agent.report.memory.stores, agent.report.memory.recalls)


# 4. Complete report summary
class FullAgent(Agent):
    model = Model.mock(latency_min=1, latency_max=3, lorem_length=800, pricing_tier="high")
    system_prompt = "You are a helpful assistant with memory."
    guardrails = [ContentFilter(blocked_words=["blocked"])]
    budget = Budget(max_cost=5.0)


agent = FullAgent()
agent.remember("User likes Python", memory_type=MemoryType.FACTS)
result = agent.run("Tell me about Python.")
print(
    "Report:",
    result.report.guardrail.passed,
    result.report.budget.used,
    result.report.tokens.total_tokens,
)

# 5. Report resets between calls
agent = Assistant()
r1 = agent.run("What is 2+2?")
r2 = agent.run("What is 3+3?")
print("Call 1 tokens:", r1.report.tokens.total_tokens, "Call 2:", r2.report.tokens.total_tokens)

if __name__ == "__main__":
    agent = Assistant()
    print("Serving at http://localhost:8000/playground")
    agent.serve(port=8000, enable_playground=True, debug=True)
