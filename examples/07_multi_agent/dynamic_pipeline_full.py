"""Dynamic Pipeline Full — Complex multi-agent system with full debugging.

Demonstrates:
- Orchestrator LLM plans which agents to spawn (real model required for valid JSON).
- 4 specialized agents (tech, finance, healthcare, summarizer) + 5 tools.
- Full observability with hooks at every stage.

Uses a real model (OpenAI gpt-4o) by default so the planner returns a valid plan
and agents are spawned. Set OPENAI_API_KEY in examples/.env.
For key-less runs: set USE_ALMOCK=1 in .env (agents may not spawn; mock returns no plan).

Run from repo root:
  python -m examples.07_multi_agent.dynamic_pipeline_full
  python examples/07_multi_agent/dynamic_pipeline_full.py
"""

import os
import sys
import time
from datetime import datetime
from pathlib import Path

# Allow running as script: add project root so "examples" package resolves
if __name__ == "__main__":
    _root = Path(__file__).resolve().parent.parent.parent
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

from dotenv import load_dotenv

from examples.models.models import almock, gpt4
from syrin import Agent
from syrin.agent.multi_agent import DynamicPipeline
from syrin.enums import Hook
from syrin.tool import tool

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

# Use a real model so the orchestrator can return a valid JSON plan and agents are spawned.
# Set OPENAI_API_KEY in examples/.env. For key-less runs, set USE_ALMOCK=1 in .env.
USE_ALMOCK = os.environ.get("USE_ALMOCK", "").strip().lower() in ("1", "true", "yes")
MODEL = almock if USE_ALMOCK else gpt4


@tool(name="search_web", description="Search the web for information")
def search_web(query: str) -> str:
    return f"[SIMULATED] Search results for: {query}\n\nFound 10 relevant articles."


@tool(name="analyze_data", description="Analyze numerical data")
def analyze_data(data: str, analysis_type: str = "statistical") -> str:
    _ = analysis_type
    return f"[SIMULATED] Analysis of: {data}\n\nTrend: Upward 15%"


@tool(name="fetch_financial", description="Fetch financial data")
def fetch_financial(symbol: str) -> str:
    return f"[SIMULATED] Financial data for {symbol}:\n\nRevenue: $2.5B"


@tool(name="generate_chart", description="Generate ASCII charts")
def generate_chart(data: str, chart_type: str = "bar") -> str:
    return f"[SIMULATED] {chart_type.upper()} Chart for: {data}"


@tool(name="export_report", description="Export report to markdown")
def export_report(title: str, content: str) -> str:
    return f"[SIMULATED] Exported: {title}"


class TechResearchAgent(Agent):
    _syrin_name = "tech_researcher"
    model = MODEL
    system_prompt = "You research technology trends."
    tools = [search_web, analyze_data]


class FinanceResearchAgent(Agent):
    _syrin_name = "finance_researcher"
    model = MODEL
    system_prompt = "You research financial markets."
    tools = [fetch_financial, analyze_data]


class HealthcareResearchAgent(Agent):
    _syrin_name = "healthcare_researcher"
    model = MODEL
    system_prompt = "You research healthcare industry."
    tools = [search_web, analyze_data]


class SummarizerAgent(Agent):
    _syrin_name = "summarizer"
    model = MODEL
    system_prompt = "You synthesize research into a clear report."
    tools = [generate_chart, export_report]


class PipelineDebugger:
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.events_log = []

    def log(self, hook: Hook, ctx: dict) -> None:
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        self.events_log.append({"timestamp": timestamp, "hook": hook.value, "data": dict(ctx)})
        if self.verbose:
            print(f"{timestamp} {hook.value}")

    def print_summary(self) -> None:
        print("\n" + "=" * 70)
        print("PIPELINE EXECUTION SUMMARY")
        print("=" * 70)
        print(f"Total events: {len(self.events_log)}")


pipeline = DynamicPipeline(
    agents=[TechResearchAgent, FinanceResearchAgent, HealthcareResearchAgent, SummarizerAgent],
    model=MODEL,
    max_parallel=4,
)
debugger = PipelineDebugger(verbose=True)
pipeline.events.on(
    Hook.DYNAMIC_PIPELINE_START, lambda ctx: debugger.log(Hook.DYNAMIC_PIPELINE_START, ctx)
)
pipeline.events.on(
    Hook.DYNAMIC_PIPELINE_AGENT_SPAWN,
    lambda ctx: debugger.log(Hook.DYNAMIC_PIPELINE_AGENT_SPAWN, ctx),
)
pipeline.events.on(
    Hook.DYNAMIC_PIPELINE_AGENT_COMPLETE,
    lambda ctx: debugger.log(Hook.DYNAMIC_PIPELINE_AGENT_COMPLETE, ctx),
)
pipeline.events.on(
    Hook.DYNAMIC_PIPELINE_END, lambda ctx: debugger.log(Hook.DYNAMIC_PIPELINE_END, ctx)
)
task = "Conduct market research on AI in Healthcare. Provide a consolidated report."
start = time.time()
result = pipeline.run(task, mode="parallel")
elapsed = time.time() - start
debugger.print_summary()
print(f"\nExecution time: {elapsed:.2f}s")
print(f"Total cost: ${result.cost:.4f}")
print(f"Preview: {result.content[:300]}...")
