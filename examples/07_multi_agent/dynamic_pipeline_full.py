"""Dynamic Pipeline Full — Complex multi-agent system with full debugging.

Demonstrates:
- Orchestrator LLM plans which agents to spawn
- 4 specialized agents (tech, finance, healthcare, summarizer) + 5 tools
- Full observability with hooks at every stage
- DynamicPipeline.run() with parallel mode

Uses Model.Almock() — no API key needed.

Run:
    python examples/07_multi_agent/dynamic_pipeline_full.py
"""

from __future__ import annotations

import time
from datetime import datetime

from syrin import Agent, Model
from syrin.agent.multi_agent import DynamicPipeline
from syrin.enums import Hook
from syrin.tool import tool

# ---------------------------------------------------------------------------
# Model — Almock orchestrator returns a valid JSON plan so agents spawn
# ---------------------------------------------------------------------------

orchestrator_model = Model.Almock(
    latency_min=0.3,
    latency_max=0.8,
    response_mode="custom",
    custom_response=(
        '[{"type":"tech_researcher","task":"Research AI trends in healthcare"},'
        '{"type":"finance_researcher","task":"Analyze healthcare market financials"},'
        '{"type":"healthcare_researcher","task":"Investigate clinical AI adoption"},'
        '{"type":"summarizer","task":"Synthesize all findings into a report"}]'
    ),
    pricing_tier="high",
)

agent_model = Model.Almock(
    latency_min=0.3,
    latency_max=0.8,
    lorem_length=200,
    pricing_tier="high",
)

# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Agents
# ---------------------------------------------------------------------------


class TechResearchAgent(Agent):
    _agent_name = "tech_researcher"
    _agent_description = "Researches technology trends"
    model = agent_model
    system_prompt = "You research technology trends."
    tools = [search_web, analyze_data]


class FinanceResearchAgent(Agent):
    _agent_name = "finance_researcher"
    _agent_description = "Researches financial markets"
    model = agent_model
    system_prompt = "You research financial markets."
    tools = [fetch_financial, analyze_data]


class HealthcareResearchAgent(Agent):
    _agent_name = "healthcare_researcher"
    _agent_description = "Researches healthcare industry"
    model = agent_model
    system_prompt = "You research healthcare industry."
    tools = [search_web, analyze_data]


class SummarizerAgent(Agent):
    _agent_name = "summarizer"
    _agent_description = "Synthesizes research into clear reports"
    model = agent_model
    system_prompt = "You synthesize research into a clear report."
    tools = [generate_chart, export_report]


# ---------------------------------------------------------------------------
# Debugger — logs every hook with timestamp
# ---------------------------------------------------------------------------


class PipelineDebugger:
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.events_log: list[dict] = []

    def log(self, hook: Hook, ctx: dict) -> None:
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        self.events_log.append({"timestamp": timestamp, "hook": hook.value, "data": dict(ctx)})
        if self.verbose:
            print(f"  [{timestamp}] {hook.value}")

    def print_summary(self) -> None:
        print("\n" + "=" * 70)
        print("PIPELINE EXECUTION SUMMARY")
        print("=" * 70)
        print(f"Total events: {len(self.events_log)}")


# ---------------------------------------------------------------------------
# Pipeline setup
# ---------------------------------------------------------------------------

pipeline = DynamicPipeline(
    agents=[TechResearchAgent, FinanceResearchAgent, HealthcareResearchAgent, SummarizerAgent],
    model=orchestrator_model,
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

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    task = "Conduct market research on AI in Healthcare. Provide a consolidated report."
    print(f"Task: {task}\n")

    start = time.time()
    result = pipeline.run(task, mode="parallel")
    elapsed = time.time() - start

    debugger.print_summary()
    print(f"\nExecution time: {elapsed:.2f}s")
    print(f"Total cost: ${result.cost:.4f}")
    print(f"Preview: {result.content[:300]}...")

    # Optional: serve the pipeline with the playground UI
    # pipeline.serve(port=8000, enable_playground=True, debug=True)
