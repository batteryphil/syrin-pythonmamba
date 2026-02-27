"""Handoff Example.

Demonstrates:
- Agent handoff between specialized agents
- Context transfer via memory
- Budget transfer between agents

Run: python -m examples.07_multi_agent.handoff
"""

from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv

from examples.models.models import almock
from syrin import Agent, prompt

load_dotenv(Path(__file__).resolve().parent.parent / ".env")


@prompt
def analyzer_prompt() -> str:
    return "You are an analyzer agent. Analyze information and provide key findings."


@prompt
def presenter_prompt() -> str:
    return "You are a presenter agent. Present information clearly and concisely."


class Analyzer(Agent):
    model = almock
    system_prompt = analyzer_prompt()


class Presenter(Agent):
    model = almock
    system_prompt = presenter_prompt()


analyzer = Analyzer()
result1 = analyzer.response("Analyze the benefits of renewable energy")
print(f"Analyzer: {result1.content[:80]}...")
result2 = analyzer.handoff(Presenter, "Present the analysis")
print(f"Presenter: {result2.content[:80]}...")
