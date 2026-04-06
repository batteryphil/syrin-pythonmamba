"""Playground: Multi-Agent Example.

Demonstrates:
- AgentRouter with multiple agents
- Agent selector in playground
- Visit http://localhost:8000/playground — choose Researcher or Writer

Run: python -m examples.16_serving.playground_multi_agent
"""

from __future__ import annotations

import sys
from pathlib import Path

_root = Path(__file__).resolve().parents[2]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from dotenv import load_dotenv

from examples.models.models import gpt4_mini
from syrin import Agent, Budget, Model
from syrin.enums import ServeProtocol
from syrin.serve import AgentRouter, ServeConfig

load_dotenv(Path(__file__).resolve().parent.parent / ".env")


class Researcher(Agent):
    name = "researcher"
    description = "Researches topics and summarizes findings"
    model = Model.mock(latency_min=1, latency_max=3, lorem_length=800, pricing_tier="high")
    system_prompt = "You are a researcher. Be thorough but concise."
    budget = Budget(max_cost=0.3)


class Writer(Agent):
    name = "writer"
    description = "Writes content in a professional style"
    model = gpt4_mini
    system_prompt = "You are a writer. Be clear and engaging."
    budget = Budget(max_cost=0.3)
    debug = True


if __name__ == "__main__":
    config = ServeConfig(enable_playground=True, debug=True)
    router = AgentRouter(agents=[Researcher(), Writer()], config=config)
    router.serve(protocol=ServeProtocol.CLI)
