"""CLI REPL Serving Example.

Demonstrates: agent.serve(protocol=ServeProtocol.CLI)

Interactive REPL: prompt, run agent, show cost/budget per turn. Ctrl+C to exit.

Requires: uv pip install syrin[serve]

Run: python -m examples.serving.cli_serve
"""

from __future__ import annotations

import sys
from pathlib import Path

_root = Path(__file__).resolve().parents[2]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from dotenv import load_dotenv

from syrin import Agent, Model
from syrin.enums import ServeProtocol

load_dotenv(Path(__file__).resolve().parent.parent / ".env")


class Assistant(Agent):
    name = "assistant"
    description = "Helpful assistant"
    model = Model.mock(latency_min=1, latency_max=3, lorem_length=800, pricing_tier="high")
    system_prompt = "You are a helpful assistant."


if __name__ == "__main__":
    agent = Assistant()
    agent.serve(protocol=ServeProtocol.CLI)
