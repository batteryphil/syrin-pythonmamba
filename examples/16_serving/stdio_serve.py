"""STDIO Serving Example.

Demonstrates: agent.serve(protocol=ServeProtocol.STDIO)

Reads JSON lines from stdin, writes JSON lines to stdout.
For background tasks, subprocess, MCP host calling your agent.

Requires: uv pip install syrin[serve]

Input (stdin): {"input": "Hello", "conversation_id": "optional"}
Output (stdout): {"content": "...", "cost": 0.0, "tokens": N, "conversation_id": "optional"}

Run: echo '{"input": "Hi"}' | python -m examples.serving.stdio_serve
"""

from __future__ import annotations

import sys
from pathlib import Path

_root = Path(__file__).resolve().parents[2]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from dotenv import load_dotenv

from examples.models.models import almock
from syrin import Agent
from syrin.enums import ServeProtocol

load_dotenv(Path(__file__).resolve().parent.parent / ".env")


class Assistant(Agent):
    _agent_name = "assistant"
    _agent_description = "Helpful assistant"
    model = almock
    system_prompt = "You are a helpful assistant."


if __name__ == "__main__":
    agent = Assistant()
    agent.serve(protocol=ServeProtocol.STDIO)
