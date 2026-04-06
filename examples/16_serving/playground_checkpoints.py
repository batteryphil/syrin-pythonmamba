"""Playground: Agent with Checkpoints Example.

Demonstrates:
- Agent with CheckpointConfig (memory storage, STEP trigger)
- State persistence across turns
- Visit http://localhost:8000/playground

Run: python -m examples.16_serving.playground_checkpoints
"""

from __future__ import annotations

import sys
from pathlib import Path

_root = Path(__file__).resolve().parents[2]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from dotenv import load_dotenv

from syrin import Agent, CheckpointConfig, CheckpointTrigger, Model

load_dotenv(Path(__file__).resolve().parent.parent / ".env")


checkpoint_config = CheckpointConfig(
    enabled=True,
    storage="memory",
    trigger=CheckpointTrigger.STEP,
    max_checkpoints=10,
)


class CheckpointAssistant(Agent):
    name = "checkpoint-assistant"
    description = "Assistant with step checkpoints"
    model = Model.mock(latency_min=1, latency_max=3, lorem_length=800, pricing_tier="high")
    system_prompt = "You are a helpful assistant. Remember context across turns."
    checkpoint = checkpoint_config


if __name__ == "__main__":
    agent = CheckpointAssistant()
    agent.serve(port=8000, enable_playground=True, debug=True)
