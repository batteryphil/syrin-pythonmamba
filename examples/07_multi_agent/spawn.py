"""Spawn Example.

Demonstrates:
- Agent spawn: parent spawns child agent
- spawn(task="...") returns Response
- spawn() without task returns child Agent instance

Run: python -m examples.07_multi_agent.spawn
"""

from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv

from examples.models.models import almock
from syrin import Agent

load_dotenv(Path(__file__).resolve().parent.parent / ".env")


class Parent(Agent):
    model = almock
    system_prompt = "You are a coordinator."


class Child(Agent):
    model = almock
    system_prompt = "You are a specialist."


parent = Parent()
result = parent.spawn(Child, task="What is AI?")
print(f"spawn(task=...): {result.content[:80]}...")

child = parent.spawn(Child)
assert child is not None
r = child.response("Summarize machine learning")
print(f"spawn() child: {r.content[:80]}...")
