"""Chatbot Example — Full-featured chatbot with context, memory, guardrails, and checkpoints.

Demonstrates:
- Context: Context(max_tokens=) for controlled context window
- Memory: Memory with SQLite backend (persistent), 4 types, Decay, auto_store
- remember tool: Explicit storage when user says "remember that X"
- Guardrails: ContentFilter + LengthGuardrail for input/output safety
- Checkpoints: CheckpointConfig with STEP trigger and memory storage
- Budget: Per-run and rate limits
- HTTP serving: agent.serve(port=8000)

Run: python -m examples.16_serving.chatbot
Visit: http://localhost:8000/chat (POST), http://localhost:8000/playground

Requires: uv pip install syrin[serve]
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_root = Path(__file__).resolve().parents[2]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from dotenv import load_dotenv

from examples.models.models import almock, gpt4_mini
from syrin import (
    Agent,
    Budget,
    CheckpointConfig,
    CheckpointTrigger,
    Decay,
    Memory,
    RateLimit,
    tool,
)
from syrin.context import Context
from syrin.enums import DecayStrategy, MemoryBackend, MemoryType, WriteMode
from syrin.guardrails import ContentFilter, LengthGuardrail

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

guardrails = [
    ContentFilter(blocked_words=["spam", "scam", "phishing"], name="NoSpam"),
    LengthGuardrail(max_length=4000, name="ResponseLength"),
]

# Persistent SQLite memory — survives restarts; write_mode=SYNC for reliable testing
MEMORY_DB = Path(__file__).resolve().parent / "chatbot_memory.db"
memory = Memory(
    backend=MemoryBackend.SQLITE,
    path=str(MEMORY_DB),
    write_mode=WriteMode.SYNC,
    types=[MemoryType.CORE, MemoryType.EPISODIC, MemoryType.SEMANTIC, MemoryType.PROCEDURAL],
    top_k=10,
    auto_store=True,
    decay=Decay(
        strategy=DecayStrategy.EXPONENTIAL,
        half_life_hours=24,
        reinforce_on_access=True,
        min_importance=0.2,
    ),
)


def _make_remember_tool(mem: Memory) -> object:
    """Create remember_fact tool that writes to the given Memory."""

    @tool
    def remember_fact(content: str, memory_type: str = "episodic") -> str:
        """Store a fact for later recall. Use when user asks to remember something.
        content: The fact to remember (e.g. 'My name is Alice').
        memory_type: core (identity/prefs), episodic (events), semantic (facts), procedural (how-to).
        """
        mt = MemoryType.EPISODIC
        if memory_type and memory_type.lower() in ("core", "episodic", "semantic", "procedural"):
            mt = MemoryType(memory_type.lower())
        ok = mem.remember(content, memory_type=mt)
        return f"Stored: {content[:80]}..." if ok else "Failed to store"

    return remember_fact


context = Context(max_tokens=16000)

checkpoint = CheckpointConfig(
    storage="memory",
    trigger=CheckpointTrigger.STEP,
    max_checkpoints=10,
)


class Chatbot(Agent):
    """Full-featured chatbot with persistent memory, remember tool, guardrails, and checkpoints."""

    name = "chatbot"
    description = "Conversational chatbot with persistent memory, guardrails, and checkpoints"
    # Use gpt4_mini for real LLM (memory recall visible); almock for no-API-key demos
    model = gpt4_mini if os.getenv("OPENAI_API_KEY") else almock
    system_prompt = (
        "You are a helpful, friendly chatbot with persistent memory. "
        "You recall past turns automatically. When the user asks you to remember something, "
        "use the remember_fact tool. Keep responses concise. Be respectful and safe."
    )
    memory = memory
    tools = [_make_remember_tool(memory)]
    context = context
    guardrails = guardrails
    checkpoint = checkpoint
    budget = Budget(run=0.50, per=RateLimit(hour=10, day=100))


if __name__ == "__main__":
    agent = Chatbot()
    print("Chatbot serving at http://localhost:8000")
    print(f"  Memory: {MEMORY_DB} (persistent SQLite)")
    print("  POST /chat  — send messages")
    print("  GET /playground — web UI")
    agent.serve(port=8000, enable_playground=True, debug=True)
