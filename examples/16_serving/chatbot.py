"""Chatbot Example -- Full-featured chatbot with context, memory, guardrails, and routing.

Demonstrates: Context, Memory (SQLite), guardrails, checkpoints, multi-model routing,
multimodal input, image/video generation.

Run:  python -m examples.16_serving.chatbot
Visit: http://localhost:8000/playground
"""

from __future__ import annotations

import os
from datetime import UTC
from pathlib import Path

from syrin import (
    Agent,
    Budget,
    CheckpointConfig,
    CheckpointTrigger,
    Decay,
    Memory,
    Model,
    RateLimit,
    tool,
)
from syrin.context import Context
from syrin.enums import DecayStrategy, Media, MemoryBackend, MemoryType, WriteMode
from syrin.generation import ImageGenerator, VideoGenerator
from syrin.guardrails import ContentFilter, LengthGuardrail
from syrin.router import RoutingConfig, RoutingMode, TaskType

_DIR = Path(__file__).resolve().parent
MEMORY_DB = _DIR / "chatbot_memory.db"
MAP_PATH = _DIR / "chatbot_context_map.json"

memory = Memory(
    backend=MemoryBackend.SQLITE,
    path=str(MEMORY_DB),
    write_mode=WriteMode.SYNC,
    types=[MemoryType.FACTS, MemoryType.HISTORY, MemoryType.KNOWLEDGE, MemoryType.INSTRUCTIONS],
    top_k=10,
    auto_store=True,
    decay=Decay(
        strategy=DecayStrategy.EXPONENTIAL,
        half_life_hours=24,
        reinforce_on_access=True,
        min_importance=0.2,
    ),
)

context = Context(
    max_tokens=16000,
    auto_compact_at=0.75,
    store_output_chunks=True,
    output_chunk_top_k=5,
    output_chunk_threshold=0.0,
    map_backend="file",
    map_path=str(MAP_PATH),
    inject_map_summary=True,
)

guardrails = [
    ContentFilter(blocked_words=["spam", "scam", "phishing"], name="NoSpam"),
    LengthGuardrail(max_length=4000, name="ResponseLength"),
]

checkpoint = CheckpointConfig(storage="memory", trigger=CheckpointTrigger.STEP, max_checkpoints=10)


@tool
def remember_fact(content: str, memory_type: str = "episodic") -> str:
    """Store a fact for later recall. content: The fact. memory_type: core, episodic, semantic, procedural."""
    mt = (
        MemoryType(memory_type.lower())
        if memory_type and memory_type.lower() in ("core", "episodic", "semantic", "procedural")
        else MemoryType.HISTORY
    )
    ok = memory.remember(content, memory_type=mt)
    return f"Stored: {content[:80]}..." if ok else "Failed to store"


@tool
def get_current_time() -> str:
    """Return current date/time. Use when user asks what time or date it is."""
    from datetime import datetime

    return datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S UTC")


@tool
def repeat_back(phrase: str) -> str:
    """Echo back a phrase. Use only when user explicitly asks to repeat."""
    return f"You said: {phrase}"


def _model_and_config():
    """Model list + RoutingConfig when OPENAI_API_KEY set; else single Almock."""
    if not os.getenv("OPENAI_API_KEY"):
        return [Model.mock()], None
    gpt4_mini = Model.OpenAI("gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"))
    gpt4 = Model.OpenAI("gpt-4o", api_key=os.getenv("OPENAI_API_KEY"))
    models = [
        gpt4_mini.with_routing(
            profile_name="general",
            strengths=[TaskType.GENERAL, TaskType.CREATIVE, TaskType.TRANSLATION],
            input_media={Media.TEXT},
            output_media={Media.TEXT},
            priority=85,
        ),
        gpt4.with_routing(
            profile_name="vision",
            strengths=[
                TaskType.GENERAL,
                TaskType.VISION,
                TaskType.VIDEO,
                TaskType.IMAGE_GENERATION,
                TaskType.VIDEO_GENERATION,
            ],
            input_media={Media.TEXT, Media.IMAGE, Media.VIDEO},
            output_media={Media.TEXT, Media.IMAGE, Media.VIDEO},
            priority=90,
        ),
        gpt4_mini.with_routing(
            profile_name="code",
            strengths=[TaskType.CODE, TaskType.REASONING, TaskType.PLANNING],
            input_media={Media.TEXT},
            output_media={Media.TEXT},
            priority=95,
        ),
    ]
    return models, RoutingConfig(routing_mode=RoutingMode.AUTO)


gen_key = (os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY") or "").strip() or None


class Chatbot(Agent):
    name = "chatbot"
    description = "Chatbot with memory, context, guardrails, routing, multimodal, image/video gen"
    model, model_router = _model_and_config()
    input_media = {Media.TEXT, Media.IMAGE}
    output_media = {Media.TEXT, Media.IMAGE, Media.VIDEO}
    system_prompt = (
        "You are a helpful chatbot with persistent memory. Recall past turns automatically. "
        "Use remember_fact when asked to remember something, get_current_time for time/date, repeat_back only when explicitly asked. "
        "Describe images when given. Use generate_image for image requests, generate_video for video. Keep responses concise."
    )
    memory = memory
    tools = [remember_fact, get_current_time, repeat_back]
    context = context
    guardrails = guardrails
    checkpoint = checkpoint
    budget = Budget(max_cost=0.50, rate_limits=RateLimit(hour=10, day=100))
    image_generation = ImageGenerator.Gemini(api_key=gen_key) if gen_key else None
    video_generation = VideoGenerator.Gemini(api_key=gen_key) if gen_key else None


if __name__ == "__main__":
    agent = Chatbot()
    print("Chatbot at http://localhost:8000 | Memory:", MEMORY_DB)
    agent.serve(port=8000, enable_playground=True, debug=True)
