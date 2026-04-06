"""Chroma Vector Memory Example.

Demonstrates:
- Memory with ChromaConfig for lightweight vector search
- Local persistent storage

Requires: pip install syrin[chroma]  # or pip install chromadb

Run: python -m examples.04_memory.chroma_memory
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from syrin import Agent, Memory, MemoryType, Model
from syrin.enums import MemoryBackend
from syrin.memory import ChromaConfig


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        chroma_path = Path(tmp) / "chroma_db"

        memory = Memory(
            backend=MemoryBackend.CHROMA,
            chroma=ChromaConfig(
                path=str(chroma_path),
                collection="syrin_memories",
            ),
        )

        agent = Agent(
            model=Model.mock(),
            system_prompt="You are a helpful assistant with memory.",
            memory=memory,
        )

        agent.remember("User likes functional programming", memory_type=MemoryType.KNOWLEDGE)
        agent.remember("Discussed list comprehensions yesterday", memory_type=MemoryType.HISTORY)

        entries = agent.recall(query="programming", limit=5)
        print(f"Recalled {len(entries)} memories:")
        for e in entries:
            print(f"  - {e.content[:60]}...")


if __name__ == "__main__":
    main()
