"""Redis Memory Example.

Demonstrates:
- Memory with RedisConfig for ultra-fast, distributed storage
- Key-prefix isolation (prefix)
- TTL for expiring memories
- Text search (substring match; for semantic search use Qdrant/Chroma)

Requires: pip install syrin[redis]  # or pip install redis

Prerequisites: Redis server running at localhost:6379
  docker run -d -p 6379:6379 redis:alpine

Run: python -m examples.04_memory.redis_memory
"""

from __future__ import annotations

from examples.models.models import almock
from syrin import Agent, Memory, MemoryType
from syrin.enums import MemoryBackend
from syrin.memory import RedisConfig


def main() -> None:
    memory = Memory(
        backend=MemoryBackend.REDIS,
        redis=RedisConfig(
            host="localhost",
            port=6379,
            db=0,
            prefix="syrin:demo:",  # Isolate demo keys
            ttl=None,  # No expiry; use ttl=3600 for 1-hour expiry
        ),
    )

    agent = Agent(
        model=almock,
        system_prompt="You are a helpful assistant with Redis-backed memory.",
        memory=memory,
    )

    agent.remember("User prefers Python over JavaScript", memory_type=MemoryType.CORE)
    agent.remember("Last discussed async/await patterns", memory_type=MemoryType.EPISODIC)
    agent.remember("Likes functional programming style", memory_type=MemoryType.SEMANTIC)

    # Recall by query (substring match; Redis does not support vector search)
    entries = agent.recall(query="programming", limit=5)
    print(f"Recalled {len(entries)} memories:")
    for e in entries:
        print(f"  - {e.content[:70]}...")

    # Custom host/port (e.g. Redis Cloud):
    # memory = Memory(
    #     backend=MemoryBackend.REDIS,
    #     redis=RedisConfig(
    #         host="your-redis-host.cloud.com",
    #         port=12345,
    #         password="your-password",
    #         prefix="syrin:prod:",
    #     ),
    # )


if __name__ == "__main__":
    main()
