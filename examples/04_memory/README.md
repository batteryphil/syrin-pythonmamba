# 04_memory — Persistent memory

- **basic_memory.py** — remember(), recall(), Memory types (CORE, EPISODIC, SEMANTIC, PROCEDURAL)
- **recall.py** — recall(query=...) to retrieve relevant memories
- **consolidate.py** — Memory.consolidate(deduplicate=True)
- **qdrant_memory.py** — QdrantConfig for vector search, namespace isolation
- **chroma_memory.py** — ChromaConfig for lightweight vector memory
- **redis_memory.py** — RedisConfig for fast, distributed storage (substring search)
- **postgres_memory.py** — PostgresConfig for production SQL-backed storage
- **test_redis_postgres_live.py** — Live test with real Redis, Postgres, and LLM
