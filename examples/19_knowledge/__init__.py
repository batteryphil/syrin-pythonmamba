"""Knowledge (RAG) examples.

- loaders_and_document: Document model and loaders
- chunking: ChunkConfig, ChunkStrategy, get_chunker
- vector_store: InMemoryKnowledgeStore, upsert, search, delete
- knowledge_agent: RAG + Agent with search_knowledge tool
- agentic_rag: Agentic RAG with search_knowledge_deep, verify_knowledge
- full_rag_lifecycle: End-to-end RAG + Agentic RAG (all features)
- postgres_backend: Knowledge with PostgreSQL (pgvector) backend
- serve_agentic_postgres: HTTP server with agentic RAG + Postgres; POST /chat to verify
"""
