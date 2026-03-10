"""Knowledge module for RAG - document loading, chunking, and retrieval.

This module provides the Knowledge class for declarative knowledge base management.

Example:
    from syrin import Knowledge, KnowledgeBackend
    from syrin.embedding import Embedding

    knowledge = Knowledge(
        sources=[
            Knowledge.PDF("./resume.pdf"),
            Knowledge.Markdown("./about.md"),
            Knowledge.Text("I have 5 years of Python experience."),
        ],
        backend=KnowledgeBackend.POSTGRES,
        embedding=Embedding.OpenAI("text-embedding-3-small"),
    )
"""

from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, cast

from syrin.enums import Hook, KnowledgeBackend
from syrin.knowledge._agentic import AgenticRAGConfig
from syrin.knowledge._chunker import Chunk, ChunkConfig, Chunker, ChunkMetadata, ChunkStrategy
from syrin.knowledge._document import Document, DocumentMetadata
from syrin.knowledge._loader import DocumentLoader
from syrin.knowledge._store import MetadataFilter, SearchResult
from syrin.knowledge.chunkers import get_chunker
from syrin.knowledge.loaders import (
    DirectoryLoader,
    GitHubLoader,
    JSONLoader,
    MarkdownLoader,
    PDFLoader,
    PythonLoader,
    RawTextLoader,
    TextLoader,
    URLLoader,
    YAMLLoader,
)
from syrin.knowledge.stores import get_knowledge_store

if TYPE_CHECKING:
    from syrin.budget import BudgetTracker
    from syrin.embedding._protocol import EmbeddingProvider
    from syrin.events import EventContext
    from syrin.knowledge._store import KnowledgeStore

_log = logging.getLogger(__name__)

__all__ = [
    "AgenticRAGConfig",
    "Chunk",
    "ChunkConfig",
    "ChunkMetadata",
    "ChunkStrategy",
    "Chunker",
    "Document",
    "DocumentMetadata",
    "DocumentLoader",
    "DirectoryLoader",
    "get_chunker",
    "GitHubLoader",
    "JSONLoader",
    "Knowledge",
    "MarkdownLoader",
    "PDFLoader",
    "PythonLoader",
    "RawTextLoader",
    "TextLoader",
    "URLLoader",
    "YAMLLoader",
]


def _default_sqlite_path() -> str:
    """Default path for SQLite backend: ~/.syrin/knowledge.db."""
    return os.path.expanduser("~/.syrin/knowledge.db")


class Knowledge:
    """Declarative knowledge base for AI agents.

    Manages the full pipeline: load → chunk → embed → store → search.
    Attach to an Agent via `knowledge=` parameter.

    Example:
        knowledge = Knowledge(
            sources=[
                Knowledge.PDF("./resume.pdf"),
                Knowledge.Markdown("./about.md"),
                Knowledge.Text("I have 8 years of Python experience."),
            ],
            embedding=Embedding.OpenAI("text-embedding-3-small"),
        )

        class MyAgent(Agent):
            knowledge = knowledge
    """

    def __init__(
        self,
        sources: list[DocumentLoader],
        *,
        embedding: EmbeddingProvider | None = None,
        backend: KnowledgeBackend = KnowledgeBackend.SQLITE,
        connection_url: str | None = None,
        path: str | None = None,
        collection: str = "default",
        chunk_config: ChunkConfig | None = None,
        chunk_strategy: ChunkStrategy | None = None,
        chunk_size: int = 512,
        top_k: int = 5,
        score_threshold: float = 0.3,
        auto_sync: bool = False,
        sync_interval: int = 0,
        agentic: bool = False,
        agentic_config: AgenticRAGConfig | None = None,
        inject_system_prompt: bool = True,
        emit: Callable[[str, EventContext], None] | None = None,
        get_budget_tracker: Callable[[], object | None] | None = None,
        get_model: Callable[[], object | None] | None = None,
    ) -> None:
        """Create a Knowledge orchestrator.

        Args:
            sources: Document loaders (Knowledge.PDF, Knowledge.Text, etc.).
            embedding: Embedding provider (required). Use Embedding.OpenAI or Embedding.Ollama.
            backend: Vector store backend. Default SQLITE.
            connection_url: Postgres connection URL (required for POSTGRES).
            path: File path for SQLite or Chroma. Default ~/.syrin/knowledge.db for SQLite.
            collection: Collection/table name for vector stores.
            chunk_config: Full chunk config. Overrides chunk_strategy/chunk_size.
            chunk_strategy: Shorthand for ChunkConfig strategy.
            chunk_size: Target tokens per chunk.
            top_k: Max results per search.
            score_threshold: Minimum similarity score for search results.
            auto_sync: Enable file watching (Step 5: not yet implemented).
            sync_interval: Seconds between sync checks (0 = file watcher).
            agentic: Enable agentic retrieval (decompose, grade, refine, verify tools).
            agentic_config: Config for agentic RAG. Defaults used when None and agentic=True.
            inject_system_prompt: Inject knowledge context into agent system prompt.
            emit: Hook emitter (agent._emit_event when attached to agent).
            get_budget_tracker: Callable to get BudgetTracker for embedding cost tracking.
            get_model: Callable to get Model for decomposition/grading. Set when attached to agent.
        """
        if embedding is None:
            raise ValueError(
                "embedding is required. Use Embedding.OpenAI(api_key=...) or "
                "Embedding.Ollama() for local embeddings."
            )
        if not sources:
            raise ValueError("sources must be a non-empty list of DocumentLoaders")

        self._agentic = agentic
        self._agentic_config = (
            agentic_config
            if agentic_config is not None
            else (AgenticRAGConfig() if agentic else None)
        )
        self._get_model = get_model

        self._embedding = embedding
        self._sources = list(sources)
        self._chunk_config = chunk_config or ChunkConfig(
            strategy=chunk_strategy or ChunkStrategy.AUTO,
            chunk_size=chunk_size,
        )
        self._top_k = top_k
        self._score_threshold = score_threshold
        self._inject_system_prompt = inject_system_prompt
        self._emit = emit
        self._get_budget_tracker = get_budget_tracker
        self._ingested = False
        self._document_ids: set[str] = set()

        if backend == KnowledgeBackend.SQLITE and path is None:
            path = _default_sqlite_path()
        self._path = path

        collection_for_store = "syrin_knowledge" if collection == "default" else collection
        store: KnowledgeStore = get_knowledge_store(
            backend,
            embedding_dimensions=embedding.dimensions,
            connection_url=connection_url,
            path=path,
            collection=collection_for_store,
        )
        self._store = store

    @property
    def embedding(self) -> EmbeddingProvider:
        """Embedding provider (read-only)."""
        return self._embedding

    async def ingest(self) -> None:
        """Load, chunk, embed, and store documents. Lazy-invoked on first search."""
        self._emit_hook(Hook.KNOWLEDGE_INGEST_START, {"source_count": len(self._sources)})
        try:
            documents: list[Document] = []
            for loader in self._sources:
                if hasattr(loader, "aload") and callable(loader.aload):
                    docs = await loader.aload()
                else:
                    loop = asyncio.get_event_loop()
                    docs = await loop.run_in_executor(None, loader.load)
                documents.extend(docs)

            if not documents:
                self._emit_hook(Hook.KNOWLEDGE_INGEST_END, {"chunk_count": 0})
                self._ingested = True
                return

            chunker = get_chunker(self._chunk_config)
            chunks = chunker.chunk(documents)

            if not chunks:
                self._emit_hook(Hook.KNOWLEDGE_INGEST_END, {"chunk_count": 0})
                self._ingested = True
                return

            texts = [c.content for c in chunks]
            bt = self._get_budget_tracker() if self._get_budget_tracker else None
            embeddings = await self._embedding.embed(
                texts,
                budget_tracker=cast("BudgetTracker | None", bt),
            )

            for c in chunks:
                self._document_ids.add(c.document_id)
            await self._store.upsert(chunks, embeddings)
            self._ingested = True
            self._emit_hook(
                Hook.KNOWLEDGE_INGEST_END,
                {"chunk_count": len(chunks), "document_count": len(documents)},
            )
        except Exception as e:
            self._emit_hook(Hook.KNOWLEDGE_INGEST_END, {"error": str(e)})
            raise

    async def search(
        self,
        query: str,
        *,
        top_k: int | None = None,
        filter: MetadataFilter | None = None,
        score_threshold: float | None = None,
    ) -> list[SearchResult]:
        """Semantic search. Triggers lazy ingest on first call."""
        if not self._ingested:
            await self.ingest()

        self._emit_hook(Hook.KNOWLEDGE_SEARCH_START, {"query": query[:200]})
        k = top_k if top_k is not None else self._top_k
        thresh = score_threshold if score_threshold is not None else self._score_threshold

        bt = self._get_budget_tracker() if self._get_budget_tracker else None
        query_embeddings = await self._embedding.embed(
            [query],
            budget_tracker=cast("BudgetTracker | None", bt),
        )
        query_emb = query_embeddings[0]
        results = await self._store.search(
            query_emb,
            top_k=k,
            filter=filter,
            score_threshold=thresh,
        )
        max_preview = 200
        results_preview = [
            {
                "rank": r.rank,
                "score": round(r.score, 3),
                "content": r.chunk.content[:max_preview]
                + ("..." if len(r.chunk.content) > max_preview else ""),
            }
            for r in results[:10]
        ]
        self._emit_hook(
            Hook.KNOWLEDGE_SEARCH_END,
            {"result_count": len(results), "results": results_preview},
        )
        return results

    def add_source(self, loader: DocumentLoader) -> None:
        """Add a source. Call ingest() to load it."""
        self._sources.append(loader)
        self._emit_hook(Hook.KNOWLEDGE_SOURCE_ADDED, {"source_count": len(self._sources)})

    async def remove_source(self, loader: DocumentLoader) -> None:
        """Remove a source and delete its chunks from the store."""
        if loader in self._sources:
            self._sources.remove(loader)
            self._emit_hook(Hook.KNOWLEDGE_SOURCE_REMOVED, {"source_count": len(self._sources)})
        docs = loader.load()
        for doc in docs:
            await self._store.delete(document_id=doc.source)
            self._document_ids.discard(doc.source)

    async def clear(self) -> None:
        """Delete all chunks from the store."""
        for doc_id in list(self._document_ids):
            await self._store.delete(document_id=doc_id)
        self._document_ids.clear()

    async def stats(self) -> dict[str, int]:
        """Return chunk count and source count."""
        count = await self._store.count()
        return {"chunk_count": count, "source_count": len(self._sources)}

    def _attach_to_agent(
        self,
        emit: Callable[[str, EventContext], None],
        get_budget_tracker: Callable[[], object | None] | None = None,
        get_model: Callable[[], object | None] | None = None,
    ) -> None:
        """Wire emit, budget tracker, and model when attached to an agent. Called by Agent."""
        self._emit = emit
        self._get_budget_tracker = get_budget_tracker
        self._get_model = get_model

    def _emit_hook(self, hook: Hook, ctx: dict[str, object]) -> None:
        """Emit hook if emitter configured. Pass Hook enum (not .value) so agent event system captures it."""
        if self._emit is not None:
            from syrin.events import EventContext

            self._emit(hook, EventContext(ctx))

    # -- File Sources --

    @staticmethod
    def Text(content: str, **metadata: object) -> RawTextLoader:
        """Create raw text source.

        Args:
            content: Text content.
            **metadata: Additional metadata (reserved for future use).

        Returns:
            RawTextLoader instance.
        """
        return RawTextLoader(
            content,
            metadata=cast(DocumentMetadata, dict(metadata)) if metadata else None,
        )

    @staticmethod
    def Texts(contents: list[str], **metadata: object) -> RawTextLoader:
        """Create multiple text sources.

        Args:
            contents: List of text contents.
            **metadata: Additional metadata (reserved for future use).

        Returns:
            RawTextLoader instance.
        """
        return RawTextLoader(
            contents,
            metadata=cast(DocumentMetadata, dict(metadata)) if metadata else None,
        )

    @staticmethod
    def TextFile(path: str | Path, **_metadata: object) -> TextLoader:
        """Create text file source.

        Args:
            path: Path to text file.
            **metadata: Additional metadata (reserved for future use).

        Returns:
            TextLoader instance.
        """
        return TextLoader(path)

    @staticmethod
    def Markdown(path: str | Path, **_metadata: object) -> MarkdownLoader:
        """Create Markdown file source.

        Args:
            path: Path to Markdown file.
            **metadata: Additional metadata (reserved for future use).

        Returns:
            MarkdownLoader instance.
        """
        return MarkdownLoader(path)

    @staticmethod
    def PDF(path: str | Path, **_metadata: object) -> PDFLoader:
        """Create PDF file source.

        Args:
            path: Path to PDF file.
            **metadata: Additional metadata (reserved for future use).

        Returns:
            PDFLoader instance.
        """
        return PDFLoader(path)

    @staticmethod
    def Python(path: str | Path, **_metadata: object) -> PythonLoader:
        """Create Python source file source.

        Args:
            path: Path to Python file.
            **metadata: Additional metadata (reserved for future use).

        Returns:
            PythonLoader instance.
        """
        return PythonLoader(path)

    @staticmethod
    def YAML(path: str | Path, **_metadata: object) -> YAMLLoader:
        """Create YAML file source.

        Args:
            path: Path to YAML file.
            **metadata: Additional metadata (reserved for future use).

        Returns:
            YAMLLoader instance.
        """
        return YAMLLoader(path)

    @staticmethod
    def JSON(
        path: str | Path,
        jq_path: str | None = None,
        **_metadata: object,
    ) -> JSONLoader:
        """Create JSON file source.

        Args:
            path: Path to JSON file.
            jq_path: Optional dot notation path to extract.
            **metadata: Additional metadata (reserved for future use).

        Returns:
            JSONLoader instance.
        """
        return JSONLoader(path, jq_path=jq_path)

    # -- Directory Sources --

    @staticmethod
    def Directory(
        path: str | Path,
        glob: str = "**/*",
        pattern: str | None = None,
        recursive: bool = True,
        **_metadata: object,
    ) -> DirectoryLoader:
        """Create directory source.

        Args:
            path: Path to directory.
            glob: Glob pattern for file matching.
            pattern: Regex pattern as alternative to glob.
            recursive: Whether to search recursively.
            **metadata: Additional metadata (reserved for future use).

        Returns:
            DirectoryLoader instance.
        """
        return DirectoryLoader(
            path,
            glob=glob,
            pattern=pattern,
            recursive=recursive,
        )

    # -- Remote Sources --

    @staticmethod
    def URL(url: str, **_metadata: object) -> URLLoader:
        """Create URL source.

        Args:
            url: URL to fetch.
            **metadata: Additional metadata (reserved for future use).

        Returns:
            URLLoader instance.
        """
        return URLLoader(url)

    @staticmethod
    def GitHub(
        username: str,
        repos: list[str] | None = None,
        include_readme: bool = True,
        include_code: bool = False,
        token: str | None = None,
        **_metadata: object,
    ) -> GitHubLoader:
        """Create GitHub repository source.

        Args:
            username: GitHub username or organization.
            repos: List of repository names (None for all repos).
            include_readme: Include README content.
            include_code: Include code files.
            token: Optional GitHub API token.
            **metadata: Additional metadata (reserved for future use).

        Returns:
            GitHubLoader instance.
        """
        return GitHubLoader(
            username,
            repos=repos,
            include_readme=include_readme,
            include_code=include_code,
            token=token,
        )
