"""MemoryBus — selective cross-agent memory sharing."""

from __future__ import annotations

import asyncio
import math
from collections.abc import Callable

from syrin.enums import Hook, MemoryType
from syrin.events import EventContext, Events
from syrin.memory.config import MemoryEntry
from syrin.swarm._agent_ref import AgentRef, _aid
from syrin.swarm.backends._memory import InMemoryBusBackend
from syrin.swarm.backends._protocol import MemoryBusBackend

# Type alias for a function that converts text to an embedding vector.
EmbeddingFn = Callable[[str], list[float]]


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Return cosine similarity between two equal-length vectors."""
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


class MemoryBus:
    """Selective, typed, audited cross-agent memory bus.

    Agents publish typed :class:`~syrin.memory.config.MemoryEntry` objects to the
    bus.  Consumers call :meth:`read` to retrieve matching entries.  Every
    publish and read operation fires lifecycle :class:`~syrin.enums.Hook` events.

    Filtering chain (applied in order):
    1. ``allow_types`` — reject entries whose :attr:`~syrin.memory.config.MemoryEntry.type`
       is not in the allow-list.
    2. ``filter`` — reject entries for which the predicate returns ``False``.

    Query modes:
    * **Substring** (default) — ``read(query="...")`` returns entries whose
      ``content`` contains the query string.  Fast; no dependencies.
    * **Semantic** (opt-in) — pass ``embedding_fn`` at construction time and
      :meth:`read` ranks *all* non-expired entries by cosine similarity to the
      query embedding.  The top ``top_k`` entries are returned.

    Args:
        allow_types: Restrict published entries to these
            :class:`~syrin.enums.MemoryType` values.  ``None`` (default) allows
            all types.
        filter: Optional predicate ``(MemoryEntry) -> bool``.  Return ``True``
            to allow the entry, ``False`` to block it.
        ttl: Default time-to-live in seconds for all published entries.
            ``None`` means entries never expire.
        backend: Storage backend implementing
            :class:`~syrin.swarm.backends.MemoryBusBackend`.  Defaults to
            :class:`~syrin.swarm.backends.InMemoryBusBackend`.
        swarm_events: Optional :class:`~syrin.events.Events` instance for
            firing lifecycle hooks.
        embedding_fn: Optional callable ``(text: str) -> list[float]``.  When
            provided, :meth:`read` uses cosine similarity ranking instead of
            substring matching.  Use any embedding provider (e.g.
            ``sentence_transformers``, OpenAI embeddings, Ollama).
        top_k: Maximum number of entries returned by :meth:`read` when
            ``embedding_fn`` is set.  Ignored for substring mode.  Defaults to
            ``10``.

    Example — substring mode (default)::

        bus = MemoryBus(allow_types=[MemoryType.KNOWLEDGE], ttl=3600)
        await bus.publish(entry, agent_id="researcher")
        results = await bus.read(query="machine learning", agent_id="writer")

    Example — semantic mode::

        from sentence_transformers import SentenceTransformer

        _model = SentenceTransformer("all-MiniLM-L6-v2")

        bus = MemoryBus(
            embedding_fn=lambda text: _model.encode(text).tolist(),
            top_k=5,
        )
        await bus.publish(entry, agent_id="researcher")
        # Returns the 5 most semantically similar entries, not just substring hits.
        results = await bus.read(query="neural networks", agent_id="writer")
    """

    def __init__(
        self,
        allow_types: list[MemoryType] | None = None,
        filter: Callable[[MemoryEntry], bool] | None = None,  # noqa: A002
        ttl: float | None = None,
        backend: MemoryBusBackend | None = None,
        swarm_events: Events | None = None,
        embedding_fn: EmbeddingFn | None = None,
        top_k: int = 10,
    ) -> None:
        """Initialise MemoryBus.

        Args:
            allow_types: Allowed :class:`~syrin.enums.MemoryType` values, or
                ``None`` to allow all types.
            filter: Optional predicate to further restrict entries.
            ttl: Default TTL in seconds for all entries.
            backend: Storage backend.  Defaults to
                :class:`~syrin.swarm.backends.InMemoryBusBackend`.
            swarm_events: Events bus for hook emission.
            embedding_fn: Optional embedding function for semantic search.
                When set, :meth:`read` ranks entries by cosine similarity
                instead of substring matching.
            top_k: Maximum results returned in semantic mode.  Defaults to 10.
        """
        self._allow_types: list[MemoryType] | None = allow_types
        self._filter: Callable[[MemoryEntry], bool] | None = filter
        self._ttl: float | None = ttl
        self._backend: MemoryBusBackend = backend or InMemoryBusBackend()
        self._events: Events | None = swarm_events
        self._embedding_fn: EmbeddingFn | None = embedding_fn
        self._top_k: int = top_k

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _fire(self, hook: Hook, data: dict[str, object]) -> None:
        """Emit *hook* via the swarm events bus (if connected)."""
        if self._events is not None:
            ctx = EventContext(data)
            ctx.scrub()
            self._events._trigger(hook, ctx)

    def _check_allowed(self, entry: MemoryEntry) -> str | None:
        """Return a filter reason string if *entry* should be blocked, else ``None``."""
        if self._allow_types is not None and entry.type not in self._allow_types:
            return f"type '{entry.type}' not in allow_types"
        if self._filter is not None and not self._filter(entry):
            return "custom filter rejected entry"
        return None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def publish(self, entry: MemoryEntry, agent_id: AgentRef | str) -> bool:
        """Publish *entry* from *agent_id* to the bus.

        Applies the filtering chain (allow_types, then custom filter).
        Fires :attr:`~syrin.enums.Hook.MEMORY_BUS_PUBLISHED` on success or
        :attr:`~syrin.enums.Hook.MEMORY_BUS_FILTERED` on rejection.

        Args:
            entry: The :class:`~syrin.memory.config.MemoryEntry` to publish.
            agent_id: Publishing agent instance or agent ID string.

        Returns:
            ``True`` if the entry was stored, ``False`` if filtered out.
        """
        aid = _aid(agent_id)
        filter_reason = self._check_allowed(entry)
        if filter_reason is not None:
            self._fire(
                Hook.MEMORY_BUS_FILTERED,
                {
                    "entry_id": entry.id,
                    "agent_id": aid,
                    "filter_reason": filter_reason,
                },
            )
            return False

        await self._backend.store(entry, agent_id=aid, ttl=self._ttl)
        self._fire(
            Hook.MEMORY_BUS_PUBLISHED,
            {
                "entry_id": entry.id,
                "agent_id": aid,
                "content": entry.content,
                "type": str(entry.type),
            },
        )
        return True

    async def read(self, query: str, agent_id: AgentRef | str) -> list[MemoryEntry]:
        """Return entries matching *query* from the bus.

        **Substring mode** (default): returns entries whose ``content`` contains
        *query* as a substring.  An empty *query* returns all non-expired entries.

        **Semantic mode** (when ``embedding_fn`` was passed at construction):
        fetches all non-expired entries, scores each by cosine similarity to the
        *query* embedding, and returns the top ``top_k`` results sorted by
        descending similarity.  An empty *query* falls back to returning all
        entries unranked.

        Fires :attr:`~syrin.enums.Hook.MEMORY_BUS_READ`.

        Args:
            query: Search string.  In substring mode: exact substring match.
                In semantic mode: natural-language query for embedding.
            agent_id: Reading agent instance or agent ID string.

        Returns:
            List of matching :class:`~syrin.memory.config.MemoryEntry` objects,
            ordered by relevance (semantic mode) or insertion order (substring).
        """
        aid = _aid(agent_id)

        if self._embedding_fn is not None and query:
            # Semantic mode: rank all non-expired entries by cosine similarity.
            all_rows = await self._backend.all_entries()
            query_vec = self._embedding_fn(query)
            scored: list[tuple[float, MemoryEntry]] = []
            for entry, _agent, _expire in all_rows:
                entry_vec = self._embedding_fn(entry.content)
                score = _cosine_similarity(query_vec, entry_vec)
                scored.append((score, entry))
            scored.sort(key=lambda t: t[0], reverse=True)
            results = [entry for _, entry in scored[: self._top_k]]
        else:
            # Substring mode (default).
            results = await self._backend.query(query=query, agent_id=aid)

        self._fire(
            Hook.MEMORY_BUS_READ,
            {
                "query": query,
                "agent_id": aid,
                "result_count": len(results),
                "semantic": self._embedding_fn is not None and bool(query),
            },
        )
        return results

    async def expire_now(self) -> list[str]:
        """Immediately remove expired entries and fire expiry hooks.

        Returns:
            IDs of entries that were removed.
        """
        expired_ids = await self._backend.clear_expired()
        for entry_id in expired_ids:
            self._fire(
                Hook.MEMORY_BUS_EXPIRED,
                {"entry_id": entry_id},
            )
        return expired_ids

    async def _expire_loop(self, interval: float = 30.0) -> None:
        """Background task: periodically clear expired entries.

        Args:
            interval: Seconds between expiry sweeps.  Defaults to 30.
        """
        while True:
            await asyncio.sleep(interval)
            await self.expire_now()


__all__ = ["MemoryBus"]
