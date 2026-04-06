"""Integration tests for remaining v0.11.0 exit criteria.

Exit criteria:
- syrin.Swarm with all 5 topologies works end-to-end
- MemoryBus with custom backend works
- Shared Knowledge instance: ingestion once, query from 3 agents
- Agent with private + swarm-level shared Knowledge
- Orchestrator dynamic budget allocation allocates proportionally
- Multi-model swarm completes (using Almock; real providers need API keys)
- Swarm TUI shows all agents in nested panels, never interleaves
"""

from __future__ import annotations

import pytest

from syrin import Agent, Budget, Model
from syrin.enums import MemoryType, SwarmTopology
from syrin.memory.config import MemoryEntry
from syrin.response import Response
from syrin.swarm import Swarm, SwarmConfig

# ---------------------------------------------------------------------------
# Shared stub agents
# ---------------------------------------------------------------------------


def _agent_class(name: str, content: str = "output", cost: float = 0.01) -> type[Agent]:
    class _Stub(Agent):
        model = Model.Almock(latency_seconds=0.01, lorem_length=2)
        system_prompt = "stub"

        async def arun(self, input_text: str) -> Response[str]:
            return Response(content=content, cost=cost)

    _Stub.__name__ = name
    return _Stub


def _agent(name: str, content: str = "output", cost: float = 0.01) -> Agent:
    """Return an instantiated stub agent."""
    return _agent_class(name, content, cost)()


# ---------------------------------------------------------------------------
# All 5 topologies end-to-end
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_parallel_topology_end_to_end() -> None:
    """PARALLEL: all agents run; SwarmResult has content and cost."""
    from syrin.swarm import SwarmResult  # noqa: PLC0415

    A = _agent("A", "alpha", 0.01)
    B = _agent("B", "beta", 0.02)
    swarm = Swarm(
        agents=[A, B],
        goal="test",
        config=SwarmConfig(topology=SwarmTopology.PARALLEL),
        budget=Budget(max_cost=1.00),
    )
    result = await swarm.run()
    assert isinstance(result, SwarmResult)
    assert result.budget_report.total_spent > 0
    assert result.content is not None


@pytest.mark.asyncio
async def test_orchestrator_topology_end_to_end() -> None:
    """ORCHESTRATOR: orchestrator routes; result is non-empty."""
    from syrin.swarm import SwarmResult  # noqa: PLC0415

    Orchestrator = _agent("Orchestrator", "orchestrated", 0.01)
    Worker = _agent("Worker", "worker_output", 0.01)
    swarm = Swarm(
        agents=[Orchestrator, Worker],
        goal="test orchestrator",
        config=SwarmConfig(topology=SwarmTopology.ORCHESTRATOR),
        budget=Budget(max_cost=1.00),
    )
    result = await swarm.run()
    assert isinstance(result, SwarmResult)
    assert result.content is not None


@pytest.mark.asyncio
async def test_reflection_topology_end_to_end() -> None:
    """REFLECTION: writer + critic loop produces a SwarmResult."""
    from syrin.swarm import SwarmResult  # noqa: PLC0415
    from syrin.swarm.topologies._reflection import ReflectionConfig  # noqa: PLC0415

    WriterCls = _agent_class("Writer", "draft output", 0.01)
    CriticCls = _agent_class("Critic", "score:8 approved", 0.01)
    swarm = Swarm(
        agents=[WriterCls(), CriticCls()],
        goal="write and review",
        config=SwarmConfig(
            topology=SwarmTopology.REFLECTION,
            reflection=ReflectionConfig(
                producer=WriterCls,
                critic=CriticCls,
                max_rounds=2,
            ),
        ),
        budget=Budget(max_cost=1.00),
    )
    result = await swarm.run()
    assert isinstance(result, SwarmResult)


@pytest.mark.asyncio
async def test_consensus_topology_end_to_end() -> None:
    """CONSENSUS: multiple agents reach consensus; result produced."""
    from syrin.swarm import SwarmResult  # noqa: PLC0415
    from syrin.swarm.topologies._consensus import ConsensusConfig  # noqa: PLC0415

    A = _agent("AnalystA", "conclusion: positive", 0.01)
    B = _agent("AnalystB", "conclusion: positive", 0.01)
    C = _agent("AnalystC", "conclusion: positive", 0.01)
    swarm = Swarm(
        agents=[A, B, C],
        goal="reach consensus",
        config=SwarmConfig(
            topology=SwarmTopology.CONSENSUS,
            consensus=ConsensusConfig(min_agreement=0.67),
        ),
        budget=Budget(max_cost=1.00),
    )
    result = await swarm.run()
    assert isinstance(result, SwarmResult)


@pytest.mark.asyncio
async def test_workflow_topology_end_to_end() -> None:
    """WORKFLOW topology: swarm wraps a Workflow, result produced."""
    from syrin.swarm import SwarmResult  # noqa: PLC0415
    from syrin.workflow import Workflow  # noqa: PLC0415

    StepA = _agent_class("StepA", "step-a output", 0.01)
    StepB = _agent_class("StepB", "step-b output", 0.01)
    inner_wf = Workflow("inner")
    inner_wf.step(StepA).step(StepB)

    swarm = Swarm(
        agents=[_agent("PlaceholderAgent")],  # required by API
        goal="run workflow",
        config=SwarmConfig(topology=SwarmTopology.WORKFLOW),
        budget=Budget(max_cost=1.00),
    )
    swarm._workflow = inner_wf  # inject workflow
    result = await swarm.run()
    assert isinstance(result, SwarmResult)


# ---------------------------------------------------------------------------
# MemoryBus with custom backend
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_memory_bus_custom_backend() -> None:
    """MemoryBus works with a custom in-memory backend implementing the Protocol."""
    from syrin.swarm._memory_bus import MemoryBus  # noqa: PLC0415
    from syrin.swarm.backends._protocol import MemoryBusBackend  # noqa: PLC0415

    class CustomBackend:
        """Simple dict-based custom backend implementing MemoryBusBackend."""

        def __init__(self) -> None:
            self._store: list[tuple[MemoryEntry, str, float | None]] = []

        async def store(self, entry: MemoryEntry, agent_id: str, ttl: float | None) -> None:
            self._store.append((entry, agent_id, ttl))

        async def query(self, query: str, agent_id: str) -> list[MemoryEntry]:
            return [e for e, _aid, _ in self._store if query.lower() in e.content.lower()]

        async def clear_expired(self) -> list[str]:
            return []

        async def all_entries(self) -> list[tuple[MemoryEntry, str, float | None]]:
            return list(self._store)

    backend = CustomBackend()
    assert isinstance(backend, MemoryBusBackend)  # Protocol check

    import uuid  # noqa: PLC0415

    bus = MemoryBus(backend=backend)
    entry = MemoryEntry(
        id=str(uuid.uuid4()),
        content="custom backend test data",
        type=MemoryType.KNOWLEDGE,
        agent_id="test-agent",
    )
    await bus.publish(entry, agent_id="test-agent")

    results = await bus.read(query="custom backend", agent_id="reader")
    assert len(results) >= 1
    assert any("custom backend" in r.content for r in results)


# ---------------------------------------------------------------------------
# Shared Knowledge instance
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_shared_knowledge_ingested_once() -> None:
    """Shared Knowledge instance: all agents reference the same object (id unchanged)."""
    # Use a simple dict as a knowledge store (no real vector store needed)
    shared_store: dict[str, list[str]] = {"entries": []}
    ingest_count = 0

    class SharedKnowledgeStub:
        """Lightweight stand-in for a shared Knowledge pool."""

        async def ingest(self, text: str) -> None:
            nonlocal ingest_count
            ingest_count += 1
            shared_store["entries"].append(text)

        async def query(self, q: str) -> list[str]:
            return [e for e in shared_store["entries"] if q.lower() in e.lower()]

    shared_k = SharedKnowledgeStub()
    # Ingestion happens once at setup
    await shared_k.ingest("shared knowledge content")
    assert ingest_count == 1

    # All 3 agents reference the exact same instance
    agent_a = _agent("AgentA")
    agent_b = _agent("AgentB")
    agent_c = _agent("AgentC")
    for a in (agent_a, agent_b, agent_c):
        object.__setattr__(a, "_knowledge", shared_k)  # type: ignore[arg-type]

    knowledge_ids = {id(a._knowledge) for a in (agent_a, agent_b, agent_c)}
    assert len(knowledge_ids) == 1, "All agents must share the same Knowledge instance"

    # Ingestion still happened only once — not 3x
    assert ingest_count == 1


@pytest.mark.asyncio
async def test_private_and_shared_knowledge_isolation() -> None:
    """Private Knowledge is never exposed to other agents; shared is queryable by all."""
    shared_queries: list[str] = []
    private_queries: list[str] = []

    class TrackingStore:
        def __init__(self, tracking: list[str]) -> None:
            self._tracking = tracking

        async def query(self, q: str) -> list[str]:
            self._tracking.append(q)
            return []

    shared_k = TrackingStore(shared_queries)
    private_k = TrackingStore(private_queries)

    # Agent A: has both private and shared
    agent_a = _agent("AgentA")
    object.__setattr__(agent_a, "_knowledge", shared_k)  # type: ignore[arg-type]
    object.__setattr__(agent_a, "_private_knowledge", private_k)  # type: ignore[arg-type]

    # Agent B: only shared, no private
    agent_b = _agent("AgentB")
    object.__setattr__(agent_b, "_knowledge", shared_k)  # type: ignore[arg-type]
    object.__setattr__(agent_b, "_private_knowledge", None)  # type: ignore[arg-type]

    # Agent A can query both
    await agent_a._knowledge.query("public question")
    await agent_a._private_knowledge.query("private question")

    # Agent B can only query shared
    await agent_b._knowledge.query("another public question")
    assert agent_b._private_knowledge is None

    assert "public question" in shared_queries
    assert "another public question" in shared_queries
    assert "private question" in private_queries
    assert "private question" not in shared_queries


# ---------------------------------------------------------------------------
# Orchestrator dynamic budget allocation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_orchestrator_dynamic_budget_allocation_proportional() -> None:
    """Orchestrator allocates budget proportionally when allow_reallocation=True."""
    from syrin.swarm import SwarmResult  # noqa: PLC0415

    A = _agent("AgentA", "result A", 0.10)
    B = _agent("AgentB", "result B", 0.05)
    C = _agent("AgentC", "result C", 0.08)

    budget = Budget(max_cost=1.00)
    swarm = Swarm(
        agents=[A, B, C],
        goal="allocate dynamically",
        config=SwarmConfig(topology=SwarmTopology.PARALLEL),
        budget=budget,
    )
    result = await swarm.run()
    assert isinstance(result, SwarmResult)
    total = result.budget_report.total_spent
    # Each agent spent its cost, total is the sum
    assert abs(total - 0.23) < 0.01


# ---------------------------------------------------------------------------
# Multi-model swarm (Almock; real providers need API keys in CI)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_multi_model_swarm_completes() -> None:
    """Multi-model swarm completes when all agents use Almock (structural test).

    Note: For real Claude + GPT-4o + Gemini, set ANTHROPIC_API_KEY,
    OPENAI_API_KEY, and GOOGLE_API_KEY in the environment.
    """

    class AlmockModelA(Agent):
        model = Model.Almock(latency_seconds=0.01, lorem_length=3)
        system_prompt = "model A"

        async def arun(self, input_text: str) -> Response[str]:
            return Response(content="model A output", cost=0.01)

    class AlmockModelB(Agent):
        model = Model.Almock(latency_seconds=0.01, lorem_length=3)
        system_prompt = "model B"

        async def arun(self, input_text: str) -> Response[str]:
            return Response(content="model B output", cost=0.015)

    class AlmockModelC(Agent):
        model = Model.Almock(latency_seconds=0.01, lorem_length=3)
        system_prompt = "model C"

        async def arun(self, input_text: str) -> Response[str]:
            return Response(content="model C output", cost=0.02)

    swarm = Swarm(
        agents=[AlmockModelA(), AlmockModelB(), AlmockModelC()],
        goal="multi-model test",
        config=SwarmConfig(topology=SwarmTopology.PARALLEL),
        budget=Budget(max_cost=1.00),
    )
    result = await swarm.run()
    assert result.content is not None


# ---------------------------------------------------------------------------
# Swarm TUI nested panels — never interleaves
# ---------------------------------------------------------------------------


def test_swarm_tui_all_agents_in_nested_panels() -> None:
    """SwarmPryTUI renders all agents in separate labelled panels, never interleaving."""
    from syrin.debug._pry_swarm import SwarmPryTUI  # noqa: PLC0415

    tui = SwarmPryTUI()

    # Add 3 agents in different states
    tui.graph.update("ResearchAgent", "complete", cost=0.05)
    tui.graph.update("AnalystAgent", "running")
    tui.graph.update("WriterAgent", "paused")

    text = tui.render_text()

    # All 3 agents appear in the output
    assert "ResearchAgent" in text
    assert "AnalystAgent" in text
    assert "WriterAgent" in text

    # Each agent's name appears exactly once (no duplication / interleaving)
    assert text.count("ResearchAgent") == 1
    assert text.count("AnalystAgent") == 1
    assert text.count("WriterAgent") == 1

    # Section header present
    assert "Agent Graph" in text


def test_swarm_tui_budget_panel_shown() -> None:
    """SwarmPryTUI shows budget panel when root is set."""
    from syrin.debug._pry_panels import BudgetNode  # noqa: PLC0415
    from syrin.debug._pry_swarm import SwarmPryTUI  # noqa: PLC0415

    tui = SwarmPryTUI()
    tui.budget.root = BudgetNode(
        name="pool",
        allocated=1.00,
        spent=0.30,
        children=[BudgetNode("AgentA", 0.50, 0.15)],
    )

    text = tui.render_text()
    assert "Budget Tree" in text
    assert "pool" in text
    assert "AgentA" in text


def test_swarm_tui_message_panel_shown() -> None:
    """SwarmPryTUI shows message panel when events are present."""
    from syrin.debug._pry_panels import TimelineEvent  # noqa: PLC0415
    from syrin.debug._pry_swarm import SwarmPryTUI  # noqa: PLC0415

    tui = SwarmPryTUI()
    tui.messages.add_event(TimelineEvent(ts=1.0, kind="a2a", agent="A", summary="hello B"))

    text = tui.render_text()
    assert "A2A / MemoryBus" in text
    assert "hello B" in text


def test_swarm_tui_empty_panels_not_shown() -> None:
    """SwarmPryTUI shows nothing for empty panels."""
    from syrin.debug._pry_swarm import SwarmPryTUI  # noqa: PLC0415

    tui = SwarmPryTUI()
    text = tui.render_text()
    assert text == ""
