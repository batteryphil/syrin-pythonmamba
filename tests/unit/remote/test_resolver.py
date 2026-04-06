"""Tests for remote config resolver: ConfigResolver, apply_overrides, ResolveResult."""

from __future__ import annotations

import threading

from syrin import Agent, Budget, Model
from syrin.budget import RateLimit
from syrin.enums import DecayStrategy
from syrin.memory import Memory
from syrin.memory.config import Decay
from syrin.remote._registry import get_registry
from syrin.remote._resolver import ConfigResolver, ResolveResult
from syrin.remote._types import ConfigOverride, OverridePayload


def _make_agent(
    name: str = "resolver_test",
    *,
    budget: Budget | None = None,
    memory: Memory | None = None,
) -> Agent:
    """Agent with optional budget and memory for resolver tests."""
    if budget is None:
        budget = Budget(max_cost=1.0)
    return Agent(
        model=Model.Almock(),
        name=name,
        budget=budget,
        memory=memory or Memory(types=[], top_k=5, decay=Decay(strategy=DecayStrategy.EXPONENTIAL)),
    )


def _make_fake_embedding(dim: int = 4):
    """Minimal EmbeddingProvider for knowledge tests."""

    class FakeEmbedding:
        dimensions = dim
        model_id = "fake"

        async def embed(self, texts, budget_tracker=None):
            return [[0.1] * dim for _ in texts]

    return FakeEmbedding()


def _make_agent_with_knowledge(
    *,
    top_k: int = 5,
    grounding: bool = False,
) -> Agent:
    """Agent with Knowledge for resolver tests."""
    from syrin.enums import KnowledgeBackend
    from syrin.knowledge import Knowledge

    knowledge = Knowledge(
        sources=[Knowledge.Text("test content")],
        embedding=_make_fake_embedding(),
        backend=KnowledgeBackend.MEMORY,
        top_k=top_k,
        grounding_enabled=grounding,
    )
    return Agent(
        model=Model.Almock(),
        name="resolver_knowledge_test",
        budget=Budget(max_cost=1.0),
        knowledge=knowledge,
    )


def _make_agent_with_model() -> Agent:
    """Agent with Model.Ollama for resolver tests."""
    return Agent(
        model=Model.Ollama("llama3"),
        name="resolver_model_test",
        budget=Budget(max_cost=1.0),
    )


def _payload(agent_id: str, *overrides: tuple[str, object]) -> OverridePayload:
    """Build OverridePayload from (path, value) pairs."""
    return OverridePayload(
        agent_id=agent_id,
        version=1,
        overrides=[ConfigOverride(path=p, value=v) for p, v in overrides],
    )


# --- ResolveResult shape ---


class TestResolveResultShape:
    """ResolveResult has accepted, rejected, pending_restart."""

    def test_resolve_result_has_accepted_rejected_pending_restart(self) -> None:
        """ResolveResult is a dataclass with accepted, rejected, pending_restart."""
        r = ResolveResult(accepted=[], rejected=[], pending_restart=[])
        assert r.accepted == []
        assert r.rejected == []
        assert r.pending_restart == []

    def test_resolve_result_accepted_list_of_paths(self) -> None:
        """accepted is list of path strings."""
        r = ResolveResult(accepted=["budget.max_cost"], rejected=[], pending_restart=[])
        assert r.accepted == ["budget.max_cost"]

    def test_resolve_result_rejected_list_of_tuples(self) -> None:
        """rejected is list of (path, reason) tuples."""
        r = ResolveResult(
            accepted=[],
            rejected=[("budget.max_cost", "validation error")],
            pending_restart=[],
        )
        assert r.rejected == [("budget.max_cost", "validation error")]


# --- Valid overrides ---


class TestValidOverrides:
    """Applying valid overrides updates agent config."""

    def test_apply_budget_run(self) -> None:
        """Apply budget.max_cost=2.0 -> agent._budget.max_cost == 2.0."""
        agent = _make_agent()
        reg = get_registry()
        reg.register(agent)
        agent_id = reg.make_agent_id(agent)
        schema = reg.get_schema(agent_id)
        assert schema is not None
        payload = _payload(agent_id, ("budget.max_cost", 2.0))
        result = ConfigResolver().apply_overrides(agent, payload, schema=schema)
        assert "budget.max_cost" in result.accepted
        assert agent._budget is not None
        assert agent._budget.max_cost == 2.0
        reg.unregister(agent_id)

    def test_apply_budget_nested_per_hour(self) -> None:
        """Apply budget.rate_limits.hour -> nested RateLimit updated."""
        agent = _make_agent(
            budget=Budget(max_cost=1.0, rate_limits=RateLimit(hour=10.0, day=100.0))
        )
        reg = get_registry()
        reg.register(agent)
        agent_id = reg.make_agent_id(agent)
        schema = reg.get_schema(agent_id)
        assert schema is not None
        payload = _payload(agent_id, ("budget.rate_limits.hour", 50.0))
        result = ConfigResolver().apply_overrides(agent, payload, schema=schema)
        assert "budget.rate_limits.hour" in result.accepted
        assert agent._budget is not None
        assert agent._budget.rate_limits is not None
        assert agent._budget.rate_limits.hour == 50.0
        assert agent._budget.rate_limits.day == 100.0
        reg.unregister(agent_id)

    def test_apply_memory_decay_strategy_enum(self) -> None:
        """Apply memory.decay.strategy='linear' -> DecayStrategy.LINEAR."""
        agent = _make_agent()
        reg = get_registry()
        reg.register(agent)
        agent_id = reg.make_agent_id(agent)
        schema = reg.get_schema(agent_id)
        assert schema is not None
        payload = _payload(agent_id, ("memory.decay.strategy", "linear"))
        result = ConfigResolver().apply_overrides(agent, payload, schema=schema)
        assert "memory.decay.strategy" in result.accepted
        assert agent._persistent_memory is not None
        assert agent._persistent_memory.decay.strategy == DecayStrategy.LINEAR
        reg.unregister(agent_id)

    def test_apply_agent_max_tool_iterations(self) -> None:
        """Apply agent.max_tool_iterations=5 -> agent._max_tool_iterations == 5."""
        agent = _make_agent()
        reg = get_registry()
        reg.register(agent)
        agent_id = reg.make_agent_id(agent)
        schema = reg.get_schema(agent_id)
        assert schema is not None
        payload = _payload(agent_id, ("agent.max_tool_iterations", 5))
        result = ConfigResolver().apply_overrides(agent, payload, schema=schema)
        assert "agent.max_tool_iterations" in result.accepted
        assert agent._max_tool_iterations == 5
        reg.unregister(agent_id)

    def test_apply_agent_max_tool_iterations_updates_correctly(self) -> None:
        """Apply agent.max_tool_iterations=5 updates the agent correctly (loop_strategy removed)."""
        agent = _make_agent()
        reg = get_registry()
        reg.register(agent)
        agent_id = reg.make_agent_id(agent)
        schema = reg.get_schema(agent_id)
        assert schema is not None
        payload = _payload(agent_id, ("agent.max_tool_iterations", 5))
        result = ConfigResolver().apply_overrides(agent, payload, schema=schema)
        assert "agent.max_tool_iterations" in result.accepted
        assert agent._max_tool_iterations == 5
        reg.unregister(agent_id)

    def test_apply_knowledge_top_k(self) -> None:
        """Apply knowledge.top_k=10 -> agent._knowledge._top_k == 10."""
        agent = _make_agent_with_knowledge()
        reg = get_registry()
        reg.register(agent)
        agent_id = reg.make_agent_id(agent)
        schema = reg.get_schema(agent_id)
        assert schema is not None
        payload = _payload(agent_id, ("knowledge.top_k", 10))
        result = ConfigResolver().apply_overrides(agent, payload, schema=schema)
        assert "knowledge.top_k" in result.accepted
        assert agent._knowledge is not None
        assert agent._knowledge._top_k == 10
        reg.unregister(agent_id)

    def test_apply_knowledge_score_threshold(self) -> None:
        """Apply knowledge.score_threshold=0.6 -> _score_threshold updated."""
        agent = _make_agent_with_knowledge()
        reg = get_registry()
        reg.register(agent)
        agent_id = reg.make_agent_id(agent)
        schema = reg.get_schema(agent_id)
        assert schema is not None
        payload = _payload(agent_id, ("knowledge.score_threshold", 0.6))
        result = ConfigResolver().apply_overrides(agent, payload, schema=schema)
        assert "knowledge.score_threshold" in result.accepted
        assert agent._knowledge is not None
        assert agent._knowledge._score_threshold == 0.6
        reg.unregister(agent_id)

    def test_apply_knowledge_grounding_extract_facts(self) -> None:
        """Apply knowledge.grounding.extract_facts=False creates/updates GroundingConfig."""
        agent = _make_agent_with_knowledge(grounding=True)
        reg = get_registry()
        reg.register(agent)
        agent_id = reg.make_agent_id(agent)
        schema = reg.get_schema(agent_id)
        assert schema is not None
        payload = _payload(agent_id, ("knowledge.grounding.extract_facts", False))
        result = ConfigResolver().apply_overrides(agent, payload, schema=schema)
        assert "knowledge.grounding.extract_facts" in result.accepted
        assert agent._knowledge is not None
        assert agent._knowledge._grounding_config is not None
        assert agent._knowledge._grounding_config.extract_facts is False
        reg.unregister(agent_id)

    def test_apply_knowledge_grounding_creates_config_when_none(self) -> None:
        """Apply knowledge.grounding.extract_facts when grounding=None creates GroundingConfig."""
        agent = _make_agent_with_knowledge(grounding=False)
        assert agent._knowledge is not None
        assert agent._knowledge._grounding_config is None
        reg = get_registry()
        reg.register(agent)
        agent_id = reg.make_agent_id(agent)
        schema = reg.get_schema(agent_id)
        assert schema is not None
        payload = _payload(agent_id, ("knowledge.grounding.extract_facts", False))
        result = ConfigResolver().apply_overrides(agent, payload, schema=schema)
        assert "knowledge.grounding.extract_facts" in result.accepted
        assert agent._knowledge._grounding_config is not None
        assert agent._knowledge._grounding_config.extract_facts is False
        reg.unregister(agent_id)

    def test_apply_knowledge_chunk_config_strategy(self) -> None:
        """Apply knowledge.chunk_config.strategy='markdown' -> ChunkStrategy.MARKDOWN."""
        from syrin.knowledge._chunker import ChunkStrategy

        agent = _make_agent_with_knowledge()
        reg = get_registry()
        reg.register(agent)
        agent_id = reg.make_agent_id(agent)
        schema = reg.get_schema(agent_id)
        assert schema is not None
        payload = _payload(agent_id, ("knowledge.chunk_config.strategy", "markdown"))
        result = ConfigResolver().apply_overrides(agent, payload, schema=schema)
        assert "knowledge.chunk_config.strategy" in result.accepted
        assert agent._knowledge is not None
        assert agent._knowledge._chunk_config.strategy == ChunkStrategy.MARKDOWN
        reg.unregister(agent_id)

    def test_apply_model_temperature(self) -> None:
        """Apply model.temperature=0.9 -> model updated via switch_model."""
        agent = _make_agent_with_model()
        reg = get_registry()
        reg.register(agent)
        agent_id = reg.make_agent_id(agent)
        schema = reg.get_schema(agent_id)
        assert schema is not None
        payload = _payload(agent_id, ("model.temperature", 0.9))
        result = ConfigResolver().apply_overrides(agent, payload, schema=schema)
        assert "model.temperature" in result.accepted
        assert agent._model is not None
        assert agent._model.settings.temperature == 0.9
        reg.unregister(agent_id)

    def test_apply_model_max_tokens(self) -> None:
        """Apply model.max_tokens=2048 -> model updated via switch_model."""
        agent = _make_agent_with_model()
        reg = get_registry()
        reg.register(agent)
        agent_id = reg.make_agent_id(agent)
        schema = reg.get_schema(agent_id)
        assert schema is not None
        payload = _payload(agent_id, ("model.max_tokens", 2048))
        result = ConfigResolver().apply_overrides(agent, payload, schema=schema)
        assert "model.max_tokens" in result.accepted
        assert agent._model is not None
        assert agent._model.settings.max_output_tokens == 2048
        reg.unregister(agent_id)

    def test_apply_unknown_field_is_rejected(self) -> None:
        """Unknown field path results in rejected override (loop_strategy removed)."""
        agent = _make_agent()
        reg = get_registry()
        reg.register(agent)
        agent_id = reg.make_agent_id(agent)
        schema = reg.get_schema(agent_id)
        assert schema is not None
        payload = _payload(agent_id, ("agent.loop_strategy", "react"))
        result = ConfigResolver().apply_overrides(agent, payload, schema=schema)
        # loop_strategy is no longer a valid field; it should be rejected or ignored
        assert "agent.loop_strategy" not in result.accepted
        reg.unregister(agent_id)

    def test_empty_overrides(self) -> None:
        """Empty payload.overrides -> empty accepted/rejected/pending_restart."""
        agent = _make_agent()
        reg = get_registry()
        reg.register(agent)
        agent_id = reg.make_agent_id(agent)
        schema = reg.get_schema(agent_id)
        payload = OverridePayload(agent_id=agent_id, version=0, overrides=[])
        result = ConfigResolver().apply_overrides(agent, payload, schema=schema)
        assert result.accepted == []
        assert result.rejected == []
        assert result.pending_restart == []
        reg.unregister(agent_id)


# --- Validation rejection ---


class TestValidationRejection:
    """Invalid values are rejected; agent config unchanged."""

    def test_budget_run_negative_rejected(self) -> None:
        """budget.max_cost=-1 -> section rejected, agent unchanged."""
        agent = _make_agent()
        original_run = agent._budget.max_cost if agent._budget else None
        reg = get_registry()
        reg.register(agent)
        agent_id = reg.make_agent_id(agent)
        schema = reg.get_schema(agent_id)
        payload = _payload(agent_id, ("budget.max_cost", -1.0))
        result = ConfigResolver().apply_overrides(agent, payload, schema=schema)
        assert "budget.max_cost" not in result.accepted
        assert any(p == "budget.max_cost" for p, _ in result.rejected)
        assert agent._budget is not None
        assert agent._budget.max_cost == original_run
        reg.unregister(agent_id)

    def test_memory_top_k_negative_rejected(self) -> None:
        """memory.top_k=-1 -> rejected (top_k has gt=0)."""
        agent = _make_agent()
        reg = get_registry()
        reg.register(agent)
        agent_id = reg.make_agent_id(agent)
        schema = reg.get_schema(agent_id)
        payload = _payload(agent_id, ("memory.top_k", -1))
        result = ConfigResolver().apply_overrides(agent, payload, schema=schema)
        assert "memory.top_k" not in result.accepted
        assert any(p == "memory.top_k" for p, _ in result.rejected)
        reg.unregister(agent_id)

    def test_invalid_chunk_strategy_rejected(self) -> None:
        """knowledge.chunk_config.strategy='invalid' -> rejected."""
        agent = _make_agent_with_knowledge()
        reg = get_registry()
        reg.register(agent)
        agent_id = reg.make_agent_id(agent)
        schema = reg.get_schema(agent_id)
        assert schema is not None
        payload = _payload(agent_id, ("knowledge.chunk_config.strategy", "invalid_strategy"))
        result = ConfigResolver().apply_overrides(agent, payload, schema=schema)
        assert "knowledge.chunk_config.strategy" not in result.accepted
        assert any(p == "knowledge.chunk_config.strategy" for p, _ in result.rejected)
        reg.unregister(agent_id)

    def test_invalid_enum_value_rejected(self) -> None:
        """memory.decay.strategy='invalid' -> rejected."""
        agent = _make_agent()
        reg = get_registry()
        reg.register(agent)
        agent_id = reg.make_agent_id(agent)
        schema = reg.get_schema(agent_id)
        payload = _payload(agent_id, ("memory.decay.strategy", "invalid_strategy"))
        result = ConfigResolver().apply_overrides(agent, payload, schema=schema)
        assert "memory.decay.strategy" not in result.accepted
        assert any(p == "memory.decay.strategy" for p, _ in result.rejected)
        assert agent._persistent_memory is not None
        assert agent._persistent_memory.decay.strategy == DecayStrategy.EXPONENTIAL
        reg.unregister(agent_id)


# --- remote_excluded ---


class TestRemoteExcluded:
    """Paths marked remote_excluded in schema are rejected."""

    def test_budget_on_exceeded_rejected(self) -> None:
        """budget.on_exceeded is callable -> remote_excluded -> rejected."""
        agent = _make_agent()
        reg = get_registry()
        reg.register(agent)
        agent_id = reg.make_agent_id(agent)
        schema = reg.get_schema(agent_id)
        payload = _payload(agent_id, ("budget.on_exceeded", None))
        result = ConfigResolver().apply_overrides(agent, payload, schema=schema)
        assert "budget.on_exceeded" not in result.accepted
        assert any("on_exceeded" in str(p) for p, _ in result.rejected)
        reg.unregister(agent_id)


# --- Unknown path ---


class TestUnknownPath:
    """Unknown paths are rejected when schema is provided."""

    def test_unknown_path_rejected(self) -> None:
        """Override for path not in schema -> rejected."""
        agent = _make_agent()
        reg = get_registry()
        reg.register(agent)
        agent_id = reg.make_agent_id(agent)
        schema = reg.get_schema(agent_id)
        payload = _payload(agent_id, ("budget.nonexistent_field", 1.0))
        result = ConfigResolver().apply_overrides(agent, payload, schema=schema)
        assert "budget.nonexistent_field" not in result.accepted
        assert any("nonexistent" in p for p, _ in result.rejected)
        reg.unregister(agent_id)


# --- Hot-swap blocklist (pending_restart) ---


class TestHotSwapBlocklist:
    """Blocklisted paths are applied but flagged pending_restart."""

    def test_memory_backend_in_pending_restart(self) -> None:
        """memory.backend override -> applied and in pending_restart."""
        agent = _make_agent()
        reg = get_registry()
        reg.register(agent)
        agent_id = reg.make_agent_id(agent)
        schema = reg.get_schema(agent_id)
        payload = _payload(agent_id, ("memory.backend", "memory"))
        result = ConfigResolver().apply_overrides(agent, payload, schema=schema)
        assert "memory.backend" in result.accepted
        assert "memory.backend" in result.pending_restart
        reg.unregister(agent_id)

    def test_checkpoint_storage_in_pending_restart(self) -> None:
        """checkpoint.storage override -> applied and in pending_restart (when checkpoint present)."""
        from syrin.checkpoint import CheckpointConfig

        agent = Agent(
            model=Model.Almock(),
            name="cp_agent",
            budget=Budget(max_cost=1.0),
            checkpoint=CheckpointConfig(storage="memory", path=None),
        )
        reg = get_registry()
        reg.register(agent)
        agent_id = reg.make_agent_id(agent)
        schema = reg.get_schema(agent_id)
        payload = _payload(agent_id, ("checkpoint.storage", "sqlite"))
        result = ConfigResolver().apply_overrides(agent, payload, schema=schema)
        assert "checkpoint.storage" in result.accepted
        assert "checkpoint.storage" in result.pending_restart
        reg.unregister(agent_id)


# --- Multiple sections: partial failure ---


class TestPartialFailure:
    """When one section fails validation, others can still be applied."""

    def test_valid_and_invalid_separate_sections(self) -> None:
        """budget.max_cost=2.0 (valid) and memory.top_k=-1 (invalid) -> budget applied, memory rejected."""
        agent = _make_agent()
        reg = get_registry()
        reg.register(agent)
        agent_id = reg.make_agent_id(agent)
        schema = reg.get_schema(agent_id)
        payload = _payload(
            agent_id,
            ("budget.max_cost", 2.0),
            ("memory.top_k", -1),
        )
        result = ConfigResolver().apply_overrides(agent, payload, schema=schema)
        assert "budget.max_cost" in result.accepted
        assert agent._budget is not None
        assert agent._budget.max_cost == 2.0
        assert "memory.top_k" not in result.accepted
        assert any(p == "memory.top_k" for p, _ in result.rejected)
        reg.unregister(agent_id)


# --- Schema from registry when not passed ---


class TestSchemaFromRegistry:
    """When schema is None, resolver can get it from registry by agent_id."""

    def test_apply_with_schema_from_registry(self) -> None:
        """apply_overrides(agent, payload, schema=None) uses registry.get_schema(payload.agent_id)."""
        agent = _make_agent()
        reg = get_registry()
        reg.register(agent)
        agent_id = reg.make_agent_id(agent)
        payload = _payload(agent_id, ("budget.max_cost", 3.0))
        result = ConfigResolver().apply_overrides(agent, payload)
        assert "budget.max_cost" in result.accepted
        assert agent._budget is not None
        assert agent._budget.max_cost == 3.0
        reg.unregister(agent_id)

    def test_apply_without_schema_and_not_registered_uses_extract_schema(self) -> None:
        """When schema=None and agent not in registry, resolver uses extract_agent_schema(agent)."""
        agent = _make_agent()
        reg = get_registry()
        agent_id = reg.make_agent_id(agent)
        # Do not register agent
        payload = _payload(agent_id, ("budget.max_cost", 3.0))
        result = ConfigResolver().apply_overrides(agent, payload)
        assert "budget.max_cost" in result.accepted
        assert agent._budget is not None
        assert agent._budget.max_cost == 3.0


# --- Stress / concurrency ---


class TestStressConcurrency:
    """Concurrent apply_overrides and register do not crash; state remains valid."""

    def test_concurrent_apply_overrides_same_agent(self) -> None:
        """Multiple threads applying overrides to the same agent; no crash, budget stays valid."""
        agent = _make_agent()
        reg = get_registry()
        reg.register(agent)
        agent_id = reg.make_agent_id(agent)
        resolver = ConfigResolver()
        errors: list[Exception] = []
        num_threads = 5
        applies_per_thread = 30

        def apply_loop() -> None:
            for i in range(applies_per_thread):
                try:
                    val = 1.0 + (i % 10) * 0.5
                    payload = _payload(agent_id, ("budget.max_cost", val))
                    resolver.apply_overrides(agent, payload)
                except Exception as e:
                    errors.append(e)

        threads = [threading.Thread(target=apply_loop) for _ in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert not errors, errors
        assert agent._budget is not None
        assert agent._budget.max_cost >= 0
        reg.unregister(agent_id)

    def test_concurrent_apply_overrides_multiple_agents(self) -> None:
        """Multiple threads each applying overrides to different agents; no crash."""
        reg = get_registry()
        resolver = ConfigResolver()
        agents: list[Agent] = []
        agent_ids: list[str] = []
        for i in range(5):
            a = _make_agent(name=f"stress_agent_{i}")
            reg.register(a)
            agents.append(a)
            agent_ids.append(reg.make_agent_id(a))
        errors: list[Exception] = []

        def apply_for_index(idx: int) -> None:
            agent = agents[idx]
            aid = agent_ids[idx]
            for j in range(20):
                try:
                    payload = _payload(aid, ("budget.max_cost", 1.0 + j * 0.1))
                    resolver.apply_overrides(agent, payload)
                except Exception as e:
                    errors.append(e)

        threads = [threading.Thread(target=apply_for_index, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert not errors, errors
        for a in agents:
            assert a._budget is not None
            assert a._budget.max_cost >= 0
        for aid in agent_ids:
            reg.unregister(aid)
