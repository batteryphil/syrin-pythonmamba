"""Tests for remote config resolver: ConfigResolver, apply_overrides, ResolveResult."""

from __future__ import annotations

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

    def test_apply_loop_strategy_rejected_since_removed(self) -> None:
        """loop_strategy is removed; applying it should not be accepted."""
        agent = _make_agent()
        reg = get_registry()
        reg.register(agent)
        agent_id = reg.make_agent_id(agent)
        schema = reg.get_schema(agent_id)
        assert schema is not None
        payload = _payload(agent_id, ("agent.loop_strategy", "single_shot"))
        result = ConfigResolver().apply_overrides(agent, payload, schema=schema)
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

    def test_apply_guardrails_disable(self) -> None:
        """Apply guardrails.{name}.enabled=false -> guardrail disabled; true -> enabled."""
        from syrin.guardrails.built_in import PIIScanner

        agent = Agent(
            model=Model.Almock(),
            name="gr_test",
            budget=Budget(max_cost=1.0),
            guardrails=[PIIScanner()],
        )
        reg = get_registry()
        reg.register(agent)
        agent_id = reg.make_agent_id(agent)
        schema = reg.get_schema(agent_id)
        assert schema is not None
        assert "guardrails" in schema.sections
        payload = _payload(agent_id, ("guardrails.PIIScanner.enabled", False))
        result = ConfigResolver().apply_overrides(agent, payload, schema=schema)
        assert "guardrails.PIIScanner.enabled" in result.accepted
        assert "PIIScanner" in agent._guardrails_disabled
        payload2 = _payload(agent_id, ("guardrails.PIIScanner.enabled", True))
        result2 = ConfigResolver().apply_overrides(agent, payload2, schema=schema)
        assert "guardrails.PIIScanner.enabled" in result2.accepted
        assert "PIIScanner" not in agent._guardrails_disabled
        reg.unregister(agent_id)

    def test_apply_template_vars(self) -> None:
        """Apply template_variables.{key}=value -> agent._template_vars updated."""
        agent = Agent(
            model=Model.Almock(),
            name="pv_test",
            budget=Budget(max_cost=1.0),
            template_variables={"env": "staging", "limit": "10"},
        )
        reg = get_registry()
        reg.register(agent)
        agent_id = reg.make_agent_id(agent)
        schema = reg.get_schema(agent_id)
        assert schema is not None
        assert "template_variables" in schema.sections
        payload = _payload(
            agent_id,
            ("template_variables.env", "prod"),
            ("template_variables.limit", "20"),
        )
        result = ConfigResolver().apply_overrides(agent, payload, schema=schema)
        assert "template_variables.env" in result.accepted
        assert "template_variables.limit" in result.accepted
        assert agent._template_vars.get("env") == "prod"
        assert agent._template_vars.get("limit") == "20"
        reg.unregister(agent_id)

    def test_apply_tools_disable(self) -> None:
        """Apply tools.{name}.enabled=false -> tool excluded from agent.tools."""
        from syrin.tool import tool

        @tool
        def alpha() -> str:
            return "a"

        @tool
        def beta() -> str:
            return "b"

        agent = Agent(
            model=Model.Almock(),
            name="tools_test",
            budget=Budget(max_cost=1.0),
            tools=[alpha, beta],
        )
        reg = get_registry()
        reg.register(agent)
        agent_id = reg.make_agent_id(agent)
        schema = reg.get_schema(agent_id)
        assert schema is not None
        assert "tools" in schema.sections
        assert len(agent.tools) == 2
        payload = _payload(agent_id, ("tools.alpha.enabled", False))
        result = ConfigResolver().apply_overrides(agent, payload, schema=schema)
        assert "tools.alpha.enabled" in result.accepted
        assert "alpha" in agent._tools_disabled
        assert len(agent.tools) == 1
        assert agent.tools[0].name == "beta"
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
