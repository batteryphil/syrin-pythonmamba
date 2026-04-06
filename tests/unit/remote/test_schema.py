"""Tests for remote config schema extraction: extract_schema, extract_agent_schema."""

from __future__ import annotations

import pytest
from pydantic import BaseModel, Field

from syrin.budget import Budget
from syrin.checkpoint import CheckpointConfig
from syrin.context.config import Context
from syrin.memory.config import Decay, Memory
from syrin.output import Output
from syrin.ratelimit.config import APIRateLimit
from syrin.remote._schema import (
    extract_agent_schema,
    extract_dataclass_schema,
    extract_plain_schema,
    extract_pydantic_schema,
    extract_schema,
)

# --- extract_pydantic_schema ---


class TestExtractPydanticSchema:
    """Pydantic model schema extraction."""

    def test_budget_run_field(self) -> None:
        """Budget.max_cost has path budget.max_cost, type float, ge=0 constraint."""
        fields = extract_pydantic_schema(Budget, "budget")
        run = next((f for f in fields if f.name == "max_cost"), None)
        assert run is not None
        assert run.path == "budget.max_cost"
        assert run.type == "float"
        assert run.constraints.get("ge") == 0
        assert run.default is None

    def test_budget_safety_margin_default(self) -> None:
        """Budget.safety_margin has default 0."""
        fields = extract_pydantic_schema(Budget, "budget")
        reserve = next((f for f in fields if f.name == "safety_margin"), None)
        assert reserve is not None
        assert reserve.default == 0
        assert reserve.constraints.get("ge") == 0

    def test_budget_exceed_policy_is_remote_configurable(self) -> None:
        """exceed_policy is a StrEnum field and must appear in the remote schema."""
        fields = extract_pydantic_schema(Budget, "budget")
        exceed_policy = next((f for f in fields if f.name == "exceed_policy"), None)
        assert exceed_policy is not None
        assert exceed_policy.remote_excluded is not True

    def test_budget_per_nested_children(self) -> None:
        """Budget.rate_limits (RateLimit) has children with budget.rate_limits.hour, etc."""
        fields = extract_pydantic_schema(Budget, "budget")
        per = next((f for f in fields if f.name == "rate_limits"), None)
        assert per is not None
        assert per.type == "object"
        assert per.children is not None
        names = {c.name for c in per.children}
        assert "hour" in names
        assert "day" in names
        child_hour = next(c for c in per.children if c.name == "hour")
        assert child_hour.path == "budget.rate_limits.hour"
        assert child_hour.type == "float"

    def test_decay_strategy_enum_values(self) -> None:
        """Decay.strategy is StrEnum; enum_values populated."""
        fields = extract_pydantic_schema(Decay, "memory.decay")
        strategy = next((f for f in fields if f.name == "strategy"), None)
        assert strategy is not None
        assert strategy.type == "str"
        assert strategy.enum_values is not None
        assert "exponential" in strategy.enum_values
        assert "linear" in strategy.enum_values
        assert "none" in strategy.enum_values

    def test_decay_rate_constraints(self) -> None:
        """Decay.rate has gt=0, le=1."""
        fields = extract_pydantic_schema(Decay, "memory.decay")
        rate = next((f for f in fields if f.name == "rate"), None)
        assert rate is not None
        assert rate.constraints.get("gt") == 0.0
        assert rate.constraints.get("le") == 1.0

    def test_memory_top_k_gt_zero(self) -> None:
        """Memory.top_k has gt=0."""
        fields = extract_pydantic_schema(Memory, "memory")
        top_k = next((f for f in fields if f.name == "top_k"), None)
        assert top_k is not None
        assert top_k.constraints.get("gt") == 0

    def test_memory_decay_nested(self) -> None:
        """Memory.decay is nested Decay; has children."""
        fields = extract_pydantic_schema(Memory, "memory")
        decay = next((f for f in fields if f.name == "decay"), None)
        assert decay is not None
        assert decay.type == "object"
        assert decay.children is not None
        strategy_child = next((c for c in decay.children if c.name == "strategy"), None)
        assert strategy_child is not None
        assert strategy_child.enum_values is not None

    def test_checkpoint_config_storage_pattern(self) -> None:
        """CheckpointConfig.storage has pattern constraint."""
        fields = extract_pydantic_schema(CheckpointConfig, "checkpoint")
        storage = next((f for f in fields if f.name == "storage"), None)
        assert storage is not None
        assert "pattern" in storage.constraints or storage.description

    def test_checkpoint_trigger_enum(self) -> None:
        """CheckpointConfig.trigger is StrEnum."""
        fields = extract_pydantic_schema(CheckpointConfig, "checkpoint")
        trigger = next((f for f in fields if f.name == "trigger"), None)
        assert trigger is not None
        assert trigger.enum_values is not None

    def test_prefix_empty_rejected(self) -> None:
        """Empty prefix is invalid."""
        with pytest.raises(ValueError, match="prefix"):
            extract_pydantic_schema(Budget, "")

    def test_description_propagated(self) -> None:
        """Field description from Pydantic Field is set."""
        fields = extract_pydantic_schema(Budget, "budget")
        run = next((f for f in fields if f.name == "max_cost"), None)
        assert run is not None
        assert run.description is not None and "cost" in run.description.lower()


# --- extract_dataclass_schema ---


class TestExtractDataclassSchema:
    """Dataclass schema extraction."""

    def test_context_max_tokens(self) -> None:
        """Context has max_tokens, reserve, etc."""
        fields = extract_dataclass_schema(Context, "context")
        names = [f.name for f in fields]
        assert "max_tokens" in names
        assert "reserve" in names
        max_tokens = next(f for f in fields if f.name == "max_tokens")
        assert max_tokens.path == "context.max_tokens"
        assert max_tokens.type in ("int", "any")

    def test_context_compactor_remote_excluded(self) -> None:
        """Protocol/callable-like fields (compactor) are remote_excluded."""
        fields = extract_dataclass_schema(Context, "context")
        compactor = next((f for f in fields if f.name == "compactor"), None)
        assert compactor is not None
        assert compactor.remote_excluded is True

    def test_output_validation_retries(self) -> None:
        """Output has validation_retries, strict."""
        fields = extract_dataclass_schema(Output, "output")
        retries = next((f for f in fields if f.name == "validation_retries"), None)
        assert retries is not None
        assert retries.type == "int"
        assert retries.default == 3
        strict = next((f for f in fields if f.name == "strict"), None)
        assert strict is not None
        assert strict.default is False

    def test_output_type_and_validator_excluded(self) -> None:
        """Output.type and Output.validator are remote_excluded (type ref, Protocol)."""
        fields = extract_dataclass_schema(Output, "output")
        type_f = next((f for f in fields if f.name == "type"), None)
        assert type_f is not None
        assert type_f.remote_excluded is True
        validator_f = next((f for f in fields if f.name == "validator"), None)
        assert validator_f is not None
        assert validator_f.remote_excluded is True

    def test_api_rate_limit_rpm_tpm(self) -> None:
        """APIRateLimit has rpm, tpm, rpd."""
        fields = extract_dataclass_schema(APIRateLimit, "rate_limit")
        rpm = next((f for f in fields if f.name == "rpm"), None)
        assert rpm is not None
        assert rpm.path == "rate_limit.rpm"
        private = next((f for f in fields if f.name.startswith("_")), None)
        if private:
            assert private.remote_excluded is True

    def test_prefix_required(self) -> None:
        """Empty prefix raises."""
        with pytest.raises(ValueError, match="prefix"):
            extract_dataclass_schema(Context, "")


# --- extract_plain_schema ---


class TestExtractPlainSchema:
    """Plain class (inspect.signature) schema extraction."""

    def test_circuit_breaker_params(self) -> None:
        """CircuitBreaker has failure_threshold, recovery_timeout, half_open_max."""
        from syrin.circuit import CircuitBreaker

        fields = extract_plain_schema(CircuitBreaker, "circuit_breaker")
        names = [f.name for f in fields]
        assert "failure_threshold" in names
        assert "recovery_timeout" in names
        assert "half_open_max" in names
        ft = next(f for f in fields if f.name == "failure_threshold")
        assert ft.path == "circuit_breaker.failure_threshold"
        assert ft.type == "int"
        assert ft.default == 5

    def test_circuit_breaker_callable_excluded(self) -> None:
        """CircuitBreaker.on_trip (callable) is remote_excluded."""
        from syrin.circuit import CircuitBreaker

        fields = extract_plain_schema(CircuitBreaker, "circuit_breaker")
        on_trip = next((f for f in fields if f.name == "on_trip"), None)
        assert on_trip is not None
        assert on_trip.remote_excluded is True

    def test_prefix_required(self) -> None:
        """Empty prefix raises."""
        from syrin.circuit import CircuitBreaker

        with pytest.raises(ValueError, match="prefix"):
            extract_plain_schema(CircuitBreaker, "")


# --- extract_schema dispatch ---


class TestExtractSchemaDispatch:
    """extract_schema auto-detects Pydantic vs dataclass vs plain."""

    def test_pydantic_dispatched(self) -> None:
        """Budget is Pydantic; extract_schema returns same as extract_pydantic_schema."""
        from_schema = extract_schema(Budget, "budget")
        from_pydantic = extract_pydantic_schema(Budget, "budget")
        assert len(from_schema) == len(from_pydantic)
        assert {f.path for f in from_schema} == {f.path for f in from_pydantic}

    def test_dataclass_dispatched(self) -> None:
        """Context is dataclass; extract_schema returns same as extract_dataclass_schema."""
        from_schema = extract_schema(Context, "context")
        from_dc = extract_dataclass_schema(Context, "context")
        assert len(from_schema) == len(from_dc)

    def test_plain_dispatched(self) -> None:
        """CircuitBreaker is plain; extract_schema returns same as extract_plain_schema."""
        from syrin.circuit import CircuitBreaker

        from_schema = extract_schema(CircuitBreaker, "circuit_breaker")
        from_plain = extract_plain_schema(CircuitBreaker, "circuit_breaker")
        assert len(from_schema) == len(from_plain)

    def test_empty_prefix_rejected(self) -> None:
        """extract_schema rejects empty prefix."""
        with pytest.raises(ValueError, match="prefix"):
            extract_schema(Budget, "")


# --- extract_agent_schema ---


class TestExtractAgentSchema:
    """Full agent schema extraction."""

    def test_agent_schema_has_agent_section(self) -> None:
        """extract_agent_schema includes agent section with max_tool_iterations, debug, etc."""
        from syrin import Agent, Model

        agent = Agent(
            model=Model.Almock(),
            system_prompt="Hi",
            max_tool_iterations=5,
            debug=True,
            human_approval_timeout=60,
        )
        schema = extract_agent_schema(agent)
        assert "agent" in schema.sections
        agent_section = schema.sections["agent"]
        assert agent_section.class_name == "Agent"
        names = [f.name for f in agent_section.fields]
        assert "max_tool_iterations" in names
        assert "debug" in names
        assert "system_prompt" in names
        assert "human_approval_timeout" in names
        assert "loop_strategy" not in names  # loop_strategy removed; use loop= directly

    def test_agent_schema_budget_section_when_budget_set(self) -> None:
        """When agent has budget, budget section present with run field."""
        from syrin import Agent, Budget, Model

        agent = Agent(
            model=Model.Almock(),
            budget=Budget(max_cost=1.0),
        )
        schema = extract_agent_schema(agent)
        assert "budget" in schema.sections
        budget_section = schema.sections["budget"]
        assert budget_section.class_name == "Budget"
        run = next((f for f in budget_section.fields if f.name == "max_cost"), None)
        assert run is not None
        assert schema.current_values.get("budget.max_cost") == 1.0

    def test_agent_schema_memory_section_when_memory_set(self) -> None:
        """When agent has memory, memory section present."""
        from syrin import Agent, Memory, Model
        from syrin.enums import MemoryType

        agent = Agent(
            model=Model.Almock(),
            memory=Memory(types=[MemoryType.FACTS], top_k=5),
        )
        schema = extract_agent_schema(agent)
        assert "memory" in schema.sections
        assert schema.current_values.get("memory.top_k") == 5

    def test_agent_schema_agent_id_and_class_name(self) -> None:
        """AgentSchema has agent_id, agent_name, class_name."""
        from syrin import Agent, Model

        agent = Agent(model=Model.Almock(), name="my_agent")
        schema = extract_agent_schema(agent)
        assert schema.agent_id != ""
        assert "my_agent" in schema.agent_id or "Agent" in schema.agent_id
        assert schema.class_name == "Agent"

    def test_agent_schema_sections_empty_when_minimal_agent(self) -> None:
        """Minimal agent still has agent section; other sections only if config present."""
        from syrin import Agent, Model

        agent = Agent(model=Model.Almock())
        schema = extract_agent_schema(agent)
        assert "agent" in schema.sections
        # budget/memory etc. may be absent or present with empty current_values
        assert schema.sections["agent"].section == "agent"

    def test_agent_schema_model_section_when_model_set(self) -> None:
        """When agent has model, model section present with model_id, temperature, max_tokens."""
        from syrin import Agent, Model

        agent = Agent(model=Model.Ollama("llama3"))
        schema = extract_agent_schema(agent)
        assert "model" in schema.sections
        model_section = schema.sections["model"]
        assert model_section.class_name == "Model"
        names = [f.name for f in model_section.fields]
        assert "model_id" in names
        assert "temperature" in names
        assert "max_tokens" in names
        assert "ollama/llama3" in str(schema.current_values.get("model.model_id", ""))

    def test_agent_schema_model_section_absent_when_no_model(self) -> None:
        """Agent with model_config only (no Model instance) has no model section."""
        from syrin import Agent
        from syrin.types import ModelConfig

        agent = Agent(
            model=ModelConfig(
                name="gpt",
                provider="openai",
                model_id="openai/gpt-4o-mini",
            ),
            system_prompt="",
        )
        schema = extract_agent_schema(agent)
        assert agent._model is None
        assert "model" not in schema.sections

    def test_agent_schema_knowledge_section_when_knowledge_set(self) -> None:
        """When agent has knowledge, knowledge section present with top_k, score_threshold."""

        def _fake_embedding():
            class F:
                dimensions = 4
                model_id = "fake"

                async def embed(self, texts, budget_tracker=None):
                    return [[0.1] * 4 for _ in texts]

            return F()

        from syrin import Agent, Model
        from syrin.enums import KnowledgeBackend
        from syrin.knowledge import Knowledge

        agent = Agent(
            model=Model.Almock(),
            knowledge=Knowledge(
                sources=[Knowledge.Text("test")],
                embedding=_fake_embedding(),
                backend=KnowledgeBackend.MEMORY,
                top_k=7,
                score_threshold=0.5,
            ),
        )
        schema = extract_agent_schema(agent)
        assert "knowledge" in schema.sections
        knowledge_section = schema.sections["knowledge"]
        assert knowledge_section.class_name == "Knowledge"
        names = [f.name for f in knowledge_section.fields]
        assert "top_k" in names
        assert "score_threshold" in names
        assert "grounding" in names
        assert "chunk_config" in names
        assert schema.current_values.get("knowledge.top_k") == 7
        assert schema.current_values.get("knowledge.score_threshold") == 0.5

    def test_agent_schema_knowledge_section_absent_when_no_knowledge(self) -> None:
        """Agent without knowledge has no knowledge section."""
        from syrin import Agent, Model

        agent = Agent(model=Model.Almock())
        schema = extract_agent_schema(agent)
        assert "knowledge" not in schema.sections


# --- Edge cases ---


class TestSchemaEdgeCases:
    """Edge cases: recursion limit, optional nested, list/dict types."""

    def test_nested_optional_decay(self) -> None:
        """Memory.decay is Optional[Decay]; children still extracted."""
        fields = extract_pydantic_schema(Memory, "memory")
        decay = next((f for f in fields if f.name == "decay"), None)
        assert decay is not None
        assert decay.children is not None

    def test_list_and_dict_types_no_children(self) -> None:
        """List/dict fields get type list/dict without recursing into element type."""
        fields = extract_pydantic_schema(Budget, "budget")
        thresholds = next((f for f in fields if f.name == "thresholds"), None)
        assert thresholds is not None
        assert thresholds.type == "list"
        assert thresholds.children is None

    def test_private_attr_not_in_pydantic_fields(self) -> None:
        """Pydantic PrivateAttr fields are not in model_fields; not extracted."""
        fields = extract_pydantic_schema(Budget, "budget")
        names = [f.name for f in fields]
        assert "_parent_budget" not in names
        assert "_spent" not in names


# --- Custom small models for isolated tests ---


class _SmallPydantic(BaseModel):
    """Minimal Pydantic model for tests."""

    x: int = Field(ge=0, le=10, description="x")
    tag: str = "default"


def test_small_pydantic_extraction() -> None:
    """Small Pydantic model: x has constraints, tag has default."""
    fields = extract_schema(_SmallPydantic, "small")
    assert len(fields) >= 2
    x = next(f for f in fields if f.name == "x")
    assert x.path == "small.x"
    assert x.type == "int"
    assert x.constraints.get("ge") == 0
    assert x.constraints.get("le") == 10
    tag = next(f for f in fields if f.name == "tag")
    assert tag.default == "default"
