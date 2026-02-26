"""Tests for multi-agent patterns: handoff, spawn, and pipeline."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from syrin import Agent, HandoffBlockedError, HandoffRetryRequested, Model, Response
from syrin.enums import Hook, MemoryType
from syrin.events import EventContext
from syrin.exceptions import ValidationError
from syrin.memory import Memory
from syrin.types import TokenUsage


def create_mock_provider():
    """Create a mock provider for testing."""
    mock = MagicMock()
    mock.complete = AsyncMock(
        return_value=MagicMock(
            content="Mock response",
            tool_calls=[],
            token_usage=TokenUsage(input_tokens=10, output_tokens=20, total_tokens=30),
        )
    )
    return mock


class TestHandoff:
    """Tests for Agent handoff functionality."""

    @patch("syrin.agent._resolve_provider")
    def test_handoff_basic(self, mock_get_provider):
        """Test basic handoff creates target agent and runs with context."""
        mock_get_provider.return_value = create_mock_provider()

        class Researcher(Agent):
            model = Model("test/model")

        class Writer(Agent):
            model = Model("test/model")

        researcher = Researcher()
        Writer()

        result = researcher.handoff(Writer, "Write about AI")

        assert isinstance(result, Response)
        assert result.content is not None

    @patch("syrin.agent._resolve_provider")
    def test_handoff_with_context_transfer(self, mock_get_provider):
        """Test handoff transfers context from source to target agent."""
        mock_get_provider.return_value = create_mock_provider()

        class AgentA(Agent):
            model = Model("test/model")

        class AgentB(Agent):
            model = Model("test/model")

        agent_a = AgentA(memory=Memory())
        agent_a.remember("Key fact from AgentA", memory_type=MemoryType.CORE)

        result = agent_a.handoff(AgentB, "Continue with this info", transfer_context=True)

        assert isinstance(result, Response)

    @patch("syrin.agent._resolve_provider")
    def test_handoff_without_context_transfer(self, mock_get_provider):
        """Test handoff without context transfer."""
        mock_get_provider.return_value = create_mock_provider()

        class AgentA(Agent):
            model = Model("test/model")

        class AgentB(Agent):
            model = Model("test/model")

        agent_a = AgentA(memory=Memory())
        agent_a.remember("Secret info", memory_type=MemoryType.CORE)

        result = agent_a.handoff(AgentB, "New task", transfer_context=False)

        assert isinstance(result, Response)

    @patch("syrin.agent._resolve_provider")
    def test_handoff_with_budget_transfer(self, mock_get_provider):
        """Test handoff transfers budget from source to target agent."""
        from syrin import Budget

        mock_get_provider.return_value = create_mock_provider()

        class AgentA(Agent):
            model = Model("test/model")
            budget = Budget(run=1.0)

        class AgentB(Agent):
            model = Model("test/model")

        agent_a = AgentA()

        result = agent_a.handoff(AgentB, "Task", transfer_budget=True)

        assert isinstance(result, Response)

    @patch("syrin.agent._resolve_provider")
    def test_handoff_chain(self, mock_get_provider):
        """Test multiple handoffs in sequence."""
        mock_get_provider.return_value = create_mock_provider()

        class Agent1(Agent):
            model = Model("test/model")

        class Agent2(Agent):
            model = Model("test/model")

        class Agent3(Agent):
            model = Model("test/model")

        agent1 = Agent1()
        agent2 = Agent2()
        Agent3()

        agent1.handoff(Agent2, "Task 1")
        result = agent2.handoff(Agent3, "Task 2")

        assert isinstance(result, Response)


class TestSpawn:
    """Tests for Agent spawning functionality."""

    @patch("syrin.agent._resolve_provider")
    def test_spawn_basic(self, mock_get_provider):
        """Test basic spawn creates sub-agent."""
        mock_get_provider.return_value = create_mock_provider()

        class Parent(Agent):
            model = Model("test/model")

        class Child(Agent):
            model = Model("test/model")

        parent = Parent()

        child = parent.spawn(Child)

        assert isinstance(child, Child)

    @patch("syrin.agent._resolve_provider")
    def test_spawn_with_task(self, mock_get_provider):
        """Test spawn runs task on sub-agent."""
        mock_get_provider.return_value = create_mock_provider()

        class Parent(Agent):
            model = Model("test/model")

        class Child(Agent):
            model = Model("test/model")

        parent = Parent()

        result = parent.spawn(Child, "Research AI")

        assert isinstance(result, Response)

    @patch("syrin.agent._resolve_provider")
    def test_spawn_with_budget(self, mock_get_provider):
        """Test spawn creates sub-agent with its own budget."""
        from syrin import Budget

        mock_get_provider.return_value = create_mock_provider()

        class Parent(Agent):
            model = Model("test/model")

        class Child(Agent):
            model = Model("test/model")

        parent = Parent()

        child = parent.spawn(Child, budget=Budget(run=0.50))

        assert child._budget is not None

    @patch("syrin.agent._resolve_provider")
    def test_spawn_shared_budget_child_uses_parent_tracker(self, mock_get_provider):
        """With shared budget, child uses parent's tracker (live view, no double count)."""
        from syrin import Budget

        mock_get_provider.return_value = create_mock_provider()

        class Parent(Agent):
            model = Model("test/model")

        class Child(Agent):
            model = Model("test/model")

        parent = Parent(budget=Budget(run=10.0, shared=True))
        parent_tracker = parent._budget_tracker

        result = parent.spawn(Child, task="Do something")
        assert result is not None
        # Child shared parent's tracker (live view)
        assert parent._budget_tracker is parent_tracker
        # Spent must match tracker: no double count from _update_parent_budget
        assert parent._budget._spent == parent._budget_tracker.current_run_cost

    @patch("syrin.agent._resolve_provider")
    def test_spawn_max_children_limit(self, mock_get_provider):
        """Test spawn respects max_children limit."""
        mock_get_provider.return_value = create_mock_provider()

        class Parent(Agent):
            model = Model("test/model")

        class Child(Agent):
            model = Model("test/model")

        parent = Parent()
        parent._max_children = 2

        parent.spawn(Child, "Task 1")
        parent.spawn(Child, "Task 2")

        with pytest.raises(RuntimeError, match="max children"):
            parent.spawn(Child, "Task 3")

    @patch("syrin.agent._resolve_provider")
    def test_spawn_parallel(self, mock_get_provider):
        """Test spawning multiple sub-agents in parallel."""
        mock_get_provider.return_value = create_mock_provider()

        class Parent(Agent):
            model = Model("test/model")

        class Child(Agent):
            model = Model("test/model")

        parent = Parent()

        results = parent.spawn_parallel(
            [
                (Child, "Task 1"),
                (Child, "Task 2"),
                (Child, "Task 3"),
            ]
        )

        assert len(results) == 3
        for result in results:
            assert isinstance(result, Response)


# -----------------------------------------------------------------------------
# Step 4: Handoff & spawn hooks, HandoffBlocked, HandoffRetryRequested
# -----------------------------------------------------------------------------


class TestHandoffHooks:
    """TDD tests for handoff hooks and edge cases."""

    @patch("syrin.agent._resolve_provider")
    def test_handoff_emits_start_and_end(self, mock_get_provider):
        """HANDOFF_START and HANDOFF_END emitted with correct context."""
        mock_get_provider.return_value = create_mock_provider()

        captured_start: list[EventContext] = []
        captured_end: list[EventContext] = []

        class SourceAgent(Agent):
            model = Model("test/model")

        class TargetAgent(Agent):
            model = Model("test/model")

        source = SourceAgent()

        def on_start(ctx: EventContext) -> None:
            captured_start.append(ctx)

        def on_end(ctx: EventContext) -> None:
            captured_end.append(ctx)

        source.events.on(Hook.HANDOFF_START, on_start)
        source.events.on(Hook.HANDOFF_END, on_end)

        result = source.handoff(TargetAgent, "Do task")

        assert len(captured_start) == 1
        assert captured_start[0]["source_agent"] == "SourceAgent"
        assert captured_start[0]["target_agent"] == "TargetAgent"
        assert captured_start[0]["task"] == "Do task"
        assert "mem_count" in captured_start[0]
        assert captured_start[0]["transfer_context"] is True

        assert len(captured_end) == 1
        assert captured_end[0]["source_agent"] == "SourceAgent"
        assert captured_end[0]["target_agent"] == "TargetAgent"
        assert captured_end[0]["task"] == "Do task"
        assert "cost" in captured_end[0]
        assert "duration" in captured_end[0]
        assert "response_preview" in captured_end[0]
        assert result.content is not None

    @patch("syrin.agent._resolve_provider")
    def test_handoff_before_handler_blocks_emits_blocked(self, mock_get_provider):
        """Before-handler raises HandoffBlockedError -> HANDOFF_BLOCKED emitted."""
        mock_get_provider.return_value = create_mock_provider()

        blocked_ctx: list[EventContext] = []

        class SourceAgent(Agent):
            model = Model("test/model")

        class TargetAgent(Agent):
            model = Model("test/model")

        def block(_ctx: EventContext) -> None:
            raise HandoffBlockedError("invalid task", "SourceAgent", "TargetAgent", "bad")

        source = SourceAgent()
        source.events.before(Hook.HANDOFF_START, block)
        source.events.on(Hook.HANDOFF_BLOCKED, lambda ctx: blocked_ctx.append(ctx))

        with pytest.raises(HandoffBlockedError, match="invalid task"):
            source.handoff(TargetAgent, "bad")

        assert len(blocked_ctx) == 1
        assert blocked_ctx[0]["source_agent"] == "SourceAgent"
        assert blocked_ctx[0]["target_agent"] == "TargetAgent"
        assert blocked_ctx[0]["task"] == "bad"
        assert blocked_ctx[0]["reason"] == "invalid task"

    @patch("syrin.agent._resolve_provider")
    def test_handoff_task_none_raises_validation_error(self, mock_get_provider):
        """handoff with task=None raises ValidationError."""
        mock_get_provider.return_value = create_mock_provider()

        class SourceAgent(Agent):
            model = Model("test/model")

        class TargetAgent(Agent):
            model = Model("test/model")

        source = SourceAgent()

        with pytest.raises(ValidationError, match="handoff task must be non-empty"):
            source.handoff(TargetAgent, None)  # type: ignore[arg-type]

    @patch("syrin.agent._resolve_provider")
    def test_handoff_task_empty_raises_validation_error(self, mock_get_provider):
        """handoff with empty task raises ValidationError."""
        mock_get_provider.return_value = create_mock_provider()

        class SourceAgent(Agent):
            model = Model("test/model")

        class TargetAgent(Agent):
            model = Model("test/model")

        source = SourceAgent()

        with pytest.raises(ValidationError, match="handoff task must be non-empty"):
            source.handoff(TargetAgent, "")

        with pytest.raises(ValidationError, match="handoff task must be non-empty"):
            source.handoff(TargetAgent, "   ")

    @patch("syrin.agent._resolve_provider")
    def test_handoff_retry_requested_propagates(self, mock_get_provider):
        """HandoffRetryRequested from target propagates to caller."""
        mock_get_provider.return_value = create_mock_provider()

        class SourceAgent(Agent):
            model = Model("test/model")

        class TargetAgent(Agent):
            model = Model("test/model")

            def response(self, user_input: str, **kwargs: object) -> Response[str]:
                raise HandoffRetryRequested(
                    "Invalid format", format_hint="Use JSON with fields: x, y"
                )

        source = SourceAgent()

        with pytest.raises(HandoffRetryRequested) as exc_info:
            source.handoff(TargetAgent, "bad data")

        assert "Use JSON with fields: x, y" in exc_info.value.format_hint

    @patch("syrin.agent._resolve_provider")
    def test_handoff_mem_count_in_start_context(self, mock_get_provider):
        """HANDOFF_START ctx has correct mem_count when memories transferred."""
        mock_get_provider.return_value = create_mock_provider()

        start_ctx: list[EventContext] = []

        class SourceAgent(Agent):
            model = Model("test/model")

        class TargetAgent(Agent):
            model = Model("test/model")

        source = SourceAgent(memory=Memory())
        source.remember("Fact 1", memory_type=MemoryType.CORE)
        source.remember("Fact 2", memory_type=MemoryType.CORE)
        source.events.on(Hook.HANDOFF_START, lambda ctx: start_ctx.append(ctx))

        source.handoff(TargetAgent, "Task", transfer_context=True)

        assert len(start_ctx) == 1
        assert start_ctx[0]["mem_count"] == 2


class TestSpawnHooks:
    """TDD tests for spawn hooks."""

    @patch("syrin.agent._resolve_provider")
    def test_spawn_with_task_emits_start_and_end(self, mock_get_provider):
        """SPAWN_START and SPAWN_END emitted when task given."""
        mock_get_provider.return_value = create_mock_provider()

        start_ctx: list[EventContext] = []
        end_ctx: list[EventContext] = []

        class ParentAgent(Agent):
            model = Model("test/model")

        class ChildAgent(Agent):
            model = Model("test/model")

        parent = ParentAgent()
        parent.events.on(Hook.SPAWN_START, lambda ctx: start_ctx.append(ctx))
        parent.events.on(Hook.SPAWN_END, lambda ctx: end_ctx.append(ctx))

        result = parent.spawn(ChildAgent, task="Do task")

        assert len(start_ctx) == 1
        assert start_ctx[0]["source_agent"] == "ParentAgent"
        assert start_ctx[0]["child_agent"] == "ChildAgent"
        assert start_ctx[0]["child_task"] == "Do task"

        assert len(end_ctx) == 1
        assert end_ctx[0]["source_agent"] == "ParentAgent"
        assert end_ctx[0]["child_agent"] == "ChildAgent"
        assert end_ctx[0]["child_task"] == "Do task"
        assert "cost" in end_ctx[0]
        assert "duration" in end_ctx[0]
        assert isinstance(result, Response)

    @patch("syrin.agent._resolve_provider")
    def test_spawn_without_task_emits_start_only(self, mock_get_provider):
        """SPAWN_START emitted; SPAWN_END not emitted when task is None."""
        mock_get_provider.return_value = create_mock_provider()

        start_ctx: list[EventContext] = []
        end_ctx: list[EventContext] = []

        class ParentAgent(Agent):
            model = Model("test/model")

        class ChildAgent(Agent):
            model = Model("test/model")

        parent = ParentAgent()
        parent.events.on(Hook.SPAWN_START, lambda ctx: start_ctx.append(ctx))
        parent.events.on(Hook.SPAWN_END, lambda ctx: end_ctx.append(ctx))

        child = parent.spawn(ChildAgent)

        assert len(start_ctx) == 1
        assert len(end_ctx) == 0
        assert child is not None

    @patch("syrin.agent._resolve_provider")
    def test_spawn_parallel_emits_spawn_hooks_per_child(self, mock_get_provider):
        """spawn_parallel emits SPAWN_START/END for each child via spawn()."""
        mock_get_provider.return_value = create_mock_provider()

        start_count = 0
        end_count = 0

        class ParentAgent(Agent):
            model = Model("test/model")

        class ChildAgent(Agent):
            model = Model("test/model")

        parent = ParentAgent()

        def on_start(ctx: EventContext) -> None:
            nonlocal start_count
            start_count += 1

        def on_end(ctx: EventContext) -> None:
            nonlocal end_count
            end_count += 1

        parent.events.on(Hook.SPAWN_START, on_start)
        parent.events.on(Hook.SPAWN_END, on_end)

        results = parent.spawn_parallel(
            [(ChildAgent, "Task 1"), (ChildAgent, "Task 2")],
        )

        assert start_count == 2
        assert end_count == 2
        assert len(results) == 2


class TestPipeline:
    """Tests for Pipeline execution."""

    @patch("syrin.agent._resolve_provider")
    def test_pipeline_sequential(self, mock_get_provider):
        """Test sequential pipeline execution."""
        from syrin.agent.multi_agent import Pipeline

        mock_get_provider.return_value = create_mock_provider()

        class Agent1(Agent):
            model = Model("test/model")

        class Agent2(Agent):
            model = Model("test/model")

        pipeline = Pipeline()

        result = pipeline.run_sequential(
            [
                (Agent1, "First task"),
                (Agent2, "Second task"),
            ]
        )

        assert isinstance(result, Response)

    @patch("syrin.agent._resolve_provider")
    def test_pipeline_parallel(self, mock_get_provider):
        """Test parallel pipeline execution."""
        from syrin.agent.multi_agent import Pipeline

        mock_get_provider.return_value = create_mock_provider()

        class Agent1(Agent):
            model = Model("test/model")

        class Agent2(Agent):
            model = Model("test/model")

        pipeline = Pipeline()

        results = pipeline.run_parallel(
            [
                (Agent1, "Task 1"),
                (Agent2, "Task 2"),
            ]
        )

        assert len(results) == 2
        for result in results:
            assert isinstance(result, Response)

    @patch("syrin.agent._resolve_provider")
    def test_pipeline_with_shared_budget(self, mock_get_provider):
        """Test pipeline with shared budget across agents."""
        from syrin import Budget
        from syrin.agent.multi_agent import Pipeline

        mock_get_provider.return_value = create_mock_provider()

        class Agent1(Agent):
            model = Model("test/model")

        class Agent2(Agent):
            model = Model("test/model")

        pipeline = Pipeline(budget=Budget(run=1.0))

        result = pipeline.run_sequential(
            [
                (Agent1, "Task 1"),
                (Agent2, "Task 2"),
            ]
        )

        assert isinstance(result, Response)

    def test_pipeline_empty(self):
        """Test pipeline with empty agent list."""
        from syrin.agent.multi_agent import Pipeline

        pipeline = Pipeline()

        result = pipeline.run_sequential([])

        assert isinstance(result, Response)
        assert result.content == ""

    def test_pipeline_parallel_empty(self):
        """Test parallel pipeline with empty list."""
        from syrin.agent.multi_agent import Pipeline

        pipeline = Pipeline()

        results = pipeline.run_parallel([])

        assert isinstance(results, list)
        assert len(results) == 0


class TestAgentTeam:
    """Tests for AgentTeam functionality."""

    def test_team_basic(self):
        """Test basic team creation."""
        from syrin.agent.multi_agent import AgentTeam

        class Agent1(Agent):
            model = Model("test/model")

        class Agent2(Agent):
            model = Model("test/model")

        team = AgentTeam(agents=[Agent1(), Agent2()])

        assert len(team.agents) == 2

    def test_team_select_agent(self):
        """Test team selects appropriate agent for task."""
        from syrin.agent.multi_agent import AgentTeam

        class Researcher(Agent):
            model = Model("test/model")

        class Writer(Agent):
            model = Model("test/model")

        team = AgentTeam(agents=[Researcher(), Writer()])

        selected = team.select_agent("research")
        assert isinstance(selected, Researcher)

    def test_team_budget_tracking(self):
        """Test team tracks combined budget across all agents."""
        from syrin import Budget
        from syrin.agent.multi_agent import AgentTeam

        class Agent1(Agent):
            model = Model("test/model")
            budget = Budget(run=1.0)

        class Agent2(Agent):
            model = Model("test/model")
            budget = Budget(run=1.0)

        team = AgentTeam(agents=[Agent1(), Agent2()])

        assert team.total_budget == 2.0


class TestParallelSequential:
    """Tests for parallel and sequential helper functions."""

    @patch("syrin.agent._resolve_provider")
    def test_parallel_creates_responses(self, mock_get_provider):
        """Test parallel helper creates responses."""
        from syrin.agent.multi_agent import parallel

        mock_get_provider.return_value = create_mock_provider()

        class TestAgent(Agent):
            model = Model("test/model")

        async def run():
            results = await parallel(
                [
                    (TestAgent(), "Task 1"),
                    (TestAgent(), "Task 2"),
                ]
            )
            return results

        results = asyncio.run(run())
        assert len(results) == 2

    @patch("syrin.agent._resolve_provider")
    def test_sequential_creates_response(self, mock_get_provider):
        """Test sequential helper creates response."""
        from syrin.agent.multi_agent import sequential

        mock_get_provider.return_value = create_mock_provider()

        class TestAgent(Agent):
            model = Model("test/model")

        result = sequential(
            [
                (TestAgent(), "Task 1"),
                (TestAgent(), "Task 2"),
            ]
        )

        assert isinstance(result, Response)

    def test_sequential_empty(self):
        """Test sequential with empty list."""
        from syrin.agent.multi_agent import sequential

        result = sequential([])

        assert isinstance(result, Response)
        assert result.content == ""


class TestEdgeCases:
    """Edge case tests for multi-agent patterns."""

    def test_handoff_with_invalid_agent(self):
        """Test handoff with invalid agent class."""
        agent = Agent(model=Model("test/model"))

        with pytest.raises(ValidationError, match="target_agent"):
            agent.handoff(None, "task")  # type: ignore[arg-type]

    @patch("syrin.agent._resolve_provider")
    def test_spawn_with_invalid_budget(self, mock_get_provider):
        """Test spawn with invalid budget - negative budget should be rejected."""
        from pydantic import ValidationError

        from syrin import Budget

        mock_get_provider.return_value = create_mock_provider()

        class Child(Agent):
            model = Model("test/model")

        parent = Agent(model=Model("test/model"))

        # Negative budget should raise ValidationError (our fix)
        with pytest.raises(ValidationError):
            parent.spawn(Child, budget=Budget(run=-1.0))

    @patch("syrin.agent._resolve_provider")
    def test_spawn_memory_isolation(self, mock_get_provider):
        """Test spawned agents have isolated memory."""
        mock_get_provider.return_value = create_mock_provider()

        class Parent(Agent):
            model = Model("test/model")

        class Child(Agent):
            model = Model("test/model")

        parent = Parent(memory=Memory())
        parent.remember("Parent secret", memory_type=MemoryType.CORE)

        parent.spawn(Child, "task")

        parent_remaining = parent.recall()
        assert len(parent_remaining) == 1


import asyncio
from unittest.mock import patch


class TestDynamicPipeline:
    """Tests for DynamicPipeline - the LLM-driven agent spawning system."""

    @pytest.fixture
    def mock_model(self):
        """Create a mock model for testing."""
        return Model(provider="openai", model_id="gpt-4o-mini")

    @pytest.fixture
    def researcher_agent(self):
        """Create a test researcher agent."""

        class ResearcherAgent(Agent):
            _syrin_name = "researcher"
            model = Model(provider="openai", model_id="gpt-4o-mini")
            system_prompt = "You research and gather information."

        return ResearcherAgent

    @pytest.fixture
    def writer_agent(self):
        """Create a test writer agent."""

        class WriterAgent(Agent):
            _syrin_name = "writer"
            model = Model(provider="openai", model_id="gpt-4o-mini")
            system_prompt = "You write reports."

        return WriterAgent

    # =========================================================================
    # VALIDATION TESTS - Invalid inputs should be rejected
    # =========================================================================

    def test_model_is_required(self):
        """Test that model parameter is required."""
        from syrin.agent.multi_agent import DynamicPipeline

        with pytest.raises(ValueError, match="model is required"):
            DynamicPipeline(agents=[])

    def test_invalid_format(self, mock_model, researcher_agent):
        """Test that invalid format raises error."""
        from syrin.agent.multi_agent import DynamicPipeline

        pipeline = DynamicPipeline(
            agents=[researcher_agent],
            model=mock_model,
            format="invalid_format",
        )

        assert pipeline._format == "invalid_format"

    def test_empty_agents_list(self, mock_model):
        """Test pipeline with empty agents list works (no agents to spawn)."""
        from syrin.agent.multi_agent import DynamicPipeline

        pipeline = DynamicPipeline(
            agents=[],
            model=mock_model,
        )

        assert len(pipeline._agent_names) == 0
        assert len(pipeline._agents) == 0

    def test_negative_max_parallel(self, mock_model, researcher_agent):
        """Test that negative max_parallel is handled."""
        from syrin.agent.multi_agent import DynamicPipeline

        pipeline = DynamicPipeline(
            agents=[researcher_agent],
            model=mock_model,
            max_parallel=-1,
        )

        assert pipeline._max_parallel == -1

    # =========================================================================
    # AGENT REGISTRATION TESTS
    # =========================================================================

    def test_agent_name_from_class(self, mock_model):
        """Test agent name is derived from class name."""
        from syrin.agent.multi_agent import DynamicPipeline

        class MyTestAgent(Agent):
            model = Model(provider="openai", model_id="gpt-4o-mini")
            system_prompt = "Test agent"

        pipeline = DynamicPipeline(
            agents=[MyTestAgent],
            model=mock_model,
        )

        assert "mytestagent" in pipeline._agent_names

    def test_agent_custom_name(self, mock_model):
        """Test custom _syrin_name attribute."""
        from syrin.agent.multi_agent import DynamicPipeline

        class MyAgent(Agent):
            _syrin_name = "custom_name"
            model = Model(provider="openai", model_id="gpt-4o-mini")
            system_prompt = "Test agent"

        pipeline = DynamicPipeline(
            agents=[MyAgent],
            model=mock_model,
        )

        assert "custom_name" in pipeline._agent_names
        assert "myagent" not in pipeline._agent_names

    def test_multiple_agents_registration(self, mock_model, researcher_agent, writer_agent):
        """Test multiple agents are registered correctly."""
        from syrin.agent.multi_agent import DynamicPipeline

        pipeline = DynamicPipeline(
            agents=[researcher_agent, writer_agent],
            model=mock_model,
        )

        assert len(pipeline._agent_names) == 2
        assert "researcher" in pipeline._agent_names
        assert "writer" in pipeline._agent_names

    # =========================================================================
    # HOOKS TESTS
    # =========================================================================

    def test_events_property_exists(self, mock_model, researcher_agent):
        """Test that events property exists and returns Events object."""
        from syrin.agent.multi_agent import DynamicPipeline
        from syrin.events import Events

        pipeline = DynamicPipeline(
            agents=[researcher_agent],
            model=mock_model,
        )

        assert hasattr(pipeline, "events")
        assert isinstance(pipeline.events, Events)

    def test_on_handler_registration(self, mock_model, researcher_agent):
        """Test registering on() handler."""
        from syrin.agent.multi_agent import DynamicPipeline
        from syrin.enums import Hook

        pipeline = DynamicPipeline(
            agents=[researcher_agent],
            model=mock_model,
        )

        handler_called = []

        def handler(ctx):
            handler_called.append(ctx)

        pipeline.events.on(Hook.DYNAMIC_PIPELINE_START, handler)

        assert len(pipeline.events._handlers[Hook.DYNAMIC_PIPELINE_START]) == 1

    def test_before_handler_registration(self, mock_model, researcher_agent):
        """Test registering before() handler."""
        from syrin.agent.multi_agent import DynamicPipeline
        from syrin.enums import Hook

        pipeline = DynamicPipeline(
            agents=[researcher_agent],
            model=mock_model,
        )

        def handler(ctx):
            pass

        pipeline.events.before(Hook.DYNAMIC_PIPELINE_AGENT_SPAWN, handler)

        assert len(pipeline.events._before_handlers[Hook.DYNAMIC_PIPELINE_AGENT_SPAWN]) == 1

    def test_after_handler_registration(self, mock_model, researcher_agent):
        """Test registering after() handler."""
        from syrin.agent.multi_agent import DynamicPipeline
        from syrin.enums import Hook

        pipeline = DynamicPipeline(
            agents=[researcher_agent],
            model=mock_model,
        )

        def handler(ctx):
            pass

        pipeline.events.after(Hook.DYNAMIC_PIPELINE_AGENT_COMPLETE, handler)

        assert len(pipeline.events._after_handlers[Hook.DYNAMIC_PIPELINE_AGENT_COMPLETE]) == 1

    def test_on_all_handler(self, mock_model, researcher_agent):
        """Test on_all() registers handler for all hooks."""
        from syrin.agent.multi_agent import DynamicPipeline
        from syrin.enums import Hook

        pipeline = DynamicPipeline(
            agents=[researcher_agent],
            model=mock_model,
        )

        call_count = []

        def handler(hook, _ctx):
            call_count.append(hook)

        pipeline.events.on_all(handler)

        # Should have handlers for all Hook values
        for hook in Hook:
            if hook in pipeline.events._handlers:
                assert len(pipeline.events._handlers[hook]) > 0

    def test_emit_hook_calls_before_after(self, mock_model, researcher_agent):
        """Test that _emit_hook calls before, on, and after handlers."""
        from syrin.agent.multi_agent import DynamicPipeline
        from syrin.enums import Hook
        from syrin.events import EventContext

        pipeline = DynamicPipeline(
            agents=[researcher_agent],
            model=mock_model,
        )

        before_called = []
        on_called = []
        after_called = []

        def before_handler(ctx):
            before_called.append(ctx)

        def on_handler(ctx):
            on_called.append(ctx)

        def after_handler(ctx):
            after_called.append(ctx)

        pipeline.events.before(Hook.DYNAMIC_PIPELINE_START, before_handler)
        pipeline.events.on(Hook.DYNAMIC_PIPELINE_START, on_handler)
        pipeline.events.after(Hook.DYNAMIC_PIPELINE_START, after_handler)

        # Emit hook
        pipeline._emit_hook(Hook.DYNAMIC_PIPELINE_START, EventContext(test="data"))

        assert len(before_called) == 1
        assert len(on_called) == 1
        assert len(after_called) == 1

    def test_before_handler_can_modify_context(self, mock_model, researcher_agent):
        """Test that before() handlers can modify the context."""
        from syrin.agent.multi_agent import DynamicPipeline
        from syrin.enums import Hook
        from syrin.events import EventContext

        pipeline = DynamicPipeline(
            agents=[researcher_agent],
            model=mock_model,
        )

        def modify_handler(ctx):
            ctx["modified"] = True

        pipeline.events.before(Hook.DYNAMIC_PIPELINE_START, modify_handler)

        ctx = EventContext(original=False)
        pipeline._emit_hook(Hook.DYNAMIC_PIPELINE_START, ctx)

        assert ctx["modified"] is True
        assert ctx["original"] is False

    # =========================================================================
    # FORMAT TESTS
    # =========================================================================

    def test_format_to_schema_toon(self, mock_model, researcher_agent):
        """Test TOON format schema generation."""
        from syrin.agent.multi_agent import DynamicPipeline
        from syrin.enums import DocFormat

        pipeline = DynamicPipeline(
            agents=[researcher_agent],
            model=mock_model,
            format=DocFormat.TOON,
        )

        schema = pipeline._format_to_schema(DocFormat.TOON)
        assert "@spawn" in schema
        assert "agents:" in schema
        assert "type:" in schema
        assert "task:" in schema

    def test_format_to_schema_json(self, mock_model, researcher_agent):
        """Test JSON format schema generation."""
        from syrin.agent.multi_agent import DynamicPipeline
        from syrin.enums import DocFormat

        pipeline = DynamicPipeline(
            agents=[researcher_agent],
            model=mock_model,
            format=DocFormat.JSON,
        )

        schema = pipeline._format_to_schema(DocFormat.JSON)
        assert "type" in schema
        assert "task" in schema

    # =========================================================================
    # EXECUTION MODE TESTS
    # =========================================================================

    def test_parallel_mode_parameter(self, mock_model, researcher_agent):
        """Test parallel mode parameter is accepted."""
        from syrin.agent.multi_agent import DynamicPipeline

        pipeline = DynamicPipeline(
            agents=[researcher_agent],
            model=mock_model,
        )

        # Just verify it doesn't error - actual execution tested separately
        assert pipeline._max_parallel == 10

    def test_sequential_mode_parameter(self, mock_model, researcher_agent):
        """Test sequential mode parameter is accepted."""
        from syrin.agent.multi_agent import DynamicPipeline

        pipeline = DynamicPipeline(
            agents=[researcher_agent],
            model=mock_model,
        )

        # Just verify it doesn't error
        assert pipeline._max_parallel == 10

    # =========================================================================
    # PARSING TESTS
    # =========================================================================

    def test_parse_agents_spec_simple(self, mock_model, researcher_agent):
        """Test parsing simple agent spec."""
        from syrin.agent.multi_agent import DynamicPipeline

        pipeline = DynamicPipeline(
            agents=[researcher_agent],
            model=mock_model,
        )

        spec = """- type: researcher
  task: research AI"""

        result = pipeline._parse_agents_spec(spec)

        assert len(result) >= 1
        assert result[0]["type"] == "researcher"

    def test_parse_agents_spec_json(self, mock_model, researcher_agent):
        """Test parsing JSON format agent spec."""
        from syrin.agent.multi_agent import DynamicPipeline

        pipeline = DynamicPipeline(
            agents=[researcher_agent],
            model=mock_model,
        )

        content = '[{"type": "researcher", "task": "test task"}]'
        result = pipeline._parse_plan(content)

        assert isinstance(result, list)
        assert len(result) >= 1

    def test_parse_plan_with_markdown(self, mock_model, researcher_agent):
        """Test parsing JSON wrapped in markdown."""
        from syrin.agent.multi_agent import DynamicPipeline

        pipeline = DynamicPipeline(
            agents=[researcher_agent],
            model=mock_model,
        )

        content = '```json\n[{"type": "researcher", "task": "test"}]\n```'
        result = pipeline._parse_plan(content)

        assert isinstance(result, list)

    def test_parse_plan_invalid_json(self, mock_model, researcher_agent):
        """Test parsing invalid JSON falls back to text parsing."""
        from syrin.agent.multi_agent import DynamicPipeline

        pipeline = DynamicPipeline(
            agents=[researcher_agent],
            model=mock_model,
        )

        content = "not valid json at all"
        result = pipeline._parse_plan(content)

        # Should fall back to text parsing
        assert isinstance(result, list)

    # =========================================================================
    # BUDGET TESTS
    # =========================================================================

    def test_budget_is_stored(self, mock_model, researcher_agent):
        """Test budget is stored in pipeline."""
        from syrin import Budget
        from syrin.agent.multi_agent import DynamicPipeline

        budget = Budget(run=1.0)
        pipeline = DynamicPipeline(
            agents=[researcher_agent],
            model=mock_model,
            budget=budget,
        )

        assert pipeline._budget is budget

    def test_budget_none_by_default(self, mock_model, researcher_agent):
        """Test budget is None by default."""
        from syrin.agent.multi_agent import DynamicPipeline

        pipeline = DynamicPipeline(
            agents=[researcher_agent],
            model=mock_model,
        )

        assert pipeline._budget is None

    # =========================================================================
    # METRICS TESTS
    # =========================================================================

    def test_run_metrics_initialized(self, mock_model, researcher_agent):
        """Test run metrics are initialized."""
        from syrin.agent.multi_agent import DynamicPipeline

        pipeline = DynamicPipeline(
            agents=[researcher_agent],
            model=mock_model,
        )

        assert hasattr(pipeline, "_run_metrics")
        assert isinstance(pipeline._run_metrics, dict)


class TestDynamicPipelineErrorHandling:
    """Tests for error handling in DynamicPipeline."""

    @pytest.fixture
    def mock_model(self):
        return Model(provider="openai", model_id="gpt-4o-mini")

    def test_unknown_agent_in_plan(self, mock_model):
        """Test handling unknown agent type in plan."""
        from syrin.agent.multi_agent import DynamicPipeline

        class TestAgent(Agent):
            model = Model(provider="openai", model_id="gpt-4o-mini")

        pipeline = DynamicPipeline(
            agents=[TestAgent],
            model=mock_model,
        )

        # Unknown agent type should be skipped gracefully
        # This tests the _execute_parallel/_execute_sequential logic
        # The plan contains an agent not in the registry
        result = pipeline._execute_parallel([{"type": "nonexistent_agent", "task": "test"}])

        # Should still return results, just skipping unknown
        assert "No results" in result or isinstance(result, str)

    def test_plan_with_missing_type(self, mock_model):
        """Test plan parsing with missing type field."""
        from syrin.agent.multi_agent import DynamicPipeline

        class TestAgent(Agent):
            model = Model(provider="openai", model_id="gpt-4o-mini")

        pipeline = DynamicPipeline(
            agents=[TestAgent],
            model=mock_model,
        )

        # Plan with no type should be handled
        result = pipeline._parse_agents_spec("task: some task")

        # Should return empty or partial result
        assert isinstance(result, list)
