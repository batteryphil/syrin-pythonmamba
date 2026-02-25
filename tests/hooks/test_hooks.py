"""Tests for the events system using Hook enums."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

from syrin import Agent, Model
from syrin.enums import Hook
from syrin.events import EventContext, Events
from syrin.types import ProviderResponse, TokenUsage


class TestEvents:
    """Test the Events class with Hook enums."""

    def test_on(self):
        """Test registering event handlers with Hook enum."""
        called = []

        def handler(ctx):
            called.append(ctx)

        events = Events(lambda _e, _c: None)
        events.on(Hook.AGENT_RUN_START, handler)

        events._trigger(Hook.AGENT_RUN_START, EventContext(test="data"))

        assert len(called) == 1
        assert called[0].test == "data"

    def test_on_all(self):
        """Test registering handler for all events."""
        called = []

        def handler(event, ctx):
            called.append((event, ctx))

        events = Events(lambda _e, _c: None)
        events.on_all(handler)

        events._trigger(Hook.AGENT_RUN_START, EventContext(test="start"))
        events._trigger(Hook.AGENT_RUN_END, EventContext(test="end"))

        assert len(called) == 2
        assert called[0][0] == Hook.AGENT_RUN_START
        assert called[1][0] == Hook.AGENT_RUN_END

    def test_before(self):
        """Test before handlers can modify context."""

        def modifier(ctx):
            ctx["modified"] = True

        events = Events(lambda _e, _c: None)
        events.before(Hook.LLM_REQUEST_START, modifier)

        ctx = EventContext(data="original")
        events._trigger_before(Hook.LLM_REQUEST_START, ctx)

        assert ctx.get("modified") is True

    def test_after(self):
        """Test after handlers are called."""

        def tracker(ctx):
            ctx["tracked"] = True

        events = Events(lambda _e, _c: None)
        events.after(Hook.AGENT_RUN_END, tracker)

        ctx = EventContext(done=True)
        events._trigger_after(Hook.AGENT_RUN_END, ctx)

        assert ctx.get("tracked") is True

    def test_shortcut_methods(self):
        """Test shortcut methods use Hook enum."""
        from syrin.enums import Hook

        called = {}

        def make_handler(name):
            def handler(ctx):
                called[name] = True

            return handler

        events = Events(lambda _e, _c: None)
        events.on_start(make_handler("start"))
        events.on_complete(make_handler("complete"))
        events.on_request(make_handler("request"))
        events.on_response(make_handler("response"))
        events.on_tool(make_handler("tool"))
        events.on_error(make_handler("error"))

        events._trigger(Hook.AGENT_RUN_START, EventContext())
        events._trigger(Hook.AGENT_RUN_END, EventContext())
        events._trigger(Hook.LLM_REQUEST_START, EventContext())
        events._trigger(Hook.LLM_REQUEST_END, EventContext())
        events._trigger(Hook.TOOL_CALL_END, EventContext())
        events._trigger(Hook.TOOL_ERROR, EventContext())

        assert called.get("start") is True
        assert called.get("complete") is True
        assert called.get("request") is True
        assert called.get("response") is True
        assert called.get("tool") is True
        assert called.get("error") is True


class TestEventContext:
    """Test EventContext class."""

    def test_dict_access(self):
        """Test dict-style access."""
        ctx = EventContext(key="value")
        assert ctx["key"] == "value"

    def test_dot_access(self):
        """Test dot-style access."""
        ctx = EventContext(key="value")
        assert ctx.key == "value"

    def test_assignment(self):
        """Test assignment via both methods."""
        ctx = EventContext()
        ctx["key"] = "value"
        assert ctx.key == "value"

        ctx.another = "data"
        assert ctx["another"] == "data"


class TestAgentEvents:
    """Test Agent events integration."""

    def test_agent_has_events(self):
        """Verify agent has events attribute."""
        agent = Agent(model=Model("test/model"))
        assert hasattr(agent, "events")
        assert isinstance(agent.events, Events)

    def test_on_start(self):
        """Test on_start event fires with Hook enum."""
        received = []

        def handler(ctx):
            received.append(ctx)

        async def mock_complete(messages, model, tools=None):
            from syrin.types import ProviderResponse, TokenUsage

            return ProviderResponse(
                content="Hello!",
                tool_calls=[],
                token_usage=TokenUsage(input_tokens=10, output_tokens=5, total_tokens=15),
            )

        agent = Agent(model=Model("test/model"))
        agent.events.on_start(handler)

        with patch.object(agent, "_provider") as mock_provider_obj:
            mock_provider_obj.complete = mock_complete
            agent.response("Hi")

        assert len(received) == 1
        assert received[0].input == "Hi"

    def test_on_complete(self):
        """Test on_complete event fires."""
        received = []

        def handler(ctx):
            received.append(ctx)

        async def mock_complete(messages, model, tools=None):
            from syrin.types import ProviderResponse, TokenUsage

            return ProviderResponse(
                content="Hello!",
                tool_calls=[],
                token_usage=TokenUsage(input_tokens=10, output_tokens=5, total_tokens=15),
            )

        agent = Agent(model=Model("test/model"))
        agent.events.on_complete(handler)

        with patch.object(agent, "_provider") as mock_provider_obj:
            mock_provider_obj.complete = mock_complete
            agent.response("Hi")

        assert len(received) == 1
        assert "cost" in received[0]

    def test_on_request(self):
        """Test on_request event fires."""
        received = []

        def handler(ctx):
            received.append(ctx)

        async def mock_complete(messages, model, tools=None):
            from syrin.types import ProviderResponse, TokenUsage

            return ProviderResponse(
                content="Hello!",
                tool_calls=[],
                token_usage=TokenUsage(input_tokens=10, output_tokens=5, total_tokens=15),
            )

        agent = Agent(model=Model("test/model"))
        agent.events.on_request(handler)

        with patch.object(agent, "_provider") as mock_provider_obj:
            mock_provider_obj.complete = mock_complete
            agent.response("Hi")

        assert len(received) == 1

    def test_on_response(self):
        """Test on_response event fires."""
        received = []

        def handler(ctx):
            received.append(ctx)

        async def mock_complete(messages, model, tools=None):
            from syrin.types import ProviderResponse, TokenUsage

            return ProviderResponse(
                content="Hello!",
                tool_calls=[],
                token_usage=TokenUsage(input_tokens=10, output_tokens=5, total_tokens=15),
            )

        agent = Agent(model=Model("test/model"))
        agent.events.on_response(handler)

        with patch.object(agent, "_provider") as mock_provider_obj:
            mock_provider_obj.complete = mock_complete
            agent.response("Hi")

        assert len(received) == 1

    def test_on_all(self):
        """Test on_all event fires for all Hooks."""
        received = []

        def handler(event, ctx):
            received.append((event, ctx))

        async def mock_complete(messages, model, tools=None):
            from syrin.types import ProviderResponse, TokenUsage

            return ProviderResponse(
                content="Hello!",
                tool_calls=[],
                token_usage=TokenUsage(input_tokens=10, output_tokens=5, total_tokens=15),
            )

        agent = Agent(model=Model("test/model"))
        agent.events.on_all(handler)

        with patch.object(agent, "_provider") as mock_provider_obj:
            mock_provider_obj.complete = mock_complete
            agent.response("Hi")

        events_received = [e for e, _ in received]
        assert Hook.AGENT_RUN_START in events_received
        assert Hook.LLM_REQUEST_START in events_received
        assert Hook.LLM_REQUEST_END in events_received
        assert Hook.AGENT_RUN_END in events_received


# =============================================================================
# HOOKS EDGE CASES - TRY TO BREAK FUNCTIONALITY
# =============================================================================


class TestHooksEdgeCases:
    """Edge cases for hooks/events."""

    def test_events_with_empty_context(self):
        """Events with empty context."""
        events = Events(lambda _e, _c: None)
        events.on(Hook.AGENT_RUN_START, lambda _ctx: None)
        events._trigger(Hook.AGENT_RUN_START, EventContext())

    def test_events_multiple_handlers_same_hook(self):
        """Multiple handlers for same hook."""
        called = []

        def handler1(_ctx):
            called.append(1)

        def handler2(_ctx):
            called.append(2)

        events = Events(lambda _e, _c: None)
        events.on(Hook.AGENT_RUN_START, handler1)
        events.on(Hook.AGENT_RUN_START, handler2)
        events._trigger(Hook.AGENT_RUN_START, EventContext())

        assert len(called) == 2

    def test_event_context_with_many_fields(self):
        """EventContext with many fields."""
        ctx = EventContext(
            field1="value1",
            field2=123,
            field3=True,
            field4=[1, 2, 3],
            field5={"key": "value"},
        )
        assert ctx.field1 == "value1"
        assert ctx.field2 == 123
        assert ctx.field3 is True

    def test_event_context_empty(self):
        """EventContext empty initialization."""
        ctx = EventContext()
        # Should be empty dict-like
        assert len(ctx.__dict__) == 0 or ctx.__dict__ == {}

    def test_events_trigger_with_valid_hook(self):
        """Trigger with valid hook."""
        events = Events(lambda _e, _c: None)
        # Should not raise with valid hook
        events._trigger(Hook.AGENT_RUN_START, EventContext())


class TestHookEmissionAudit:
    """Regression: agent/loop emit only Hook enum values (or mapped legacy strings)."""

    def test_agent_run_emits_only_hook_enum_values(self) -> None:
        """During response(), every emitted hook is a Hook enum member."""
        emitted: list[Hook] = []

        def capture(hook: Hook, _ctx: EventContext) -> None:
            emitted.append(hook)

        model = Model("anthropic/claude-3-5-sonnet")
        agent = Agent(model=model, system_prompt="Test.")
        agent.events.on_all(capture)

        mock_resp = ProviderResponse(
            content="Hi",
            tool_calls=[],
            token_usage=TokenUsage(
                input_tokens=5,
                output_tokens=10,
                total_tokens=15,
            ),
        )
        with patch.object(
            agent._provider,
            "complete",
            new_callable=AsyncMock,
            return_value=mock_resp,
        ):
            agent.response("Hello")

        for h in emitted:
            assert isinstance(h, Hook), f"Expected Hook enum, got {type(h).__name__}: {h}"
            assert h in Hook
