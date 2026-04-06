"""Tests for observability module (spans, tracing, exporters)."""

from __future__ import annotations

import pytest

from syrin.observability import (
    InMemoryExporter,
    JSONLExporter,
    SemanticAttributes,
    Session,
    Span,
    SpanContext,
    SpanKind,
    SpanStatus,
    Tracer,
)

# =============================================================================
# SPAN TESTS
# =============================================================================


class TestSpan:
    """Tests for Span class."""

    def test_span_creation(self):
        """Test basic span creation."""
        context = SpanContext.create()
        span = Span(
            name="test_span",
            kind=SpanKind.INTERNAL,
            context=context,
        )
        assert span.name == "test_span"
        assert span.kind == SpanKind.INTERNAL
        assert span.status == SpanStatus.PENDING

    def test_span_set_attribute(self):
        """Test setting attributes on span."""
        context = SpanContext.create()
        span = Span(name="test", kind=SpanKind.INTERNAL, context=context)

        span.set_attribute("key", "value")
        assert span.attributes["key"] == "value"

    def test_span_set_attributes(self):
        """Test setting multiple attributes."""
        context = SpanContext.create()
        span = Span(name="test", kind=SpanKind.INTERNAL, context=context)

        span.set_attributes({"a": 1, "b": 2})
        assert span.attributes["a"] == 1
        assert span.attributes["b"] == 2

    def test_span_add_event(self):
        """Test adding events to span."""
        context = SpanContext.create()
        span = Span(name="test", kind=SpanKind.INTERNAL, context=context)

        span.add_event("test_event", {"detail": "info"})
        assert len(span.events) == 1
        assert span.events[0]["name"] == "test_event"

    def test_span_duration(self):
        """Test span duration calculation."""
        context = SpanContext.create()
        span = Span(name="test", kind=SpanKind.INTERNAL, context=context)

        import time

        time.sleep(0.01)  # 10ms
        span.end()

        assert span.duration_ms >= 10

    def test_span_end_with_status(self):
        """Test ending span with status."""
        context = SpanContext.create()
        span = Span(name="test", kind=SpanKind.INTERNAL, context=context)

        span.end(status=SpanStatus.OK)
        assert span.status == SpanStatus.OK
        assert span.end_time is not None

    def test_span_record_exception(self):
        """Test recording exception on span."""
        context = SpanContext.create()
        span = Span(name="test", kind=SpanKind.INTERNAL, context=context)

        try:
            raise ValueError("Test error")
        except Exception as e:
            span.record_exception(e)

        assert span.status == SpanStatus.ERROR
        assert "Test error" in span.status_message


# =============================================================================
# SPAN CONTEXT TESTS
# =============================================================================


class TestSpanContext:
    """Tests for SpanContext."""

    def test_create_root_context(self):
        """Test creating root span context."""
        context = SpanContext.create()
        assert context.trace_id is not None
        assert context.span_id is not None
        assert context.parent_span_id is None

    def test_create_child_context(self):
        """Test creating child span context."""
        parent = SpanContext.create()
        child = SpanContext.create(parent)

        assert child.trace_id == parent.trace_id
        assert child.parent_span_id == parent.span_id
        assert child.span_id != parent.span_id

    def test_is_sampled(self):
        """Test sampling check."""
        context = SpanContext.create()
        assert context.is_sampled() is True


# =============================================================================
# SESSION TESTS
# =============================================================================


class TestSession:
    """Tests for Session."""

    def test_session_creation(self):
        """Test session creation."""
        session = Session(id="test-session")
        assert session.id == "test-session"
        assert session.start_time is not None

    def test_session_to_dict(self):
        """Test session serialization."""
        session = Session(id="test", metadata={"key": "value"})
        data = session.to_dict()

        assert data["id"] == "test"
        assert data["metadata"]["key"] == "value"


# =============================================================================
# TRACER TESTS
# =============================================================================


class TestTracer:
    """Tests for Tracer."""

    def test_tracer_creation(self):
        """Test tracer creation."""
        tracer = Tracer()
        assert tracer is not None

    def test_tracer_span_context_manager(self):
        """Test using span as context manager."""
        tracer = Tracer()

        with tracer.span("test") as span:
            span.set_attribute("key", "value")

        assert span.status == SpanStatus.OK
        assert span.attributes["key"] == "value"

    def test_tracer_span_error(self):
        """Test span with exception."""
        tracer = Tracer()

        with pytest.raises(ValueError), tracer.span("test"):
            raise ValueError("Test error")

        # After exception, span should have error status
        # Note: The span object reference is still valid after context exit

    def test_tracer_session_context_manager(self):
        """Test using session as context manager."""
        tracer = Tracer()

        with tracer.session("session-1") as session:
            pass

        assert session.id == "session-1"

    def test_tracer_add_exporter(self):
        """Test adding exporter to tracer."""
        tracer = Tracer()
        exporter = InMemoryExporter()

        tracer.add_exporter(exporter)

        with tracer.span("test"):
            pass

        assert len(exporter.spans) == 1

    def test_tracer_debug_mode(self):
        """Test debug mode toggle."""
        tracer = Tracer()
        assert tracer.debug_mode is False

        tracer.set_debug_mode(True)
        assert tracer.debug_mode is True

    def test_tracer_clear(self):
        """Test clearing tracer."""
        tracer = Tracer()
        exporter = InMemoryExporter()
        tracer.add_exporter(exporter)

        with tracer.span("test"):
            pass

        tracer.clear()
        assert len(tracer._spans) == 0


# =============================================================================
# EXPORTER TESTS
# =============================================================================


class TestInMemoryExporter:
    """Tests for InMemoryExporter."""

    def test_export_span(self):
        """Test exporting span to memory."""
        exporter = InMemoryExporter()
        context = SpanContext.create()
        span = Span(name="test", kind=SpanKind.INTERNAL, context=context)

        exporter.export(span)
        assert len(exporter.spans) == 1

    def test_get_root_spans(self):
        """Test getting root spans."""
        exporter = InMemoryExporter()

        root_context = SpanContext.create()
        root_span = Span(name="root", kind=SpanKind.INTERNAL, context=root_context)

        child_context = SpanContext.create(root_context)
        child_span = Span(name="child", kind=SpanKind.INTERNAL, context=child_context)

        exporter.export(root_span)
        exporter.export(child_span)

        roots = exporter.get_root_spans()
        assert len(roots) == 1
        assert roots[0].name == "root"

    def test_clear(self):
        """Test clearing exporter."""
        exporter = InMemoryExporter()
        context = SpanContext.create()
        span = Span(name="test", kind=SpanKind.INTERNAL, context=context)

        exporter.export(span)
        exporter.clear()

        assert len(exporter.spans) == 0


class TestJSONLExporter:
    """Tests for JSONLExporter."""

    def test_export_to_file(self, tmp_path):
        """Test exporting span to JSONL file."""
        filepath = tmp_path / "spans.jsonl"
        exporter = JSONLExporter(str(filepath))

        context = SpanContext.create()
        span = Span(name="test", kind=SpanKind.INTERNAL, context=context)
        span.end()

        exporter.export(span)

        assert filepath.exists()
        content = filepath.read_text()
        assert "test" in content


# =============================================================================
# SEMANTIC ATTRIBUTES TESTS
# =============================================================================


class TestSemanticAttributes:
    """Tests for SemanticAttributes."""

    def test_agent_attributes(self):
        """Test agent attribute constants."""
        assert SemanticAttributes.AGENT_NAME == "agent.name"
        assert SemanticAttributes.AGENT_CLASS == "agent.class"

    def test_llm_attributes(self):
        """Test LLM attribute constants."""
        assert SemanticAttributes.LLM_MODEL == "llm.model"
        assert SemanticAttributes.LLM_TOKENS_TOTAL == "llm.tokens.total"

    def test_tool_attributes(self):
        """Test tool attribute constants."""
        assert SemanticAttributes.TOOL_NAME == "tool.name"
        assert SemanticAttributes.TOOL_INPUT == "tool.input"

    def test_memory_attributes(self):
        """Test memory attribute constants."""
        assert SemanticAttributes.MEMORY_OPERATION == "memory.operation"
        assert SemanticAttributes.MEMORY_TYPE == "memory.type"

    def test_budget_attributes(self):
        """Test budget attribute constants."""
        assert SemanticAttributes.BUDGET_LIMIT == "budget.limit"
        assert SemanticAttributes.BUDGET_USED == "budget.used"


# =============================================================================
# CONVENIENCE FUNCTIONS TESTS
# =============================================================================


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_llm_span(self):
        """Test llm_span convenience function."""
        from syrin.observability import llm_span

        with llm_span("gpt-4", prompt="Hello") as span:
            span.set_attribute("completion", "Hi")

        assert span.kind == SpanKind.LLM
        assert span.attributes[SemanticAttributes.LLM_MODEL] == "gpt-4"

    def test_tool_span(self):
        """Test tool_span convenience function."""
        from syrin.observability import tool_span

        with tool_span("calculator", input={"x": 1, "y": 2}) as span:
            pass

        assert span.kind == SpanKind.TOOL
        assert span.attributes[SemanticAttributes.TOOL_NAME] == "calculator"

    def test_memory_span(self):
        """Test memory_span convenience function."""
        from syrin.observability import memory_span

        with memory_span("recall", memory_type="history") as span:
            pass

        assert span.kind == SpanKind.MEMORY
        assert span.attributes[SemanticAttributes.MEMORY_OPERATION] == "recall"

    def test_budget_span(self):
        """Test budget_span convenience function."""
        from syrin.observability import budget_span

        with budget_span("check", limit=10.0, used=5.0) as span:
            pass

        assert span.kind == SpanKind.BUDGET
        assert span.attributes[SemanticAttributes.BUDGET_LIMIT] == 10.0

    def test_agent_span(self):
        """Test agent_span convenience function."""
        from syrin.observability import agent_span

        with agent_span("my_agent") as span:
            pass

        assert span.kind == SpanKind.AGENT
        assert span.attributes[SemanticAttributes.AGENT_NAME] == "my_agent"

    def test_handoff_span(self):
        """Test handoff_span convenience function."""
        from syrin.observability import handoff_span

        with handoff_span("agent_a", "agent_b") as span:
            pass

        assert span.kind == SpanKind.HANDOFF
        assert span.attributes[SemanticAttributes.HANDOFF_SOURCE] == "agent_a"


# =============================================================================
# OBSERVABILITY EDGE CASES
# =============================================================================


class TestObservabilityEdgeCases:
    """Edge cases for observability."""

    def test_span_with_many_attributes(self):
        """Span with many attributes."""
        context = SpanContext.create()
        span = Span(name="test", kind=SpanKind.INTERNAL, context=context)

        for i in range(1000):
            span.set_attribute(f"key_{i}", f"value_{i}")

        assert len(span.attributes) == 1000

    def test_span_with_many_events(self):
        """Span with many events."""
        context = SpanContext.create()
        span = Span(name="test", kind=SpanKind.INTERNAL, context=context)

        for i in range(100):
            span.add_event(f"event_{i}")

        assert len(span.events) == 100

    def test_nested_spans(self):
        """Test deeply nested spans."""
        tracer = Tracer()

        with (
            tracer.span("level_1") as span1,
            tracer.span("level_2") as span2,
            tracer.span("level_3") as span3,
        ):
            pass

        # All spans should be ended
        assert span1.end_time is not None
        assert span2.end_time is not None
        assert span3.end_time is not None

    def test_span_walk(self):
        """Test span tree walking."""
        root_context = SpanContext.create()
        root = Span(name="root", kind=SpanKind.INTERNAL, context=root_context)

        child_context = SpanContext.create(root_context)
        child = Span(name="child", kind=SpanKind.INTERNAL, context=child_context)

        root.add_child(child)

        spans = list(root.walk())
        assert len(spans) == 2

    def test_span_to_dict(self):
        """Test span serialization."""
        context = SpanContext.create()
        span = Span(name="test", kind=SpanKind.INTERNAL, context=context)
        span.set_attribute("key", "value")
        span.end()

        data = span.to_dict()
        assert data["name"] == "test"
        assert data["attributes"]["key"] == "value"

    def test_span_unicode_attributes(self):
        """Span with unicode attributes."""
        context = SpanContext.create()
        span = Span(name="test", kind=SpanKind.INTERNAL, context=context)

        span.set_attribute("message", "Hello 🌍 你好 🔥")
        assert "🌍" in span.attributes["message"]

    def test_exporter_exception_handling(self):
        """Test exporter exception handling."""

        class FailingExporter:
            def export(self, _span):
                raise RuntimeError("Export failed")

        tracer = Tracer()
        tracer.add_exporter(FailingExporter())

        # Should not raise even though exporter fails
        with tracer.span("test"):
            pass

    def test_tracer_with_sampler(self):
        """Test tracer with sampler."""
        tracer = Tracer()

        class AlwaysSample:
            def should_sample(self, _span):
                return True

        tracer.set_sampler(AlwaysSample())

        with tracer.span("test"):
            pass

        # Span should be sampled

    def test_tracer_never_sample(self):
        """Test tracer with never-sampling sampler."""
        tracer = Tracer()
        exporter = InMemoryExporter()
        tracer.add_exporter(exporter)

        class NeverSample:
            def should_sample(self, _span):
                return False

        tracer.set_sampler(NeverSample())

        with tracer.span("test"):
            pass

        # Span should not be exported
        assert len(exporter.spans) == 0
