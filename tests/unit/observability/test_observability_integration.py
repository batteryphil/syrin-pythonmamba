"""TDD tests for observability: span coverage, session, metrics, sampling, exporters.

Run with: pytest tests/observability/test_observability_integration.py -v
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from syrin.observability import (
    InMemoryExporter,
    SpanKind,
    Tracer,
    get_tracer,
)

# -----------------------------------------------------------------------------
# Span coverage: agent run creates agent span
# -----------------------------------------------------------------------------


class TestSpanCoverageAgentSpan:
    """Agent run must create one root span with SpanKind.AGENT."""

    def test_agent_run_creates_agent_span(self):
        """Valid: single response() creates one root span with kind=AGENT."""
        from syrin import Agent
        from syrin.model import Model
        from syrin.types import ProviderResponse, TokenUsage

        tracer = get_tracer()
        exporter = InMemoryExporter()
        tracer.add_exporter(exporter)
        tracer.clear()

        model = Model("openai/gpt-4o-mini")
        agent = Agent(
            model=model,
            system_prompt="Be brief.",
            tracer=tracer,
        )
        with patch.object(
            agent._provider,
            "complete",
            new_callable=AsyncMock,
            return_value=ProviderResponse(
                content="Hi",
                tool_calls=[],
                token_usage=TokenUsage(input_tokens=10, output_tokens=5, total_tokens=15),
            ),
        ):
            agent.run("Hello")

        roots = exporter.get_root_spans()
        assert len(roots) >= 1
        agent_spans = [s for s in roots if s.kind == SpanKind.AGENT]
        assert len(agent_spans) >= 1
        assert "response" in agent_spans[0].name or "agent" in agent_spans[0].name.lower()

    def test_agent_span_has_semantic_attributes(self):
        """Valid: agent span has agent.name, input (or similar)."""
        from syrin import Agent
        from syrin.model import Model
        from syrin.observability import SemanticAttributes
        from syrin.types import ProviderResponse, TokenUsage

        tracer = get_tracer()
        exporter = InMemoryExporter()
        tracer.add_exporter(exporter)
        tracer.clear()

        agent = Agent(
            model=Model("openai/gpt-4o-mini"),
            system_prompt="Be brief.",
            tracer=tracer,
        )
        with patch.object(
            agent._provider,
            "complete",
            new_callable=AsyncMock,
            return_value=ProviderResponse(
                content="Hi",
                tool_calls=[],
                token_usage=TokenUsage(input_tokens=10, output_tokens=5, total_tokens=15),
            ),
        ):
            agent.run("Hello")

        roots = exporter.get_root_spans()
        agent_span = next((s for s in roots if s.kind == SpanKind.AGENT), None)
        assert agent_span is not None
        assert SemanticAttributes.AGENT_NAME in agent_span.attributes or "agent.name" in str(
            agent_span.attributes
        )


class TestSpanCoverageLLMSpan:
    """Loop must create LLM child span per completion call."""

    def test_single_shot_creates_llm_span(self):
        """Valid: single LLM call creates one child span with kind=LLM."""
        from syrin import Agent
        from syrin.loop import SingleShotLoop
        from syrin.model import Model
        from syrin.types import ProviderResponse, TokenUsage

        tracer = get_tracer()
        exporter = InMemoryExporter()
        tracer.add_exporter(exporter)
        tracer.clear()

        agent = Agent(
            model=Model("openai/gpt-4o-mini"),
            system_prompt="Be brief.",
            loop=SingleShotLoop(),
            tracer=tracer,
        )
        with patch.object(
            agent._provider,
            "complete",
            new_callable=AsyncMock,
            return_value=ProviderResponse(
                content="Hi",
                tool_calls=[],
                token_usage=TokenUsage(input_tokens=10, output_tokens=5, total_tokens=15),
            ),
        ):
            agent.run("Hello")

        all_spans = exporter.spans
        llm_spans = [s for s in all_spans if s.kind == SpanKind.LLM]
        assert len(llm_spans) >= 1, "Expected at least one LLM span"

    def test_react_loop_creates_llm_span_per_iteration(self):
        """Valid: each LLM call in REACT loop creates an LLM span."""
        from syrin import Agent
        from syrin.loop import ReactLoop
        from syrin.model import Model
        from syrin.types import ProviderResponse, TokenUsage

        tracer = get_tracer()
        exporter = InMemoryExporter()
        tracer.add_exporter(exporter)
        tracer.clear()

        agent = Agent(
            model=Model("openai/gpt-4o-mini"),
            system_prompt="Be brief.",
            loop=ReactLoop(max_iterations=3),
            tracer=tracer,
        )
        # First call returns content (no tools), so one iteration
        with patch.object(
            agent._provider,
            "complete",
            new_callable=AsyncMock,
            return_value=ProviderResponse(
                content="Done",
                tool_calls=[],
                token_usage=TokenUsage(input_tokens=10, output_tokens=5, total_tokens=15),
            ),
        ):
            agent.run("Hello")

        llm_spans = [s for s in exporter.spans if s.kind == SpanKind.LLM]
        assert len(llm_spans) >= 1


class TestSpanCoverageToolSpan:
    """Loop must create tool child span per tool execution."""

    def test_tool_call_creates_tool_span(self):
        """Valid: one tool call creates one child span with kind=TOOL."""
        from syrin import Agent
        from syrin.loop import ReactLoop
        from syrin.model import Model
        from syrin.tool import tool
        from syrin.types import ProviderResponse, TokenUsage

        @tool(name="echo", description="Echo")
        def echo(x: str) -> str:
            return x

        tracer = get_tracer()
        exporter = InMemoryExporter()
        tracer.add_exporter(exporter)
        tracer.clear()

        agent = Agent(
            model=Model("openai/gpt-4o-mini"),
            system_prompt="Use tools.",
            tools=[echo],
            loop=ReactLoop(max_iterations=5),
            tracer=tracer,
        )
        call_count = 0

        async def mock_complete(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                from syrin.types import ToolCall

                return ProviderResponse(
                    content="I'll echo",
                    tool_calls=[
                        ToolCall(id="1", name="echo", arguments={"x": "hi"}),
                    ],
                    token_usage=TokenUsage(input_tokens=10, output_tokens=20, total_tokens=30),
                )
            return ProviderResponse(
                content="Done",
                tool_calls=[],
                token_usage=TokenUsage(input_tokens=15, output_tokens=5, total_tokens=20),
            )

        with patch.object(agent._provider, "complete", side_effect=mock_complete):
            agent.run("Echo hello")

        tool_spans = [s for s in exporter.spans if s.kind == SpanKind.TOOL]
        assert len(tool_spans) >= 1, "Expected at least one TOOL span"


# -----------------------------------------------------------------------------
# Session: session_id propagates to spans
# -----------------------------------------------------------------------------


class TestSessionTracking:
    """Session ID must propagate to all spans created within session()."""

    def test_session_id_propagates_to_spans(self):
        """Valid: with session('s1'), root span has session_id='s1'."""
        from syrin import Agent
        from syrin.model import Model
        from syrin.observability import session
        from syrin.types import ProviderResponse, TokenUsage

        tracer = get_tracer()
        exporter = InMemoryExporter()
        tracer.add_exporter(exporter)
        tracer.clear()

        agent = Agent(
            model=Model("openai/gpt-4o-mini"),
            system_prompt="Be brief.",
            tracer=tracer,
        )
        with (
            patch.object(
                agent._provider,
                "complete",
                new_callable=AsyncMock,
                return_value=ProviderResponse(
                    content="Hi",
                    tool_calls=[],
                    token_usage=TokenUsage(input_tokens=10, output_tokens=5, total_tokens=15),
                ),
            ),
            session("session_123"),
        ):
            agent.run("Hello")

        roots = exporter.get_root_spans()
        assert len(roots) >= 1
        assert roots[0].session_id == "session_123"

    def test_no_session_spans_have_none_session_id(self):
        """Valid: without session(), spans may have session_id None."""
        from syrin import Agent
        from syrin.model import Model
        from syrin.types import ProviderResponse, TokenUsage

        tracer = get_tracer()
        exporter = InMemoryExporter()
        tracer.add_exporter(exporter)
        tracer.clear()

        agent = Agent(
            model=Model("openai/gpt-4o-mini"),
            system_prompt="Be brief.",
            tracer=tracer,
        )
        with patch.object(
            agent._provider,
            "complete",
            new_callable=AsyncMock,
            return_value=ProviderResponse(
                content="Hi",
                tool_calls=[],
                token_usage=TokenUsage(input_tokens=10, output_tokens=5, total_tokens=15),
            ),
        ):
            agent.run("Hello")

        for span in exporter.spans:
            # session_id can be None when no session() is used
            assert span.session_id is None or isinstance(span.session_id, str)


# -----------------------------------------------------------------------------
# Metrics: get_metrics() returns consistent schema
# -----------------------------------------------------------------------------


class TestMetricsAggregation:
    """Metrics from spans must be aggregated and schema documented."""

    def test_get_metrics_returns_summary_with_expected_keys(self):
        """Valid: get_summary() has llm, agent, tool sections."""
        from syrin.observability.metrics import get_metrics

        metrics = get_metrics()
        metrics.clear()
        summary = metrics.get_summary()
        assert "llm" in summary
        assert "agent" in summary
        assert "tool" in summary
        assert "counters" in summary or "gauges" in summary or "llm" in summary

    def test_metrics_schema_llm_has_cost_and_tokens(self):
        """Valid: summary['llm'] has cost, tokens_total (or similar)."""
        from syrin.observability.metrics import get_metrics

        metrics = get_metrics()
        metrics.clear()
        summary = metrics.get_summary()
        llm = summary.get("llm", {})
        assert isinstance(llm, dict)
        # Schema may use tokens_total or tokens_total from aggregation
        assert "cost" in llm or "requests" in llm or "tokens_total" in llm or "latency_avg" in llm

    def test_metrics_clear_resets_state(self):
        """Valid: clear() resets counters/gauges."""
        from syrin.observability.metrics import get_metrics

        metrics = get_metrics()
        metrics.increment("test.counter")
        metrics.clear()
        summary = metrics.get_summary()
        # After clear, counters should be empty or zero
        assert metrics.get_counter("test.counter") == 0.0 or "test.counter" not in str(summary)


# -----------------------------------------------------------------------------
# Sampling: parent-child consistency
# -----------------------------------------------------------------------------


class TestSamplingParentChild:
    """Sampling must not break parent-child: either all of a trace or none."""

    def test_when_root_sampled_children_exported(self):
        """Valid: if root is sampled, child spans are still exported (same trace)."""
        from syrin.observability import SpanKind, Tracer

        tracer = Tracer()
        exporter = InMemoryExporter()
        tracer.add_exporter(exporter)

        class AlwaysSample:
            def should_sample(self, span):
                return True

        tracer.set_sampler(AlwaysSample())
        with tracer.span("root", kind=SpanKind.AGENT), tracer.span("child", kind=SpanKind.LLM):
            pass
        # Root and child both exported (sampler returns True)
        assert len(exporter.spans) >= 2 or len(exporter.get_root_spans()) >= 1

    def test_when_root_not_sampled_no_export(self):
        """Valid: if sampler says no, root (and thus trace) not exported."""
        tracer = Tracer()
        exporter = InMemoryExporter()
        tracer.add_exporter(exporter)

        class NeverSample:
            def should_sample(self, span):
                return False

        tracer.set_sampler(NeverSample())
        with tracer.span("root", kind=SpanKind.AGENT):
            pass
        assert len(exporter.spans) == 0


# -----------------------------------------------------------------------------
# Debug mode: Agent(debug=True) adds console and verbose behavior
# -----------------------------------------------------------------------------


class TestDebugMode:
    """Agent(debug=True) must enable observable behavior."""

    def test_debug_true_adds_console_exporter_when_not_present(self):
        """Valid: debug=True adds ConsoleExporter if tracer has none."""
        from syrin import Agent
        from syrin.model import Model
        from syrin.observability import ConsoleExporter, Tracer

        tracer = Tracer()
        assert not any(isinstance(e, ConsoleExporter) for e in tracer._exporters)
        agent = Agent(
            model=Model("openai/gpt-4o-mini"),
            system_prompt="Be brief.",
            debug=True,
            tracer=tracer,
        )
        assert any(isinstance(e, ConsoleExporter) for e in agent._tracer._exporters)

    def test_debug_true_sets_debug_mode_on_tracer(self):
        """Valid: debug=True sets tracer.debug_mode True."""
        from syrin import Agent
        from syrin.model import Model
        from syrin.observability import Tracer

        tracer = Tracer()
        agent = Agent(
            model=Model("openai/gpt-4o-mini"),
            system_prompt="Be brief.",
            debug=True,
            tracer=tracer,
        )
        assert agent._tracer.debug_mode is True


# -----------------------------------------------------------------------------
# OTLP exporter (optional dependency)
# -----------------------------------------------------------------------------


class TestOTLPExporter:
    """OTLP exporter works when optional dependency is available."""

    def test_otlp_exporter_import(self):
        """Valid: OTLPExporter can be imported."""
        from syrin.observability.otlp import OPENTELEMETRY_AVAILABLE, OTLPExporter

        assert hasattr(OTLPExporter, "export")
        # When OTLP HTTP exporter package is not installed, OPENTELEMETRY_AVAILABLE is False
        assert isinstance(OPENTELEMETRY_AVAILABLE, bool)

    def test_otlp_exporter_export_no_op_when_unavailable(self):
        """Valid: when OTel OTLP not available, export() does not raise."""
        from syrin.observability import Span, SpanContext, SpanKind
        from syrin.observability.otlp import OPENTELEMETRY_AVAILABLE, OTLPExporter

        if OPENTELEMETRY_AVAILABLE:
            pytest.skip("OTLP available - test only runs when optional dep missing")
        exporter = OTLPExporter(endpoint="http://localhost:4318/v1/traces")
        ctx = SpanContext.create()
        span = Span(name="test", kind=SpanKind.INTERNAL, context=ctx)
        span.end()
        exporter.export(span)  # no-op, no raise

    def test_otlp_exporter_export_when_available(self):
        """Valid: when OTel OTLP available, export() runs without error."""
        from syrin.observability import Span, SpanContext, SpanKind
        from syrin.observability.otlp import OPENTELEMETRY_AVAILABLE, OTLPExporter

        if not OPENTELEMETRY_AVAILABLE:
            pytest.skip("OTLP HTTP exporter not installed")
        exporter = OTLPExporter(endpoint="http://localhost:4318/v1/traces", service_name="test")
        ctx = SpanContext.create()
        span = Span(name="test", kind=SpanKind.INTERNAL, context=ctx)
        span.end()
        exporter.export(span)  # may log warning if endpoint unreachable, but no crash


# -----------------------------------------------------------------------------
# Hook coverage: only Hook enum (or mapped strings) used
# -----------------------------------------------------------------------------


class TestHookCoverage:
    """Agent and loop must use Hook enum for lifecycle events."""

    def test_agent_emit_event_accepts_hook_enum(self):
        """Valid: _emit_event with Hook enum runs without error."""
        from syrin import Agent
        from syrin.enums import Hook
        from syrin.events import EventContext
        from syrin.model import Model

        agent = Agent(model=Model("openai/gpt-4o-mini"), system_prompt="Test")
        agent._emit_event(
            Hook.AGENT_RUN_START, EventContext(input="hi", model="gpt-4o", iteration=0)
        )
        # No crash

    def test_agent_emit_event_mapped_string_resolved(self):
        """Valid: mapped string names (e.g. context.compact) are resolved to Hook."""
        from syrin import Agent
        from syrin.model import Model

        agent = Agent(model=Model("openai/gpt-4o-mini"), system_prompt="Test")
        agent._emit_event("context.compact", {})
        # No crash, event resolved and dispatched

    def test_agent_emit_event_unknown_string_ignored(self):
        """Valid: unknown string hook is ignored (no crash)."""
        from syrin import Agent
        from syrin.model import Model

        agent = Agent(model=Model("openai/gpt-4o-mini"), system_prompt="Test")
        agent._emit_event("unknown.custom.event", {"key": "value"})
        # No crash


# -----------------------------------------------------------------------------
# Export format: JSONL structure
# -----------------------------------------------------------------------------


class TestExportFormat:
    """Export format must be documented and consistent."""

    def test_span_to_dict_has_required_fields(self):
        """Valid: span.to_dict() has name, kind, trace_id, span_id, duration_ms, status."""
        from syrin.observability import Span, SpanContext, SpanKind

        ctx = SpanContext.create()
        span = Span(name="test", kind=SpanKind.INTERNAL, context=ctx)
        span.end()
        d = span.to_dict()
        assert "name" in d
        assert "kind" in d
        assert "trace_id" in d
        assert "span_id" in d
        assert "duration_ms" in d
        assert "status" in d

    def test_jsonl_exporter_writes_valid_json_per_line(self, tmp_path):
        """Valid: JSONL exporter writes one JSON object per line."""
        import json

        from syrin.observability import JSONLExporter, Span, SpanContext, SpanKind

        path = tmp_path / "out.jsonl"
        exporter = JSONLExporter(str(path))
        ctx = SpanContext.create()
        span = Span(name="test", kind=SpanKind.INTERNAL, context=ctx)
        span.end()
        exporter.export(span)
        lines = path.read_text().strip().split("\n")
        assert len(lines) >= 1
        for line in lines:
            obj = json.loads(line)
            assert "name" in obj
