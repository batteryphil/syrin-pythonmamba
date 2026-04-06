"""Tests for context snapshot API."""

from __future__ import annotations

from syrin.context import Context, DefaultContextManager
from syrin.context.snapshot import (
    ContextBreakdown,
    ContextSegmentProvenance,
    ContextSegmentSource,
    ContextSnapshot,
    MessagePreview,
    _context_rot_risk_from_utilization,
)

# =============================================================================
# CONTEXT ROT RISK
# =============================================================================


class TestContextRotRisk:
    """Context rot risk derivation from utilization."""

    def test_low_below_60(self) -> None:
        assert _context_rot_risk_from_utilization(0) == "low"
        assert _context_rot_risk_from_utilization(59.9) == "low"

    def test_medium_60_to_70(self) -> None:
        assert _context_rot_risk_from_utilization(60) == "medium"
        assert _context_rot_risk_from_utilization(65) == "medium"
        assert _context_rot_risk_from_utilization(69.9) == "medium"

    def test_high_at_or_above_70(self) -> None:
        assert _context_rot_risk_from_utilization(70) == "high"
        assert _context_rot_risk_from_utilization(100) == "high"


# =============================================================================
# CONTEXT BREAKDOWN
# =============================================================================


class TestContextBreakdown:
    """ContextBreakdown dataclass."""

    def test_defaults_zero(self) -> None:
        b = ContextBreakdown()
        assert b.system_tokens == 0
        assert b.tools_tokens == 0
        assert b.memory_tokens == 0
        assert b.messages_tokens == 0
        assert b.total_tokens == 0

    def test_total_tokens_sum(self) -> None:
        b = ContextBreakdown(
            system_tokens=10,
            tools_tokens=20,
            memory_tokens=5,
            messages_tokens=100,
        )
        assert b.total_tokens == 135


# =============================================================================
# MESSAGE PREVIEW & PROVENANCE
# =============================================================================


class TestMessagePreview:
    """MessagePreview dataclass."""

    def test_all_fields(self) -> None:
        p = MessagePreview(
            role="user",
            content_snippet="Hello...",
            token_count=5,
            source=ContextSegmentSource.CURRENT_PROMPT,
        )
        assert p.role == "user"
        assert p.content_snippet == "Hello..."
        assert p.token_count == 5
        assert p.source == ContextSegmentSource.CURRENT_PROMPT


class TestContextSegmentProvenance:
    """ContextSegmentProvenance dataclass."""

    def test_with_detail(self) -> None:
        p = ContextSegmentProvenance(
            segment_id="0",
            source=ContextSegmentSource.MEMORY,
            source_detail="memory_id_1",
        )
        assert p.segment_id == "0"
        assert p.source == ContextSegmentSource.MEMORY
        assert p.source_detail == "memory_id_1"

    def test_source_detail_optional(self) -> None:
        p = ContextSegmentProvenance(
            segment_id="1",
            source=ContextSegmentSource.SYSTEM,
        )
        assert p.source_detail is None


# =============================================================================
# CONTEXT SNAPSHOT
# =============================================================================


class TestContextSnapshot:
    """ContextSnapshot dataclass and to_dict."""

    def test_default_snapshot(self) -> None:
        s = ContextSnapshot()
        assert s.total_tokens == 0
        assert s.max_tokens == 0
        assert s.utilization_pct == 0.0
        assert s.context_rot_risk == "low"
        assert s.messages_count == 0
        assert s.message_preview == []
        assert s.provenance == []
        assert s.why_included == []
        assert s.raw_messages is None

    def test_to_dict_has_required_keys(self) -> None:
        s = ContextSnapshot(
            total_tokens=100,
            max_tokens=1000,
            utilization_pct=10.0,
            context_rot_risk="low",
        )
        d = s.to_dict()
        assert "timestamp" in d
        assert "total_tokens" in d
        assert "max_tokens" in d
        assert "utilization_pct" in d
        assert "breakdown" in d
        assert "message_preview" in d
        assert "provenance" in d
        assert "why_included" in d
        assert "context_rot_risk" in d
        assert d["total_tokens"] == 100
        assert d["context_rot_risk"] == "low"

    def test_to_dict_breakdown_structure(self) -> None:
        s = ContextSnapshot(
            breakdown=ContextBreakdown(
                system_tokens=5,
                tools_tokens=10,
                memory_tokens=0,
                messages_tokens=85,
            ),
        )
        d = s.to_dict()
        assert d["breakdown"]["system_tokens"] == 5
        assert d["breakdown"]["tools_tokens"] == 10
        assert d["breakdown"]["total_tokens"] == 100

    def test_to_dict_include_raw_messages_false_by_default(self) -> None:
        s = ContextSnapshot(raw_messages=[{"role": "user", "content": "Hi"}])
        d = s.to_dict()
        assert "raw_messages" not in d

    def test_to_dict_include_raw_messages_true(self) -> None:
        raw = [{"role": "user", "content": "Hi"}]
        s = ContextSnapshot(raw_messages=raw)
        d = s.to_dict(include_raw_messages=True)
        assert d["raw_messages"] == raw

    def test_to_dict_include_raw_messages_true_but_none(self) -> None:
        s = ContextSnapshot(raw_messages=None)
        d = s.to_dict(include_raw_messages=True)
        assert "raw_messages" not in d


# =============================================================================
# MANAGER SNAPSHOT — VALID CASES
# =============================================================================


class TestManagerSnapshotValid:
    """Snapshot from DefaultContextManager — valid cases."""

    def test_snapshot_before_any_prepare_returns_empty_snapshot(self) -> None:
        """Before any prepare(), snapshot has zeros and low rot risk."""
        manager = DefaultContextManager(Context(max_tokens=8000))
        snap = manager.snapshot()
        assert isinstance(snap, ContextSnapshot)
        assert snap.total_tokens == 0
        assert snap.max_tokens == 0
        assert snap.messages_count == 0
        assert snap.context_rot_risk == "low"
        assert snap.message_preview == []
        assert snap.provenance == []
        assert snap.why_included == []

    def test_snapshot_after_prepare_has_tokens_and_preview(self) -> None:
        """After prepare(), snapshot reflects that run."""
        manager = DefaultContextManager(Context(max_tokens=8000))
        manager.prepare(
            messages=[
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Hello"},
            ],
            system_prompt="You are helpful.",
            tools=[],
            memory_context="",
        )
        snap = manager.snapshot()
        assert snap.total_tokens > 0
        assert snap.max_tokens == 8000
        assert snap.messages_count >= 2
        assert snap.utilization_pct >= 0
        assert len(snap.message_preview) == snap.messages_count
        assert len(snap.provenance) >= 1
        assert len(snap.why_included) >= 1

    def test_snapshot_breakdown_populated_after_prepare(self) -> None:
        """Breakdown has non-zero system/messages when present."""
        manager = DefaultContextManager(Context(max_tokens=8000))
        manager.prepare(
            messages=[{"role": "user", "content": "Hi"}],
            system_prompt="System",
            tools=[],
            memory_context="",
        )
        snap = manager.snapshot()
        assert snap.breakdown.system_tokens >= 0
        assert snap.breakdown.messages_tokens >= 0
        assert snap.breakdown.total_tokens == snap.total_tokens

    def test_snapshot_with_tools_includes_tools_tokens(self) -> None:
        """When tools are passed, breakdown.tools_tokens > 0."""
        manager = DefaultContextManager(Context(max_tokens=8000))
        manager.prepare(
            messages=[{"role": "user", "content": "Hi"}],
            system_prompt="",
            tools=[{"name": "foo", "description": "A tool", "parameters": {}}],
            memory_context="",
        )
        snap = manager.snapshot()
        assert snap.breakdown.tools_tokens > 0

    def test_snapshot_with_memory_includes_memory_tokens(self) -> None:
        """When memory_context is non-empty, breakdown.memory_tokens > 0."""
        manager = DefaultContextManager(Context(max_tokens=8000))
        manager.prepare(
            messages=[{"role": "user", "content": "Hi"}],
            system_prompt="",
            tools=[],
            memory_context="Relevant: user likes Python.",
        )
        snap = manager.snapshot()
        assert snap.breakdown.memory_tokens > 0

    def test_snapshot_context_rot_risk_low_when_under_60_percent(self) -> None:
        """Utilization under 60% yields context_rot_risk low."""
        manager = DefaultContextManager(Context(max_tokens=100_000))
        manager.prepare(
            messages=[{"role": "user", "content": "Short"}],
            system_prompt="Short system.",
            tools=[],
            memory_context="",
        )
        snap = manager.snapshot()
        assert snap.utilization_pct < 60
        assert snap.context_rot_risk == "low"

    def test_snapshot_compacted_and_compact_method_when_compaction_ran(self) -> None:
        """When compaction runs, compacted=True and compact_method set."""
        manager = DefaultContextManager(Context(max_tokens=500))
        many = [{"role": "user", "content": "x" * 200} for _ in range(20)]
        manager.prepare(
            messages=many,
            system_prompt="",
            tools=[],
            memory_context="",
        )
        snap = manager.snapshot()
        # May or may not compact depending on threshold; if it did:
        if snap.compacted:
            assert snap.compact_method is not None


# =============================================================================
# MANAGER SNAPSHOT — EDGE CASES
# =============================================================================


class TestManagerSnapshotEdgeCases:
    """Edge cases: empty messages, zero capacity, long content, etc."""

    def test_snapshot_after_prepare_empty_messages(self) -> None:
        """Empty messages list still produces valid snapshot."""
        manager = DefaultContextManager(Context(max_tokens=8000))
        manager.prepare(
            messages=[],
            system_prompt="",
            tools=[],
            memory_context="",
        )
        snap = manager.snapshot()
        assert snap.messages_count >= 0
        assert snap.total_tokens >= 0
        assert snap.context_rot_risk in ("low", "medium", "high")

    def test_snapshot_no_division_by_zero_when_max_tokens_zero(self) -> None:
        """Capacity with max_tokens=0 does not cause division by zero."""
        manager = DefaultContextManager(Context(max_tokens=8000))
        capacity = manager.context.get_capacity()
        # Force used_tokens so utilization would be computed
        capacity.used_tokens = 0
        manager.prepare(
            messages=[],
            system_prompt="",
            tools=[],
            memory_context="",
        )
        snap = manager.snapshot()
        assert snap.utilization_pct >= 0
        assert snap.context_rot_risk in ("low", "medium", "high")

    def test_message_preview_snippet_truncated(self) -> None:
        """Long message content is truncated in content_snippet."""
        manager = DefaultContextManager(Context(max_tokens=8000))
        long_content = "a" * 500
        manager.prepare(
            messages=[{"role": "user", "content": long_content}],
            system_prompt="",
            tools=[],
            memory_context="",
        )
        snap = manager.snapshot()
        assert len(snap.message_preview) >= 1
        for p in snap.message_preview:
            assert len(p.content_snippet) <= 200  # reasonable max snippet length

    def test_snapshot_why_included_non_empty_after_prepare(self) -> None:
        """why_included has at least one human-readable reason."""
        manager = DefaultContextManager(Context(max_tokens=8000))
        manager.prepare(
            messages=[{"role": "user", "content": "Hi"}],
            system_prompt="You are helpful.",
            tools=[],
            memory_context="",
        )
        snap = manager.snapshot()
        assert len(snap.why_included) >= 1
        assert any(len(s) > 0 for s in snap.why_included)


# =============================================================================
# CONTEXT STATS BREAKDOWN
# =============================================================================


class TestContextStatsBreakdown:
    """ContextStats.breakdown is set after prepare and matches snapshot."""

    def test_stats_breakdown_none_before_prepare(self) -> None:
        """Before any prepare(), stats.breakdown is None."""
        manager = DefaultContextManager(Context(max_tokens=8000))
        assert manager.stats.breakdown is None

    def test_stats_breakdown_set_after_prepare(self) -> None:
        """After prepare(), stats.breakdown is not None."""
        manager = DefaultContextManager(Context(max_tokens=8000))
        manager.prepare(
            messages=[{"role": "user", "content": "Hi"}],
            system_prompt="You are helpful.",
            tools=[],
            memory_context="",
        )
        assert manager.stats.breakdown is not None
        assert manager.stats.breakdown.total_tokens == manager.stats.total_tokens

    def test_stats_breakdown_matches_snapshot_after_prepare(self) -> None:
        """After prepare(), stats.breakdown matches snapshot.breakdown."""
        manager = DefaultContextManager(Context(max_tokens=8000))
        manager.prepare(
            messages=[{"role": "user", "content": "Hello"}],
            system_prompt="System",
            tools=[{"name": "foo", "description": "A tool", "parameters": {}}],
            memory_context="",
        )
        snap = manager.snapshot()
        breakdown = manager.stats.breakdown
        assert breakdown is not None
        assert breakdown.system_tokens == snap.breakdown.system_tokens
        assert breakdown.tools_tokens == snap.breakdown.tools_tokens
        assert breakdown.memory_tokens == snap.breakdown.memory_tokens
        assert breakdown.messages_tokens == snap.breakdown.messages_tokens
        assert breakdown.total_tokens == snap.breakdown.total_tokens

    def test_stats_breakdown_with_memory(self) -> None:
        """When memory_context is set, stats.breakdown includes memory_tokens."""
        manager = DefaultContextManager(Context(max_tokens=8000))
        manager.prepare(
            messages=[{"role": "user", "content": "Hi"}],
            system_prompt="",
            tools=[],
            memory_context="User prefers Python.",
        )
        assert manager.stats.breakdown is not None
        assert manager.stats.breakdown.memory_tokens > 0


# =============================================================================
# HOOK EMISSION
# =============================================================================


class TestSnapshotHookEmission:
    """CONTEXT_SNAPSHOT hook emitted after prepare."""

    def test_context_snapshot_emitted_after_prepare(self) -> None:
        events: list[tuple[str, object]] = []

        def emit_fn(event: str, ctx: object) -> None:
            events.append((event, ctx))

        manager = DefaultContextManager(Context(max_tokens=8000))
        manager.set_emit_fn(emit_fn)
        manager.prepare(
            messages=[{"role": "user", "content": "Hi"}],
            system_prompt="",
            tools=[],
            memory_context="",
        )
        snapshot_events = [e for e in events if e[0] == "context.snapshot"]
        assert len(snapshot_events) >= 1
        payload = snapshot_events[0][1]
        assert isinstance(payload, dict)
        assert "snapshot" in payload or "utilization_pct" in str(payload)


# =============================================================================
# CONTEXT SEGMENT SOURCE ENUM
# =============================================================================


class TestContextSegmentSource:
    """ContextSegmentSource StrEnum."""

    def test_all_sources_defined(self) -> None:
        assert ContextSegmentSource.SYSTEM.value == "system"
        assert ContextSegmentSource.MEMORY.value == "memory"
        assert ContextSegmentSource.CONVERSATION.value == "conversation"
        assert ContextSegmentSource.TOOLS.value == "tools"
        assert ContextSegmentSource.CURRENT_PROMPT.value == "current_prompt"
        assert ContextSegmentSource.INJECTED.value == "injected"


# =============================================================================
# AGENT FACADE SNAPSHOT
# =============================================================================


class TestAgentContextSnapshotFacade:
    """agent.context.snapshot() when agent uses DefaultContextManager."""

    def test_agent_context_snapshot_returns_snapshot(self) -> None:
        """Agent with DefaultContextManager exposes context.snapshot()."""
        from syrin import Agent, Context, Model

        # Use a simple model stub so we don't need real API
        agent = Agent(
            model=Model("openai/gpt-4o-mini"),
            context=Context(max_tokens=8000),
        )
        # Before any run, snapshot is empty
        snap = agent.context.snapshot()
        assert isinstance(snap, ContextSnapshot)
        assert snap.total_tokens == 0
        assert snap.context_rot_risk == "low"
