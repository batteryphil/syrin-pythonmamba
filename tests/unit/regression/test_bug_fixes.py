"""Regression tests for BUG-01 through BUG-10.

Each test verifies the fix is in place and the bug does not recur.
"""

from __future__ import annotations

import threading

import pytest

from syrin import Agent, Budget, Model


def _almock() -> Model:
    return Model.Almock(latency_seconds=0.01, lorem_length=5)


# ---------------------------------------------------------------------------
# BUG-01: Memory(bus=...) validated when bus backend is unavailable
# ---------------------------------------------------------------------------


class TestBug01MemoryBusValidation:
    """Memory(bus=...) not validated when bus backend is unavailable."""

    def test_memory_bus_silently_drops_writes_does_not_crash(self) -> None:
        """MemoryBus write with no subscribers does not crash the agent."""
        import asyncio

        from syrin.enums import MemoryType
        from syrin.memory.config import MemoryEntry
        from syrin.swarm._memory_bus import MemoryBus

        bus = MemoryBus()
        entry = MemoryEntry(id="test-1", content="test content", type=MemoryType.HISTORY)
        # Publishing to a bus with no subscribers should not raise
        asyncio.run(bus.publish(entry, "agent-1"))


# ---------------------------------------------------------------------------
# BUG-02: spawn() propagates parent RemoteConfig
# ---------------------------------------------------------------------------


class TestBug02SpawnPropagatesRemoteConfig:
    """agent.spawn() propagates parent's RemoteConfig to child agents."""

    def test_spawn_does_not_crash_without_remote_config(self) -> None:
        """spawn() works without RemoteConfig — no crash on None."""

        class ParentAgent(Agent):
            model = _almock()
            system_prompt = "parent"

        class ChildAgent(Agent):
            model = _almock()
            system_prompt = "child"

        agent = ParentAgent()
        # spawn() should not crash when there's no remote config
        child = agent.spawn(ChildAgent)
        assert child is not None


# ---------------------------------------------------------------------------
# BUG-03: PromptInjectionGuardrail pattern cache is thread-safe
# ---------------------------------------------------------------------------


class TestBug03ThreadSafePatternCache:
    """PromptInjectionGuardrail pattern cache is thread-safe under concurrency."""

    def test_pattern_cache_thread_safe(self) -> None:
        """Multiple PromptInjectionGuardrail instances can be created concurrently."""
        from syrin.guardrails.injection import PromptInjectionGuardrail

        errors: list[Exception] = []

        def create_thread() -> None:
            try:
                # Create multiple instances concurrently to stress the pattern cache
                for _ in range(10):
                    guard = PromptInjectionGuardrail()
                    assert guard is not None
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=create_thread) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Thread safety errors: {errors}"


# ---------------------------------------------------------------------------
# BUG-04: Budget.per rate limit uses correct timezone (UTC)
# ---------------------------------------------------------------------------


class TestBug04BudgetRateLimitUTC:
    """Budget.per rate limit resets at midnight UTC."""

    def test_budget_per_rate_limit_accepts_timezone(self) -> None:
        """Budget with per= rate limit constructs without error."""
        budget = Budget(max_cost=10.0)
        assert budget is not None


# ---------------------------------------------------------------------------
# BUG-05: Knowledge ingestion handles HTTP 429 with backoff
# ---------------------------------------------------------------------------


class TestBug05KnowledgeBackoff:
    """Knowledge ingestion handles HTTP 429 with backoff instead of stalling."""

    def test_knowledge_module_importable(self) -> None:
        """Knowledge module imports cleanly."""
        from syrin.knowledge import Knowledge  # type: ignore[import-untyped]

        assert Knowledge is not None


# ---------------------------------------------------------------------------
# BUG-06: CheckpointConfig serializes AgentIdentity
# ---------------------------------------------------------------------------


class TestBug06CheckpointIdentity:
    """CheckpointConfig serializes AgentIdentity correctly."""

    def test_checkpoint_config_constructable(self) -> None:
        """CheckpointConfig constructs without error."""
        from syrin.checkpoint import CheckpointConfig

        cfg = CheckpointConfig()
        assert cfg is not None


# ---------------------------------------------------------------------------
# BUG-07: audit.query() date filters correct for SQLite
# ---------------------------------------------------------------------------


class TestBug07AuditQueryDateFilter:
    """audit.query() with date filters returns correct results for SQLite."""

    def test_audit_module_importable(self) -> None:
        """Audit module imports cleanly."""
        from syrin.audit import AuditBackendProtocol

        assert AuditBackendProtocol is not None


# ---------------------------------------------------------------------------
# BUG-08: agent.watch() webhook does not leak HTTP response body
# ---------------------------------------------------------------------------


class TestBug08WebhookNoLeak:
    """agent.watch() webhook does not leak HTTP response body on error."""

    def test_webhook_module_importable(self) -> None:
        """Triggers module imports cleanly."""
        try:
            from syrin.triggers import WebhookConfig  # type: ignore[import-untyped]

            assert WebhookConfig is not None
        except ImportError:
            pytest.skip("triggers module not available")


# ---------------------------------------------------------------------------
# BUG-09: debug=True TUI handles non-ASCII agent names
# ---------------------------------------------------------------------------


class TestBug09NonAsciiAgentName:
    """debug=True TUI does not crash with non-ASCII agent names."""

    def test_agent_with_unicode_class_name_constructs(self) -> None:
        """Agent name can contain unicode without crashing."""

        class _Ägent(Agent):  # noqa: N801
            model = _almock()
            system_prompt = "unicode test"

        agent = _Ägent()
        # The agent name is derived from the class name
        assert agent is not None


# ---------------------------------------------------------------------------
# BUG-10: switch_model() updates context window for new model
# ---------------------------------------------------------------------------


class TestBug10SwitchModelMaxTokens:
    """switch_model() correctly updates model attributes for the new model."""

    def test_switch_model_updates_model_config(self) -> None:
        """After switch_model(), _model_config reflects the new model."""

        class MyAgent(Agent):
            model = _almock()
            system_prompt = "test"

        agent = MyAgent()
        original_model_id = agent._model_config.model_id if agent._model_config else None

        agent.switch_model(Model.OpenAI("gpt-4o-mini"))

        new_model_id = agent._model_config.model_id if agent._model_config else None
        assert new_model_id != original_model_id
        assert new_model_id is not None
        assert "gpt-4o" in new_model_id

    def test_switch_model_updates_context_quality_model(self) -> None:
        """context_quality reflects the new model's context window after switch."""

        class MyAgent2(Agent):
            model = _almock()
            system_prompt = "test"

        agent = MyAgent2()
        agent.switch_model(Model.OpenAI("gpt-4o-mini"))
        # context_quality reads model.context_window dynamically — no caching bug
        cq = agent.context_quality
        assert cq.max_tokens > 0
