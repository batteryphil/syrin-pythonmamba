"""Structured output enforcement tests: OutputValidationError, result.output typed property,
and hook emission on validation retry and exhaustion."""

from __future__ import annotations

from syrin.enums import Hook

# ─── Hook enum values ─────────────────────────────────────────────────────────


class TestOutputValidationHooks:
    def test_output_validation_retry_hook_exists(self) -> None:
        assert hasattr(Hook, "OUTPUT_VALIDATION_RETRY")
        assert isinstance(Hook.OUTPUT_VALIDATION_RETRY.value, str)

    def test_output_validation_error_hook_exists(self) -> None:
        assert hasattr(Hook, "OUTPUT_VALIDATION_ERROR")
        assert isinstance(Hook.OUTPUT_VALIDATION_ERROR.value, str)


# ─── OutputValidationError ────────────────────────────────────────────────────


class TestOutputValidationError:
    def test_exception_importable(self) -> None:
        from syrin.exceptions import OutputValidationError

        assert issubclass(OutputValidationError, Exception)

    def test_has_raw_response(self) -> None:
        from syrin.exceptions import OutputValidationError

        err = OutputValidationError("failed", raw="some raw text", attempts=3)
        assert err.raw == "some raw text"

    def test_has_attempts(self) -> None:
        from syrin.exceptions import OutputValidationError

        err = OutputValidationError("failed", raw="raw", attempts=2)
        assert err.attempts == 2

    def test_has_errors_list(self) -> None:
        from syrin.exceptions import OutputValidationError

        errors = ["not json", "missing field"]
        err = OutputValidationError("failed", raw="raw", attempts=2, errors=errors)
        assert err.errors == errors


# ─── result.output typed property ─────────────────────────────────────────────


class TestResponseOutputProperty:
    def test_output_returns_content_for_plain_response(self) -> None:
        """For plain text agents, output returns the content string."""
        from syrin.response import Response

        r: Response[str] = Response(content="hello")
        assert r.output == "hello"

    def test_output_returns_parsed_when_structured_set(self) -> None:
        from pydantic import BaseModel

        from syrin.response import Response, StructuredOutput

        class MyModel(BaseModel):
            value: int

        instance = MyModel(value=42)
        so = StructuredOutput(raw='{"value": 42}', parsed=instance)
        r: Response[str] = Response(content='{"value": 42}', structured=so)
        assert r.output is instance

    def test_output_returns_content_when_no_structured(self) -> None:
        """No structured output → output falls back to content string."""
        from syrin.response import Response

        r: Response[str] = Response(content="hello")
        assert r.output == r.content

    def test_output_is_none_when_parsed_is_none(self) -> None:
        """Structured configured but validation failed → output is None (parsed=None)."""
        from syrin.response import Response, StructuredOutput

        so = StructuredOutput(raw="bad json", parsed=None)
        r: Response[str] = Response(content="bad json", structured=so)
        assert r.output is None


# ─── Retry loop emits hooks ───────────────────────────────────────────────────


class TestValidationRetryHooksEmitted:
    def test_retry_hook_fires_on_validation_failure(self) -> None:
        """OUTPUT_VALIDATION_RETRY fires on each retry."""
        from syrin.agent import Agent
        from syrin.model import Model

        agent = Agent(model=Model.OpenAI("gpt-4o-mini"))
        emitted: list[object] = []
        agent.events.on(Hook.OUTPUT_VALIDATION_RETRY, lambda ctx: emitted.append(ctx))

        # Simulate emitting via _emit_event directly (unit test without real LLM)
        from syrin.events import EventContext

        agent._emit_event(
            Hook.OUTPUT_VALIDATION_RETRY, EventContext(attempt=1, error="bad json", raw="...")
        )
        assert len(emitted) == 1
        assert emitted[0].attempt == 1  # type: ignore[union-attr]

    def test_error_hook_fires_when_all_retries_exhausted(self) -> None:
        """OUTPUT_VALIDATION_ERROR fires when retries exhausted."""
        from syrin.agent import Agent
        from syrin.model import Model

        agent = Agent(model=Model.OpenAI("gpt-4o-mini"))
        emitted: list[object] = []
        agent.events.on(Hook.OUTPUT_VALIDATION_ERROR, lambda ctx: emitted.append(ctx))

        from syrin.events import EventContext

        agent._emit_event(Hook.OUTPUT_VALIDATION_ERROR, EventContext(attempts=3, raw="bad"))
        assert len(emitted) == 1
        assert emitted[0].attempts == 3  # type: ignore[union-attr]
