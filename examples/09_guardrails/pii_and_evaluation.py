"""PII Detection and Parallel Evaluation — Content filters, PII scanning, and chains.

Demonstrates:
- ContentFilter for blocked words
- PIIScanner for PII detection and redaction
- ParallelEvaluationEngine for running guardrails concurrently
- GuardrailChain for sequential evaluation
- Custom guardrails
- GuardrailDecision inspection (confidence, alternatives, metadata)

Run:
    python examples/09_guardrails/pii_and_evaluation.py
"""

from __future__ import annotations

import asyncio

from syrin import Agent, Model
from syrin.enums import GuardrailStage
from syrin.guardrails import (
    ContentFilter,
    Guardrail,
    GuardrailChain,
    GuardrailContext,
    GuardrailDecision,
    ParallelEvaluationEngine,
    PIIScanner,
)

model = Model.Almock()


async def example_content_filter() -> None:
    print("\n" + "=" * 55)
    print("1. ContentFilter — blocked words")
    print("=" * 55)

    guardrail = ContentFilter(
        blocked_words=["password", "secret", "api_key"], name="security_filter"
    )

    safe = GuardrailContext(text="Hello, how are you?", stage=GuardrailStage.INPUT)
    result = await guardrail.evaluate(safe)
    print(f"Safe text passed: {result.passed}")

    blocked = GuardrailContext(text="My password is 123456", stage=GuardrailStage.INPUT)
    result = await guardrail.evaluate(blocked)
    print(f"Blocked text passed: {result.passed}")
    print(f"Reason: {result.reason}")
    print(f"Alternatives: {result.alternatives}")


async def example_pii_scanner() -> None:
    print("\n" + "=" * 55)
    print("2. PIIScanner — detect and redact PII")
    print("=" * 55)

    scanner = PIIScanner(name="pii_scanner", redact=True, redaction_char="*")

    context = GuardrailContext(
        text="Contact me at john@example.com or call 555-123-4567",
        stage=GuardrailStage.INPUT,
    )
    result = await scanner.evaluate(context)
    print(f"PII detected: {not result.passed}")
    print(f"Reason: {result.reason}")
    print(f"Findings: {result.metadata.get('findings', [])}")
    print(f"Redacted: {result.metadata.get('redacted_text', '')}")


async def example_parallel_evaluation() -> None:
    print("\n" + "=" * 55)
    print("3. ParallelEvaluationEngine")
    print("=" * 55)

    engine = ParallelEvaluationEngine(timeout=5.0)

    context = GuardrailContext(text="This is a normal message", stage=GuardrailStage.INPUT)
    result = await engine.evaluate(context, [ContentFilter(blocked_words=["spam"]), PIIScanner()])
    print(f"All passed: {result.passed}")
    print(f"Latency: {result.total_latency_ms:.2f}ms")
    print(f"Checks: {len(result.decisions)}")

    context = GuardrailContext(text="This is spam and test@test.com", stage=GuardrailStage.INPUT)
    result = await engine.evaluate(context, [ContentFilter(blocked_words=["spam"]), PIIScanner()])
    print(f"\nBlocked: {not result.passed}")
    print(f"Rule: {result.rule}")


async def example_guardrail_chain() -> None:
    print("\n" + "=" * 55)
    print("4. GuardrailChain — sequential, stops on first failure")
    print("=" * 55)

    chain = GuardrailChain([ContentFilter(blocked_words=["blocked"]), PIIScanner()])

    context = GuardrailContext(
        text="Hello, my email is user@example.com", stage=GuardrailStage.INPUT
    )
    result = await chain.evaluate(context)
    print(f"Chain result: {result.passed}, decisions: {len(result.decisions)}")

    context = GuardrailContext(text="This is blocked content", stage=GuardrailStage.INPUT)
    result = await chain.evaluate(context)
    print(f"Blocked early: {not result.passed}, decisions: {len(result.decisions)}")


async def example_custom_guardrail() -> None:
    print("\n" + "=" * 55)
    print("5. Custom Guardrail")
    print("=" * 55)

    class LengthGuardrail(Guardrail):
        def __init__(self, min_length: int = 1, max_length: int = 1000, name: str | None = None):
            super().__init__(name)
            self.min_length = min_length
            self.max_length = max_length

        async def evaluate(self, context: GuardrailContext) -> GuardrailDecision:
            length = len(context.text)
            if length < self.min_length:
                return GuardrailDecision(
                    passed=False,
                    rule="too_short",
                    reason=f"Too short: {length} (min: {self.min_length})",
                )
            if length > self.max_length:
                return GuardrailDecision(
                    passed=False,
                    rule="too_long",
                    reason=f"Too long: {length} (max: {self.max_length})",
                )
            return GuardrailDecision(passed=True, rule="length_ok")

    guardrail = LengthGuardrail(min_length=5, max_length=50)

    for text, label in [
        ("Hi", "too short"),
        ("Hello world!", "just right"),
        ("A" * 100, "too long"),
    ]:
        ctx = GuardrailContext(text=text, stage=GuardrailStage.INPUT)
        result = await guardrail.evaluate(ctx)
        print(f"{label}: passed={result.passed}, length={len(text)}")


async def example_decision_inspection() -> None:
    print("\n" + "=" * 55)
    print("6. Decision Inspection")
    print("=" * 55)

    guardrail = ContentFilter(blocked_words=["forbidden"])
    context = GuardrailContext(text="This contains a forbidden word", stage=GuardrailStage.INPUT)
    result = await guardrail.evaluate(context)

    print(f"Passed: {result.passed}")
    print(f"Rule: {result.rule}")
    print(f"Reason: {result.reason}")
    print(f"Confidence: {result.confidence}")
    print(f"Action: {result.action}")
    print(f"Metadata: {result.metadata}")
    print(f"JSON: {result.to_json()}")


async def _run() -> None:
    await example_content_filter()
    await example_pii_scanner()
    await example_parallel_evaluation()
    await example_guardrail_chain()
    await example_custom_guardrail()
    await example_decision_inspection()


# Agent with guardrails applied as a list
class GuardrailDemoAgent(Agent):
    _agent_name = "guardrail-demo"
    _agent_description = "Agent with ContentFilter and PIIScanner guardrails"
    model = model
    system_prompt = "You are a helpful assistant."
    guardrails = [ContentFilter(blocked_words=["password", "secret"]), PIIScanner()]


if __name__ == "__main__":
    asyncio.run(_run())

    # --- Optional: serve with playground ---
    # agent = GuardrailDemoAgent()
    # print("Serving at http://localhost:8000/playground")
    # agent.serve(port=8000, enable_playground=True, debug=True)
