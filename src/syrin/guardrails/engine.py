"""Parallel evaluation engine for guardrails."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import cast

from syrin.guardrails.base import Guardrail
from syrin.guardrails.context import GuardrailContext
from syrin.guardrails.decision import GuardrailDecision
from syrin.guardrails.enums import DecisionAction


@dataclass
class EvaluationResult:
    """Result of evaluating multiple guardrails."""

    passed: bool
    """Overall result - True if all guardrails passed."""

    decisions: list[GuardrailDecision]
    """Individual decisions from each guardrail."""

    rule: str = ""
    """Primary rule that caused failure (if any)."""

    reason: str = ""
    """Primary reason for failure (if any)."""

    metadata: dict[str, object] = field(default_factory=dict)
    """Additional metadata about the evaluation."""

    total_latency_ms: float = 0.0
    """Total time for evaluation."""

    total_budget_consumed: float = 0.0
    """Total budget consumed across all guardrails."""

    def to_dict(self) -> dict[str, object]:
        """Convert result to dictionary."""
        return {
            "passed": self.passed,
            "rule": self.rule,
            "reason": self.reason,
            "decisions": [d.to_dict() for d in self.decisions],
            "metadata": self.metadata,
            "total_latency_ms": self.total_latency_ms,
            "total_budget_consumed": self.total_budget_consumed,
        }


class ParallelEvaluationEngine:
    """Engine for evaluating guardrails in parallel.

    Runs all guardrails concurrently to minimize latency.
    Supports timeouts and short-circuit on first failure.

    Example:
        >>> engine = ParallelEvaluationEngine(timeout=1.0)
        >>> result = await engine.evaluate(context, [guardrail1, guardrail2])
        >>> if not result.passed:
        ...     print(f"Blocked: {result.reason}")
    """

    def __init__(
        self, timeout: float = 30.0, short_circuit: bool = False, fail_on_timeout: bool = True
    ):
        """Initialize evaluation engine.

        Args:
            timeout: Maximum time (seconds) to wait for all guardrails. Default 30.
            short_circuit: If True, stop as soon as one guardrail fails.
            fail_on_timeout: If True, timeout counts as failure.
        """
        self.timeout = timeout
        self.short_circuit = short_circuit
        self.fail_on_timeout = fail_on_timeout

    async def evaluate(
        self, context: GuardrailContext, guardrails: list[Guardrail]
    ) -> EvaluationResult:
        """Evaluate all guardrails in parallel.

        Args:
            context: Context to evaluate against.
            guardrails: List of guardrails to evaluate.

        Returns:
            EvaluationResult with combined results.
        """
        start_time = time.time()

        # Handle empty guardrail list
        if not guardrails:
            return EvaluationResult(
                passed=True, decisions=[], total_latency_ms=0.0, total_budget_consumed=0.0
            )

        # Create evaluation tasks
        tasks: list[asyncio.Task[GuardrailDecision]] = []
        for guardrail in guardrails:
            task = asyncio.create_task(self._evaluate_guardrail(guardrail, context))
            tasks.append(task)

        # Run all tasks with timeout
        try:
            if self.short_circuit:
                # Short-circuit: return as soon as one fails
                decisions = await self._evaluate_with_short_circuit(tasks)
            else:
                # Normal: wait for all to complete
                results: list[GuardrailDecision | BaseException] = await asyncio.wait_for(
                    asyncio.gather(*tasks, return_exceptions=True),
                    timeout=self.timeout,
                )
                decisions = results
        except TimeoutError:
            # Timeout occurred
            if self.fail_on_timeout:
                elapsed = (time.time() - start_time) * 1000
                timeout_decision = GuardrailDecision(
                    passed=False,
                    rule="timeout",
                    reason=f"Guardrail evaluation timed out after {self.timeout}s",
                    action=DecisionAction.BLOCK,
                )
                return EvaluationResult(
                    passed=False,
                    decisions=[timeout_decision],
                    rule="timeout",
                    reason=f"Evaluation timed out after {self.timeout}s",
                    total_latency_ms=elapsed,
                    total_budget_consumed=0.0,
                )
            else:
                # Timeout passes
                decisions = []

        # Process results
        processed_decisions: list[GuardrailDecision] = []
        has_failure = False
        first_failure_rule = ""
        first_failure_reason = ""
        total_budget = 0.0

        for decision in decisions:
            if isinstance(decision, Exception):
                # Exception during evaluation
                error_decision = GuardrailDecision(
                    passed=False,
                    rule="error",
                    reason=f"Guardrail evaluation error: {str(decision)}",
                    action=DecisionAction.BLOCK,
                    metadata={"error": str(decision)},
                )
                processed_decisions.append(error_decision)
                has_failure = True
                if not first_failure_rule:
                    first_failure_rule = "error"
                    first_failure_reason = str(decision)
            else:
                decision_typed = cast(GuardrailDecision, decision)
                processed_decisions.append(decision_typed)
                total_budget += decision_typed.budget_consumed
                if not decision_typed.passed:
                    has_failure = True
                    if not first_failure_rule:
                        first_failure_rule = decision_typed.rule
                        first_failure_reason = decision_typed.reason

        elapsed = (time.time() - start_time) * 1000

        return EvaluationResult(
            passed=not has_failure,
            decisions=processed_decisions,
            rule=first_failure_rule,
            reason=first_failure_reason,
            total_latency_ms=elapsed,
            total_budget_consumed=total_budget,
        )

    async def _evaluate_guardrail(
        self, guardrail: Guardrail, context: GuardrailContext
    ) -> GuardrailDecision:
        """Evaluate a single guardrail with timing.

        Args:
            guardrail: Guardrail to evaluate.
            context: Evaluation context.

        Returns:
            GuardrailDecision with timing information.
        """
        start = time.time()

        try:
            decision = await guardrail.evaluate(context)
        except Exception as e:
            # Wrap exceptions in decision
            decision = GuardrailDecision(
                passed=False,
                rule="exception",
                reason=f"Guardrail '{guardrail.name}' raised exception: {str(e)}",
                action=DecisionAction.BLOCK,
                metadata={"exception": str(e), "guardrail": guardrail.name},
            )

        # Add timing
        decision.latency_ms = (time.time() - start) * 1000

        return decision

    async def _evaluate_with_short_circuit(
        self, tasks: list[asyncio.Task[GuardrailDecision]]
    ) -> list[GuardrailDecision | BaseException]:
        """Evaluate tasks and short-circuit on first failure.

        Args:
            tasks: List of evaluation tasks.

        Returns:
            List of decisions (may be incomplete if short-circuited).
        """
        # Run all tasks and return as soon as one fails
        pending = set(tasks)
        completed: list[GuardrailDecision | BaseException] = []

        while pending:
            # Wait for the next task to complete
            done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)

            for task in done:
                try:
                    result = await task
                    completed.append(result)

                    # Short-circuit on failure
                    if isinstance(result, GuardrailDecision) and not result.passed:
                        # Cancel remaining tasks
                        for p in pending:
                            p.cancel()
                        return completed

                except Exception as e:
                    completed.append(e)
                    # Short-circuit on exception
                    for p in pending:
                        p.cancel()
                    return completed

        return completed
