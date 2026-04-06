"""ModelRouter and RoutingReason — intelligent model selection."""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from syrin.budget import Budget
from syrin.enums import Media
from syrin.exceptions import NoMatchingProfileError
from syrin.model import Model
from syrin.router.classifier import PromptClassifier
from syrin.router.enums import ComplexityTier, RoutingMode, TaskType
from syrin.router.modality import ModalityDetector
from syrin.router.protocols import ClassifierProtocol
from syrin.tool import ToolSpec
from syrin.types import Message

if TYPE_CHECKING:
    pass


@dataclass
class _RoutingProfile:
    """Internal routing profile. Built from Model via _profiles_from_models."""

    model: Model
    name: str
    strengths: list[TaskType]
    input_media: set[Media]
    output_media: set[Media]
    supports_tools: bool
    priority: int


logger = logging.getLogger(__name__)

# Only warn once per process when classification fails (e.g. missing sentence-transformers)
_classification_warned: set[str] = set()
_classification_warned_lock: threading.Lock = threading.Lock()


@dataclass
class RoutingReason:
    """Explains why a model was selected. Returned by ModelRouter.route().

    Example::

        model, task, reason = router.route("hello")
        print(reason.selected_model, reason.task_type, reason.reason)

    Attributes:
        selected_model: Profile name chosen (e.g., "claude-code").
        task_type: Detected or overridden task type.
        reason: Human-readable explanation.
        cost_estimate: Estimated cost in USD for the call.
        alternatives: Other profile names that could have been used.
        classification_confidence: 0.0–1.0; confidence in task classification.
        complexity_tier: LOW/MEDIUM/HIGH when classify_extended used. None otherwise.
        system_alignment_score: Prompt vs system alignment [0,1] when available.
        routing_latency_ms: Time taken for the routing decision (classification + selection).
    """

    selected_model: str
    task_type: TaskType
    reason: str
    cost_estimate: float
    alternatives: list[str]
    classification_confidence: float
    complexity_tier: ComplexityTier | None = None
    system_alignment_score: float | None = None
    routing_latency_ms: float = 0.0


class ModelRouter:
    """Intelligent model router. Selects the best model based on task, modality, cost, budget.

    Example::

        router = ModelRouter(
            models=[
                Model.Anthropic("claude-sonnet-4-5", strengths=[TaskType.CODE, TaskType.REASONING], profile_name="code"),
                Model.OpenAI("gpt-4o-mini", strengths=[TaskType.GENERAL], profile_name="general"),
            ],
            routing_mode=RoutingMode.AUTO,
        )
        model, task, reason = router.route("write a sort function")
    """

    def __init__(
        self,
        models: list[Model],
        *,
        routing_mode: RoutingMode = RoutingMode.AUTO,
        classifier: ClassifierProtocol | None = None,
        budget: Budget | None = None,
        budget_optimisation: bool = True,
        prefer_cheaper_below_budget_ratio: float = 0.20,
        force_cheapest_below_budget_ratio: float = 0.10,
        force_model: Model | None = None,
        routing_rule_callback: Callable[[str, TaskType, list[str]], str | None] | None = None,
    ) -> None:
        if not models and force_model is None:
            raise ValueError(
                "ModelRouter requires at least one model when force_model is not set. "
                "Add models or set force_model to bypass routing."
            )
        from syrin.router.agent_integration import _profiles_from_models

        self._profiles: list[_RoutingProfile] = _profiles_from_models(models)
        self._routing_mode = routing_mode
        self._classifier = classifier
        self._budget = budget
        self._prefer_cheap = budget_optimisation
        self._budget_low = prefer_cheaper_below_budget_ratio
        self._budget_critical = force_cheapest_below_budget_ratio
        self._force_model = force_model
        self._routing_rule_callback = routing_rule_callback
        self._modality_detector = ModalityDetector()
        self._pricing_cache: dict[str, tuple[float, float]] = {}
        for p in self._profiles:
            pricing = p.model.get_pricing() if hasattr(p.model, "get_pricing") else None
            if pricing is not None:
                self._pricing_cache[p.name] = (pricing.input_per_1m, pricing.output_per_1m)
            else:
                self._pricing_cache[p.name] = (0.0, 0.0)

    def _estimate_cost(
        self, profile: _RoutingProfile, input_tokens: int, output_tokens: int
    ) -> float:
        """Estimate cost in USD using cached pricing. Zero overhead per route."""
        inp, out = self._pricing_cache.get(profile.name, (0.0, 0.0))
        return round(
            (input_tokens / 1_000_000) * inp + (output_tokens / 1_000_000) * out,
            6,
        )

    def _get_classifier(self) -> ClassifierProtocol:
        if self._classifier is not None:
            return self._classifier
        return PromptClassifier()

    def _required_media(self, messages: list[Message] | None) -> set[Media]:
        if not messages:
            return {Media.TEXT}
        return self._modality_detector.detect(messages)

    def _filter_by_media(
        self, profiles: list[_RoutingProfile], required: set[Media]
    ) -> list[_RoutingProfile]:
        return [p for p in profiles if (p.input_media or {Media.TEXT}) >= required]

    def _filter_by_tools(
        self, profiles: list[_RoutingProfile], tools: list[ToolSpec] | None
    ) -> list[_RoutingProfile]:
        if not tools:
            return profiles
        return [p for p in profiles if p.supports_tools]

    def _filter_by_task(
        self, profiles: list[_RoutingProfile], task_type: TaskType
    ) -> list[_RoutingProfile]:
        return [p for p in profiles if task_type in p.strengths]

    def _budget_ratio(self) -> float | None:
        """Remaining budget / limit. None when budget.max_cost is None (per_hour/per_day not supported)."""
        if self._budget is None or self._budget.max_cost is None:
            return None
        remaining = self._budget.remaining
        limit = (self._budget.max_cost or 0) - self._budget.safety_margin
        if limit <= 0:
            return None
        return remaining / limit if remaining is not None else None

    def _estimate_tokens(self, prompt: str, context: dict[str, object] | None) -> tuple[int, int]:
        in_tok: int | None = None
        if context:
            val = context.get("input_tokens_estimate")
            if isinstance(val, int):
                in_tok = val
        if in_tok is None:
            in_tok = max(1, len(prompt or "") // 4)
        out_tok = 1024
        if context:
            val = context.get("max_output_tokens")
            if isinstance(val, int):
                out_tok = val
        return (in_tok, out_tok)

    def _primary_selection_reason(
        self,
        by_task: list[_RoutingProfile],
        ratio: float | None,
        complexity_tier: ComplexityTier | None,
        task_type: TaskType,
        callback_used: bool = False,
    ) -> str:
        """Human-readable reason for primary selection. Used by route_ordered."""
        if callback_used:
            return "Custom routing rule selected profile"
        selected = by_task[0]
        if self._prefer_cheap and ratio is not None and ratio < self._budget_critical:
            return "Budget critical; using cheapest capable model"
        if self._prefer_cheap and ratio is not None and ratio < self._budget_low:
            return "Budget low; preferring cheaper capable model"
        if complexity_tier == ComplexityTier.HIGH:
            return f"Complexity HIGH; selected highest-priority ({selected.name})"
        if self._routing_mode == RoutingMode.COST_FIRST:
            return "COST_FIRST: selected cheapest capable model"
        if self._routing_mode == RoutingMode.QUALITY_FIRST:
            return f"QUALITY_FIRST: selected highest-priority ({selected.name})"
        return f"Model specializes in {task_type.value} tasks"

    def select_model(
        self,
        prompt: str,
        *,
        context: dict[str, object] | None = None,
    ) -> Model:
        """Select the best model for the given prompt. Simplified; use route() for full reason."""
        model, _task, _reason = self.route(prompt, context=context)
        return model

    def route_ordered(
        self,
        prompt: str,
        *,
        tools: list[ToolSpec] | None = None,
        context: dict[str, object] | None = None,
        messages: list[Message] | None = None,
        task_override: TaskType | None = None,
        max_alternatives: int | None = None,
    ) -> list[tuple[Model, TaskType, RoutingReason]]:
        """Return ranked list of (model, task_type, reason). Use for fallback: try each until one succeeds.

        Same filtering and selection logic as route(), but returns all candidates in preference order.
        max_alternatives: cap total (1=primary only). None = all candidates.
        """
        t0 = time.perf_counter()
        if self._force_model is not None:
            in_tok, out_tok = self._estimate_tokens(prompt, context)
            cost = 0.0
            if hasattr(self._force_model, "estimate_cost"):
                cost = self._force_model.estimate_cost(in_tok, out_tok)
            latency_ms = round((time.perf_counter() - t0) * 1000, 2)
            return [
                (
                    self._force_model,
                    task_override or TaskType.GENERAL,
                    RoutingReason(
                        selected_model="force_model",
                        task_type=task_override or TaskType.GENERAL,
                        reason="Routing bypassed via force_model",
                        cost_estimate=cost,
                        alternatives=[],
                        classification_confidence=1.0,
                        complexity_tier=None,
                        system_alignment_score=None,
                        routing_latency_ms=latency_ms,
                    ),
                )
            ]
        if self._routing_mode == RoutingMode.MANUAL and task_override is None:
            raise ValueError(
                "RoutingMode.MANUAL requires task_override. "
                "Pass task_override=TaskType.CODE (or similar) to route()."
            )
        required_media = self._required_media(messages)
        candidates = self._filter_by_media(self._profiles, required_media)
        candidates = self._filter_by_tools(candidates, tools)
        if task_override is not None:
            task_type = task_override
            confidence = 1.0
            complexity_tier = None
            system_alignment_score = None
        else:
            classifier = self._get_classifier()
            system_prompt = None
            if messages:
                for m in messages:
                    role = getattr(m, "role", None)
                    content = getattr(m, "content", None)
                    if role == "system" and isinstance(content, str) and content.strip():
                        system_prompt = content.strip()
                        break
            try:
                if hasattr(classifier, "classify_extended"):
                    ext = classifier.classify_extended(prompt, system_prompt)
                    task_type = ext.task_type
                    confidence = ext.confidence
                    complexity_tier = ext.complexity_tier
                    system_alignment_score = ext.system_alignment_score
                else:
                    task_type, confidence = classifier.classify(prompt)
                    complexity_tier = None
                    system_alignment_score = None
            except Exception:
                task_type = classifier.low_confidence_fallback
                confidence = 0.0
                complexity_tier = None
                system_alignment_score = None
        by_task = self._filter_by_task(candidates, task_type)
        if not by_task:
            names = [p.name for p in self._profiles]
            raise NoMatchingProfileError(
                f"No profile supports TaskType.{task_type.name} and media {required_media}. "
                "Add a profile with matching strengths and input_media.",
                required_task_type=task_type,
                required_modalities=required_media,
                available_profiles=names,
            )
        profile_names = [p.name for p in by_task]
        callback_used = False
        if self._routing_rule_callback is not None:
            chosen = self._routing_rule_callback(prompt, task_type, profile_names)
            if chosen is not None:
                chosen_profile = next((p for p in by_task if p.name == chosen), None)
                if chosen_profile is not None:
                    rest = [p for p in by_task if p.name != chosen]
                    by_task = [chosen_profile] + rest
                    callback_used = True
                else:
                    logger.warning(
                        "Callback returned unknown profile %r; using default routing", chosen
                    )
        in_tok, out_tok = self._estimate_tokens(prompt, context)
        ratio = self._budget_ratio()
        if (
            self._prefer_cheap
            and ratio is not None
            and ratio < self._budget_critical
            or self._prefer_cheap
            and ratio is not None
            and ratio < self._budget_low
        ):
            by_task = sorted(by_task, key=lambda p: self._estimate_cost(p, in_tok, out_tok))
        elif complexity_tier == ComplexityTier.HIGH:
            by_task = sorted(by_task, key=lambda p: -p.priority)
        elif self._routing_mode == RoutingMode.COST_FIRST:
            by_task = sorted(by_task, key=lambda p: self._estimate_cost(p, in_tok, out_tok))
        elif self._routing_mode == RoutingMode.QUALITY_FIRST:
            by_task = sorted(by_task, key=lambda p: -p.priority)
        else:
            by_task = sorted(
                by_task,
                key=lambda p: (-p.priority, self._estimate_cost(p, in_tok, out_tok)),
            )
        primary_reason = self._primary_selection_reason(
            by_task, ratio, complexity_tier, task_type, callback_used
        )
        results: list[tuple[Model, TaskType, RoutingReason]] = []
        for i, p in enumerate(by_task):
            reason_str = primary_reason if i == 0 else f"Ranked candidate: {p.name}"
            model, tt, reason = self._make_result(
                p,
                by_task,
                task_type,
                confidence,
                prompt,
                context,
                reason_str,
                complexity_tier=complexity_tier,
                system_alignment_score=system_alignment_score,
            )
            results.append((model, tt, reason))
        if max_alternatives is not None and max_alternatives > 0:
            results = results[:max_alternatives]
        latency_ms = round((time.perf_counter() - t0) * 1000, 2)
        # Inject routing_latency_ms into the first result (primary)
        if results:
            model, task_t, reason_obj = results[0]
            results[0] = (
                model,
                task_t,
                RoutingReason(
                    selected_model=reason_obj.selected_model,
                    task_type=reason_obj.task_type,
                    reason=reason_obj.reason,
                    cost_estimate=reason_obj.cost_estimate,
                    alternatives=reason_obj.alternatives,
                    classification_confidence=reason_obj.classification_confidence,
                    complexity_tier=reason_obj.complexity_tier,
                    system_alignment_score=reason_obj.system_alignment_score,
                    routing_latency_ms=latency_ms,
                ),
            )
        return results

    def route(
        self,
        prompt: str,
        *,
        tools: list[ToolSpec] | None = None,
        context: dict[str, object] | None = None,
        messages: list[Message] | None = None,
        task_override: TaskType | None = None,
    ) -> tuple[Model, TaskType, RoutingReason]:
        """Full routing decision. Returns (model, task_type, routing_reason)."""
        results = self.route_ordered(
            prompt,
            tools=tools,
            context=context,
            messages=messages,
            task_override=task_override,
            max_alternatives=1,
        )
        return results[0]

    def _make_result(
        self,
        selected: _RoutingProfile,
        candidates: list[_RoutingProfile],
        task_type: TaskType,
        confidence: float,
        prompt: str,
        context: dict[str, object] | None,
        reason: str,
        *,
        complexity_tier: ComplexityTier | None = None,
        system_alignment_score: float | None = None,
        routing_latency_ms: float = 0.0,
    ) -> tuple[Model, TaskType, RoutingReason]:
        in_tok, out_tok = self._estimate_tokens(prompt, context)
        cost = self._estimate_cost(selected, in_tok, out_tok)
        alternatives = [p.name for p in candidates if p.name != selected.name]
        return (
            selected.model,
            task_type,
            RoutingReason(
                selected_model=selected.name,
                task_type=task_type,
                reason=reason,
                cost_estimate=cost,
                alternatives=alternatives,
                classification_confidence=confidence,
                complexity_tier=complexity_tier,
                system_alignment_score=system_alignment_score,
                routing_latency_ms=routing_latency_ms,
            ),
        )
