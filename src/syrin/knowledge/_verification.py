"""Shared verification: grade_results and verify_claim.

Used by both agentic RAG (_agentic) and grounding layer (_grounding).
"""

from __future__ import annotations

import asyncio
import re
from collections.abc import Callable
from typing import TYPE_CHECKING

from syrin.enums import Hook, MessageRole
from syrin.types import CostInfo, Message, TokenUsage

if TYPE_CHECKING:
    from syrin.budget import BudgetTracker
    from syrin.events import EventContext
    from syrin.knowledge._store import SearchResult
    from syrin.model import Model

_GRADE_PROMPT = """Rate the relevance of these search results for answering the question, from 0.0 (irrelevant) to 1.0 (highly relevant).
Reply with only a single float, e.g. 0.75

Question: {query}

Results:
{results}

Relevance (0.0-1.0):"""

_VERIFY_PROMPT = """Does the following evidence support, contradict, or provide no information about the claim?
Reply with exactly one word: SUPPORTED, CONTRADICTED, or NOT_FOUND.

Claim: {claim}

Evidence:
{evidence}

Verdict:"""


def _parse_grade(text: str) -> float:
    """Parse relevance score from LLM output."""
    if not text:
        return 0.0
    match = re.search(r"0?\.\d+|\d+\.\d+|\d+", text.strip())
    if match:
        try:
            v = float(match.group())
            return max(0.0, min(1.0, v))
        except ValueError:
            pass
    return 0.0


def _parse_verdict(text: str) -> str:
    """Parse SUPPORTED, CONTRADICTED, or NOT_FOUND from LLM output."""
    if not text:
        return "NOT_FOUND"
    upper = text.strip().upper()
    if "SUPPORTED" in upper:
        return "SUPPORTED"
    if "CONTRADICTED" in upper:
        return "CONTRADICTED"
    return "NOT_FOUND"


async def _call_model(
    model: Model,
    prompt: str,
    *,
    budget_tracker: BudgetTracker | None = None,
    timeout: float = 30.0,
) -> str:
    """Call model with a simple user prompt, return content string.

    Shared by agentic RAG and grounding layer. Records cost and tokens to
    budget_tracker when provided. On timeout returns empty string (callers fall back).
    """
    messages = [Message(role=MessageRole.USER, content=prompt)]

    async def _do_complete() -> str:
        response = await model.acomplete(messages, max_tokens=512, stream=False)
        if response is None or not hasattr(response, "content"):
            return ""
        if budget_tracker is not None:
            from syrin.cost import calculate_cost

            token_usage = getattr(response, "token_usage", None) or TokenUsage()
            if not isinstance(token_usage, TokenUsage):
                token_usage = TokenUsage(
                    input_tokens=getattr(token_usage, "input_tokens", 0),
                    output_tokens=getattr(token_usage, "output_tokens", 0),
                    total_tokens=getattr(token_usage, "total_tokens", 0),
                )
            model_id = getattr(model, "model_id", None) or str(
                getattr(model, "_model_id", "unknown")
            )
            pricing = getattr(model, "pricing", None)
            cost_usd = calculate_cost(model_id, token_usage, pricing_override=pricing)
            cost_info = CostInfo(
                token_usage=token_usage,
                cost_usd=cost_usd,
                model_name=model_id,
            )
            budget_tracker.record(cost_info)
        content = getattr(response, "content", None)
        return (content or "").strip() if isinstance(content, str) else ""

    try:
        return await asyncio.wait_for(_do_complete(), timeout=timeout)
    except TimeoutError:
        return ""


async def grade_results(
    query: str,
    results: list[SearchResult],
    model: Model,
    *,
    emit: Callable[[str, EventContext], None] | None = None,
    budget_tracker: BudgetTracker | None = None,
) -> float:
    """Grade relevance of search results for a query. Returns score in [0, 1]."""
    if not results:
        return 0.0
    previews = [
        f"[{r.rank}] (score={r.score:.2f}) {r.chunk.content[:200]}..."
        if len(r.chunk.content) > 200
        else f"[{r.rank}] (score={r.score:.2f}) {r.chunk.content}"
        for r in results[:5]
    ]
    results_text = "\n\n".join(previews)
    prompt = _GRADE_PROMPT.format(query=query, results=results_text)
    out = await _call_model(model, prompt, budget_tracker=budget_tracker)
    score = _parse_grade(out)
    if emit:
        from syrin.events import EventContext

        emit(Hook.KNOWLEDGE_AGENTIC_GRADE, EventContext({"query": query, "grade": score}))
    return score


async def verify_claim(
    claim: str,
    results: list[SearchResult],
    model: Model,
    *,
    emit: Callable[[str, EventContext], None] | None = None,
    budget_tracker: BudgetTracker | None = None,
) -> str:
    """Verify if evidence supports, contradicts, or does not address the claim.

    Returns "SUPPORTED", "CONTRADICTED", or "NOT_FOUND".
    """
    if not results:
        if emit:
            from syrin.events import EventContext

            emit(
                Hook.KNOWLEDGE_AGENTIC_VERIFY,
                EventContext({"claim": claim, "verdict": "NOT_FOUND"}),
            )
        return "NOT_FOUND"
    evidence = "\n\n".join(r.chunk.content[:300] for r in results[:5])
    prompt = _VERIFY_PROMPT.format(claim=claim, evidence=evidence)
    out = await _call_model(model, prompt, budget_tracker=budget_tracker)
    verdict = _parse_verdict(out)
    if emit:
        from syrin.events import EventContext

        emit(Hook.KNOWLEDGE_AGENTIC_VERIFY, EventContext({"claim": claim, "verdict": verdict}))
    return verdict
