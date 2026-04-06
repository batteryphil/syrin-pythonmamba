"""Serve adapter — converts AgentRouter to an agent-like serveable interface.

Enables build_router() to serve AgentRouter instances via HTTP/CLI/STDIO without
requiring users to write a wrapper agent.  Plain :class:`~syrin.agent.Agent`
instances are returned as-is.
"""

from __future__ import annotations

import asyncio
import contextlib
from typing import TYPE_CHECKING, cast

from syrin.budget import BudgetState
from syrin.events import EventContext, Events
from syrin.response import Response, StreamChunk
from syrin.types import TokenUsage

if TYPE_CHECKING:
    from syrin.agent import Agent
    from syrin.agent.agent_router import AgentRouter


def to_serveable(obj: Agent | AgentRouter) -> Agent:
    """Convert an Agent or AgentRouter to a serveable Agent-compatible object.

    Agent instances are returned as-is.  AgentRouter instances are wrapped in
    :class:`_AgentRouterAdapter` which exposes ``arun``, ``astream``, ``events``,
    and ``budget_state`` so they can be served via HTTP/CLI/STDIO.

    Args:
        obj: Agent or AgentRouter to convert.

    Returns:
        Agent-compatible object.

    Raises:
        TypeError: If ``obj`` is not an Agent or AgentRouter.
    """
    from syrin.agent import Agent
    from syrin.agent.agent_router import AgentRouter

    if isinstance(obj, Agent):
        return obj
    if isinstance(obj, AgentRouter):
        return cast("Agent", _AgentRouterAdapter(obj))
    raise TypeError(f"Expected Agent or AgentRouter, got {type(obj).__name__}")


def _budget_state_from_budget(budget: object) -> BudgetState | None:
    """Build BudgetState from a Budget object."""
    if budget is None or getattr(budget, "max_cost", None) is None:
        return None
    effective = budget.max_cost - getattr(budget, "reserve", 0)  # type: ignore[attr-defined]
    if effective <= 0:
        effective = budget.max_cost  # type: ignore[attr-defined]
    spent = getattr(budget, "_spent", 0.0)
    remaining = max(0.0, effective - spent)
    percent = (spent / effective * 100.0) if effective > 0 else 0.0
    return BudgetState(
        limit=effective,
        remaining=remaining,
        spent=spent,
        percent_used=round(percent, 2),
    )


class _AgentRouterAdapter:
    """Internal adapter: AgentRouter → agent-like interface for HTTP serving.

    Wraps an :class:`~syrin.agent.agent_router.AgentRouter` so it can be
    mounted via :func:`~syrin.serve.http.build_router` without subclassing
    :class:`~syrin.agent.Agent`.
    """

    def __init__(self, router: AgentRouter) -> None:
        self._router = router
        self.name = "agent-router"
        self.description = "AgentRouter — LLM-driven dynamic multi-agent orchestration"
        self.internal_agents = list(getattr(router, "_agent_names", {}).keys())
        self.tools: list[object] = []
        self._events = Events(lambda _h, _c: None)
        router.events.on_all(lambda h, c: self._forward_event(h, c))

    def _forward_event(self, hook: object, ctx: EventContext) -> None:
        self._events._trigger_before(hook, ctx)  # type: ignore[arg-type]
        self._events._trigger(hook, ctx)  # type: ignore[arg-type]
        self._events._trigger_after(hook, ctx)  # type: ignore[arg-type]

    @property
    def events(self) -> Events:
        """Router lifecycle events."""
        return self._events

    @property
    def budget_state(self) -> BudgetState | None:
        """Current budget utilisation, or ``None`` if no budget is set."""
        return _budget_state_from_budget(getattr(self._router, "_budget", None))

    async def arun(
        self,
        user_input: str,
        context: object = None,
        template_variables: dict[str, object] | None = None,
    ) -> Response[str]:
        """Async run delegating to ``AgentRouter.run()`` in a thread.

        Args:
            user_input: Task string for the router.
            context: Ignored (AgentRouter does not use context).
            template_variables: Ignored.

        Returns:
            :class:`~syrin.response.Response` from the router.
        """
        del context, template_variables
        return await asyncio.to_thread(self._router.run, user_input, "parallel")

    def run(self, user_input: str) -> Response[str]:
        """Synchronous run. Blocks until complete."""
        return asyncio.run(self.arun(user_input))

    async def astream(
        self,
        user_input: str,
        context: object = None,
        template_variables: dict[str, object] | None = None,
    ) -> object:
        """Async streaming run.  Emits intermediate hook chunks then the final result.

        Args:
            user_input: Task string for the router.
            context: Ignored.
            template_variables: Ignored.

        Yields:
            :class:`~syrin.response.StreamChunk` objects.
        """
        del context, template_variables

        hook_queue: asyncio.Queue[tuple[str, dict[str, object]]] = asyncio.Queue()

        def on_hook(h: object, c: object) -> None:
            h_val = getattr(h, "value", str(h))
            c_dict: dict[str, object] = dict(c) if hasattr(c, "items") else {}  # type: ignore[call-overload]
            with contextlib.suppress(asyncio.QueueFull):
                hook_queue.put_nowait((h_val, c_dict))

        self._router.events.on_all(on_hook)
        result_container: list[Response[str]] = []

        async def run_router() -> None:
            result_container.append(
                await asyncio.to_thread(self._router.run, user_input, "parallel")
            )

        task = asyncio.create_task(run_router())

        while not task.done():
            try:
                h, c = await asyncio.wait_for(hook_queue.get(), timeout=0.05)
                chunk = StreamChunk()
                object.__setattr__(chunk, "_hook", (h, c))
                yield chunk
            except TimeoutError:
                continue

        while not hook_queue.empty():
            try:
                h, c = hook_queue.get_nowait()
                chunk = StreamChunk()
                object.__setattr__(chunk, "_hook", (h, c))
                yield chunk
            except asyncio.QueueEmpty:
                break

        result = result_container[0]
        yield StreamChunk(
            index=0,
            text=result.content,
            accumulated_text=result.content,
            cost_so_far=result.cost,
            tokens_so_far=result.tokens or TokenUsage(),
            is_final=True,
            response=result,
        )
