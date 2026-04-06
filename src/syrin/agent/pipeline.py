"""Multi-agent helper utilities: parallel() and sequential() convenience functions.

These are lightweight helpers for running multiple agent-task pairs without
constructing a full :class:`~syrin.workflow.Workflow`.  Use them in scripts
and notebooks where a one-liner is preferable.

For production multi-agent orchestration use the canonical classes:

- :class:`~syrin.workflow.Workflow` — deterministic sequential/branching DAG.
- :class:`~syrin.swarm.Swarm` — concurrent agents with shared goal and budget.
- :class:`~syrin.agent.agent_router.AgentRouter` — LLM-driven dynamic routing.

Example::

    from syrin.agent.pipeline import parallel, sequential

    # Run two agents in parallel
    import asyncio
    results = asyncio.run(parallel([
        (researcher, "Research AI trends"),
        (analyst, "Analyse market data"),
    ]))

    # Run two agents sequentially, passing output forward
    result = sequential([
        (researcher, "Research AI trends"),
        (writer, "Write a report"),
    ])
    print(result.content)
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from syrin.enums import StopReason
from syrin.types import TokenUsage

if TYPE_CHECKING:
    from syrin.agent._core import Agent
    from syrin.response import Response


async def parallel(
    agents: list[tuple[Agent, str]],
) -> list[Response[str]]:
    """Run multiple agent tasks in parallel.

    Each ``(agent, task)`` pair runs concurrently.  Results are returned in
    the same order as the input list.

    Args:
        agents: List of ``(agent_instance, task_string)`` tuples.

    Returns:
        List of :class:`~syrin.response.Response` objects, one per agent, in
        input order.

    Example::

        import asyncio
        from syrin.agent.pipeline import parallel

        results = asyncio.run(parallel([
            (researcher, "Find AI trends"),
            (analyst, "Analyse findings"),
        ]))
        for r in results:
            print(r.content)
    """

    async def _run_one(agent: Agent, task: str) -> Response[str]:
        return await agent.arun(task)

    return await asyncio.gather(*[_run_one(agent, task) for agent, task in agents])


def sequential(
    agents: list[tuple[Agent, str]],
    pass_previous: bool = True,
) -> Response[str]:
    """Run multiple agent tasks sequentially.

    Agents run one after another.  When ``pass_previous=True`` (default), the
    previous agent's output is appended to the next agent's task as context.

    Args:
        agents: List of ``(agent_instance, task_string)`` tuples.
        pass_previous: Whether to append prior output as context to the next
            task.  Defaults to ``True``.

    Returns:
        :class:`~syrin.response.Response` from the last agent.

    Example::

        from syrin.agent.pipeline import sequential

        result = sequential([
            (researcher, "Research renewable energy"),
            (writer, "Write a summary"),
        ])
        print(result.content)
    """
    if not agents:
        return _empty_response()

    result: Response[str] | None = None
    for agent, task in agents:
        if pass_previous and result and result.content:
            full_task = f"{task}\n\nPrevious results:\n{result.content}"
        else:
            full_task = task
        result = agent.run(full_task)

    return result or _empty_response()


def _empty_response(model: str = "") -> Response[str]:
    """Return an empty Response with zero cost/tokens."""
    from syrin.response import Response  # noqa: PLC0415

    return Response(
        content="",
        raw="",
        cost=0.0,
        tokens=TokenUsage(),
        model=model,
        stop_reason=StopReason.END_TURN,
        trace=[],
    )


__all__ = ["parallel", "sequential"]
