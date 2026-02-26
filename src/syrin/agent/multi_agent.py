"""Multi-agent patterns: Pipeline, Team, and orchestration utilities."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, TypeVar, cast

from syrin.agent import Agent
from syrin.audit import AuditHookHandler, AuditLog
from syrin.budget import Budget
from syrin.enums import DocFormat, Hook, StopReason
from syrin.events import EventContext, Events
from syrin.model import Model
from syrin.response import Response
from syrin.types import TokenUsage

_log = logging.getLogger(__name__)

T = TypeVar("T")


class PipelineBuilder:
    """Builder for pipeline execution. Executes sequentially by default when used.

    Usage:
        result = pipeline.run([...])          # Sequential (Response)
        result = pipeline.run([...]).parallel()  # Parallel (list[Response])
    """

    def __init__(
        self,
        pipeline: Pipeline,
        agents: list[type[Agent] | tuple[type[Agent], str]],
    ) -> None:
        self._pipeline = pipeline
        self._agents = agents

    def __call__(self) -> Response[str]:
        """Execute sequentially (default)."""
        return self._pipeline.run_sequential(self._agents)

    @property
    def content(self) -> str:
        """Get content from sequential execution."""
        return self._pipeline.run_sequential(self._agents).content

    @property
    def cost(self) -> float:
        """Get cost from sequential execution."""
        return self._pipeline.run_sequential(self._agents).cost

    @property
    def budget(self) -> Any:
        """Get budget from sequential execution."""
        return self._pipeline.run_sequential(self._agents).budget

    def sequential(self) -> Response[str]:
        """Execute sequentially (explicit)."""
        return self._pipeline.run_sequential(self._agents)

    def parallel(self) -> list[Response[str]]:
        """Execute in parallel."""
        return self._pipeline.run_parallel(self._agents)


class PipelineRun:
    """Builder for pipeline execution with fluent API.

    Provides chaining methods to specify execution mode:
        pipeline.run(agents)               # Run one after another (default)
        pipeline.run(agents).sequential()  # Run one after another (explicit)
        pipeline.run(agents).parallel()    # Run simultaneously
    """

    def __init__(
        self,
        pipeline: Pipeline,
        agents: list[type[Agent] | tuple[type[Agent], str]],
    ) -> None:
        """Initialize pipeline run.

        Args:
            pipeline: Parent pipeline instance
            agents: List of agent classes or (agent_class, task) tuples
        """
        self._pipeline = pipeline
        self._agents = agents

    def __call__(self) -> Response[str]:
        """Execute pipeline sequentially (default behavior).

        This allows calling pipeline.run(agents) directly without
        specifying .sequential().

        Returns:
            Response from the last agent
        """
        return self.sequential()

    def sequential(self) -> Response[str]:
        """Run agents sequentially, passing output of each as input to next.

        Returns:
            Response from the last agent
        """
        if not self._agents:
            return Response(
                content="",
                raw="",
                cost=0.0,
                tokens=TokenUsage(),
                model="",
                stop_reason=StopReason.END_TURN,
                trace=[],
            )

        pipeline = self._pipeline
        if hasattr(pipeline, "_emit_pipeline_hook"):
            pipeline._emit_pipeline_hook(
                Hook.PIPELINE_START,
                EventContext(agents=len(self._agents)),
            )

        result: Response[str] | None = None
        budget = pipeline._budget
        total_cost = 0.0

        for idx, item in enumerate(self._agents):
            if isinstance(item, tuple):
                agent_class, task = item
            else:
                agent_class = item
                task = ""

            agent_name = agent_class.__name__
            if hasattr(pipeline, "_emit_pipeline_hook"):
                pipeline._emit_pipeline_hook(
                    Hook.PIPELINE_AGENT_START,
                    EventContext(agent_type=agent_name, task=task, index=idx),
                )

            agent = agent_class(budget=budget) if budget else agent_class()

            if result and task:
                combined_input = f"{task}\n\nPrevious context: {result.content}"
                result = agent.response(combined_input)
            elif task:
                result = agent.response(task)

            if result:
                total_cost += result.cost
            if hasattr(pipeline, "_emit_pipeline_hook"):
                pipeline._emit_pipeline_hook(
                    Hook.PIPELINE_AGENT_COMPLETE,
                    EventContext(
                        agent_type=agent_name,
                        cost=result.cost if result else 0.0,
                        content_preview=(result.content[:200] if result and result.content else ""),
                    ),
                )

        if hasattr(pipeline, "_emit_pipeline_hook"):
            pipeline._emit_pipeline_hook(
                Hook.PIPELINE_END,
                EventContext(total_cost=total_cost),
            )

        return result or Response(
            content="",
            raw="",
            cost=0.0,
            tokens=TokenUsage(),
            model="",
            stop_reason=StopReason.END_TURN,
            trace=[],
        )

    def parallel(self) -> list[Response[str]]:
        """Run agents in parallel.

        Returns:
            List of responses from all agents
        """
        if not self._agents:
            return []

        pipeline = self._pipeline
        if hasattr(pipeline, "_emit_pipeline_hook"):
            pipeline._emit_pipeline_hook(
                Hook.PIPELINE_START,
                EventContext(agents=len(self._agents)),
            )

        results: list[Response[str]] = []
        budget = pipeline._budget

        for idx, item in enumerate(self._agents):
            if isinstance(item, tuple):
                agent_class, task = item
            else:
                agent_class = item
                task = ""

            agent_name = agent_class.__name__
            if hasattr(pipeline, "_emit_pipeline_hook"):
                pipeline._emit_pipeline_hook(
                    Hook.PIPELINE_AGENT_START,
                    EventContext(agent_type=agent_name, task=task, index=idx),
                )

            agent = agent_class(budget=budget) if budget else agent_class()

            if task:
                result = agent.response(task)
            else:
                result = Response(
                    content="",
                    raw="",
                    cost=0.0,
                    tokens=TokenUsage(),
                    model=agent_class.model.model_id if hasattr(agent_class, "model") else "",
                    stop_reason=StopReason.END_TURN,
                    trace=[],
                )

            results.append(result)
            if hasattr(pipeline, "_emit_pipeline_hook"):
                pipeline._emit_pipeline_hook(
                    Hook.PIPELINE_AGENT_COMPLETE,
                    EventContext(
                        agent_type=agent_name,
                        cost=result.cost,
                        content_preview=(result.content[:200] if result.content else ""),
                    ),
                )

        if hasattr(pipeline, "_emit_pipeline_hook"):
            total_cost = sum(r.cost for r in results)
            pipeline._emit_pipeline_hook(
                Hook.PIPELINE_END,
                EventContext(total_cost=total_cost),
            )

        return results

    async def sequential_async(self) -> Response[str]:
        """Run agents sequentially with async support."""
        if not self._agents:
            return Response(
                content="",
                raw="",
                cost=0.0,
                tokens=TokenUsage(),
                model="",
                stop_reason=StopReason.END_TURN,
                trace=[],
            )

        pipeline = self._pipeline
        if hasattr(pipeline, "_emit_pipeline_hook"):
            pipeline._emit_pipeline_hook(
                Hook.PIPELINE_START,
                EventContext(agents=len(self._agents)),
            )

        result: Response[str] | None = None
        budget = pipeline._budget
        total_cost = 0.0

        for idx, item in enumerate(self._agents):
            if isinstance(item, tuple):
                agent_class, task = item
            else:
                agent_class = item
                task = ""

            agent_name = agent_class.__name__
            if hasattr(pipeline, "_emit_pipeline_hook"):
                pipeline._emit_pipeline_hook(
                    Hook.PIPELINE_AGENT_START,
                    EventContext(agent_type=agent_name, task=task, index=idx),
                )

            agent = agent_class(budget=budget) if budget else agent_class()

            if result and task:
                combined_input = f"{task}\n\nPrevious context: {result.content}"
                result = await agent.arun(combined_input)
            elif task:
                result = await agent.arun(task)

            if result:
                total_cost += result.cost
            if hasattr(pipeline, "_emit_pipeline_hook"):
                pipeline._emit_pipeline_hook(
                    Hook.PIPELINE_AGENT_COMPLETE,
                    EventContext(
                        agent_type=agent_name,
                        cost=result.cost if result else 0.0,
                        content_preview=(result.content[:200] if result and result.content else ""),
                    ),
                )

        if hasattr(pipeline, "_emit_pipeline_hook"):
            pipeline._emit_pipeline_hook(
                Hook.PIPELINE_END,
                EventContext(total_cost=total_cost),
            )

        return result or Response(
            content="",
            raw="",
            cost=0.0,
            tokens=TokenUsage(),
            model="",
            stop_reason=StopReason.END_TURN,
            trace=[],
        )

    async def parallel_async(self) -> list[Response[str]]:
        """Run agents in parallel with async support."""
        if not self._agents:
            return []

        pipeline = self._pipeline
        if hasattr(pipeline, "_emit_pipeline_hook"):
            pipeline._emit_pipeline_hook(
                Hook.PIPELINE_START,
                EventContext(agents=len(self._agents)),
            )

        async def run_one(
            item: type[Agent] | tuple[type[Agent], str],
            idx: int,
        ) -> tuple[Response[str], str]:
            if isinstance(item, tuple):
                agent_class, task = item
            else:
                agent_class = item
                task = ""

            agent_name = agent_class.__name__
            if hasattr(pipeline, "_emit_pipeline_hook"):
                pipeline._emit_pipeline_hook(
                    Hook.PIPELINE_AGENT_START,
                    EventContext(agent_type=agent_name, task=task, index=idx),
                )

            budget = pipeline._budget
            agent = agent_class(budget=budget) if budget else agent_class()

            if task:
                result = await agent.arun(task)
            else:
                result = Response(
                    content="",
                    raw="",
                    cost=0.0,
                    tokens=TokenUsage(),
                    model="",
                    stop_reason=StopReason.END_TURN,
                    trace=[],
                )

            if hasattr(pipeline, "_emit_pipeline_hook"):
                pipeline._emit_pipeline_hook(
                    Hook.PIPELINE_AGENT_COMPLETE,
                    EventContext(
                        agent_type=agent_name,
                        cost=result.cost,
                        content_preview=(result.content[:200] if result.content else ""),
                    ),
                )
            return (result, agent_name)

        tasks = [run_one(item, idx) for idx, item in enumerate(self._agents)]
        gathered = await asyncio.gather(*tasks)
        results = [r[0] for r in gathered]

        if hasattr(pipeline, "_emit_pipeline_hook"):
            total_cost = sum(r.cost for r in results)
            pipeline._emit_pipeline_hook(
                Hook.PIPELINE_END,
                EventContext(total_cost=total_cost),
            )

        return results


class Pipeline:
    """Pipeline for running multiple agents sequentially or in parallel.

    Supports fluent API via run() method:
        pipeline.run(agents)               # Run one after another (default)
        pipeline.run(agents).sequential()  # Run one after another (explicit)
        pipeline.run(agents).parallel()    # Run simultaneously

    Also supports traditional API:
        pipeline.run_sequential(agents)
        pipeline.run_parallel(agents)
    """

    def __init__(
        self,
        budget: Budget | None = None,
        timeout: float | None = None,
        agents: list[type[Agent] | tuple[type[Agent], str]] | None = None,
        sequential: bool = True,
        debug: bool = False,  # Enable debug logging
        audit: AuditLog | None = None,
    ) -> None:
        """Initialize pipeline.

        Args:
            budget: Optional shared budget for all agents in pipeline
            timeout: Optional timeout in seconds for each agent run
            agents: Optional list of agents to run immediately
            sequential: Whether to run agents sequentially (default True) or in parallel
            debug: Enable debug logging to console
            audit: Optional AuditLog for pipeline-level audit
        """
        self._budget = budget
        self._timeout = timeout
        self._agents = agents
        self._sequential = sequential
        self._debug = debug
        self._audit = audit
        self._events = Events(self._emit_pipeline_hook)
        if audit is not None:
            if not isinstance(audit, AuditLog):
                raise TypeError(f"audit must be AuditLog or None, got {type(audit).__name__}.")
            audit_handler = AuditHookHandler(source="Pipeline", config=audit)
            self._events.on_all(audit_handler)

    def _emit_pipeline_hook(self, hook: Hook, ctx: EventContext) -> None:
        """Emit pipeline hook for observability and audit."""
        ctx["timestamp"] = time.time()
        self._events._trigger_before(hook, ctx)
        self._events._trigger(hook, ctx)
        self._events._trigger_after(hook, ctx)

    @property
    def events(self) -> Events:
        """Pipeline events for hooks (PIPELINE_START, PIPELINE_END, etc.)."""
        return self._events

    @property
    def agents(self) -> list[type[Agent] | tuple[type[Agent], str]] | None:
        """Get agents in the pipeline."""
        return self._agents

    def run(
        self,
        agents: list[type[Agent] | tuple[type[Agent], str]],
    ) -> PipelineBuilder:
        """Run agents. Use .parallel() for parallel execution.

        Args:
            agents: List of agent classes or (agent_class, task) tuples

        Returns:
            PipelineBuilder. Access .content/.cost for sequential result,
            or call .parallel() for parallel execution.

        Example:
            # Sequential execution (default) - access .content
            result = pipeline.run([
                (Researcher, "Research topic"),
                (Writer, "Write article"),
            ])
            print(result.content)

            # Parallel execution
            results = pipeline.run([
                (Agent1, "Task 1"),
                (Agent2, "Task 2"),
            ]).parallel()
        """
        return PipelineBuilder(self, agents)

    def run_sequential(
        self,
        agents: list[type[Agent] | tuple[type[Agent], str]],
    ) -> Response[str]:
        """Run agents sequentially (traditional API).

        Use pipeline.run(agents) for fluent API.
        """
        return PipelineRun(self, agents).sequential()

    def run_parallel(
        self,
        agents: list[type[Agent] | tuple[type[Agent], str]],
    ) -> list[Response[str]]:
        """Run agents in parallel (traditional API).

        Use pipeline.run(agents).parallel() for fluent API.
        """
        return PipelineRun(self, agents).parallel()

    async def run_sequential_async(
        self,
        agents: list[type[Agent] | tuple[type[Agent], str]],
    ) -> Response[str]:
        """Run agents sequentially with async support (traditional API)."""
        return await PipelineRun(self, agents).sequential_async()

    async def run_parallel_async(
        self,
        agents: list[type[Agent] | tuple[type[Agent], str]],
    ) -> list[Response[str]]:
        """Run agents in parallel with async support (traditional API)."""
        return await PipelineRun(self, agents).parallel_async()


class AgentTeam:
    """Team of agents that can collaborate and share context/budget."""

    def __init__(
        self,
        agents: list[Agent],
        budget: Budget | None = None,
        shared_memory: bool = False,
        max_agents: int | None = None,
    ) -> None:
        """Initialize agent team.

        Args:
            agents: List of agent instances
            budget: Optional budget. If budget.shared=True, all agents share this budget.
            shared_memory: Whether agents share persistent memory
            max_agents: Maximum number of agents allowed in the team (optional limit)
        """
        self._agents = agents
        self._budget = budget
        self._shared_memory = shared_memory
        self._shared_memory_backend = None
        self._max_agents = max_agents

        if budget and budget.shared:
            for agent in agents:
                agent._budget = budget

    @property
    def agents(self) -> list[Agent]:
        """Get all agents in the team."""
        return self._agents

    @property
    def max_agents(self) -> int | None:
        """Get maximum agents allowed in the team."""
        return self._max_agents

    @property
    def total_budget(self) -> float:
        """Get total budget across all agents."""
        if self._budget:
            return self._budget.run or 0.0
        total = 0.0
        for agent in self._agents:
            if agent._budget and agent._budget.run:
                total += agent._budget.run
        return total

    def select_agent(self, task: str) -> Agent:
        """Select the most appropriate agent for a task.

        Currently uses simple keyword matching. Can be extended with LLM-based selection.
        """
        task_lower = task.lower()

        agent_keywords = {
            "research": ["research", "find", "search", "investigate", "analyze"],
            "write": ["write", "compose", "draft", "create", "generate"],
            "code": ["code", "program", "implement", "debug", "fix"],
            "review": ["review", "check", "validate", "verify", "test"],
        }

        for agent in self._agents:
            agent_name = agent.__class__.__name__.lower()
            for keyword, matches in agent_keywords.items():
                if keyword in agent_name or any(m in task_lower for m in matches):
                    return agent

        return self._agents[0]

    def run_task(self, task: str, agent: Agent | None = None) -> Response[str]:
        """Run a task using selected agent or automatic selection."""
        target = agent or self.select_agent(task)
        return target.response(task)

    async def run_task_async(self, task: str, agent: Agent | None = None) -> Response[str]:
        """Run a task asynchronously."""
        target = agent or self.select_agent(task)
        return await target.arun(task)


async def parallel(
    agents: list[tuple[Agent, str]],
) -> list[Response[str]]:
    """Run multiple agent tasks in parallel.

    Args:
        agents: List of (agent, task) tuples

    Returns:
        List of responses
    """

    async def run_one(agent: Agent, task: str) -> Response[str]:
        return await agent.arun(task)

    tasks = [run_one(agent, task) for agent, task in agents]
    return await asyncio.gather(*tasks)


def sequential(
    agents: list[tuple[Agent, str]],
    pass_previous: bool = True,
) -> Response[str]:
    """Run multiple agent tasks sequentially.

    Args:
        agents: List of (agent, task) tuples
        pass_previous: Whether to pass previous agent's output to next

    Returns:
        Response from the last agent
    """
    if not agents:
        return Response(
            content="",
            raw="",
            cost=0.0,
            tokens=TokenUsage(),
            model="",
            stop_reason=StopReason.END_TURN,
            trace=[],
        )

    result: Response[str] | None = None

    for agent, task in agents:
        if result and pass_previous:
            combined = f"{task}\n\nContext: {result.content}"
            result = agent.response(combined)
        else:
            result = agent.response(task)

    return result or Response(
        content="",
        raw="",
        cost=0.0,
        tokens=TokenUsage(),
        model="",
        stop_reason=StopReason.END_TURN,
        trace=[],
    )


# =============================================================================
# Dynamic Pipeline - LLM decides agents to spawn with configurable format
# =============================================================================


class DynamicPipeline:
    """Pipeline where LLM decides how many and what agents to spawn.

    This is a truly agentic feature - the LLM analyzes the task and decides:
    1. What specialized agents are needed
    2. How many agents to spawn
    3. What each agent should do
    4. Execution order (parallel or sequential)

    Usage:
        pipeline = DynamicPipeline(
            agents=[ResearcherAgent, AnalystAgent, WriterAgent],
            format=DocFormat.TOON,  # Use TOON for ~40% fewer tokens
            max_parallel=5,  # Max agents to spawn in parallel
        )
        result = pipeline.run("Research AI market and create a report")

    Or with custom agent names:
        class MyAgent(Agent):
            _syrin_name = "research"  # Custom name
    """

    def __init__(
        self,
        agents: list[type[Agent]] | None = None,
        budget: Budget | None = None,
        model: Model | None = None,
        format: DocFormat = DocFormat.TOON,  # Default to TOON
        max_parallel: int = 10,  # Max agents to spawn in parallel
        debug: bool = False,  # Enable debug logging
        audit: AuditLog | None = None,
    ):
        """Initialize DynamicPipeline.

        Args:
            agents: List of Agent classes available for spawning.
                Each agent's name is:
                - The `_syrin_name` attribute if set, otherwise
                - The class name (lowercase)
                Example: ResearcherAgent → "researcher"
            budget: Optional budget for all spawned agents
            model: REQUIRED - Model to use for the orchestrator agent.
                Must be a Model instance (e.g., Model(provider="openai", model_id="gpt-4o-mini"))
            format: Format for internal agent communication (TOON, JSON, YAML).
                Default is TOON for ~40% fewer tokens.
            max_parallel: Maximum number of agents to spawn in parallel.
                This is a limit, not a guarantee - LLM decides how many to actually use.
            debug: Enable debug logging to console
            audit: Optional AuditLog for dynamic pipeline audit
        """
        if model is None:
            raise ValueError(
                "model is required - pass a Model instance (e.g., Model(provider='openai', model_id='gpt-4o-mini'))"
            )

        self._agents = agents or []
        self._budget = budget
        self._model = model
        self._format = format
        self._max_parallel = max_parallel
        self._debug = debug
        self._run_metrics: dict[str, Any] = {}

        # Build agent name mapping
        self._agent_names: dict[str, type[Agent]] = {}
        for agent_class in self._agents:
            name = getattr(agent_class, "_syrin_name", None)
            if name is None:
                name = agent_class.__name__.lower()
            self._agent_names[name] = agent_class

        # Events system for observability
        self._events = Events(self._emit_hook)

        # Audit logging
        if audit is not None:
            if not isinstance(audit, AuditLog):
                raise TypeError(f"audit must be AuditLog or None, got {type(audit).__name__}.")
            audit_handler = AuditHookHandler(source="DynamicPipeline", config=audit)
            self._events.on_all(audit_handler)

    @property
    def events(self) -> Events:
        """Access events for registering hooks.

        Usage:
            pipeline.events.on(Hook.DYNAMIC_PIPELINE_START, lambda ctx: print(f"Starting: {ctx.task}"))
            pipeline.events.on(Hook.DYNAMIC_PIPELINE_AGENT_SPAWN, lambda ctx: print(f"Spawned: {ctx.agent_type}"))
            pipeline.events.on(Hook.DYNAMIC_PIPELINE_END, lambda ctx: print(f"Done! Cost: {ctx.total_cost}"))
        """
        return self._events

    def _emit_hook(self, hook: Hook, ctx: EventContext) -> None:
        """Emit a hook event with before/after handlers."""
        ctx["pipeline_id"] = id(self)
        ctx["timestamp"] = time.time()

        # Print debug event to console
        if self._debug:
            self._print_event(hook, ctx)

        # Trigger before handlers (can modify ctx)
        self._events._trigger_before(hook, ctx)

        # Trigger main handlers
        self._events._trigger(hook, ctx)

        # Trigger after handlers
        self._events._trigger_after(hook, ctx)

        _log.debug(
            f"Pipeline hook: {hook.value} - before:{len(self._events._before_handlers[hook])} main:{len(self._events._handlers[hook])} after:{len(self._events._after_handlers[hook])}"
        )

    def _print_event(self, hook: Hook, ctx: EventContext) -> None:
        """Print event to console when debug=True."""
        import sys
        from datetime import datetime

        is_tty = sys.stdout.isatty()

        RESET = "\033[0m" if is_tty else ""
        GREEN = "\033[92m" if is_tty else ""
        BLUE = "\033[94m" if is_tty else ""
        YELLOW = "\033[93m" if is_tty else ""
        CYAN = "\033[96m" if is_tty else ""
        RED = "\033[91m" if is_tty else ""

        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]

        event_str = str(hook.value)
        if "start" in event_str or "init" in event_str:
            color = GREEN
            symbol = "▶"
        elif "end" in event_str or "complete" in event_str:
            color = BLUE
            symbol = "✓"
        elif "spawn" in event_str or "handoff" in event_str:
            color = CYAN
            symbol = "→"
        elif "plan" in event_str:
            color = YELLOW
            symbol = "◉"
        elif "error" in event_str:
            color = RED
            symbol = "✗"
        else:
            color = ""
            symbol = "•"

        print(f"{color}{symbol} {timestamp} {hook.value}{RESET}")

        indent = "     "

        if "task" in ctx:
            task = ctx["task"]
            if isinstance(task, str) and len(task) > 60:
                task = task[:57] + "..."
            print(f"{indent}Task: {task}")

        if "agent_type" in ctx:
            print(f"{indent}Agent: {ctx['agent_type']}")

        if "model" in ctx:
            print(f"{indent}Model: {ctx['model']}")

        if "total_cost" in ctx and ctx["total_cost"] is not None:
            cost_val = float(ctx["total_cost"])
            if cost_val > 0:
                print(f"{indent}Total cost: ${cost_val:.6f}")

        if "error" in ctx:
            print(f"{indent}{RED}Error: {ctx['error']}{RESET}")

        print()

    def _get_agent_description(self, agent_class: type[Agent]) -> str:
        """Get description for an agent class."""
        name = getattr(agent_class, "_syrin_name", None) or agent_class.__name__.lower()
        prompt = getattr(agent_class, "system_prompt", "Specialized agent")[:100]
        return f"- {name}: {prompt}"

    def _format_to_schema(self, format: str) -> str:
        """Convert agent spec to the configured format."""
        if format.lower() == "toon" or (hasattr(format, "value") and format.value == "toon"):
            # TOON format
            return """```
@spawn
agents:
- type: <agent_name>
  task: <what this agent should do>
```"""
        elif format.lower() == "json" or (hasattr(format, "value") and format.value == "json"):
            return """```json
[
  {"type": "<agent_name>", "task": "<what this agent should do>"}
]
```"""
        else:
            # Default to TOON
            return """```
@spawn
agents:
- type: <agent_name>
  task: <what this agent should do>
```"""

    def run(
        self,
        task: str,
        mode: str = "parallel",  # "parallel" or "sequential"
    ) -> Response[str]:
        """Run dynamic agent pipeline using two-step approach.

        Step 1: Ask LLM to plan which agents to spawn (returns JSON)
        Step 2: Execute the planned agents explicitly

        Args:
            task: The task to complete
            mode: Execution mode - "parallel" (default) or "sequential"
                - parallel: All agents run simultaneously, results combined
                - sequential: Agents run one after another, each gets previous output

        Returns:
            Response with consolidated results from all agents
        """
        self._run_metrics = {
            "task": task,
            "mode": mode,
            "start_time": time.time(),
            "agents_spawned": [],
            "total_cost": 0.0,
            "total_tokens": 0,
        }

        # Emit DYNAMIC_PIPELINE_START
        self._emit_hook(
            Hook.DYNAMIC_PIPELINE_START,
            EventContext(
                task=task,
                mode=mode,
                model=self._model.model_id,
                available_agents=list(self._agent_names.keys()),
                budget_remaining=self._budget.remaining if self._budget else None,
            ),
        )

        try:
            # Step 1: Get agent plan from LLM
            plan = self._get_agent_plan(task)

            # Emit DYNAMIC_PIPELINE_PLAN
            self._emit_hook(
                Hook.DYNAMIC_PIPELINE_PLAN,
                EventContext(
                    task=task,
                    plan=plan,
                    plan_count=len(plan),
                ),
            )

            # Step 2: Execute the planned agents
            result = self._execute_plan(plan, mode)

            # Emit DYNAMIC_PIPELINE_END
            self._run_metrics["end_time"] = time.time()
            self._emit_hook(
                Hook.DYNAMIC_PIPELINE_END,
                EventContext(
                    task=task,
                    mode=mode,
                    agents_spawned=self._run_metrics["agents_spawned"],
                    total_cost=self._run_metrics["total_cost"],
                    total_tokens=self._run_metrics["total_tokens"],
                    duration=self._run_metrics["end_time"] - self._run_metrics["start_time"],
                    result_preview=result.content[:200] if result.content else "",
                ),
            )

            return result

        except Exception as e:
            self._emit_hook(
                Hook.DYNAMIC_PIPELINE_ERROR,
                EventContext(
                    task=task,
                    mode=mode,
                    error=str(e),
                    error_type=type(e).__name__,
                    agents_spawned=self._run_metrics["agents_spawned"],
                    total_cost=self._run_metrics["total_cost"],
                ),
            )
            raise

    def _get_agent_plan(self, task: str) -> list[dict[str, Any]]:
        """Step 1: Ask LLM to plan which agents to spawn.

        Returns a list of agent specs in JSON format.
        """
        from syrin.tool import tool

        agent_descriptions = [self._get_agent_description(a) for a in self._agents]
        agent_list_str = "\n".join(agent_descriptions) or "No agents available"

        @tool(name="plan", description="Plan which agents to spawn for this task")  # type: ignore[untyped-decorator,operator]
        def plan_agents(plan: str) -> str:
            """Return the agent plan as JSON."""
            return plan

        system_prompt = f"""Analyze this task and decide which agents to spawn.

Available agents:
{agent_list_str}

Maximum agents you can spawn: {self._max_parallel}

Return your plan as JSON array:
[
  {{"type": "agent_name", "task": "what this agent should do"}},
  ...
]

IMPORTANT: Return ONLY valid JSON, no other text."""

        orchestrator = Agent(
            model=self._model,
            system_prompt=system_prompt,
            tools=[plan_agents],
            budget=self._budget,
        )

        # Ask LLM to generate plan - it will use the tool to return JSON
        response = orchestrator.response(task)

        # Parse the JSON from response
        return self._parse_plan(response.content)

    def _parse_plan(self, content: str) -> list[dict[str, Any]]:
        """Parse agent plan from LLM response."""
        import json

        # Try to extract JSON from response
        content = content.strip()

        # Handle JSON wrapped in markdown
        if "```json" in content:
            start = content.find("```json") + 7
            end = content.find("```", start)
            content = content[start:end].strip()
        elif "```" in content:
            start = content.find("```") + 3
            end = content.find("```", start)
            content = content[start:end].strip()

        try:
            plan = json.loads(content)
            if isinstance(plan, list):
                return cast(list[dict[str, Any]], plan)
            elif isinstance(plan, dict) and "agents" in plan:
                agents = cast(list[dict[str, Any]], plan["agents"])
                return agents
        except json.JSONDecodeError:
            pass

        # Fallback: try to parse from text
        return self._parse_agents_spec(content)

    def _execute_plan(self, plan: list[dict[str, Any]], mode: str) -> Response[str]:
        """Step 2: Execute the planned agents."""
        if not plan:
            return Response(content="No agents to spawn", cost=0, tokens=TokenUsage())

        # Emit DYNAMIC_PIPELINE_EXECUTE
        self._emit_hook(
            Hook.DYNAMIC_PIPELINE_EXECUTE,
            EventContext(
                plan=plan,
                plan_count=len(plan),
                mode=mode,
            ),
        )

        if mode == "sequential":
            result_content, cost, tokens = self._execute_sequential(plan)
        else:
            result_content, cost, tokens = self._execute_parallel(plan)

        # Return a consolidated response
        return Response(
            content=result_content,
            cost=cost,
            tokens=tokens,
        )

    def _parse_agents_spec(self, spec: str) -> list[dict[str, Any]]:
        """Parse agent specification string."""
        agents = []
        lines = spec.strip().split("\n")

        current = {}
        in_agents = False

        for line in lines:
            line = line.strip()

            if line.startswith("-"):
                in_agents = True
                if "type:" in line:
                    parts = line.split("type:")
                    if len(parts) > 1:
                        agent_type = parts[1].strip().strip("-").strip().strip('"').strip("'")
                        current = {"type": agent_type}
            elif in_agents and "task:" in line:
                parts = line.split("task:")
                if len(parts) > 1:
                    current["task"] = parts[1].strip()
                    agents.append(current)
                    current = {}
            elif in_agents and line and not line.startswith("#") and not line.startswith("```"):
                if current and "task" in current:
                    current["task"] += " " + line

        return agents

    def _execute_parallel(self, agents_spec: list[dict[str, Any]]) -> tuple[str, float, TokenUsage]:
        """Execute agents in parallel."""

        async def run() -> list[Response[str]]:
            tasks: list[tuple[Agent, str]] = []
            for spec in agents_spec[: self._max_parallel]:
                agent_type = spec.get("type", "").lower()
                task = spec.get("task", "")

                agent_class = self._agent_names.get(agent_type)
                if not agent_class:
                    continue

                # Emit SPAWN hook
                spawn_time = time.time()
                self._emit_hook(
                    Hook.DYNAMIC_PIPELINE_AGENT_SPAWN,
                    EventContext(
                        agent_type=agent_type,
                        task=task,
                        spawn_time=spawn_time,
                        execution_mode="parallel",
                    ),
                )

                tasks.append((agent_class(budget=self._budget), task))

            return await parallel(tasks)

        results = asyncio.run(run())

        total_cost = 0.0
        total_tokens = 0
        for i, result in enumerate(results):
            if i < len(agents_spec):
                spec = agents_spec[i]
                agent_type = spec.get("type", "").lower()

                # Emit COMPLETE hook
                self._emit_hook(
                    Hook.DYNAMIC_PIPELINE_AGENT_COMPLETE,
                    EventContext(
                        agent_type=agent_type,
                        task=spec.get("task", ""),
                        result_preview=result.content[:200] if result.content else "",
                        cost=result.cost,
                        tokens=result.tokens.total_tokens,
                        duration=time.time() - self._run_metrics.get("start_time", time.time()),
                    ),
                )

                self._run_metrics["agents_spawned"].append(agent_type)
                self._run_metrics["total_cost"] += result.cost
                self._run_metrics["total_tokens"] += result.tokens.total_tokens
                total_cost += result.cost
                total_tokens += result.tokens.total_tokens

        return self._consolidate_results(results), total_cost, TokenUsage(total_tokens=total_tokens)

    def _execute_sequential(
        self, agents_spec: list[dict[str, Any]]
    ) -> tuple[str, float, TokenUsage]:
        """Execute agents sequentially, passing context to next."""
        results = []
        previous_output = ""
        total_cost = 0.0
        total_tokens = 0

        for spec in agents_spec[: self._max_parallel]:
            agent_type = spec.get("type", "").lower()
            task = spec.get("task", "")
            agent_start_time = time.time()

            agent_class = self._agent_names.get(agent_type)
            if not agent_class:
                continue

            # Emit SPAWN hook
            self._emit_hook(
                Hook.DYNAMIC_PIPELINE_AGENT_SPAWN,
                EventContext(
                    agent_type=agent_type,
                    task=task,
                    spawn_time=agent_start_time,
                    execution_mode="sequential",
                    previous_output_preview=previous_output[:100] if previous_output else None,
                ),
            )

            # Add previous output to context
            if previous_output:
                full_task = f"{task}\n\nPrevious results:\n{previous_output}"
            else:
                full_task = task

            agent = agent_class(budget=self._budget)
            result = agent.response(full_task)
            results.append(result)

            agent_duration = time.time() - agent_start_time

            # Emit COMPLETE hook
            self._emit_hook(
                Hook.DYNAMIC_PIPELINE_AGENT_COMPLETE,
                EventContext(
                    agent_type=agent_type,
                    task=task,
                    result_preview=result.content[:200] if result.content else "",
                    cost=result.cost,
                    tokens=result.tokens.total_tokens,
                    duration=agent_duration,
                    passed_context=bool(previous_output),
                ),
            )

            self._run_metrics["agents_spawned"].append(agent_type)
            self._run_metrics["total_cost"] += result.cost
            self._run_metrics["total_tokens"] += result.tokens.total_tokens
            total_cost += result.cost
            total_tokens += result.tokens.total_tokens
            previous_output = result.content

        return self._consolidate_results(results), total_cost, TokenUsage(total_tokens=total_tokens)

    def _consolidate_results(self, results: list[Response[str]]) -> str:
        """Consolidate results from multiple agents."""
        if not results:
            return "No results"

        consolidated = "=== AGENT RESULTS ===\n\n"

        for i, result in enumerate(results):
            consolidated += f"--- Agent {i + 1} ---\n"
            consolidated += result.content + "\n\n"
            consolidated += f"[Cost: ${result.cost:.6f}]\n\n"

        total_cost = sum(r.cost for r in results)
        total_tokens = sum(r.tokens.total_tokens for r in results)

        consolidated += f"=== TOTAL: {len(results)} agents ===\n"
        consolidated += f"Total cost: ${total_cost:.6f}\n"
        consolidated += f"Total tokens: {total_tokens}\n"

        return consolidated


__all__ = [
    "Pipeline",
    "PipelineBuilder",
    "PipelineRun",
    "AgentTeam",
    "DynamicPipeline",
    "parallel",
    "sequential",
]
