"""AgentRouter — LLM-driven dynamic multi-agent orchestration.

The LLM analyses the task and decides which agents to spawn, how many, and in
what order.

.. note:: **AgentRouter vs. ModelRouter**

    ``AgentRouter`` (this module) routes *tasks* to different *agent classes* —
    the LLM picks which specialist agents to use.

    :class:`~syrin.router.router.ModelRouter` (``syrin.router``) routes *LLM calls*
    to different *models* — cheapest capable model wins at inference time.

    They are orthogonal and can be combined.

.. note:: **AgentRouter vs. Workflow vs. Swarm**

    * :class:`~syrin.workflow._core.Workflow` — deterministic DAG with explicit
      steps, branching, and lifecycle control.  Use when agent order is known.
    * :class:`~syrin.swarm._core.Swarm` — concurrent agents sharing a goal and
      budget, with multiple topologies (PARALLEL, CONSENSUS, REFLECTION, etc.).
    * ``AgentRouter`` — dynamic: the LLM decides which agents to invoke and in
      what order.  Use when agent selection depends on task content.

Example::

    from syrin import AgentRouter, Model

    router = AgentRouter(
        agents=[ResearchAgent, AnalystAgent, WriterAgent],
        model=Model.Anthropic("claude-haiku-4-5-20251001"),
    )
    result = router.run("Research AI market and write a report")
    print(result.content)
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable
from typing import TypedDict, Unpack, cast

from syrin.agent._core import Agent
from syrin.audit import AuditHookHandler, AuditLog
from syrin.budget import Budget
from syrin.enums import DocFormat, Hook
from syrin.events import EventContext, Events
from syrin.model import Model
from syrin.response import Response
from syrin.serve.config import ServeConfig, ServeConfigKwargs
from syrin.serve.servable import Servable
from syrin.types import TokenUsage
from syrin.watch import Watchable

_log = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Internal TypedDicts (private to this module)
# ──────────────────────────────────────────────────────────────────────────────


class _AgentSpec(TypedDict, total=False):
    """Parsed agent spec produced by the LLM planning step.

    Attributes:
        type: Agent name as registered in the router's ``_agent_names`` map.
        task: Task description for this agent instance.
    """

    type: str
    task: str


class _RunMetrics(TypedDict, total=False):
    """Runtime metrics accumulated during a single :meth:`AgentRouter.run` call.

    Attributes:
        task: The original task string.
        mode: Execution mode — ``"parallel"`` or ``"sequential"``.
        agents_spawned: Names of agents that were spawned.
        total_cost: Accumulated cost across all spawned agents.
        total_tokens: Accumulated token count across all spawned agents.
        start_time: Unix timestamp when the run started.
        end_time: Unix timestamp when the run finished.
    """

    task: str
    mode: str
    agents_spawned: list[str]
    total_cost: float
    total_tokens: int
    start_time: float
    end_time: float


# ──────────────────────────────────────────────────────────────────────────────
# AgentRouter
# ──────────────────────────────────────────────────────────────────────────────


class AgentRouter(Watchable, Servable):
    """Pipeline where the LLM decides which agents to spawn and how to run them.

    This is a fully agentic orchestration primitive: the LLM analyses the task
    and decides:

    1. Which specialised agents are needed.
    2. How many agents to spawn.
    3. What each agent should do.
    4. Execution order (parallel or sequential).

    Usage::

        router = AgentRouter(
            agents=[ResearcherAgent, AnalystAgent, WriterAgent],
            model=Model.Anthropic("claude-haiku-4-5-20251001"),
            max_parallel=5,
        )
        result = router.run("Research AI market and create a report")

    Custom agent names::

        class MyAgent(Agent):
            _agent_name = "research"  # Used for routing and discovery.

    Attributes:
        events: Lifecycle hooks for all dynamic pipeline events.
    """

    def __init__(
        self,
        agents: list[type[Agent]] | None = None,
        budget: Budget | None = None,
        model: Model | None = None,
        format: DocFormat = DocFormat.TOON,
        max_parallel: int = 10,
        debug: bool = False,
        audit: AuditLog | None = None,
        output_format: str = "clean",
    ) -> None:
        """Initialise AgentRouter.

        Args:
            agents: Agent classes available for spawning.  Each agent's name is
                ``Agent._agent_name`` if set, otherwise the lowercase class name.
            budget: Optional shared budget for all spawned agents.
            model: **Required.** Orchestrator model that plans and spawns agents.
            format: Tool-schema format used in agent descriptions.
                :attr:`~syrin.enums.DocFormat.TOON` uses ~40 % fewer tokens.
            max_parallel: Maximum agents to spawn at once.
            debug: Print hook events to stdout while running.
            audit: Optional :class:`~syrin.audit.AuditLog` for all events.
            output_format: ``"clean"`` (chat-friendly) or ``"verbose"`` (debug
                format with per-agent headers and cost breakdown).

        Raises:
            ValueError: If *model* is ``None``.
            TypeError: If *audit* is not an :class:`~syrin.audit.AuditLog`.
        """
        if model is None:
            raise ValueError(
                "model is required — pass a Model instance "
                "(e.g., Model.Anthropic('claude-haiku-4-5-20251001'))"
            )

        self._agents: list[type[Agent]] = agents or []
        self._budget = budget
        self._model = model
        self._format = format
        self._max_parallel = max_parallel
        self._debug = debug
        self._output_format = output_format
        self._run_metrics: _RunMetrics = {}

        self._agent_names: dict[str, type[Agent]] = {}
        for agent_class in self._agents:
            name = getattr(agent_class, "_syrin_default_name", None)
            if name is None:
                name = agent_class.__name__.lower()
            self._agent_names[name] = agent_class

        self._events = Events(self._emit_hook)

        if audit is not None:
            if not isinstance(audit, AuditLog):
                raise TypeError(f"audit must be AuditLog or None, got {type(audit).__name__}.")
            audit_handler = AuditHookHandler(source="AgentRouter", config=audit)
            self._events.on_all(audit_handler)

        Watchable.__init__(self)

    @property
    def events(self) -> Events:
        """Lifecycle hooks: DYNAMIC_PIPELINE_START/END, AGENT_SPAWN, etc.

        Example::

            router.events.on(Hook.DYNAMIC_PIPELINE_START, lambda ctx: print(ctx))
        """
        return self._events

    @property
    def estimated_cost(self) -> object | None:
        """Pre-flight cost estimate for all agents registered with this router.

        Returns ``None`` when ``estimation=False`` on the budget, or when no
        budget is set.  Access is synchronous.

        Example::

            router = AgentRouter(
                agents=[ResearchAgent, WriterAgent],
                model=Model.Anthropic("claude-haiku-4-5-20251001"),
                budget=Budget(max_cost=5.0, estimation=True),
            )
            est = router.estimated_cost
            if est is not None:
                print(f"p50=${est.p50:.4f}  p95=${est.p95:.4f}")
        """
        import logging

        from syrin.budget._estimate import CostEstimate
        from syrin.budget._preflight import InsufficientBudgetError
        from syrin.enums import EstimationPolicy

        budget = self._budget
        if budget is None or not budget.estimation:
            return None

        estimator = budget._effective_estimator()
        result: CostEstimate = estimator.estimate_many(list(self._agents), budget)

        policy = budget.estimation_policy
        if not result.sufficient:
            if policy == EstimationPolicy.RAISE:
                max_cost = budget.max_cost or 0.0
                raise InsufficientBudgetError(
                    total_p50=result.p50,
                    total_p95=result.p95,
                    budget_configured=max_cost,
                    policy=budget.estimation_policy,
                )
            elif policy == EstimationPolicy.WARN_ONLY:
                logging.getLogger(__name__).warning(
                    "AgentRouter pre-flight estimation: budget $%.4f may be insufficient "
                    "(p50=$%.4f, p95=$%.4f). Run may exceed budget.",
                    budget.max_cost or 0.0,
                    result.p50,
                    result.p95,
                )

        return result

    # ──────────────────────────────────────────────────────────────────────────
    def _format_to_schema(self, format: object) -> str:
        """Convert agent spawn specification to the configured output format.

        Args:
            format: :class:`~syrin.enums.DocFormat` value or ``str`` format
                name (``"toon"`` or ``"json"``).

        Returns:
            Template string showing agents how to format their spawn payload.
        """
        fmt_str = format.value if hasattr(format, "value") else str(format)
        if fmt_str.lower() == "json":
            return """```json\n[\n  {"type": "<agent_name>", "task": "<what this agent should do>"}\n]\n```"""
        # Default: TOON format
        return """```\n@spawn\nagents:\n- type: <agent_name>\n  task: <what this agent should do>\n```"""

    # Public run interface
    # ──────────────────────────────────────────────────────────────────────────

    def run(
        self,
        task: str,
        mode: str = "parallel",
    ) -> Response[str]:
        """Run dynamic agent pipeline.

        Asks the orchestrator LLM to plan which agents to spawn, then executes
        those agents.

        Args:
            task: The task to complete.
            mode: Execution mode — ``"parallel"`` (default) or ``"sequential"``.

                - **parallel**: All agents run simultaneously; results combined.
                - **sequential**: Agents run one after another; each receives
                  the previous agent's output as additional context.

        Returns:
            Consolidated :class:`~syrin.response.Response` from all agents.
        """
        self._run_metrics = {
            "task": task,
            "mode": mode,
            "start_time": time.time(),
            "agents_spawned": [],
            "total_cost": 0.0,
            "total_tokens": 0,
        }

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
            plan = self._get_agent_plan(task)

            self._emit_hook(
                Hook.DYNAMIC_PIPELINE_PLAN,
                EventContext(task=task, plan=plan, plan_count=len(plan)),
            )

            result = self._execute_plan(plan, mode)

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
                    agents_spawned=self._run_metrics.get("agents_spawned", []),
                    total_cost=self._run_metrics.get("total_cost", 0.0),
                ),
            )
            raise

    def as_router(
        self,
        config: ServeConfig | None = None,
        **config_kwargs: Unpack[ServeConfigKwargs],
    ) -> object:
        """Return a FastAPI ``APIRouter`` for this router. Mount on your app.

        Args:
            config: Optional :class:`~syrin.serve.config.ServeConfig`.
            **config_kwargs: Keyword args forwarded to :class:`ServeConfig`.

        Returns:
            FastAPI ``APIRouter`` instance.
        """
        from syrin.serve.config import ServeConfig as _ServeConfig  # noqa: PLC0415
        from syrin.serve.http import build_router  # noqa: PLC0415

        cfg = config if isinstance(config, _ServeConfig) else _ServeConfig(**config_kwargs)
        return build_router(self, cfg)

    def visualize(self) -> None:
        """Print a rich summary of this router to stdout.

        Shows the header "AgentRouter — LLM-routed" and the list of candidate
        agents available for dynamic spawning.

        Example::

            router.visualize()
            # AgentRouter — LLM-routed
            # Agents (3):
            #   ResearchAgent
            #   WriterAgent
            #   EditorAgent
        """
        try:
            from rich import print as rprint  # noqa: PLC0415
            from rich.tree import Tree  # noqa: PLC0415

            tree = Tree("[bold cyan]AgentRouter[/bold cyan] — LLM-routed")
            agents_branch = tree.add(f"[bold]Agents ({len(self._agents)}):[/bold]")
            for agent_class in self._agents:
                name = getattr(agent_class, "_syrin_default_name", None) or agent_class.__name__
                agents_branch.add(f"[green]{name}[/green]")

            if self._model is not None:
                tree.add(f"[bold]Orchestrator model:[/bold] {self._model.model_id}")

            rprint(tree)

        except ImportError:
            print("AgentRouter — LLM-routed")
            print(f"Agents ({len(self._agents)}):")
            for agent_class in self._agents:
                print(f"  {agent_class.__name__}")

    async def _arun_for_trigger(self, input: str) -> object:  # noqa: A002
        """Run the router with a trigger input string.

        Args:
            input: Task string from the trigger.

        Returns:
            Response from ``run()``.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.run, input)

    # ──────────────────────────────────────────────────────────────────────────
    # Private helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _emit_hook(self, hook: Hook, ctx: EventContext) -> None:
        """Emit a hook event with before/after handlers.

        Args:
            hook: Hook to emit.
            ctx: Event context payload.
        """
        ctx["pipeline_id"] = id(self)
        ctx["timestamp"] = time.time()

        if self._debug:
            self._print_event(hook, ctx)

        self._events._trigger_before(hook, ctx)
        self._events._trigger(hook, ctx)
        self._events._trigger_after(hook, ctx)

        _log.debug(
            "AgentRouter hook: %s — before:%d main:%d after:%d",
            hook.value,
            len(self._events._before_handlers[hook]),
            len(self._events._handlers[hook]),
            len(self._events._after_handlers[hook]),
        )

    def _print_event(self, hook: Hook, ctx: EventContext) -> None:
        """Print a hook event to stdout when ``debug=True``.

        Args:
            hook: The hook that fired.
            ctx: Event context payload.
        """
        import sys  # noqa: PLC0415
        from datetime import datetime  # noqa: PLC0415

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
            color, symbol = GREEN, "▶"
        elif "end" in event_str or "complete" in event_str:
            color, symbol = BLUE, "✓"
        elif "spawn" in event_str or "handoff" in event_str:
            color, symbol = CYAN, "→"
        elif "plan" in event_str:
            color, symbol = YELLOW, "◉"
        elif "error" in event_str:
            color, symbol = RED, "✗"
        else:
            color, symbol = "", "•"

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
        total_cost_val = ctx.get("total_cost")
        if total_cost_val is not None:
            cost_num = float(cast(float | int, total_cost_val))
            if cost_num > 0:
                print(f"{indent}Total cost: ${cost_num:.6f}")
        if "error" in ctx:
            print(f"{indent}{RED}Error: {ctx['error']}{RESET}")
        print()

    def _get_agent_description(self, agent_class: type[Agent]) -> str:
        """Return a short description string for an agent class.

        Args:
            agent_class: Agent class to describe.

        Returns:
            Formatted description line, e.g. ``"- research: Searches the web"``.
        """
        name = getattr(agent_class, "_syrin_default_name", None) or agent_class.__name__.lower()
        prompt = getattr(agent_class, "system_prompt", "Specialised agent")[:100]
        return f"- {name}: {prompt}"

    def _get_agent_plan(self, task: str) -> list[_AgentSpec]:
        """Ask the orchestrator LLM to plan which agents to spawn.

        Args:
            task: User task string.

        Returns:
            List of :class:`_AgentSpec` dicts.
        """
        from syrin.tool import ToolSpec, tool  # noqa: PLC0415

        agent_descriptions = [self._get_agent_description(a) for a in self._agents]
        agent_list_str = "\n".join(agent_descriptions) or "No agents available"

        def _plan_agents_fn(plan: str) -> str:
            """Return the agent plan as JSON."""
            return plan

        _tool_decorator = cast(
            "Callable[[Callable[[str], str]], ToolSpec]",
            tool(name="plan", description="Plan which agents to spawn for this task"),
        )
        plan_tool: ToolSpec = _tool_decorator(_plan_agents_fn)

        system_prompt = (
            f"Analyse this task and decide which agents to spawn.\n\n"
            f"Available agents:\n{agent_list_str}\n\n"
            f"Maximum agents you can spawn: {self._max_parallel}\n\n"
            f"Return your plan as JSON array:\n"
            f'[\n  {{"type": "agent_name", "task": "what this agent should do"}},\n  ...\n]\n\n'
            f"IMPORTANT: Return ONLY valid JSON, no other text."
        )
        orchestrator = Agent(
            model=self._model,
            system_prompt=system_prompt,
            tools=[plan_tool],
            budget=self._budget,
        )
        response = orchestrator.run(task)
        return self._parse_plan(response.content)

    def _parse_plan(self, content: str) -> list[_AgentSpec]:
        """Parse agent plan from the LLM response.

        Args:
            content: Raw LLM response string.

        Returns:
            List of :class:`_AgentSpec` dicts.
        """
        import json  # noqa: PLC0415

        content = content.strip()
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
                return cast(list[_AgentSpec], plan)
            if isinstance(plan, dict) and "agents" in plan:
                return cast(list[_AgentSpec], plan["agents"])
        except json.JSONDecodeError:
            pass

        return self._parse_agents_spec(content)

    def _parse_agents_spec(self, spec: str) -> list[_AgentSpec]:
        """Parse an agent specification string (TOON fallback).

        Args:
            spec: Raw specification string.

        Returns:
            List of :class:`_AgentSpec` dicts.
        """
        agents: list[_AgentSpec] = []
        lines = spec.strip().split("\n")
        current: _AgentSpec = {}
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

    def _build_no_agents_message(self) -> str:
        """Build a helpful message when the orchestrator spawns no agents.

        Returns:
            Multi-line message string describing available agents.
        """
        lines: list[str] = [
            "No agents were spawned for this request.",
            "",
            "**Available agents:**",
        ]
        for agent_class in self._agents:
            lines.append(f"  {self._get_agent_description(agent_class)}")
        lines.extend(
            ["", "Provide a specific task and the orchestrator will choose the right agent(s)."]
        )
        return "\n".join(lines)

    def _execute_plan(self, plan: list[_AgentSpec], mode: str) -> Response[str]:
        """Execute the planned agents.

        Args:
            plan: List of :class:`_AgentSpec` dicts from the LLM planner.
            mode: ``"parallel"`` or ``"sequential"``.

        Returns:
            Consolidated :class:`~syrin.response.Response`.
        """
        if not plan:
            content = self._build_no_agents_message()
            return Response(content=content, cost=0.0, tokens=TokenUsage())

        self._emit_hook(
            Hook.DYNAMIC_PIPELINE_EXECUTE,
            EventContext(plan=plan, plan_count=len(plan), mode=mode),
        )

        if mode == "sequential":
            result_content, cost, tokens = self._execute_sequential(plan)
        else:
            result_content, cost, tokens = self._execute_parallel(plan)

        return Response(content=result_content, cost=cost, tokens=tokens)

    def _execute_parallel(self, agents_spec: list[_AgentSpec]) -> tuple[str, float, TokenUsage]:
        """Execute agents in parallel via ``asyncio.run``.

        Args:
            agents_spec: Agent specs from the planner.

        Returns:
            Tuple of ``(consolidated_content, total_cost, total_token_usage)``.
        """

        async def _run() -> list[Response[str]]:
            from syrin.agent.pipeline import parallel as _parallel  # noqa: PLC0415

            tasks: list[tuple[Agent, str]] = []
            for spec in agents_spec[: self._max_parallel]:
                agent_type = spec.get("type", "").lower()
                task = spec.get("task", "")
                agent_class = self._agent_names.get(agent_type)
                if not agent_class:
                    continue
                self._emit_hook(
                    Hook.DYNAMIC_PIPELINE_AGENT_SPAWN,
                    EventContext(
                        agent_type=agent_type,
                        task=task,
                        spawn_time=time.time(),
                        execution_mode="parallel",
                    ),
                )
                tasks.append((agent_class(budget=self._budget), task))
            return await _parallel(tasks)

        results = asyncio.run(_run())

        total_cost = 0.0
        total_tokens = 0
        for i, result in enumerate(results):
            if i < len(agents_spec):
                spec = agents_spec[i]
                agent_type = spec.get("type", "").lower()
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
                self._run_metrics["total_cost"] = (
                    self._run_metrics.get("total_cost", 0.0) + result.cost
                )
                self._run_metrics["total_tokens"] = (
                    self._run_metrics.get("total_tokens", 0) + result.tokens.total_tokens
                )
                total_cost += result.cost
                total_tokens += result.tokens.total_tokens

        return self._consolidate_results(results), total_cost, TokenUsage(total_tokens=total_tokens)

    def _execute_sequential(self, agents_spec: list[_AgentSpec]) -> tuple[str, float, TokenUsage]:
        """Execute agents sequentially, passing context between them.

        Args:
            agents_spec: Agent specs from the planner.

        Returns:
            Tuple of ``(consolidated_content, total_cost, total_token_usage)``.
        """
        results: list[Response[str]] = []
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

            full_task = (
                f"{task}\n\nPrevious results:\n{previous_output}" if previous_output else task
            )
            agent = agent_class(budget=self._budget)
            result = agent.run(full_task)
            results.append(result)

            self._emit_hook(
                Hook.DYNAMIC_PIPELINE_AGENT_COMPLETE,
                EventContext(
                    agent_type=agent_type,
                    task=task,
                    result_preview=result.content[:200] if result.content else "",
                    cost=result.cost,
                    tokens=result.tokens.total_tokens,
                    duration=time.time() - agent_start_time,
                    passed_context=bool(previous_output),
                ),
            )

            self._run_metrics["agents_spawned"].append(agent_type)
            self._run_metrics["total_cost"] = self._run_metrics.get("total_cost", 0.0) + result.cost
            self._run_metrics["total_tokens"] = (
                self._run_metrics.get("total_tokens", 0) + result.tokens.total_tokens
            )
            total_cost += result.cost
            total_tokens += result.tokens.total_tokens
            previous_output = result.content

        return self._consolidate_results(results), total_cost, TokenUsage(total_tokens=total_tokens)

    def _consolidate_results(self, results: list[Response[str]]) -> str:
        """Consolidate results from multiple agents into a single string.

        Args:
            results: Individual agent responses.

        Returns:
            - ``"clean"`` format: plain concatenation, chat-friendly.
            - ``"verbose"`` format: per-agent headers + cost summary.
        """
        if not results:
            return "No results"

        if self._output_format == "verbose":
            consolidated = "=== AGENT RESULTS ===\n\n"
            for i, result in enumerate(results):
                consolidated += (
                    f"--- Agent {i + 1} ---\n{result.content}\n\n[Cost: ${result.cost:.6f}]\n\n"
                )
            total_cost = sum(r.cost for r in results)
            total_tokens = sum(r.tokens.total_tokens for r in results)
            consolidated += f"=== TOTAL: {len(results)} agents ===\nTotal cost: ${total_cost:.6f}\nTotal tokens: {total_tokens}\n"
            return consolidated

        parts = [r.content.strip() for r in results if r.content.strip()]
        return "\n\n".join(parts) if parts else "No results"


# ──────────────────────────────────────────────────────────────────────────────
__all__ = [
    "AgentRouter",
]
