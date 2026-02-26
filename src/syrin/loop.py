"""Simplified Loop strategies for Syrin.

This module provides simple loop strategies that control how an agent
iterates when handling a task. The Loop Protocol is preserved for
custom implementations, but built-in loops are simplified.

Usage:
    # Simple - just use the built-in
    agent = Agent(loop=ReactLoop())

    # Human in the loop - simplified
    async def approve(tool): return True
    agent = Agent(loop=HumanInTheLoop(approve))

    # Single shot - no iteration
    agent = Agent(loop=SingleShotLoop())
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Any

from syrin.agent._run_context import AgentRunContext
from syrin.enums import Hook, LoopStrategy, MessageRole, StopReason


def _get_tracer(ctx: Any) -> Any:
    """Return tracer from context if available, else None (no span creation)."""
    return getattr(ctx, "tracer", None)


def _llm_span_context(ctx: Any, iteration: int, model_id: str) -> Any:
    """Context manager for LLM span when ctx.tracer is set; else no-op."""
    tracer = _get_tracer(ctx)
    if tracer is None:
        return nullcontext()
    from syrin.observability import SemanticAttributes, SpanKind

    return tracer.span(
        f"llm.iteration_{iteration}",
        kind=SpanKind.LLM,
        attributes={
            SemanticAttributes.LLM_MODEL: model_id,
            SemanticAttributes.AGENT_ITERATION: iteration,
        },
    )


def _tool_span_context(ctx: Any, tool_name: str, tool_args: dict[str, Any], iteration: int) -> Any:
    """Context manager for tool span when ctx.tracer is set; else no-op."""
    tracer = _get_tracer(ctx)
    if tracer is None:
        return nullcontext()
    import json

    from syrin.observability import SemanticAttributes, SpanKind

    return tracer.span(
        f"tool.{tool_name}",
        kind=SpanKind.TOOL,
        attributes={
            SemanticAttributes.TOOL_NAME: tool_name,
            SemanticAttributes.TOOL_INPUT: json.dumps(tool_args) if tool_args else "{}",
            SemanticAttributes.AGENT_ITERATION: iteration,
        },
    )


@dataclass
class LoopResult:
    """Result from a loop execution."""

    content: str
    stop_reason: str
    iterations: int
    tools_used: list[str] = field(default_factory=list)
    cost_usd: float = 0.0
    latency_ms: float = 0.0
    token_usage: dict[str, int] = field(
        default_factory=lambda: {"input": 0, "output": 0, "total": 0}
    )
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    raw_response: Any = None


# Type for tool approval callback
ToolApprovalFn = Callable[[str, dict[str, Any]], Awaitable[bool]]


class Loop:
    """Protocol for custom loop strategies.

    Implement this to create your own loop. The run context provides
    build_messages, complete, execute_tool, emit_event, budget/rate-limit
    checks, and model_id/tools/max_output_tokens for cost calculation.

    class MyLoop:
        name = "my_loop"

        async def run(self, ctx: AgentRunContext, user_input: str) -> LoopResult:
            ...
    """

    name: str = "base"

    async def run(self, ctx: AgentRunContext | Any, user_input: str) -> LoopResult:
        """Execute the loop. Override in subclasses. ctx is AgentRunContext."""
        raise NotImplementedError


class SingleShotLoop(Loop):
    """One-shot execution - single LLM call, no tools iteration.

    Use for: Simple questions, one-step tasks
    """

    name = "single_shot"

    async def run(self, ctx: AgentRunContext | Any, user_input: str) -> LoopResult:
        """Execute single LLM call."""
        from syrin.cost import calculate_cost
        from syrin.events import EventContext

        messages = ctx.build_messages(user_input)
        run_start = time.perf_counter()

        ctx.emit_event(
            Hook.AGENT_RUN_START,
            EventContext(
                input=user_input,
                model=ctx.model_id,
                iteration=0,
            ),
        )

        ctx.check_and_apply_rate_limit()
        ctx.pre_call_budget_check(messages, max_output_tokens=ctx.max_output_tokens)
        with _llm_span_context(ctx, 1, ctx.model_id) as llm_span:
            response = await ctx.complete(messages)
            if llm_span is not None:
                from syrin.observability import SemanticAttributes

                u = response.token_usage
                llm_span.set_attribute(
                    SemanticAttributes.LLM_TOKENS_TOTAL,
                    getattr(
                        u,
                        "total_tokens",
                        getattr(u, "input_tokens", 0) + getattr(u, "output_tokens", 0),
                    ),
                )
        if ctx.has_rate_limit:
            ctx.record_rate_limit_usage(response.token_usage)
        if ctx.has_budget:
            ctx.record_cost(response.token_usage, ctx.model_id)

        latency_ms = (time.perf_counter() - run_start) * 1000
        content = response.content or ""

        u = response.token_usage
        cost_usd = calculate_cost(ctx.model_id, u, pricing_override=ctx.pricing_override)

        tool_calls = []
        tool_names = []
        if response.tool_calls:
            for tc in response.tool_calls:
                tool_calls.append(
                    {
                        "id": tc.id,
                        "name": tc.name,
                        "arguments": tc.arguments,
                    }
                )
                tool_names.append(tc.name)

        # Extract stop_reason from response, default to "end_turn"
        stop_reason = response.stop_reason or "end_turn"

        total_tokens = u.input_tokens + u.output_tokens
        ctx.emit_event(
            Hook.AGENT_RUN_END,
            EventContext(
                content=content,
                iterations=1,
                cost=cost_usd,
                tokens=total_tokens,
                duration=latency_ms / 1000.0,
                stop_reason=stop_reason,
            ),
        )

        return LoopResult(
            content=content,
            stop_reason=stop_reason,
            iterations=1,
            latency_ms=latency_ms,
            cost_usd=cost_usd,
            token_usage={
                "input": u.input_tokens,
                "output": u.output_tokens,
                "total": u.total_tokens,
            },
            tool_calls=tool_calls,
            tools_used=tool_names,
            raw_response=response.raw_response,
        )


class ReactLoop(Loop):
    """Think → Act → Observe loop (default).

    Use for: Multi-step tasks requiring tools
    """

    name = "react"

    def __init__(self, max_iterations: int = 10):
        if not isinstance(max_iterations, int) or max_iterations < 1:
            raise ValueError(
                f"max_iterations must be int >= 1, got {max_iterations!r}. "
                "Use at least 1 to allow at least one LLM call."
            )
        self.max_iterations = max_iterations

    async def run(self, ctx: AgentRunContext | Any, user_input: str) -> LoopResult:
        """Execute REACT loop."""
        from syrin.cost import calculate_cost
        from syrin.events import EventContext
        from syrin.types import Message, TokenUsage

        messages = ctx.build_messages(user_input)
        tools = ctx.tools
        iteration = 0
        tools_used = []
        tool_calls_all = []
        run_start = time.perf_counter()

        ctx.emit_event(
            Hook.AGENT_RUN_START,
            EventContext(
                input=user_input,
                model=ctx.model_id,
                iteration=0,
            ),
        )

        while iteration < self.max_iterations:
            iteration += 1
            ctx.check_and_apply_budget()
            ctx.check_and_apply_rate_limit()
            ctx.pre_call_budget_check(messages, max_output_tokens=ctx.max_output_tokens)

            ctx.emit_event(Hook.LLM_REQUEST_START, EventContext(iteration=iteration))

            with _llm_span_context(ctx, iteration, ctx.model_id) as llm_span:
                response = await ctx.complete(messages, tools)
                if llm_span is not None:
                    from syrin.observability import SemanticAttributes

                    u = response.token_usage
                    total_tokens = getattr(u, "total_tokens", None) or (
                        getattr(u, "input_tokens", 0) + getattr(u, "output_tokens", 0)
                    )
                    llm_span.set_attribute(SemanticAttributes.LLM_TOKENS_TOTAL, total_tokens)
            if ctx.has_rate_limit:
                ctx.record_rate_limit_usage(response.token_usage)
            if ctx.has_budget:
                ctx.record_cost(response.token_usage, ctx.model_id)

            stop_reason = getattr(response, "stop_reason", None) or StopReason.END_TURN

            ctx.emit_event(
                Hook.LLM_REQUEST_END,
                EventContext(
                    content=response.content or "",
                    iteration=iteration,
                ),
            )

            if not response.tool_calls:
                break

            messages.append(
                Message(
                    role=MessageRole.ASSISTANT,
                    content=response.content or "",
                    tool_calls=response.tool_calls,
                )
            )

            for tc in response.tool_calls:
                tool_name = tc.name
                tool_args = tc.arguments or {}
                tool_spec = next((t for t in (tools or []) if t.name == tool_name), None)
                needs_approval = tool_spec is not None and getattr(
                    tool_spec, "requires_approval", False
                )

                approved = True
                if needs_approval:
                    gate = getattr(ctx, "approval_gate", None)
                    timeout = getattr(ctx, "hitl_timeout", 300)
                    ctx.emit_event(
                        Hook.HITL_PENDING,
                        EventContext(
                            name=tool_name,
                            arguments=tool_args,
                            message=f"Tool {tool_name} requires approval",
                            iteration=iteration,
                        ),
                    )
                    if gate is not None:
                        approved = await gate.request(
                            message=f"Tool {tool_name!r} requested with args: {tool_args}",
                            timeout=timeout,
                            context={"tool_name": tool_name, "arguments": tool_args},
                        )
                    else:
                        approved = False
                    ctx.emit_event(
                        Hook.HITL_APPROVED if approved else Hook.HITL_REJECTED,
                        EventContext(
                            name=tool_name,
                            arguments=tool_args,
                            approved=approved,
                            iteration=iteration,
                        ),
                    )
                if not approved:
                    messages.append(
                        Message(
                            role=MessageRole.TOOL,
                            content=f"Tool '{tool_name}' not approved.",
                            tool_call_id=tc.id,
                        )
                    )
                    continue

                tools_used.append(tool_name)
                tool_calls_all.append(
                    {
                        "id": tc.id,
                        "name": tc.name,
                        "arguments": tc.arguments,
                    }
                )

                ctx.emit_event(
                    Hook.TOOL_CALL_START,
                    EventContext(
                        name=tool_name,
                        arguments=tool_args,
                        iteration=iteration,
                    ),
                )

                try:
                    with _tool_span_context(ctx, tool_name, tool_args, iteration) as tool_span:
                        result = await ctx.execute_tool(tool_name, tool_args)
                        if tool_span is not None:
                            from syrin.observability import SemanticAttributes

                            tool_span.set_attribute(
                                SemanticAttributes.TOOL_OUTPUT, str(result)[:500]
                            )
                    messages.append(
                        Message(
                            role=MessageRole.TOOL,
                            content=result,
                            tool_call_id=tc.id,
                        )
                    )
                except Exception as e:
                    ctx.emit_event(
                        Hook.TOOL_ERROR,
                        EventContext(
                            error=str(e),
                            tool=tool_name,
                            iteration=iteration,
                        ),
                    )
                    messages.append(
                        Message(
                            role=MessageRole.TOOL,
                            content=f"Error: {str(e)}",
                            tool_call_id=tc.id,
                        )
                    )

        latency_ms = (time.perf_counter() - run_start) * 1000

        total_input = 0
        total_output = 0

        for msg in messages:
            if hasattr(msg, "tokens"):
                total_input += msg.tokens.get("input", 0)
                total_output += msg.tokens.get("output", 0)

        u = response.token_usage
        total_input += u.input_tokens
        total_output += u.output_tokens

        total_cost = calculate_cost(
            ctx.model_id,
            TokenUsage(
                input_tokens=total_input,
                output_tokens=total_output,
                total_tokens=total_input + total_output,
            ),
            pricing_override=ctx.pricing_override,
        )

        total_tokens = total_input + total_output
        ctx.emit_event(
            Hook.AGENT_RUN_END,
            EventContext(
                content=response.content or "",
                iterations=iteration,
                cost=total_cost,
                tokens=total_tokens,
                duration=latency_ms / 1000.0,
                stop_reason=getattr(stop_reason, "value", str(stop_reason)),
            ),
        )

        return LoopResult(
            content=response.content or "",
            stop_reason=getattr(stop_reason, "value", str(stop_reason)),
            iterations=iteration,
            tools_used=tools_used,
            latency_ms=latency_ms,
            cost_usd=total_cost,
            token_usage={
                "input": total_input,
                "output": total_output,
                "total": total_input + total_output,
            },
            tool_calls=tool_calls_all,
            raw_response=response.raw_response,
        )


class HumanInTheLoop(Loop):
    """Human approval before every tool execution.

    Use for: Safety-critical applications where all tools need approval.

    Args:
        approval_gate: ApprovalGate for async request/approve. Use ApprovalGate(callback=fn).
        approve: Legacy: async (tool_name, args) -> bool. Wrapped into ApprovalGate if set.
        timeout: Seconds to wait for approval. On timeout, reject. Default 300.
        max_iterations: Max tool-call loops.
    """

    name = "human_in_the_loop"

    def __init__(
        self,
        approval_gate: Any = None,
        approve: ToolApprovalFn | None = None,
        timeout: int = 300,
        max_iterations: int = 10,
    ) -> None:
        from syrin.hitl import ApprovalGate

        if approval_gate is not None:
            self._gate = approval_gate
        elif approve is not None:

            async def _wrap(msg: str, t: int, ctx: dict[str, Any]) -> bool:
                return await approve(ctx.get("tool_name", ""), ctx.get("arguments", {}))

            self._gate = ApprovalGate(_wrap)
        else:
            raise ValueError("HumanInTheLoop requires approval_gate or approve")
        self._timeout = timeout
        self.max_iterations = max_iterations

    async def run(self, ctx: AgentRunContext | Any, user_input: str) -> LoopResult:
        """Execute loop with human approval."""
        from syrin.cost import calculate_cost
        from syrin.events import EventContext
        from syrin.types import Message, TokenUsage

        messages = ctx.build_messages(user_input)
        tools = ctx.tools
        iteration = 0
        tools_used = []
        tool_calls_all = []
        run_start = time.perf_counter()

        ctx.emit_event(
            Hook.AGENT_RUN_START,
            EventContext(
                input=user_input,
                model=ctx.model_id,
                iteration=0,
            ),
        )

        while iteration < self.max_iterations:
            iteration += 1
            ctx.check_and_apply_rate_limit()
            ctx.pre_call_budget_check(messages, max_output_tokens=ctx.max_output_tokens)

            response = await ctx.complete(messages, tools)

            if ctx.has_rate_limit:
                ctx.record_rate_limit_usage(response.token_usage)
            if ctx.has_budget:
                ctx.record_cost(response.token_usage, ctx.model_id)

            if not response.tool_calls:
                break

            messages.append(
                Message(
                    role=MessageRole.ASSISTANT,
                    content=response.content or "",
                    tool_calls=response.tool_calls,
                )
            )

            for tc in response.tool_calls:
                tool_name = tc.name
                tool_args = tc.arguments or {}

                ctx.emit_event(
                    Hook.HITL_PENDING,
                    EventContext(
                        name=tool_name,
                        arguments=tool_args,
                        message=f"Tool {tool_name} requires approval",
                        iteration=iteration,
                    ),
                )
                try:
                    approved = await asyncio.wait_for(
                        self._gate.request(
                            message=f"Tool {tool_name!r} with args: {tool_args}",
                            timeout=self._timeout,
                            context={"tool_name": tool_name, "arguments": tool_args},
                        ),
                        timeout=self._timeout,
                    )
                except asyncio.TimeoutError:
                    approved = False

                ctx.emit_event(
                    Hook.HITL_APPROVED if approved else Hook.HITL_REJECTED,
                    EventContext(
                        name=tool_name,
                        arguments=tool_args,
                        approved=approved,
                        iteration=iteration,
                    ),
                )

                ctx.emit_event(
                    Hook.TOOL_CALL_START,
                    EventContext(
                        name=tool_name,
                        arguments=tool_args,
                        iteration=iteration,
                        approved=approved,
                    ),
                )

                if not approved:
                    messages.append(
                        Message(
                            role=MessageRole.TOOL,
                            content=f"Tool '{tool_name}' not approved.",
                            tool_call_id=tc.id,
                        )
                    )
                    continue

                tools_used.append(tool_name)
                tool_calls_all.append(
                    {
                        "id": tc.id,
                        "name": tc.name,
                        "arguments": tc.arguments,
                    }
                )

                try:
                    result = await ctx.execute_tool(tool_name, tool_args)
                    messages.append(
                        Message(
                            role=MessageRole.TOOL,
                            content=result,
                            tool_call_id=tc.id,
                        )
                    )
                except Exception as e:
                    messages.append(
                        Message(
                            role=MessageRole.TOOL,
                            content=f"Error: {str(e)}",
                            tool_call_id=tc.id,
                        )
                    )

        latency_ms = (time.perf_counter() - run_start) * 1000

        u = response.token_usage
        total_input = u.input_tokens
        total_output = u.output_tokens

        total_cost = calculate_cost(
            ctx.model_id,
            TokenUsage(
                input_tokens=total_input,
                output_tokens=total_output,
                total_tokens=total_input + total_output,
            ),
            pricing_override=ctx.pricing_override,
        )

        total_tokens = total_input + total_output
        ctx.emit_event(
            Hook.AGENT_RUN_END,
            EventContext(
                content=response.content or "",
                iterations=iteration,
                cost=total_cost,
                tokens=total_tokens,
                duration=latency_ms / 1000.0,
                stop_reason="end_turn",
            ),
        )

        return LoopResult(
            content=response.content or "",
            stop_reason="end_turn",
            iterations=iteration,
            tools_used=tools_used,
            latency_ms=latency_ms,
            cost_usd=total_cost,
            token_usage={
                "input": total_input,
                "output": total_output,
                "total": total_input + total_output,
            },
            tool_calls=tool_calls_all,
            raw_response=response.raw_response,
        )


class PlanExecuteLoop(Loop):
    """Plan → Execute → Review loop.

    First generates a plan with specific steps, then executes each step
    sequentially, and finally reviews the results.

    Use for: Complex multi-step tasks that benefit from upfront planning
    """

    name = "plan_execute"

    def __init__(
        self,
        max_plan_iterations: int = 5,
        max_execution_iterations: int = 20,
    ):
        self.max_plan_iterations = max_plan_iterations
        self.max_execution_iterations = max_execution_iterations

    async def run(self, ctx: AgentRunContext | Any, user_input: str) -> LoopResult:
        """Execute PLAN → EXECUTE → REVIEW loop."""
        from syrin.cost import calculate_cost
        from syrin.events import EventContext
        from syrin.types import Message, TokenUsage

        messages = ctx.build_messages(
            user_input
            + "\n\nPlease provide a detailed plan with numbered steps to accomplish this task."
        )
        tools = ctx.tools
        run_start = time.perf_counter()
        total_input = 0
        total_output = 0

        ctx.emit_event(
            Hook.AGENT_RUN_START,
            EventContext(
                input=user_input,
                model=ctx.model_id,
                iteration=0,
            ),
        )

        # Phase 1: Planning
        plan_iteration = 0
        plan_response = None

        while plan_iteration < self.max_plan_iterations:
            plan_iteration += 1
            ctx.check_and_apply_rate_limit()
            ctx.pre_call_budget_check(messages, max_output_tokens=ctx.max_output_tokens)
            ctx.emit_event(Hook.LLM_REQUEST_START, EventContext(iteration=plan_iteration))

            response = await ctx.complete(messages, tools)

            if ctx.has_rate_limit:
                ctx.record_rate_limit_usage(response.token_usage)
            if ctx.has_budget:
                ctx.record_cost(response.token_usage, ctx.model_id)

            u = response.token_usage
            total_input += u.input_tokens
            total_output += u.output_tokens

            if response.tool_calls:
                for tc in response.tool_calls:
                    messages.append(
                        Message(
                            role=MessageRole.ASSISTANT,
                            content=response.content or "",
                            tool_calls=response.tool_calls,
                        )
                    )
                    tool_result = await ctx.execute_tool(tc.name, tc.arguments or {})
                    messages.append(
                        Message(
                            role=MessageRole.TOOL,
                            content=tool_result,
                            tool_call_id=tc.id,
                        )
                    )
            else:
                plan_response = response
                break

            ctx.emit_event(Hook.LLM_REQUEST_END, EventContext(iteration=plan_iteration))

        if plan_response is None:
            plan_response = response

        # Phase 2: Execution - ask for final execution
        messages.append(
            Message(
                role=MessageRole.USER,
                content="Now please execute the plan and provide the final result.",
            )
        )

        exec_iteration = 0
        final_response = None

        while exec_iteration < self.max_execution_iterations:
            exec_iteration += 1
            ctx.check_and_apply_budget()
            ctx.check_and_apply_rate_limit()
            ctx.pre_call_budget_check(messages, max_output_tokens=ctx.max_output_tokens)
            ctx.emit_event(
                Hook.LLM_REQUEST_START, EventContext(iteration=plan_iteration + exec_iteration)
            )

            response = await ctx.complete(messages, tools)

            if ctx.has_rate_limit:
                ctx.record_rate_limit_usage(response.token_usage)
            if ctx.has_budget:
                ctx.record_cost(response.token_usage, ctx.model_id)

            u = response.token_usage
            total_input += u.input_tokens
            total_output += u.output_tokens

            if not response.tool_calls:
                final_response = response
                break

            messages.append(
                Message(
                    role=MessageRole.ASSISTANT,
                    content=response.content or "",
                    tool_calls=response.tool_calls,
                )
            )

            for tc in response.tool_calls:
                tool_result = await ctx.execute_tool(tc.name, tc.arguments or {})
                messages.append(
                    Message(
                        role=MessageRole.TOOL,
                        content=tool_result,
                        tool_call_id=tc.id,
                    )
                )

            ctx.emit_event(
                Hook.LLM_REQUEST_END, EventContext(iteration=plan_iteration + exec_iteration)
            )

        if final_response is None:
            final_response = response

        latency_ms = (time.perf_counter() - run_start) * 1000

        total_cost = calculate_cost(
            ctx.model_id,
            TokenUsage(
                input_tokens=total_input,
                output_tokens=total_output,
                total_tokens=total_input + total_output,
            ),
            pricing_override=ctx.pricing_override,
        )

        total_tokens = total_input + total_output
        ctx.emit_event(
            Hook.AGENT_RUN_END,
            EventContext(
                content=final_response.content or "",
                iterations=plan_iteration + exec_iteration,
                cost=total_cost,
                tokens=total_tokens,
                duration=latency_ms / 1000.0,
                stop_reason=getattr(final_response, "stop_reason", "end_turn") or "end_turn",
            ),
        )

        return LoopResult(
            content=final_response.content or "",
            stop_reason=getattr(final_response, "stop_reason", "end_turn") or "end_turn",
            iterations=plan_iteration + exec_iteration,
            latency_ms=latency_ms,
            cost_usd=total_cost,
            token_usage={
                "input": total_input,
                "output": total_output,
                "total": total_input + total_output,
            },
            tool_calls=[],
            raw_response=final_response.raw_response,
        )


class CodeActionLoop(Loop):
    """Generate code → Execute → Interpret results loop.

    Uses the LLM to generate Python code to solve the problem,
    executes it, and interprets the results.

    Use for: Mathematical computations, data processing, code generation tasks
    """

    name = "code_action"

    def __init__(
        self,
        max_iterations: int = 10,
        timeout_seconds: int = 60,
    ):
        self.max_iterations = max_iterations
        self.timeout_seconds = timeout_seconds

    async def run(self, ctx: AgentRunContext | Any, user_input: str) -> LoopResult:
        """Execute CODE → EXECUTE → INTERPRET loop."""
        import re

        from syrin.cost import calculate_cost
        from syrin.events import EventContext
        from syrin.types import Message, TokenUsage

        system_msg = (
            "You are a code-generating AI. When asked to solve a problem, "
            "generate Python code in a code block. The code will be executed "
            "and you will receive the output. Use the output to provide your final answer.\n\n"
            "Example format:\n```python\n# Your code here\nresult = calculation\nprint(result)\n```"
        )

        messages = [
            Message(role=MessageRole.SYSTEM, content=system_msg),
            Message(role=MessageRole.USER, content=user_input),
        ]
        tools = None  # Code action doesn't use tools
        iteration = 0
        run_start = time.perf_counter()
        total_input = 0
        total_output = 0

        ctx.emit_event(
            Hook.AGENT_RUN_START,
            EventContext(
                input=user_input,
                model=ctx.model_id,
                iteration=0,
            ),
        )

        while iteration < self.max_iterations:
            iteration += 1
            ctx.check_and_apply_budget()
            ctx.check_and_apply_rate_limit()
            ctx.pre_call_budget_check(messages, max_output_tokens=ctx.max_output_tokens)
            ctx.emit_event(Hook.LLM_REQUEST_START, EventContext(iteration=iteration))

            response = await ctx.complete(messages, tools)

            if ctx.has_rate_limit:
                ctx.record_rate_limit_usage(response.token_usage)
            if ctx.has_budget:
                ctx.record_cost(response.token_usage, ctx.model_id)

            u = response.token_usage
            total_input += u.input_tokens
            total_output += u.output_tokens

            ctx.emit_event(
                Hook.LLM_REQUEST_END,
                EventContext(
                    content=response.content or "",
                    iteration=iteration,
                ),
            )

            content = response.content or ""

            code_blocks = re.findall(r"```(?:python)?\n(.*?)```", content, re.DOTALL)

            if code_blocks:
                code = code_blocks[0].strip()

                try:
                    import io
                    from contextlib import redirect_stdout

                    output_buffer = io.StringIO()
                    try:
                        with redirect_stdout(output_buffer):
                            exec(code, {"print": lambda x: print(x)}, {})
                        code_output = output_buffer.getvalue()
                    except Exception as e:
                        code_output = f"Error: {str(e)}"

                    messages.append(
                        Message(
                            role=MessageRole.USER,
                            content=f"Code output:\n```{code_output}```\n\nPlease provide your final answer based on this output.",
                        )
                    )
                except Exception as e:
                    messages.append(
                        Message(
                            role=MessageRole.USER,
                            content=f"Code execution error: {str(e)}. Please fix and try again.",
                        )
                    )
            else:
                break

            if response.tool_calls:
                messages.append(
                    Message(
                        role=MessageRole.ASSISTANT,
                        content=content,
                        tool_calls=response.tool_calls,
                    )
                )
            else:
                messages.append(Message(role=MessageRole.ASSISTANT, content=content))

        latency_ms = (time.perf_counter() - run_start) * 1000

        total_cost = calculate_cost(
            ctx.model_id,
            TokenUsage(
                input_tokens=total_input,
                output_tokens=total_output,
                total_tokens=total_input + total_output,
            ),
            pricing_override=ctx.pricing_override,
        )

        total_tokens = total_input + total_output
        ctx.emit_event(
            Hook.AGENT_RUN_END,
            EventContext(
                content=response.content or "",
                iterations=iteration,
                cost=total_cost,
                tokens=total_tokens,
                duration=latency_ms / 1000.0,
                stop_reason=getattr(response, "stop_reason", "end_turn") or "end_turn",
            ),
        )

        return LoopResult(
            content=response.content or "",
            stop_reason=getattr(response, "stop_reason", "end_turn") or "end_turn",
            iterations=iteration,
            latency_ms=latency_ms,
            cost_usd=total_cost,
            token_usage={
                "input": total_input,
                "output": total_output,
                "total": total_input + total_output,
            },
            tool_calls=[],
            raw_response=response.raw_response,
        )


class LoopStrategyMapping:
    """Maps LoopStrategy enum to loop implementations."""

    _MAPPING = {
        LoopStrategy.REACT: ReactLoop,
        LoopStrategy.PLAN_EXECUTE: PlanExecuteLoop,
        LoopStrategy.CODE_ACTION: CodeActionLoop,
        LoopStrategy.SINGLE_SHOT: SingleShotLoop,
    }

    @classmethod
    def get_loop(cls, strategy: LoopStrategy) -> type[Loop]:
        """Get loop class for strategy."""
        return cls._MAPPING.get(strategy, ReactLoop)

    @classmethod
    def create_loop(cls, strategy: LoopStrategy | str, max_iterations: int = 10) -> Loop:
        """Create loop instance from strategy."""
        if isinstance(strategy, str):
            strategy = LoopStrategy(strategy)
        loop_class = cls.get_loop(strategy)
        if loop_class in (ReactLoop, HumanInTheLoop, CodeActionLoop):
            return loop_class(max_iterations=max_iterations)  # type: ignore[call-arg]
        if loop_class is PlanExecuteLoop:
            return loop_class(max_execution_iterations=max_iterations)  # type: ignore[call-arg]
        return loop_class()


PLAN_EXECUTE = PlanExecuteLoop
CODE_ACTION = CodeActionLoop

# Simple constants
REACT = ReactLoop
SINGLE_SHOT = SingleShotLoop
HITL = HumanInTheLoop


__all__ = [
    "Loop",
    "LoopResult",
    "ReactLoop",
    "SingleShotLoop",
    "HumanInTheLoop",
    "PlanExecuteLoop",
    "CodeActionLoop",
    "LoopStrategyMapping",
    "REACT",
    "SINGLE_SHOT",
    "HITL",
    "PLAN_EXECUTE",
    "CODE_ACTION",
    "ToolApprovalFn",
]
