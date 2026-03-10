"""Tool execution use case: execute tools by name.

Agent delegates to execute_tool. Public API stays on Agent.
Tools may be sync (return str) or async (return coroutine); caller must await.
"""

from __future__ import annotations

import asyncio
from typing import Any, cast

from syrin.exceptions import ToolExecutionError
from syrin.run_context import RunContext


def execute_tool(agent: Any, name: str, arguments: dict[str, Any]) -> str | Any:
    """Execute a tool by name. Returns result string or raises ToolExecutionError."""
    for spec in agent.tools:
        if spec.name == name:
            try:
                if spec.inject_run_context:
                    if agent._dependencies is None:
                        raise ToolExecutionError(
                            f"Tool {name!r} expects ctx: RunContext but Agent has no dependencies. "
                            "Pass dependencies=MyDeps(...) to Agent."
                        )
                    ctx = RunContext(
                        deps=agent._dependencies,
                        agent_name=cast(str, agent._agent_name),
                        conversation_id=getattr(agent, "_conversation_id", None),
                        budget_state=agent.budget_state,
                        retry_count=0,
                    )
                    result = spec.func(ctx=ctx, **arguments)
                else:
                    result = spec.func(**arguments)
                if asyncio.iscoroutine(result):
                    return result  # Caller must await
                return str(result) if result is not None else ""
            except ToolExecutionError:
                raise
            except Exception as e:
                raise ToolExecutionError(f"Tool {name!r} failed: {e}") from e
    if name == "generate_image":
        return (
            "Image generation is not available. Provide api_key via a Google model or "
            "image_generation=ImageGenerator.Gemini(api_key=...). Install: pip install syrin[generation]"
        )
    if name == "generate_video":
        return (
            "Video generation is not available. Provide api_key via a Google model or "
            "video_generation=VideoGenerator.Gemini(api_key=...). Install: pip install syrin[generation]"
        )
    raise ToolExecutionError(f"Unknown tool: {name!r}")
