"""HTTP serving — FastAPI routes for /chat, /stream, /health, /ready, /budget, /describe."""

from __future__ import annotations

import json
import time
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from syrin.agent import Agent
    from syrin.serve.config import ServeConfig


def _add_startup_endpoint_logging(app: Any) -> None:
    """Add startup event that prints endpoints with methods."""

    @app.on_event("startup")  # type: ignore[untyped-decorator]
    def _log_endpoints() -> None:
        lines: list[str] = ["Syrin endpoints:"]
        for route in app.routes:
            if hasattr(route, "methods") and hasattr(route, "path"):
                methods = ", ".join(sorted(m for m in route.methods if m != "HEAD"))
                lines.append(f"  {methods:6} {route.path}")
        print("\n".join(lines) + "\n", flush=True)


def _ensure_serve_deps() -> None:
    """Ensure FastAPI and uvicorn are installed. Raise with install hint if not."""
    try:
        import fastapi  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "HTTP serving requires FastAPI. Install with: uv pip install syrin[serve]"
        ) from e


def build_router(
    agent: Agent,
    config: ServeConfig,
) -> Any:
    """Build a FastAPI APIRouter for the given agent and config.

    Routes: POST /chat, POST /stream, GET /health, GET /ready, GET /budget, GET /describe.
    Use agent.as_router() or mount the router on an existing FastAPI app.

    Requires syrin[serve] (fastapi, uvicorn).
    """
    _ensure_serve_deps()
    from fastapi import APIRouter, Body
    from fastapi.responses import JSONResponse, StreamingResponse

    from syrin.response import Response

    router = APIRouter()

    prefix = (config.route_prefix or "").strip().rstrip("/")
    if prefix and not prefix.startswith("/"):
        prefix = "/" + prefix

    def _route(path: str) -> str:
        return f"{prefix}{path}" if prefix else path

    def _chat_body(r: dict[str, Any]) -> tuple[str, str | None]:
        message = r.get("message") or r.get("input") or r.get("content")
        if isinstance(message, str):
            return message.strip(), r.get("thread_id")
        return "", None

    @router.post(_route("/chat"))
    async def chat(body: dict[str, Any] | None = Body(default=None)) -> Any:  # noqa: B008
        """Run agent and return full response. POST body: {message: str}."""
        msg, thread_id = _chat_body(body or {})
        if not msg:
            return JSONResponse(
                status_code=400,
                content={"error": "Missing 'message', 'input', or 'content' in body"},
            )
        start = time.perf_counter()
        try:
            r: Response[str] = await agent.arun(msg)
            elapsed = time.perf_counter() - start
            out: dict[str, Any] = {"content": str(r.content)}
            if config.include_metadata:
                out["cost"] = r.cost
                out["tokens"] = {
                    "input": r.tokens.input_tokens,
                    "output": r.tokens.output_tokens,
                    "total": r.tokens.total_tokens,
                }
                out["model"] = r.model
                out["stop_reason"] = str(r.stop_reason)
                out["duration"] = round(elapsed, 4)
            return JSONResponse(content=out)
        except Exception as e:
            return JSONResponse(
                status_code=500,
                content={"error": str(e)},
            )

    @router.post(_route("/stream"))
    async def stream(body: dict[str, Any] | None = Body(default=None)) -> Any:  # noqa: B008
        """Stream response as SSE. POST body: {message: str}."""
        msg, _ = _chat_body(body or {})
        if not msg:
            return JSONResponse(
                status_code=400,
                content={"error": "Missing 'message', 'input', or 'content' in body"},
            )

        async def sse_gen() -> Any:
            accumulated = ""
            async for chunk in agent.astream(msg):
                text = getattr(chunk, "text", "") or getattr(chunk, "content", "")
                accumulated += text
                line = json.dumps({"text": text, "accumulated": accumulated}) + "\n"
                yield f"data: {line}\n\n"

        return StreamingResponse(
            sse_gen(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    @router.get(_route("/health"))
    async def health() -> dict[str, str]:
        """Liveness probe."""
        return {"status": "ok"}

    @router.get(_route("/ready"))
    async def ready() -> dict[str, bool]:
        """Readiness probe (agent initialized, model reachable). Minimal check for now."""
        return {"ready": True}

    @router.get(_route("/budget"))
    async def budget() -> Any:
        """Budget state if configured."""
        state = agent.budget_state
        if state is None:
            return JSONResponse(status_code=404, content={"error": "No budget configured"})
        return JSONResponse(
            content={
                "limit": state.limit,
                "remaining": state.remaining,
                "spent": state.spent,
                "percent_used": state.percent_used,
            }
        )

    @router.get(_route("/describe"))
    async def describe() -> dict[str, Any]:
        """Runtime introspection: name, tools, budget config."""
        tool_names = [t.name for t in agent.tools]
        budget_state = agent.budget_state
        return {
            "name": agent.name,
            "description": agent.description,
            "tools": tool_names,
            "budget": (
                {
                    "limit": budget_state.limit,
                    "remaining": budget_state.remaining,
                    "spent": budget_state.spent,
                    "percent_used": budget_state.percent_used,
                }
                if budget_state is not None
                else None
            ),
        }

    return router
