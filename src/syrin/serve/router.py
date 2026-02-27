"""Multi-agent HTTP routing — AgentRouter for multiple agents on one server."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from syrin.agent import Agent
    from syrin.serve.config import ServeConfig


def _ensure_serve_deps() -> None:
    """Ensure FastAPI and uvicorn are installed. Raise with install hint if not."""
    try:
        import fastapi  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "HTTP serving requires FastAPI. Install with: uv pip install syrin[serve]"
        ) from e


class AgentRouter:
    """Router for multiple agents on one HTTP server.

    Creates routes per agent: /agent/{name}/chat, /agent/{name}/stream,
    /agent/{name}/health, etc. Mount on an existing FastAPI app or call serve().

    Example:
        >>> from syrin.serve import AgentRouter
        >>> router = AgentRouter(agents=[researcher, writer])
        >>> router.serve(port=8000)
        >>> # Or: app.include_router(router.fastapi_router(), prefix="/api/v1")
    """

    def __init__(
        self,
        agents: list[Agent],
        *,
        config: ServeConfig | None = None,
        agent_prefix: str = "/agent",
    ) -> None:
        """Create a router for multiple agents.

        Args:
            agents: List of Agent instances to serve.
            config: Optional ServeConfig. Defaults used if None.
            agent_prefix: URL prefix for agent routes, e.g. "/agent" yields /agent/{name}/chat.
        """
        _ensure_serve_deps()
        if not agents:
            raise ValueError("AgentRouter requires at least one agent")
        names = [a.name for a in agents]
        if len(names) != len(set(names)):
            raise ValueError("Agent names must be unique")
        self._agents = agents
        self._config = config
        self._agent_prefix = agent_prefix.strip().rstrip("/") or "/agent"
        if not self._agent_prefix.startswith("/"):
            self._agent_prefix = "/" + self._agent_prefix

    def fastapi_router(self) -> Any:
        """Return a FastAPI APIRouter with all agents mounted under /agent/{name}."""
        from fastapi import APIRouter

        from syrin.serve.config import ServeConfig
        from syrin.serve.http import build_router

        cfg = self._config or ServeConfig()
        main = APIRouter()
        for agent in self._agents:
            sub_config = ServeConfig(
                protocol=cfg.protocol,
                host=cfg.host,
                port=cfg.port,
                route_prefix=f"{self._agent_prefix}/{agent.name}",
                stream=cfg.stream,
                include_metadata=cfg.include_metadata,
                debug=cfg.debug,
                enable_playground=cfg.enable_playground,
                enable_discovery=cfg.enable_discovery,
            )
            router = build_router(agent, sub_config)
            main.include_router(router)
        return main

    def serve(self, config: ServeConfig | None = None, **config_kwargs: Any) -> None:
        """Run uvicorn with all agents. Blocks until stopped."""
        from fastapi import FastAPI

        from syrin.serve.config import ServeConfig

        cfg = (
            config
            if isinstance(config, ServeConfig)
            else (self._config or ServeConfig(**config_kwargs))
        )
        try:
            import uvicorn
        except ImportError as e:
            raise ImportError(
                "HTTP serving requires uvicorn. Install with: uv pip install syrin[serve]"
            ) from e
        from syrin.serve.http import _add_startup_endpoint_logging

        app = FastAPI(
            title="Syrin Multi-Agent",
            description=f"Agents: {', '.join(a.name for a in self._agents)}",
        )
        app.include_router(self.fastapi_router(), prefix=cfg.route_prefix or "")
        _add_startup_endpoint_logging(app)
        uvicorn.run(app, host=cfg.host, port=cfg.port)
