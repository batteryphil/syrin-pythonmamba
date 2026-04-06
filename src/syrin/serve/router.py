"""Multi-agent HTTP routing — AgentRouter for multiple agents on one server."""

from __future__ import annotations

from typing import TYPE_CHECKING

from syrin.serve.servable import Servable

if TYPE_CHECKING:
    from syrin.agent import Agent
    from syrin.agent.agent_router import AgentRouter as DynamicAgentRouter
    from syrin.serve.config import ServeConfig


def _ensure_serve_deps() -> None:
    """Ensure FastAPI and uvicorn are installed. Raise with install hint if not."""
    try:
        import fastapi  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "HTTP serving requires FastAPI. Install with: uv pip install syrin[serve]"
        ) from e


class AgentRouter(Servable):
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
        agents: list[Agent | DynamicAgentRouter],
        *,
        config: ServeConfig | None = None,
        agent_prefix: str = "/agent",
    ) -> None:
        """Create a router for multiple agents.

        Args:
            agents: List of Agent or AgentRouter instances to serve.
            config: Optional ServeConfig. Defaults used if None.
            agent_prefix: URL prefix for agent routes, e.g. "/agent" yields /agent/{name}/chat.
        """
        _ensure_serve_deps()
        from syrin.serve.adapter import to_serveable

        if not agents:
            raise ValueError("AgentRouter requires at least one agent or pipeline")
        self._raw = agents
        self._serveables = [to_serveable(a) for a in agents]
        names = [s.name for s in self._serveables]
        if len(names) != len(set(names)):
            raise ValueError("Agent/pipeline names must be unique")
        self._config = config
        self._agent_prefix = agent_prefix.strip().rstrip("/") or "/agent"
        if not self._agent_prefix.startswith("/"):
            self._agent_prefix = "/" + self._agent_prefix

    def fastapi_router(self) -> object:
        """Return a FastAPI APIRouter with all agents mounted under /agent/{name}."""
        from fastapi import APIRouter

        from syrin.serve.config import ServeConfig
        from syrin.serve.discovery import (
            AGENT_CARD_PATH,
            build_agent_card_json,
            should_enable_discovery,
        )
        from syrin.serve.http import build_router

        cfg = self._config or ServeConfig()
        main = APIRouter()
        for raw, serveable in zip(self._raw, self._serveables, strict=True):
            sub_config = ServeConfig(
                protocol=cfg.protocol,
                host=cfg.host,
                port=cfg.port,
                route_prefix=f"{self._agent_prefix}/{serveable.name}",
                stream=cfg.stream,
                include_metadata=cfg.include_metadata,
                debug=cfg.debug,
                enable_playground=cfg.enable_playground,
                enable_discovery=cfg.enable_discovery,
            )
            router = build_router(raw, sub_config)
            main.include_router(router)  # type: ignore[arg-type]
        # Root registry at /.well-known/agent-card.json — lists all agents when discovery enabled
        if cfg.enable_discovery is not False:
            host = cfg.host or "0.0.0.0"
            port = cfg.port or 8000
            display_host = "localhost" if host == "0.0.0.0" else host
            base = f"http://{display_host}:{port}"
            if cfg.route_prefix:
                base = (base.rstrip("/") + "/" + cfg.route_prefix.strip("/").lstrip("/")).rstrip(
                    "/"
                )
            prefix = (self._agent_prefix or "/agent").rstrip("/")

            @main.get(AGENT_CARD_PATH)
            async def registry() -> dict[str, object]:
                """Multi-agent registry: agents with name, description, url."""
                agents_list: list[dict[str, object]] = []
                for serveable in self._serveables:
                    if should_enable_discovery(serveable, cfg):
                        card = build_agent_card_json(
                            serveable,
                            base_url=f"{base}{prefix}/{serveable.name}",
                        )
                        agents_list.append(
                            {
                                "name": card["name"],
                                "description": card["description"],
                                "url": card["url"],
                            }
                        )
                return {"agents": agents_list}

        # Playground — when enable_playground=True (multi-agent: agent selector)
        if cfg.enable_playground:
            from syrin.serve.playground import _playground_static_dir, get_playground_html

            route_prefix_str = (cfg.route_prefix or "").strip().rstrip("/")
            route_prefix_str = (
                "/" + route_prefix_str
                if route_prefix_str and not route_prefix_str.startswith("/")
                else route_prefix_str or ""
            )
            base_path = f"{route_prefix_str}/playground" if route_prefix_str else "/playground"
            api_base = f"{route_prefix_str}{self._agent_prefix}".strip("/")
            api_base = f"/{api_base}/" if api_base else "/"
            agents_data = [{"name": a.name, "description": a.description} for a in self._serveables]

            @main.get("/playground/config")
            async def playground_config() -> dict[str, object]:
                """Playground config: apiBase, agents, debug, setup_type."""
                setup_type = "multi" if len(agents_data) > 1 else "single"
                return {
                    "apiBase": api_base,
                    "agents": agents_data,
                    "debug": cfg.debug,
                    "setup_type": setup_type,
                }

            # Static files must be mounted on the main app (router.mount is not transferred).
            # serve() calls add_playground_static_mount() after include_router.
            static_dir = _playground_static_dir()
            if static_dir is None:
                from fastapi.responses import HTMLResponse

                @main.get("/playground", response_class=HTMLResponse)
                async def playground() -> str:
                    """Web playground (inline fallback)."""
                    return get_playground_html(
                        base_path=base_path,
                        api_base=api_base,
                        agents=agents_data,  # type: ignore[arg-type]
                        debug=cfg.debug,
                    )

        return main

    def serve(self, config: ServeConfig | None = None, **config_kwargs: object) -> None:
        """Run HTTP server or CLI REPL based on protocol. Blocks until stopped."""
        from syrin.enums import ServeProtocol
        from syrin.serve.config import ServeConfig

        base = config if isinstance(config, ServeConfig) else (self._config or ServeConfig())
        cfg = ServeConfig(**{**vars(base), **config_kwargs}) if config_kwargs else base

        if cfg.protocol == ServeProtocol.CLI:
            self._serve_cli(cfg)
            return
        if cfg.protocol == ServeProtocol.STDIO:
            self._serve_stdio(cfg, config_kwargs.get("stdin"), config_kwargs.get("stdout"))
            return

        # HTTP (default)
        self._serve_http(cfg)

    def _serve_http(self, cfg: ServeConfig) -> None:
        """Run uvicorn with all agents."""
        from fastapi import FastAPI

        try:
            import uvicorn
        except ImportError as e:
            raise ImportError(
                "HTTP serving requires uvicorn. Install with: uv pip install syrin[serve]"
            ) from e
        from syrin.serve.http import _add_startup_endpoint_logging
        from syrin.serve.playground import add_playground_static_mount

        app = FastAPI(
            title="Syrin Multi-Agent",
            description=f"Agents: {', '.join(a.name for a in self._serveables)}",
        )
        prefix = (cfg.route_prefix or "").strip().rstrip("/")
        app.include_router(self.fastapi_router(), prefix=prefix or "")  # type: ignore[arg-type]
        if cfg.enable_playground:
            mount_path = f"/{prefix}/playground" if prefix else "/playground"
            add_playground_static_mount(app, mount_path)
        _add_startup_endpoint_logging(app)
        uvicorn.run(app, host=cfg.host, port=cfg.port, workers=1)

    def _select_agent_cli(self) -> object:
        """Prompt user to select an agent. Returns the selected serveable."""
        print("\nSelect agent:")
        for i, s in enumerate(self._serveables, 1):
            desc = getattr(s, "description", "") or ""
            if desc and len(desc) > 50:
                desc = desc[:47] + "..."
            print(f"  {i}) {s.name}" + (f" — {desc}" if desc else ""))
        while True:
            try:
                line = input("> ").strip()
            except (KeyboardInterrupt, EOFError):
                print("\nBye.")
                raise SystemExit(0) from None
            if not line:
                continue
            try:
                idx = int(line)
                if 1 <= idx <= len(self._serveables):
                    return self._serveables[idx - 1]
            except ValueError:
                pass
            print("Invalid choice. Enter a number from the list.")

    def _serve_cli(self, cfg: ServeConfig) -> None:
        """Run CLI REPL with agent selection."""
        from syrin.serve.cli import run_cli_repl

        print("[Syrin] Multi-agent CLI. Choose an agent to chat with.")
        serveable = self._select_agent_cli()
        run_cli_repl(serveable, cfg)  # type: ignore[arg-type]

    def _serve_stdio(self, cfg: ServeConfig, stdin: object = None, stdout: object = None) -> None:
        """Run STDIO protocol with agent selection."""
        import sys

        from syrin.serve.stdio import run_stdio_protocol

        out = stdout if stdout is not None else sys.stdout
        print("[Syrin] Multi-agent STDIO. Choose an agent.", file=out)  # type: ignore[arg-type]
        serveable = self._select_agent_cli()
        run_stdio_protocol(serveable, cfg, stdin=stdin, stdout=stdout)  # type: ignore[arg-type]
