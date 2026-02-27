"""Configuration for agent serving."""

from __future__ import annotations

from dataclasses import dataclass

from syrin.enums import ServeProtocol


@dataclass
class ServeConfig:
    """Configuration for agent.serve() and serving layer.

    Use when calling agent.serve(**config) or agent.serve(config=ServeConfig(...)).
    MCP routes are driven by syrin.MCP in tools — no enable_mcp flag.
    Discovery is auto-detected when agent has name; set enable_discovery=False to force off.

    CORS and auth are not handled here. Mount agent.as_router() on your own FastAPI app
    and add CORSMiddleware, OAuth, etc. from Starlette or other libraries.
    """

    protocol: ServeProtocol = ServeProtocol.HTTP
    host: str = "0.0.0.0"
    port: int = 8000
    route_prefix: str = ""
    stream: bool = True
    include_metadata: bool = True
    debug: bool = False
    enable_playground: bool = False
    enable_discovery: bool | None = None
