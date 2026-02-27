"""Agent serving layer — HTTP, CLI, STDIO protocols."""

from syrin.serve.config import ServeConfig
from syrin.serve.http import build_router
from syrin.serve.router import AgentRouter

__all__ = [
    "AgentRouter",
    "build_router",
    "ServeConfig",
]
