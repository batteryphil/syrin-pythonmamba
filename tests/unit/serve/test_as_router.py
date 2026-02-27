"""Tests for agent.as_router() — returns FastAPI APIRouter."""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi")

from starlette.testclient import TestClient

from syrin.agent import Agent
from syrin.model import Model
from syrin.serve.config import ServeConfig


class _TestAgent(Agent):
    name = "test-agent"
    description = "Test agent"
    model = Model.Almock()


def test_as_router_returns_router() -> None:
    """agent.as_router() returns an APIRouter with routes."""
    agent = _TestAgent()
    router = agent.as_router()
    assert router is not None
    assert hasattr(router, "routes")
    assert len(router.routes) >= 6


def test_as_router_with_config() -> None:
    """agent.as_router(config=ServeConfig(...)) uses the config."""
    agent = _TestAgent()
    config = ServeConfig(route_prefix="/api/v1")
    router = agent.as_router(config=config)
    from fastapi import FastAPI

    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)
    r = client.get("/api/v1/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_as_router_with_kwargs() -> None:
    """agent.as_router(route_prefix=...) uses kwargs as ServeConfig."""
    agent = _TestAgent()
    router = agent.as_router(route_prefix="/agent/test")
    from fastapi import FastAPI

    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)
    r = client.get("/agent/test/health")
    assert r.status_code == 200


def test_as_router_mount_on_existing_app() -> None:
    """agent.as_router() can be mounted on an existing FastAPI app."""
    from fastapi import FastAPI

    agent = _TestAgent()
    app = FastAPI(title="My API")
    app.include_router(agent.as_router(), prefix="/agent")
    client = TestClient(app)
    r = client.get("/agent/health")
    assert r.status_code == 200
    r2 = client.get("/agent/describe")
    assert r2.status_code == 200
    assert r2.json()["name"] == "test-agent"
