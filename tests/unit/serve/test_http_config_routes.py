"""Tests for HTTP config routes: GET/PATCH /config, GET /config/stream."""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi")

from starlette.testclient import TestClient

from syrin import Agent, Budget, Model
from syrin.serve.config import ServeConfig
from syrin.serve.http import build_router


class _TestAgentWithBudget(Agent):
    """Agent with budget for config override tests."""

    name = "config-test-agent"
    description = "Config test"
    model = Model.Almock()


def test_get_config_returns_schema_and_values() -> None:
    """GET /config returns schema, baseline_values, overrides, current_values, and per-field values."""
    agent = _TestAgentWithBudget(budget=Budget(max_cost=1.0))
    config = ServeConfig()
    router = build_router(agent, config)
    from fastapi import FastAPI

    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)
    r = client.get("/config")
    assert r.status_code == 200
    data = r.json()
    assert "sections" in data and "agent_id" in data
    assert "baseline_values" in data
    assert "overrides" in data
    assert "current_values" in data
    assert data["agent_id"] == "config-test-agent:Agent" or "config-test-agent" in data["agent_id"]
    assert data["current_values"].get("budget.max_cost") == 1.0
    assert data["baseline_values"].get("budget.max_cost") == 1.0
    assert data["overrides"] == {}
    budget_section = data["sections"].get("budget", {})
    run_field = next(
        (f for f in budget_section.get("fields", []) if f.get("path") == "budget.max_cost"),
        None,
    )
    assert run_field is not None
    assert run_field.get("baseline_value") == 1.0
    assert run_field.get("current_value") == 1.0
    assert run_field.get("overridden") is False
    # Agent section must NOT expose loop_strategy (removed; use loop= directly)
    agent_section = data["sections"].get("agent", {})
    loop_field = next(
        (f for f in agent_section.get("fields", []) if f.get("path") == "agent.loop_strategy"),
        None,
    )
    assert loop_field is None, "agent.loop_strategy field must be removed"


def test_get_config_with_route_prefix() -> None:
    """GET /config with route_prefix uses prefixed path."""
    agent = _TestAgentWithBudget()
    config = ServeConfig(route_prefix="/api/v1")
    router = build_router(agent, config)
    from fastapi import FastAPI

    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)
    r = client.get("/api/v1/config")
    assert r.status_code == 200


def test_patch_config_applies_overrides() -> None:
    """PATCH /config with OverridePayload applies overrides via resolver."""
    agent = _TestAgentWithBudget(budget=Budget(max_cost=0.5))
    config = ServeConfig()
    router = build_router(agent, config)
    from fastapi import FastAPI

    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)
    # Get agent_id from GET config first, or use convention
    get_r = client.get("/config")
    assert get_r.status_code == 200
    get_data = get_r.json()
    agent_id = get_data.get("agent_id", "config-test-agent:Agent")
    payload = {
        "agent_id": agent_id,
        "version": 1,
        "overrides": [{"path": "budget.max_cost", "value": 2.0}],
    }
    r = client.patch("/config", json=payload)
    assert r.status_code == 200
    # Agent's budget.max_cost should now be 2.0
    assert agent._budget.max_cost == 2.0


def test_patch_agent_max_tool_iterations_via_config_route() -> None:
    """PATCH agent.max_tool_iterations via /config route is accepted."""
    agent = _TestAgentWithBudget(budget=Budget(max_cost=1.0))
    config = ServeConfig()
    router = build_router(agent, config)
    from fastapi import FastAPI

    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)
    get_r = client.get("/config")
    assert get_r.status_code == 200
    agent_id = get_r.json().get("agent_id", "config-test-agent:Agent")
    payload = {
        "agent_id": agent_id,
        "version": 1,
        "overrides": [{"path": "agent.max_tool_iterations", "value": 5}],
    }
    r = client.patch("/config", json=payload)
    assert r.status_code == 200
    data = r.json()
    assert "agent.max_tool_iterations" in data.get("accepted", [])
    assert agent._max_tool_iterations == 5


def test_patch_config_invalid_path_rejected() -> None:
    """PATCH /config with unknown path returns 4xx or success with rejected in body."""
    agent = _TestAgentWithBudget(budget=Budget(max_cost=0.5))
    config = ServeConfig()
    router = build_router(agent, config)
    from fastapi import FastAPI

    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)
    get_r = client.get("/config")
    agent_id = get_r.json().get("agent_id", "config-test-agent:Agent")
    payload = {
        "agent_id": agent_id,
        "version": 1,
        "overrides": [{"path": "nonexistent.foo", "value": 1}],
    }
    r = client.patch("/config", json=payload)
    # Either 400/422 or 200 with result indicating rejected
    assert r.status_code in (200, 400, 422)
    if r.status_code == 200:
        data = r.json()
        if "rejected" in data:
            assert len(data["rejected"]) > 0
    assert agent._budget.max_cost == 0.5  # unchanged


def test_patch_config_invalid_value_rejected() -> None:
    """PATCH /config with value that fails validation leaves agent unchanged and rolls back store."""
    agent = _TestAgentWithBudget(budget=Budget(max_cost=0.5))
    config = ServeConfig()
    router = build_router(agent, config)
    from fastapi import FastAPI

    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)
    get_r = client.get("/config")
    agent_id = get_r.json().get("agent_id", "config-test-agent:Agent")
    # budget.max_cost has ge=0; -1 should be rejected
    payload = {
        "agent_id": agent_id,
        "version": 1,
        "overrides": [{"path": "budget.max_cost", "value": -1.0}],
    }
    r = client.patch("/config", json=payload)
    assert r.status_code in (200, 400, 422)
    assert agent._budget.max_cost == 0.5  # unchanged
    # Rejected path rolled back from store; GET reflects baseline
    get_after = client.get("/config").json()
    assert get_after["current_values"].get("budget.max_cost") == 0.5
    assert "budget.max_cost" not in get_after.get("overrides", {})


@pytest.mark.skip(
    reason="TestClient blocks until stream ends; /config/stream is infinite. Tested manually or via integration."
)
def test_config_stream_returns_sse() -> None:
    """GET /config/stream returns 200 with text/event-stream and yields SSE (heartbeat, then override events)."""
    agent = _TestAgentWithBudget(budget=Budget(max_cost=0.5))
    config = ServeConfig()
    router = build_router(agent, config)
    from fastapi import FastAPI

    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)
    r = client.get("/config/stream")
    assert r.status_code == 200
    assert "text/event-stream" in r.headers.get("content-type", "")


def test_patch_config_empty_overrides_succeeds() -> None:
    """PATCH /config with empty overrides list returns 200."""
    agent = _TestAgentWithBudget(budget=Budget(max_cost=0.5))
    config = ServeConfig()
    router = build_router(agent, config)
    from fastapi import FastAPI

    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)
    get_r = client.get("/config")
    agent_id = get_r.json().get("agent_id", "config-test-agent:Agent")
    r = client.patch("/config", json={"agent_id": agent_id, "version": 1, "overrides": []})
    assert r.status_code == 200
    assert agent._budget.max_cost == 0.5


def test_patch_then_revert_restores_baseline() -> None:
    """PATCH with value then PATCH with value=null (revert) restores baseline; GET reflects it."""
    agent = _TestAgentWithBudget(budget=Budget(max_cost=1.0))
    config = ServeConfig()
    router = build_router(agent, config)
    from fastapi import FastAPI

    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)
    get_r = client.get("/config")
    assert get_r.status_code == 200
    data = get_r.json()
    agent_id = data["agent_id"]
    assert data["baseline_values"].get("budget.max_cost") == 1.0
    # Override budget.max_cost to 2.0
    patch_r = client.patch(
        "/config",
        json={
            "agent_id": agent_id,
            "version": 1,
            "overrides": [{"path": "budget.max_cost", "value": 2.0}],
        },
    )
    assert patch_r.status_code == 200
    assert agent._budget.max_cost == 2.0
    get2 = client.get("/config").json()
    assert get2["current_values"].get("budget.max_cost") == 2.0
    assert get2["overrides"].get("budget.max_cost") == 2.0
    run_field = next(
        (f for f in get2["sections"]["budget"]["fields"] if f["path"] == "budget.max_cost"),
        None,
    )
    assert run_field is not None and run_field["overridden"] is True
    # Revert: value=null removes override
    patch2 = client.patch(
        "/config",
        json={
            "agent_id": agent_id,
            "version": 2,
            "overrides": [{"path": "budget.max_cost", "value": None}],
        },
    )
    assert patch2.status_code == 200
    assert agent._budget.max_cost == 1.0
    get3 = client.get("/config").json()
    assert get3["current_values"].get("budget.max_cost") == 1.0
    assert "budget.max_cost" not in get3["overrides"]
    run_field3 = next(
        (f for f in get3["sections"]["budget"]["fields"] if f["path"] == "budget.max_cost"),
        None,
    )
    assert run_field3 is not None and run_field3["overridden"] is False
    assert run_field3["current_value"] == run_field3["baseline_value"] == 1.0


class TestRemoteConfigE2EFullFeatures:
    """E2E: serve agent with guardrails, template_variables, tools; override via PATCH; assert state."""

    def test_full_features_get_config_has_all_sections(self) -> None:
        """GET /config returns guardrails, template_variables, tools sections when agent has them."""
        from syrin.guardrails.built_in import PIIScanner
        from syrin.tool import tool

        @tool
        def alpha() -> str:
            return "a"

        agent = Agent(
            model=Model.Almock(),
            name="e2e_agent",
            budget=Budget(max_cost=1.0),
            system_prompt="Hi.",
            tools=[alpha],
            guardrails=[PIIScanner()],
            template_variables={"x": "y"},
        )
        config = ServeConfig()
        router = build_router(agent, config)
        from fastapi import FastAPI

        app = FastAPI()
        app.include_router(router)
        client = TestClient(app)
        r = client.get("/config")
        assert r.status_code == 200
        data = r.json()
        assert "sections" in data
        sections = data["sections"]
        assert "guardrails" in sections
        assert "template_variables" in sections
        assert "tools" in sections
        assert "budget" in sections
        assert "agent" in sections
        assert "current_values" in data
        cv = data["current_values"]
        assert cv.get("template_variables.x") == "y"
        assert cv.get("tools.alpha.enabled") is True
        assert cv.get("guardrails.PIIScanner.enabled") is True

    def test_full_features_patch_overrides_reflected_on_agent(self) -> None:
        """PATCH guardrails/tools/template_variables overrides; agent state and GET /config reflect them."""
        from syrin.guardrails.built_in import PIIScanner
        from syrin.tool import tool

        @tool
        def alpha() -> str:
            return "a"

        @tool
        def beta() -> str:
            return "b"

        agent = Agent(
            model=Model.Almock(),
            name="e2e_patch_agent",
            budget=Budget(max_cost=1.0),
            system_prompt="Hi. Env: {env}.",
            tools=[alpha, beta],
            guardrails=[PIIScanner()],
            template_variables={"env": "staging"},
        )
        config = ServeConfig()
        router = build_router(agent, config)
        from fastapi import FastAPI

        app = FastAPI()
        app.include_router(router)
        client = TestClient(app)

        # 1) GET config, obtain agent_id
        get_r = client.get("/config")
        assert get_r.status_code == 200
        get_data = get_r.json()
        agent_id = get_data["agent_id"]
        assert "e2e_patch_agent" in agent_id

        # 2) Apply overrides: disable PIIScanner, set template_variables.env, disable tool alpha
        payload = {
            "agent_id": agent_id,
            "version": 1,
            "overrides": [
                {"path": "guardrails.PIIScanner.enabled", "value": False},
                {"path": "template_variables.env", "value": "prod"},
                {"path": "tools.alpha.enabled", "value": False},
            ],
        }
        patch_r = client.patch("/config", json=payload)
        assert patch_r.status_code == 200
        patch_data = patch_r.json()
        assert "guardrails.PIIScanner.enabled" in patch_data["accepted"]
        assert "template_variables.env" in patch_data["accepted"]
        assert "tools.alpha.enabled" in patch_data["accepted"]
        assert len(patch_data["rejected"]) == 0

        # 3) Assert agent state
        assert "PIIScanner" in agent._guardrails_disabled
        assert agent._template_vars.get("env") == "prod"
        assert "alpha" in agent._tools_disabled
        assert len(agent.tools) == 1
        assert agent.tools[0].name == "beta"

        # 4) GET config again — current_values should reflect overrides
        get2_r = client.get("/config")
        assert get2_r.status_code == 200
        cv = get2_r.json()["current_values"]
        assert cv.get("guardrails.PIIScanner.enabled") is False
        assert cv.get("template_variables.env") == "prod"
        assert cv.get("tools.alpha.enabled") is False
        assert cv.get("tools.beta.enabled") is True
