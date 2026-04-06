#!/usr/bin/env python3
"""Stress test remote config against a running chatbot (or any agent) server.

Run the chatbot first in another terminal:
  python -m examples.16_serving.chatbot

Then run this script:
  python -m examples.16_serving.stress_test_remote_config

Tests:
- GET /config: schema has agent.max_tool_iterations field
- PATCH agent.max_tool_iterations
- PATCH tools.<name>.enabled (toggle tool off/on)
- PATCH budget.max_cost
- Revert (value: null)
- Stress: many PATCHes in sequence

Note: budget.on_exceeded (raise/warn/stop) is a callable in code and is not exposed
as an enum in remote config; only run, reserve, per, etc. are configurable.
loop_strategy has been removed; use loop= directly on Agent(...).
"""

from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request

BASE = "http://localhost:8000"
CONFIG_URL = f"{BASE}/config"


def _read_response(resp) -> tuple[int, dict]:
    """Read response body and return (status_code, json_body)."""
    raw = resp.read().decode()
    try:
        return (int(resp.status), json.loads(raw) if raw else {})
    except json.JSONDecodeError:
        return (int(resp.status), {"_raw": raw})


def req(method: str, url: str, data: dict | None = None) -> tuple[int, dict]:
    """GET or PATCH; return (status_code, json_body)."""
    if data is not None:
        body = json.dumps(data).encode("utf-8")
        r = urllib.request.Request(url, data=body, method=method)
        r.add_header("Content-Type", "application/json")
    else:
        r = urllib.request.Request(url, method=method)
    try:
        with urllib.request.urlopen(r, timeout=10) as resp:
            return _read_response(resp)
    except urllib.error.HTTPError as e:
        raw = e.read().decode() if e.fp else "{}"
        try:
            return (e.code, json.loads(raw))
        except json.JSONDecodeError:
            return (e.code, {"error": raw})
    except Exception as e:
        return (0, {"error": str(e)})


def get_config() -> tuple[int, dict]:
    """GET /config."""
    r = urllib.request.Request(CONFIG_URL, method="GET")
    try:
        with urllib.request.urlopen(r, timeout=10) as resp:
            return _read_response(resp)
    except urllib.error.HTTPError as e:
        raw = e.read().decode() if e.fp else "{}"
        return (e.code, json.loads(raw) if raw.strip() else {})


def patch_config(agent_id: str, overrides: list[dict], version: int = 1) -> tuple[int, dict]:
    """PATCH /config with overrides (list of {path, value}). value=null for revert."""
    payload = {"agent_id": agent_id, "version": version, "overrides": overrides}
    return req("PATCH", CONFIG_URL, payload)


def main() -> int:
    print("Remote config stress test (expect server at", BASE, ")")
    print()

    # 1) GET config, assert schema has agent.max_tool_iterations
    status, data = get_config()
    if status != 200:
        print("FAIL: GET /config returned", status, data)
        return 1
    agent_id = data.get("agent_id")
    if not agent_id:
        print("FAIL: GET /config missing agent_id")
        return 1
    print("agent_id:", agent_id)

    sections = data.get("sections", {})
    agent_section = sections.get("agent", {})
    fields = agent_section.get("fields") or []
    iter_field = next((f for f in fields if f.get("path") == "agent.max_tool_iterations"), None)
    if not iter_field:
        print("FAIL: sections.agent has no field agent.max_tool_iterations")
        return 1
    # loop_strategy must NOT be present (removed)
    loop_field = next((f for f in fields if f.get("path") == "agent.loop_strategy"), None)
    if loop_field:
        print("FAIL: agent.loop_strategy must not be present (removed from remote config)")
        return 1
    print("OK: agent.max_tool_iterations present, agent.loop_strategy absent")

    # 2) Set max_tool_iterations
    status, res = patch_config(
        agent_id, [{"path": "agent.max_tool_iterations", "value": 5}], version=2
    )
    if status != 200:
        print("FAIL: PATCH max_tool_iterations -> 5:", status, res)
        return 1
    _, data2 = get_config()
    if data2.get("current_values", {}).get("agent.max_tool_iterations") != 5:
        print(
            "FAIL: current_values.agent.max_tool_iterations not 5 after PATCH:",
            data2.get("current_values"),
        )
        return 1
    print("OK: agent.max_tool_iterations -> 5 applied")

    # 3) Disable tool remember_fact
    status, res = patch_config(
        agent_id, [{"path": "tools.remember_fact.enabled", "value": False}], version=3
    )
    if status != 200:
        print("FAIL: PATCH tools.remember_fact.enabled false:", status, res)
        return 1
    _, data3 = get_config()
    if data3.get("current_values", {}).get("tools.remember_fact.enabled") is not False:
        print(
            "FAIL: tools.remember_fact.enabled not false after PATCH:", data3.get("current_values")
        )
        return 1
    print("OK: tools.remember_fact.enabled -> false applied")

    # 4) Change budget.max_cost
    status, res = patch_config(agent_id, [{"path": "budget.max_cost", "value": 0.25}], version=4)
    if status != 200:
        print("FAIL: PATCH budget.max_cost:", status, res)
        return 1
    _, data4 = get_config()
    if data4.get("current_values", {}).get("budget.max_cost") != 0.25:
        print("FAIL: budget.max_cost not 0.25 after PATCH:", data4.get("current_values"))
        return 1
    print("OK: budget.max_cost -> 0.25 applied")

    # 5) Revert max_tool_iterations (value: null)
    status, res = patch_config(
        agent_id, [{"path": "agent.max_tool_iterations", "value": None}], version=5
    )
    if status != 200:
        print("FAIL: PATCH revert max_tool_iterations:", status, res)
        return 1
    _, data5 = get_config()
    print("OK: reverted agent.max_tool_iterations")

    # 6) Re-enable tool
    status, res = patch_config(
        agent_id, [{"path": "tools.remember_fact.enabled", "value": True}], version=6
    )
    if status != 200:
        print("FAIL: PATCH tools.remember_fact.enabled true:", status, res)
        return 1
    print("OK: tools.remember_fact.enabled -> true applied")

    # 7) Stress: many PATCHes (max_tool_iterations flip, budget flip, tool flip)
    print()
    print("Stress: 20 PATCHes (max_tool_iterations, budget.max_cost, tool toggle)...")
    for i in range(20):
        v = 100 + i
        which = i % 3
        if which == 0:
            val = 5 if (i // 3) % 2 == 0 else 10
            ov = [{"path": "agent.max_tool_iterations", "value": val}]
        elif which == 1:
            ov = [{"path": "budget.max_cost", "value": 0.1 + (i % 5) * 0.1}]
        else:
            ov = [{"path": "tools.remember_fact.enabled", "value": i % 2 == 0}]
        status, res = patch_config(agent_id, ov, version=v)
        if status != 200:
            print("FAIL stress step", i, status, res)
            return 1
    print("OK: stress 20 PATCHes completed")

    print()
    print("All checks passed. Config API is working correctly.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
