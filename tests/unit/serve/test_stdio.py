"""Tests for STDIO JSON lines protocol (ServeProtocol.STDIO)."""

from __future__ import annotations

import io
import json

from syrin.agent import Agent
from syrin.enums import ServeProtocol
from syrin.model import Model
from syrin.serve.config import ServeConfig
from syrin.serve.stdio import run_stdio_protocol


class _TestAgent(Agent):
    _agent_name = "test-agent"
    _agent_description = "Test agent"
    model = Model.Almock()
    system_prompt = "You are helpful."


def test_stdio_one_input_one_output() -> None:
    """STDIO reads one JSON line, runs agent, writes one JSON line."""
    agent = _TestAgent()
    config = ServeConfig()
    stdin = io.StringIO('{"input": "Hi"}\n')
    stdout = io.StringIO()
    run_stdio_protocol(agent, config, stdin=stdin, stdout=stdout)
    lines = [ln for ln in stdout.getvalue().strip().split("\n") if ln]
    assert len(lines) >= 1
    out = json.loads(lines[0])
    assert "content" in out
    assert isinstance(out["content"], str)
    assert len(out["content"]) > 0
    assert "cost" in out
    assert "tokens" in out


def test_stdio_with_conversation_id() -> None:
    """STDIO passes conversation_id through to output."""
    agent = _TestAgent()
    config = ServeConfig()
    stdin = io.StringIO('{"message": "Hello", "conversation_id": "conv-123"}\n')
    stdout = io.StringIO()
    run_stdio_protocol(agent, config, stdin=stdin, stdout=stdout)
    out = json.loads(stdout.getvalue().strip())
    assert out.get("conversation_id") == "conv-123"
    assert "content" in out


def test_stdio_invalid_json_returns_error() -> None:
    """STDIO returns error for invalid JSON."""
    agent = _TestAgent()
    config = ServeConfig()
    stdin = io.StringIO("{invalid}\n")
    stdout = io.StringIO()
    run_stdio_protocol(agent, config, stdin=stdin, stdout=stdout)
    out = json.loads(stdout.getvalue().strip())
    assert "error" in out
    assert "Invalid JSON" in out["error"]


def test_stdio_missing_input_returns_error() -> None:
    """STDIO returns error when input/message/content is missing."""
    agent = _TestAgent()
    config = ServeConfig()
    stdin = io.StringIO('{"foo": "bar"}\n')
    stdout = io.StringIO()
    run_stdio_protocol(agent, config, stdin=stdin, stdout=stdout)
    out = json.loads(stdout.getvalue().strip())
    assert "error" in out
    assert "Missing" in out["error"]


def test_agent_serve_stdio_runs_protocol() -> None:
    """agent.serve(protocol=STDIO) runs STDIO protocol."""
    agent = _TestAgent()
    stdin = io.StringIO('{"input": "Test"}\n')
    stdout = io.StringIO()
    agent.serve(protocol=ServeProtocol.STDIO, stdin=stdin, stdout=stdout)
    out = json.loads(stdout.getvalue().strip())
    assert "content" in out or "error" in out
