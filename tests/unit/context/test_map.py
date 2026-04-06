"""Tests for persistent context map.

ContextMap, FileContextMapBackend, get_map/update_map, inject_map_summary.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from syrin.context import Context, ContextMap, FileContextMapBackend, create_context_manager


def test_context_map_to_dict_roundtrip() -> None:
    m = ContextMap(
        topics=["a", "b"],
        decisions=["x"],
        segment_ids=["s1"],
        summary="Session summary.",
        last_updated=123.0,
    )
    d = m.to_dict()
    assert d["topics"] == ["a", "b"]
    assert d["decisions"] == ["x"]
    assert d["segment_ids"] == ["s1"]
    assert d["summary"] == "Session summary."
    assert d["last_updated"] == 123.0

    m2 = ContextMap.from_dict(d)
    assert m2.topics == m.topics
    assert m2.decisions == m.decisions
    assert m2.segment_ids == m.segment_ids
    assert m2.summary == m.summary
    assert m2.last_updated == m.last_updated


def test_context_map_from_dict_none_returns_empty() -> None:
    m = ContextMap.from_dict(None)
    assert m.topics == []
    assert m.decisions == []
    assert m.summary == ""


def test_context_map_from_dict_invalid_returns_empty() -> None:
    m = ContextMap.from_dict("not a dict")
    assert m.topics == []
    assert m.decisions == []


def test_file_context_map_backend_save_load() -> None:
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "map.json"
        backend = FileContextMapBackend(path)
        m = ContextMap(topics=["t1"], summary="Summary", last_updated=1.0)
        backend.save(m)
        loaded = backend.load()
        assert loaded.topics == ["t1"]
        assert loaded.summary == "Summary"
        assert loaded.last_updated == 1.0


def test_file_context_map_backend_missing_file_returns_empty() -> None:
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "nonexistent.json"
        backend = FileContextMapBackend(path)
        m = backend.load()
        assert m.topics == []
        assert m.summary == ""


def test_context_map_backend_creates_parent_dirs() -> None:
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "sub" / "nested" / "map.json"
        backend = FileContextMapBackend(path)
        backend.save(ContextMap(summary="x"))
        assert path.exists()


def test_context_accepts_map_config() -> None:
    ctx = Context(
        map_backend="file",
        map_path="/tmp/ctx_map.json",
        inject_map_summary=True,
    )
    assert ctx.map_backend == "file"
    assert ctx.map_path == "/tmp/ctx_map.json"
    assert ctx.inject_map_summary is True


def test_context_map_backend_invalid_raises() -> None:
    with pytest.raises(ValueError, match="map_backend must be 'file' or None"):
        Context(map_backend="redis", map_path="/tmp/x")


def test_context_map_path_required_when_file() -> None:
    with pytest.raises(ValueError, match="map_path is required"):
        Context(map_backend="file", map_path="")


def test_get_map_no_backend_returns_empty() -> None:
    ctx = Context()
    mgr = create_context_manager(ctx)
    m = mgr.get_map()
    assert isinstance(m, ContextMap)
    assert m.topics == []
    assert m.summary == ""


def test_get_map_loads_from_file() -> None:
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "map.json"
        path.write_text(json.dumps({"topics": ["x"], "summary": "Loaded"}))
        ctx = Context(map_backend="file", map_path=str(path))
        mgr = create_context_manager(ctx)
        m = mgr.get_map()
        assert m.topics == ["x"]
        assert m.summary == "Loaded"


def test_update_map_no_backend_no_op() -> None:
    ctx = Context()
    mgr = create_context_manager(ctx)
    mgr.update_map(ContextMap(summary="x"))  # no-op
    m = mgr.get_map()
    assert m.summary == ""


def test_update_map_merges_and_persists() -> None:
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "map.json"
        ctx = Context(map_backend="file", map_path=str(path))
        mgr = create_context_manager(ctx)

        mgr.update_map(ContextMap(topics=["t1"], summary="S1"))
        m = mgr.get_map()
        assert m.topics == ["t1"]
        assert m.summary == "S1"

        mgr.update_map(ContextMap(summary="S2"))  # update summary only
        m = mgr.get_map()
        assert m.topics == ["t1"]  # unchanged
        assert m.summary == "S2"


def test_update_map_accepts_dict() -> None:
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "map.json"
        ctx = Context(map_backend="file", map_path=str(path))
        mgr = create_context_manager(ctx)
        mgr.update_map({"topics": ["a"], "summary": "From dict"})
        m = mgr.get_map()
        assert m.topics == ["a"]
        assert m.summary == "From dict"


def test_inject_map_summary_injects_when_non_empty() -> None:
    """When inject_map_summary=True and map has summary, it is prepended to injected messages."""
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "map.json"
        ctx = Context(
            map_backend="file",
            map_path=str(path),
            inject_map_summary=True,
        )
        mgr = create_context_manager(ctx)
        mgr.update_map(ContextMap(summary="Prior session: user asked about Python."))

        payload = mgr.prepare(
            messages=[
                {"role": "user", "content": "Continue"},
            ],
            system_prompt="You help.",
            tools=[],
        )
        # Map summary is injected as system message (placement=BEFORE_CURRENT_TURN)
        msgs = payload.messages
        session_summary_msgs = [
            m
            for m in msgs
            if m.get("role") == "system" and "[Session summary]" in m.get("content", "")
        ]
        assert len(session_summary_msgs) >= 1
        content = session_summary_msgs[0].get("content", "")
        assert "Prior session: user asked about Python" in content


def test_inject_map_summary_empty_summary_no_inject() -> None:
    """When map summary is empty, no extra system message is added."""
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "map.json"
        ctx = Context(
            map_backend="file",
            map_path=str(path),
            inject_map_summary=True,
        )
        mgr = create_context_manager(ctx)
        # No update_map call - map is empty

        payload = mgr.prepare(
            messages=[{"role": "user", "content": "Hi"}],
            system_prompt="You help.",
            tools=[],
        )
        msgs = payload.messages
        # Only system + user; no [Session summary] block
        sys_contents = [m.get("content", "") for m in msgs if m.get("role") == "system"]
        assert not any("[Session summary]" in c for c in sys_contents)


def test_agent_context_get_map_update_map() -> None:
    """agent.context.get_map() and agent.context.update_map() work via facade."""
    from syrin import Agent, Model

    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "map.json"
        agent = Agent(
            model=Model.Almock(latency_seconds=0.01),
            context=Context(
                map_backend="file",
                map_path=str(path),
            ),
        )
        agent.context.update_map({"summary": "Test session."})
        m = agent.context.get_map()
        assert m.summary == "Test session."


def test_inject_map_summary_false_no_inject() -> None:
    """When inject_map_summary=False, map summary is never injected."""
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "map.json"
        ctx = Context(
            map_backend="file",
            map_path=str(path),
            inject_map_summary=False,
        )
        mgr = create_context_manager(ctx)
        mgr.update_map(ContextMap(summary="Should not appear."))

        payload = mgr.prepare(
            messages=[{"role": "user", "content": "Hi"}],
            system_prompt="You help.",
            tools=[],
        )
        sys_contents = [m.get("content", "") for m in payload.messages if m.get("role") == "system"]
        assert not any("[Session summary]" in c for c in sys_contents)
