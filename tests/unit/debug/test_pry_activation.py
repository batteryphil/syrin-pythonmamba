"""Tests that pry=True activates correctly on all primitives.

Exit criteria:
- pry=True on Workflow, Swarm, and single Agent — all activate correctly
"""

from __future__ import annotations

from syrin import Agent, Model
from syrin.enums import SwarmTopology
from syrin.swarm import Swarm, SwarmConfig
from syrin.workflow._core import Workflow


def _model() -> Model:
    return Model.Almock(latency_seconds=0.01, lorem_length=2)


class _StubAgent(Agent):
    model = _model()
    system_prompt = "stub"


# ---------------------------------------------------------------------------
# Workflow pry=True
# ---------------------------------------------------------------------------


def test_workflow_pry_true_stores_flag() -> None:
    """Workflow(pry=True) stores _pry=True."""
    wf = Workflow("wf", pry=True)
    assert wf._pry is True


def test_workflow_pry_false_is_default() -> None:
    """Workflow() has pry=False by default."""
    wf = Workflow("wf")
    assert wf._pry is False


# ---------------------------------------------------------------------------
# Swarm pry=True
# ---------------------------------------------------------------------------


def test_swarm_pry_true_stores_flag() -> None:
    """Swarm(..., pry=True) stores pry=True."""
    swarm = Swarm(
        agents=[_StubAgent],
        goal="test",
        config=SwarmConfig(topology=SwarmTopology.PARALLEL),
        pry=True,
    )
    assert swarm.pry is True


def test_swarm_pry_false_is_default() -> None:
    """Swarm() has pry=False by default."""
    swarm = Swarm(agents=[_StubAgent], goal="test")
    assert swarm.pry is False


# ---------------------------------------------------------------------------
# Agent pry=True (instance parameter)
# ---------------------------------------------------------------------------


def test_agent_pry_true_stores_flag() -> None:
    """Agent(pry=True) stores _pry=True."""
    agent = _StubAgent(pry=True)
    assert agent._pry is True


def test_agent_pry_false_is_default() -> None:
    """Agent() has pry=False by default."""
    agent = _StubAgent()
    assert agent._pry is False


# ---------------------------------------------------------------------------
# All primitives: pry=True does not cause errors at construction time
# ---------------------------------------------------------------------------


def test_all_primitives_pry_true_no_crash() -> None:
    """pry=True on all supported primitives constructs without raising."""
    Workflow("wf", pry=True).step(_StubAgent)
    Swarm(agents=[_StubAgent], goal="test", pry=True)
    _StubAgent(pry=True)
