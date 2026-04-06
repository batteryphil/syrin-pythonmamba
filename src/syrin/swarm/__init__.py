"""syrin.swarm — Multi-agent Swarm orchestration.

A :class:`Swarm` groups multiple agents under a shared goal with shared
budget pools, selective cross-agent memory sharing via :class:`MemoryBus`,
direct agent-to-agent messaging, and graceful degradation on failure.

Topologies:
    - :attr:`~syrin.enums.SwarmTopology.PARALLEL` — all agents run concurrently.
    - :attr:`~syrin.enums.SwarmTopology.CONSENSUS` — agents vote; winner by strategy.
    - :attr:`~syrin.enums.SwarmTopology.REFLECTION` — producer–critic iterative loop.

Authority and control:
    - :class:`SwarmAuthorityGuard` — RBAC permission gate for all control actions.
    - :class:`SwarmController` — Control actions (pause/resume/kill/skip/change_context).
    - :class:`BroadcastBus` — Topic-based pub-sub for in-swarm broadcasts.
    - :class:`MonitorLoop` — Async supervisor loop with heartbeat and intervention.

Which messaging API should I use?
----------------------------------

.. list-table::
   :header-rows: 1

   * - Situation
     - Use
     - Methods
   * - Agent A sends a typed message to Agent B (pull-based)
     - :class:`A2ARouter`
     - ``.send()`` / ``.receive()`` / ``.send_with_ack()``
   * - One agent announces to *every* registered agent
     - :class:`A2ARouter` with ``channel=A2AChannel.BROADCAST``
     - ``.send(..., channel=A2AChannel.BROADCAST)``
   * - Fan-out to agents subscribed to a named topic (pull-based)
     - :class:`A2ARouter` with ``channel=A2AChannel.TOPIC``
     - ``.subscribe()`` + ``.send(..., channel=A2AChannel.TOPIC)``
   * - Fire-and-forget pub-sub with *push* callbacks (``fnmatch`` patterns)
     - :class:`BroadcastBus`
     - ``.subscribe()`` / ``.broadcast()``
   * - Share typed memory facts across agents (persistent, TTL-aware)
     - :class:`MemoryBus`
     - ``.publish()`` / ``.read()``

**A2ARouter and MemoryBus do NOT require a Swarm.**  Use them standalone
for any 2+ agent system::

    from syrin.swarm import A2ARouter
    from pydantic import BaseModel

    class Ping(BaseModel):
        text: str

    router = A2ARouter()
    router.register_agent("alice")
    router.register_agent("bob")

    await router.send(from_agent="alice", to_agent="bob", message=Ping(text="hi"))

    envelope = await router.receive(agent_id="bob", timeout=5.0)
    if envelope:
        msg = envelope.get_typed_payload(Ping)   # type-safe access
        print(msg.text)                           # "hi"
        await router.ack(agent_id="bob", message_id=envelope.message_id)
"""

from syrin.enums import ControlAction
from syrin.swarm._a2a import (
    A2AAuditEntry,
    A2ABudgetExceededError,
    A2AConfig,
    A2AMessageEnvelope,
    A2AMessageTooLarge,
    A2ARouter,
    A2ATimeoutError,
)
from syrin.swarm._agent_ref import AgentRef
from syrin.swarm._authority import (
    AgentPermissionError,
    AuditEntry,
    SwarmAuthorityGuard,
    build_guard_from_agents,
)
from syrin.swarm._broadcast import (
    BroadcastBus,
    BroadcastConfig,
    BroadcastEvent,
    BroadcastPayloadTooLarge,
)
from syrin.swarm._config import FallbackStrategy, SwarmConfig
from syrin.swarm._control import AgentStateSnapshot, SwarmController
from syrin.swarm._core import Swarm, SwarmRunHandle
from syrin.swarm._handoff import SwarmHandoffContext
from syrin.swarm._memory_bus import MemoryBus
from syrin.swarm._monitor import MaxInterventionsExceeded, MonitorEvent, MonitorLoop
from syrin.swarm._registry import AgentRegistry, AgentSummary
from syrin.swarm._result import (
    AgentBudgetSummary,
    AgentStatusEntry,
    SwarmBudgetReport,
    SwarmResult,
)
from syrin.swarm._spawn import SpawnResult, SpawnSpec
from syrin.swarm.topologies._consensus import (
    ConsensusConfig,
    ConsensusResult,
    ConsensusVote,
)
from syrin.swarm.topologies._reflection import (
    ReflectionConfig,
    ReflectionResult,
    RoundOutput,
)

__all__ = [
    "A2AAuditEntry",
    "AgentRef",
    "A2ABudgetExceededError",
    "A2AConfig",
    "A2AMessageEnvelope",
    "A2AMessageTooLarge",
    "A2ARouter",
    "A2ATimeoutError",
    "AgentBudgetSummary",
    "AgentPermissionError",
    "AgentRegistry",
    "AgentStateSnapshot",
    "AgentStatusEntry",
    "AgentSummary",
    "AuditEntry",
    "BroadcastBus",
    "BroadcastConfig",
    "BroadcastEvent",
    "BroadcastPayloadTooLarge",
    "ConsensusConfig",
    "ConsensusResult",
    "ConsensusVote",
    "ControlAction",
    "FallbackStrategy",
    "MaxInterventionsExceeded",
    "MemoryBus",
    "MonitorEvent",
    "MonitorLoop",
    "ReflectionConfig",
    "ReflectionResult",
    "RoundOutput",
    "SpawnResult",
    "SpawnSpec",
    "Swarm",
    "SwarmAuthorityGuard",
    "SwarmBudgetReport",
    "SwarmConfig",
    "SwarmController",
    "SwarmResult",
    "SwarmHandoffContext",
    "SwarmRunHandle",
    "build_guard_from_agents",
]
