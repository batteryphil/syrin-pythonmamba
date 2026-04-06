"""SwarmConfig — swarm-level configuration and validation."""

from __future__ import annotations

from typing import TYPE_CHECKING

from syrin.enums import FallbackStrategy, SwarmTopology

if TYPE_CHECKING:
    from syrin.swarm.topologies._consensus import ConsensusConfig
    from syrin.swarm.topologies._reflection import ReflectionConfig


class SwarmConfig:
    """Configuration for a :class:`~syrin.swarm.Swarm`.

    Topology-specific configuration is co-located here via the ``consensus``
    and ``reflection`` attributes.  This keeps all swarm configuration in one
    place instead of splitting it across ``Swarm()`` constructor arguments.

    Attributes:
        on_agent_failure: How to handle a failing agent.  Defaults to
            :attr:`~syrin.enums.FallbackStrategy.SKIP_AND_CONTINUE`.
        max_parallel_agents: Maximum agents allowed to run concurrently.
            Must be > 0.  Defaults to ``10``.
        timeout: Total swarm timeout in seconds, or ``None`` for no limit.
        topology: Execution topology for this swarm.  Defaults to
            :attr:`~syrin.enums.SwarmTopology.ORCHESTRATOR`.
        agent_timeout: Per-agent execution timeout in seconds.  Must be > 0
            when set.  ``None`` means no per-agent limit.
        max_agent_retries: Number of automatic retries per failing agent.
            Must be ≥ 0.  Defaults to ``0`` (no retries).
        debug: Enable verbose debug logging.  Defaults to ``False``.
        consensus: :class:`~syrin.swarm.topologies._consensus.ConsensusConfig`
            for :attr:`~syrin.enums.SwarmTopology.CONSENSUS` topology.
            Required when ``topology=SwarmTopology.CONSENSUS``.
        reflection: :class:`~syrin.swarm.topologies._reflection.ReflectionConfig`
            for :attr:`~syrin.enums.SwarmTopology.REFLECTION` topology.
            Required when ``topology=SwarmTopology.REFLECTION``.

    Examples::

        # ORCHESTRATOR — default, LLM routes tasks to agents
        config = SwarmConfig(
            topology=SwarmTopology.ORCHESTRATOR,
            max_parallel_agents=5,
            agent_timeout=30.0,
        )

        # CONSENSUS — agents vote, majority wins
        from syrin.swarm import ConsensusConfig
        config = SwarmConfig(
            topology=SwarmTopology.CONSENSUS,
            consensus=ConsensusConfig(min_agreement=0.67),
        )

        # REFLECTION — iterative writer + critic loop
        from syrin.swarm import ReflectionConfig
        config = SwarmConfig(
            topology=SwarmTopology.REFLECTION,
            reflection=ReflectionConfig(
                producer=WriterAgent,
                critic=EditorAgent,
                max_rounds=3,
                score_threshold=0.85,
            ),
        )
    """

    def __init__(
        self,
        on_agent_failure: FallbackStrategy = FallbackStrategy.SKIP_AND_CONTINUE,
        max_parallel_agents: int = 10,
        timeout: float | None = None,
        topology: SwarmTopology = SwarmTopology.ORCHESTRATOR,
        agent_timeout: float | None = None,
        max_agent_retries: int = 0,
        debug: bool = False,
        consensus: ConsensusConfig | None = None,
        reflection: ReflectionConfig | None = None,
    ) -> None:
        """Initialise SwarmConfig.

        Args:
            on_agent_failure: Fallback strategy when an agent raises.
            max_parallel_agents: Maximum concurrent agents.  Must be > 0.
            timeout: Swarm-level wall-clock timeout in seconds.
            topology: Execution topology.
            agent_timeout: Per-agent wall-clock timeout in seconds.
                Must be > 0 when provided.
            max_agent_retries: Automatic retries per failing agent (≥ 0).
            debug: Enable verbose debug output.
            consensus: Topology config for
                :attr:`~syrin.enums.SwarmTopology.CONSENSUS` runs.
                Required when ``topology=SwarmTopology.CONSENSUS``.
            reflection: Topology config for
                :attr:`~syrin.enums.SwarmTopology.REFLECTION` runs.
                Required when ``topology=SwarmTopology.REFLECTION``.

        Raises:
            ValueError: If ``max_parallel_agents`` ≤ 0, ``agent_timeout`` ≤ 0,
                or ``max_agent_retries`` < 0.
        """
        if max_parallel_agents <= 0:
            raise ValueError(f"max_parallel_agents must be > 0, got {max_parallel_agents}")
        if agent_timeout is not None and agent_timeout <= 0:
            raise ValueError(f"agent_timeout must be > 0 when set, got {agent_timeout}")
        if max_agent_retries < 0:
            raise ValueError(f"max_agent_retries must be >= 0, got {max_agent_retries}")

        self.on_agent_failure: FallbackStrategy = on_agent_failure
        self.max_parallel_agents: int = max_parallel_agents
        self.timeout: float | None = timeout
        self.topology: SwarmTopology = topology
        self.agent_timeout: float | None = agent_timeout
        self.max_agent_retries: int = max_agent_retries
        self.debug: bool = debug
        self.consensus: ConsensusConfig | None = consensus
        self.reflection: ReflectionConfig | None = reflection


__all__ = ["SwarmConfig", "FallbackStrategy"]
