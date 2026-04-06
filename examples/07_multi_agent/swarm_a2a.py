"""Agent-to-agent (A2A) messaging — typed, audited coordination between agents.

Sharing a goal is not always enough. Agents sometimes need to pass structured
data to specific peers, publish findings to a shared knowledge bus, or broadcast
phase-transition signals to multiple subscribers. A2A messaging provides a typed,
audited communication channel that operates independently of the swarm's shared goal.

A research pipeline illustrates all three patterns:
  - An OrchestratorAgent assigns tasks directly to a ResearchAgent (direct messaging).
  - The ResearchAgent publishes findings to a MemoryBus (broadcast to consumers).
  - The AnalysisAgent reads findings from the bus without needing to know who published them.
  - The orchestrator broadcasts a phase-complete signal to all topic subscribers.

Use A2A when:
  - Agents must pass structured data (not free-text) to specific peers.
  - An agent's logic depends on another agent's output, not just the shared goal.
  - You need a full audit trail of every inter-agent message.
  - You want topic-based fan-out without hardcoding subscriber lists in the sender.

Run:
    uv run python examples/07_multi_agent/swarm_a2a.py
"""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from syrin import Agent, Model
from syrin.enums import A2AChannel, MemoryType
from syrin.memory.config import MemoryEntry
from syrin.swarm import A2AConfig, A2ARouter, MemoryBus

_MODEL = Model.OpenAI("gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"))

# ── Message types ─────────────────────────────────────────────────────────────
#
# Use @dataclass for A2A message payloads — one type per message keeps the
# protocol explicit and self-documenting.


@dataclass
class ResearchTask:
    """Sent from orchestrator to researcher to kick off a research assignment."""

    task_id: str
    topic: str
    priority: int = 1


@dataclass
class ResearchFindings:
    """Sent from researcher back to orchestrator once research is complete."""

    task_id: str
    summary: str
    confidence: float = 0.9


@dataclass
class PhaseComplete:
    """Broadcast from orchestrator to all agents when a pipeline phase finishes."""

    phase: str
    next_phase: str


# ── Agent definitions ─────────────────────────────────────────────────────────
#
# These agents represent roles in the coordination pipeline. Their real LLM
# work happens via arun() when integrated into a Swarm. The demos below
# exercise the messaging protocol directly, independent of LLM execution.


class OrchestratorAgent(Agent):
    """Assigns research tasks and coordinates the pipeline phases."""

    model = _MODEL
    system_prompt = (
        "You are a research pipeline orchestrator. "
        "Assign tasks to specialist agents, track progress, and synthesise findings "
        "into a structured research brief."
    )


class ResearchAgent(Agent):
    """Gathers primary sources and publishes findings to the shared MemoryBus."""

    model = _MODEL
    system_prompt = (
        "You are a research specialist. Gather credible primary sources on your "
        "assigned topic and publish a concise findings summary with a confidence score."
    )


class AnalysisAgent(Agent):
    """Reads research from the MemoryBus and synthesises strategic insights."""

    model = _MODEL
    system_prompt = (
        "You are a strategic analyst. Read research findings from shared memory "
        "and synthesise them into three actionable recommendations with supporting evidence."
    )


# ── Demo 1: Direct messaging ───────────────────────────────────────────────────
#
# Use direct messaging to send a specific instruction or result to exactly one
# other agent. The router queues the message in the recipient's inbox.


async def demo_direct_messaging(router: A2ARouter) -> None:
    print("\n── Demo 1: Direct messaging (orchestrator ↔ researcher) ─────────")

    # Orchestrator assigns a research task
    await router.send(
        from_agent="orchestrator",
        to_agent="researcher",
        message=ResearchTask(
            task_id="t-001",
            topic="LLM inference optimisation: speculative decoding and continuous batching",
            priority=2,
        ),
    )

    envelope = await router.receive(agent_id="researcher", timeout=1.0)
    if envelope and isinstance(envelope.payload, ResearchTask):
        task = envelope.payload
        print(f"  Researcher received task: {task.topic!r}")
        print(f"  Priority: {task.priority}")

        # Researcher reports findings back
        await router.send(
            from_agent="researcher",
            to_agent="orchestrator",
            message=ResearchFindings(
                task_id=task.task_id,
                summary=(
                    "Speculative decoding reduces p50 latency by 2-3x on autoregressive models. "
                    "Continuous batching yields 10-20x throughput improvement over static batching "
                    "for production serving. Both are now standard in vLLM and TGI."
                ),
                confidence=0.91,
            ),
        )

    envelope = await router.receive(agent_id="orchestrator", timeout=1.0)
    if envelope and isinstance(envelope.payload, ResearchFindings):
        findings = envelope.payload
        print(f"  Orchestrator received: {findings.summary[:80]}...")
        print(f"  Confidence: {findings.confidence:.0%}")


# ── Demo 2: MemoryBus — publish once, query by any agent ──────────────────────
#
# Use the MemoryBus when findings should be accessible to multiple consumers.
# Producers publish without knowing who will read; consumers query by topic.
# This decouples producers from consumers — add new consumers without touching
# publisher code.


async def demo_memory_bus(bus: MemoryBus) -> None:
    print("\n── Demo 2: MemoryBus — publish findings, read by semantic query ──")

    # Researcher publishes a semantic finding
    finding = MemoryEntry(
        id="mem-inference-001",
        content=(
            "LLM inference optimisation: speculative decoding achieves 2-3x latency reduction "
            "by using a small draft model to propose tokens verified by the larger model in parallel. "
            "Continuous batching improves GPU utilisation from ~30% (static) to ~85% under load."
        ),
        type=MemoryType.KNOWLEDGE,
        importance=0.88,
        keywords=["inference", "latency", "speculative-decoding", "batching"],
        created_at=datetime.now(),
    )
    stored = await bus.publish(finding, agent_id="researcher")
    print(f"  Finding published: {stored}")

    # Analyst queries without knowing who published or where it's stored
    results = await bus.read(query="speculative decoding latency", agent_id="analyst")
    print(f"  Analyst found {len(results)} relevant finding(s):")
    for entry in results:
        print(f"    [{entry.type}] {entry.content[:100]}...")


# ── Demo 3: Topic broadcast — phase transition fan-out ────────────────────────
#
# Use broadcasts to notify multiple agents simultaneously without the sender
# needing to know the subscriber list. Subscribers opt in by topic.


async def demo_topic_broadcast(router: A2ARouter) -> None:
    print("\n── Demo 3: Topic broadcast (phase transition) ───────────────────")

    # Both analyst and reviewer subscribe to the "pipeline" topic
    router.subscribe("analyst", topic="pipeline")
    router.subscribe("reviewer", topic="pipeline")

    # Orchestrator broadcasts that the research phase is complete
    await router.send(
        from_agent="orchestrator",
        to_agent="pipeline",
        message=PhaseComplete(phase="research", next_phase="analysis"),
        channel=A2AChannel.TOPIC,
    )

    # Each subscriber receives independently — no coupling between them
    for agent_id in ("analyst", "reviewer"):
        envelope = await router.receive(agent_id=agent_id, timeout=0.5)
        if envelope and isinstance(envelope.payload, PhaseComplete):
            evt = envelope.payload
            print(
                f"  {agent_id:12s} notified: phase '{evt.phase}' complete "
                f"→ starting '{evt.next_phase}'"
            )


# ── Demo 4: Audit log ─────────────────────────────────────────────────────────
#
# Every message sent through an audit-enabled router is recorded automatically.
# Audit logs provide evidence for compliance reviews and simplify debugging
# of coordination issues in complex multi-agent pipelines.


async def demo_audit_log(router: A2ARouter) -> None:
    print("\n── Demo 4: Audit log — full inter-agent message history ─────────")
    for entry in router.audit_log():
        print(
            f"  [{entry.timestamp.strftime('%H:%M:%S.%f')[:-3]}] "
            f"{entry.from_agent:<14} → {entry.to_agent:<12} "
            f"{entry.message_type} ({entry.size_bytes}B)"
        )


# ── Main ──────────────────────────────────────────────────────────────────────


async def main() -> None:
    print("=== A2A Messaging and MemoryBus ===")

    # Router with full audit logging — every send() is recorded
    config = A2AConfig(audit_all=True, max_queue_depth=100)
    router = A2ARouter(config=config)

    for agent_id in ("orchestrator", "researcher", "analyst", "reviewer"):
        router.register_agent(agent_id)

    # MemoryBus restricted to SEMANTIC knowledge entries
    # Entries tagged "private" are blocked from the shared bus
    bus = MemoryBus(
        allow_types=[MemoryType.KNOWLEDGE],
        filter=lambda e: "private" not in e.keywords,
    )

    await demo_direct_messaging(router)
    await demo_memory_bus(bus)
    await demo_topic_broadcast(router)
    await demo_audit_log(router)

    print("\nAll A2A demos completed.")


if __name__ == "__main__":
    asyncio.run(main())
