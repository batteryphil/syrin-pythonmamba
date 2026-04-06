# 07_multi_agent — Multi-agent orchestration

## Spawn & Handoff

- **spawn.py** — Parent spawns child agent (task delegation vs. no-task for independent use)
- **handoff.py** — Transfer control to a specialist agent mid-conversation

## Workflow (deterministic sequential/branching)

- **workflow_sequential.py** — Chain agents where each step feeds the next
- **workflow_parallel.py** — Run independent steps concurrently inside a workflow
- **workflow_conditional.py** — Branch to different agents based on output
- **workflow_dynamic.py** — Runtime-constructed workflow steps
- **workflow_nested.py** — Workflows inside workflows for complex orchestration
- **workflow_lifecycle.py** — Lifecycle hooks on a workflow (start, step, end)
- **workflow_visualization.py** — Render a workflow execution graph

## Swarm

- **swarm_parallel.py** — Multiple agents share a goal, run simultaneously
- **swarm_orchestrator.py** — LLM lead agent delegates to specialists dynamically
- **swarm_consensus.py** — Independent agents vote; majority/threshold wins
- **swarm_reflection.py** — Producer/critic loop with iterative quality improvement
- **swarm_multi_model.py** — Mix models across agents (cheap for extraction, best for synthesis)
- **hierarchical_swarm.py** — Nested swarms with parent/child budget delegation
- **swarm_authority.py** — Authority levels and trust between agents
- **swarm_a2a.py** — Agent-to-agent typed messaging (A2A protocol)

## AgentRouter (LLM-planned dynamic execution)

- **agent_router.py** — LLM picks which agents to run and assigns tasks dynamically

## Team & Budget

- **team.py** — Shared-goal swarm with per-agent budgets and hooks
- **budget_delegation.py** — Parent assigns budget slices to children

## Observability & Control

- **agent_broadcast.py** — Broadcast messages to all agents in a group
- **agent_authority.py** — Role-based authority and permission levels
- **monitor_loop.py** — Watch agent activity in real time
