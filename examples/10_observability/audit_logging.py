"""Audit logging for compliance and observability.

AuditLog persists lifecycle events (LLM calls, tool calls, handoffs, spawns)
to JSONL files or custom backends. Use for compliance, debugging, and
cost attribution.

Run: python examples/10_observability/audit_logging.py
Visit: http://localhost:8000/playground
Requires: uv pip install syrin[serve]
"""

import syrin
from syrin import Agent, AgentRouter, AuditLog, Model


def main() -> None:
    # Agent with audit - writes to ./audit_agent.jsonl
    audit = AuditLog(path="./audit_agent.jsonl")
    agent = Agent(
        model=Model.mock(),
        system_prompt="You are helpful.",
        audit=audit,
    )
    agent.run("What is 2+2?")
    print("Agent audit written to ./audit_agent.jsonl")

    # AgentRouter with audit - writes to ./audit_router.jsonl
    router_audit = AuditLog(path="./audit_router.jsonl")

    class Writer(Agent):
        model = Model.mock()
        system_prompt = "Write concisely."

    class Analyst(Agent):
        model = Model.mock()
        system_prompt = "Analyse data concisely."

    router = AgentRouter(
        agents=[Writer, Analyst],
        model=Model.mock(),
        audit=router_audit,
    )
    router.run("Greet the user")
    print("AgentRouter audit written to ./audit_router.jsonl")

    # Query entries (built-in JSONL backend)
    backend = audit.get_backend()
    entries = backend.query(syrin.AuditFilters(limit=5))
    print(f"\nLast 5 agent audit entries: {len(entries)}")
    for e in entries[:3]:
        print(f"  {e.timestamp} | {e.event} | {e.source}")


class AuditDemoAgent(syrin.Agent):
    name = "audit-agent"
    description = "Agent with audit logging"
    model = syrin.Model.mock()
    system_prompt = "You are helpful."


if __name__ == "__main__":
    main()
    audit = syrin.AuditLog(path="./audit_serve.jsonl")
    agent = AuditDemoAgent(audit=audit)
    print("Serving at http://localhost:8000/playground")
    agent.serve(port=8000, enable_playground=True, debug=True)
