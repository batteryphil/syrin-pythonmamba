"""Audit logging for compliance and observability.

AuditLog persists lifecycle events (LLM calls, tool calls, handoffs, spawns)
to JSONL files or custom backends. Use for compliance, debugging, and
cost attribution.

Run: python examples/10_observability/audit_logging.py
"""

import syrin
from syrin import Agent, AuditLog, Model, Pipeline
from syrin.agent.multi_agent import DynamicPipeline


def main() -> None:
    # Agent with audit - writes to ./audit_agent.jsonl
    audit = AuditLog(path="./audit_agent.jsonl")
    agent = Agent(
        model=Model.Almock(),
        system_prompt="You are helpful.",
        audit=audit,
    )
    agent.response("What is 2+2?")
    print("Agent audit written to ./audit_agent.jsonl")

    # Pipeline with audit
    pipeline_audit = AuditLog(path="./audit_pipeline.jsonl")

    class Writer(Agent):
        model = Model.Almock()
        system_prompt = "Write concisely."

    pipeline = Pipeline(audit=pipeline_audit)
    pipeline.run([(Writer, "Say hello in one word")])
    print("Pipeline audit written to ./audit_pipeline.jsonl")

    # DynamicPipeline with audit
    dyn_audit = AuditLog(path="./audit_dynamic.jsonl")
    dyn = DynamicPipeline(
        agents=[Writer],
        model=Model.Almock(),
        audit=dyn_audit,
    )
    dyn.run("Greet the user")
    print("DynamicPipeline audit written to ./audit_dynamic.jsonl")

    # Query entries (built-in JSONL backend)
    backend = audit.get_backend()
    entries = backend.query(syrin.AuditFilters(limit=5))
    print(f"\nLast 5 agent audit entries: {len(entries)}")
    for e in entries[:3]:
        print(f"  {e.timestamp} | {e.event} | {e.source}")


if __name__ == "__main__":
    main()
