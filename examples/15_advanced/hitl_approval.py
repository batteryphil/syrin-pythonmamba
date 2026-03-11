"""Human-in-the-Loop Approval Example.

Demonstrates:
- Per-tool requires_approval with ApprovalGate
- Approval callback that controls whether a tool call proceeds
- Hooks: HITL_PENDING, HITL_APPROVED, HITL_REJECTED

Run: python examples/15_advanced/hitl_approval.py
"""

from syrin import Agent, ApprovalGate, Hook, Model, tool

mock = Model.Almock(latency_seconds=0.01, lorem_length=40)


# --- Tools: delete_record requires approval, search does not ---

@tool(requires_approval=True, description="Delete a record by ID")
def delete_record(id: str) -> str:
    return f"Deleted record {id}"


@tool(description="Search for records")
def search(query: str) -> str:
    return f"Results for: {query}"


# --- Approval callback ---

def approve_cb(msg: str, timeout: int, ctx: dict) -> bool:
    """In production this would prompt a human, post to Slack, etc."""
    print(f"  [HITL] Approval requested: {msg[:60]}")
    return True  # Auto-approve for demo


gate = ApprovalGate(callback=approve_cb)


class HITLAgent(Agent):
    _agent_name = "hitl-agent"
    _agent_description = "Agent with human-in-the-loop approval"
    model = mock
    system_prompt = "Use delete_record to delete, search to find."
    tools = [delete_record, search]
    approval_gate = gate
    human_approval_timeout = 60


if __name__ == "__main__":
    agent = HITLAgent()

    # Listen for HITL lifecycle hooks
    agent.events.on(Hook.HITL_PENDING, lambda ctx: print(f"  [HITL PENDING]  {ctx.get('name')}"))
    agent.events.on(Hook.HITL_APPROVED, lambda ctx: print(f"  [HITL APPROVED] {ctx.get('name')}"))

    print("--- Human-in-the-Loop Approval Example ---")
    r = agent.response("Delete record abc123")
    print(f"Result: {r.content[:120]}")
    print(f"Cost: ${r.cost:.6f}")
    print("Done.")

    # Optional: serve the agent with playground UI
    # agent.serve(port=8000, enable_playground=True, debug=True)
