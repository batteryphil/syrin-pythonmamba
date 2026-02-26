"""Human-in-the-Loop Approval Example (v0.3.0+).

Demonstrates:
- Per-tool requires_approval with ApprovalGate
- HumanInTheLoop loop for all-tools approval
- Hooks: HITL_PENDING, HITL_APPROVED, HITL_REJECTED

Run: python -m examples.15_advanced.hitl_approval
"""

from pathlib import Path

from dotenv import load_dotenv

from examples.models.models import almock
from syrin import Agent, ApprovalGate, Hook, tool

load_dotenv(Path(__file__).resolve().parent.parent / ".env")


# Option 1: Per-tool approval (ReactLoop + requires_approval)
@tool(requires_approval=True, description="Delete a record by ID")
def delete_record(id: str) -> str:
    return f"Deleted record {id}"


@tool(description="Search for records")
def search(query: str) -> str:
    return f"Results for: {query}"


def approve_cb(msg: str, timeout: int, ctx: dict) -> bool:
    print(f"  [HITL] {msg[:60]}...")
    return True  # Auto-approve for demo; in production: prompt, Slack, etc.


gate = ApprovalGate(callback=approve_cb)
agent = Agent(
    model=almock,
    system_prompt="Use delete_record to delete, search to find.",
    tools=[delete_record, search],
    approval_gate=gate,
    hitl_timeout=60,
)

agent.events.on(Hook.HITL_PENDING, lambda ctx: print(f"  [HITL PENDING] {ctx.get('name')}"))
agent.events.on(Hook.HITL_APPROVED, lambda ctx: print(f"  [HITL APPROVED] {ctx.get('name')}"))

r = agent.response("Delete record abc123")
print(f"Result: {r.content[:80]}...")
