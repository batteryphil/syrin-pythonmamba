"""Authority Layer — Permission checks, budget enforcement, and approval flows.

Demonstrates:
- AuthorityCheck — permission-based authorization
- BudgetEnforcer — budget enforcement for actions
- HumanApproval — human-in-the-loop approval
- ThresholdApproval — K-of-N threshold consensus
- CapabilityToken and CapabilityIssuer

Run:
    python examples/09_guardrails/authority_layer.py
"""

from __future__ import annotations

import asyncio
from unittest.mock import Mock

from syrin import Agent, Model
from syrin.guardrails import ContentFilter


async def example_authority_check() -> None:
    print("\n" + "=" * 55)
    print("1. AuthorityCheck — permission-based")
    print("=" * 55)

    from syrin.enums import GuardrailStage
    from syrin.guardrails import AuthorityCheck, GuardrailContext

    user = Mock()
    user.id = "trader123"
    user.has_permission = Mock(return_value=True)

    context = GuardrailContext(text="Transfer $5000", stage=GuardrailStage.ACTION, user=user)
    guardrail = AuthorityCheck(requires="finance:transfer")
    result = await guardrail.evaluate(context)
    print(f"Has 'finance:transfer': {result.passed}")

    user.has_permission.return_value = False
    result = await guardrail.evaluate(context)
    print(f"Lacks permission: {not result.passed} — {result.reason}")


async def example_budget_enforcer() -> None:
    print("\n" + "=" * 55)
    print("2. BudgetEnforcer — budget limits on actions")
    print("=" * 55)

    from syrin.guardrails import BudgetEnforcer, GuardrailContext

    budget = Mock()
    budget.remaining = 1000.0

    action = Mock()
    action.name = "transfer_funds"
    action.amount = 500.0

    context = GuardrailContext(text="Transfer $500", budget=budget, action=action)
    guardrail = BudgetEnforcer(max_amount=1000)
    result = await guardrail.evaluate(context)
    print(f"Transfer $500 (limit $1000): passed={result.passed}")

    action.amount = 1500.0
    result = await guardrail.evaluate(context)
    print(f"Transfer $1500 (limit $1000): passed={result.passed} — {result.reason}")


async def example_threshold_approval() -> None:
    print("\n" + "=" * 55)
    print("3. ThresholdApproval — K-of-N consensus")
    print("=" * 55)

    from syrin.guardrails import GuardrailContext, ThresholdApproval

    guardrail = ThresholdApproval(
        k=2,
        n=3,
        approvers=["alice@example.com", "bob@example.com", "charlie@example.com"],
    )
    context = GuardrailContext(
        text="Delete production database",
        metadata={"request_id": "req_001"},
    )

    result = await guardrail.evaluate(context)
    print(f"0 approvals: passed={result.passed}")

    guardrail.add_approval("req_001", "alice@example.com")
    result = await guardrail.evaluate(context)
    print(f"1 approval: passed={result.passed}")

    guardrail.add_approval("req_001", "bob@example.com")
    result = await guardrail.evaluate(context)
    print(f"2 approvals (threshold met): passed={result.passed}")


async def example_human_approval() -> None:
    print("\n" + "=" * 55)
    print("4. HumanApproval — human-in-the-loop")
    print("=" * 55)

    from syrin.guardrails import GuardrailContext, HumanApproval

    guardrail = HumanApproval(
        approver="admin@example.com",
        requires_justification=True,
        timeout=300,
    )
    context = GuardrailContext(
        text="Deploy to production",
        metadata={"request_id": "req_deploy"},
    )

    result = await guardrail.evaluate(context)
    print(f"Before approval: passed={result.passed}")

    guardrail.approve(
        request_id="req_deploy",
        approver="admin@example.com",
        justification="Code reviewed and tested",
    )
    result = await guardrail.evaluate(context)
    print(f"After approval: passed={result.passed}")
    print(f"Approved by: {result.metadata.get('approved_by')}")


async def example_capability_tokens() -> None:
    print("\n" + "=" * 55)
    print("5. CapabilityToken — scoped tokens with budget")
    print("=" * 55)

    from syrin.guardrails import CapabilityIssuer

    issuer = CapabilityIssuer()
    token = issuer.issue(scope="reports:generate", budget=5, ttl=3600, issued_to="analyst789")

    print(f"Scope: {token.scope}")
    print(f"Budget: {token.budget}")
    print(f"Valid: {token.is_valid()}")
    print(f"Can generate reports: {token.can('reports:generate')}")
    print(f"Can delete data: {token.can('data:delete')}")

    for i in range(6):
        success = token.consume(1)
        print(f"  Use {i + 1}: {'ok' if success else 'exhausted'} (budget: {token.budget})")


async def _run() -> None:
    await example_authority_check()
    await example_budget_enforcer()
    await example_threshold_approval()
    await example_human_approval()
    await example_capability_tokens()


if __name__ == "__main__":
    asyncio.run(_run())

    # --- Optional: serve with playground ---
    # class AuthorityDemoAgent(Agent):
    #     _agent_name = "authority-demo"
    #     _agent_description = "Agent with authority layer guardrails demo"
    #     model = Model.Almock()
    #     system_prompt = "You are a helpful assistant."
    #     guardrails = [ContentFilter(blocked_words=["unauthorized"])]
    #
    # agent = AuthorityDemoAgent()
    # print("Serving at http://localhost:8000/playground")
    # agent.serve(port=8000, enable_playground=True, debug=True)
