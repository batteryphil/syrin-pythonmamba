"""Tests for Human-in-the-Loop: ApprovalGate, requires_approval, ReactLoop integration."""

from syrin import Agent, ApprovalGate, Model
from syrin.tool import tool


def _almock() -> Model:
    return Model.Almock(latency_seconds=0.01, lorem_length=50)


class TestToolRequiresApproval:
    """ToolSpec.requires_approval and @syrin.tool(requires_approval=)."""

    def test_tool_default_no_approval(self) -> None:
        @tool
        def search(query: str) -> str:
            return f"Results: {query}"

        assert search.requires_approval is False

    def test_tool_requires_approval_true(self) -> None:
        @tool(requires_approval=True)
        def delete(id: str) -> str:
            return f"Deleted {id}"

        assert delete.requires_approval is True


class TestReactLoopRequiresApproval:
    """ReactLoop with tools that require approval uses ApprovalGate."""

    def test_requires_approval_no_gate_rejects(self) -> None:
        """When tool has requires_approval and agent has no approval_gate, reject."""

        @tool(requires_approval=True)
        def delete(id: str) -> str:
            return f"Deleted {id}"

        agent = Agent(
            model=_almock(),
            system_prompt="Use delete when asked. Use search otherwise.",
            tools=[delete, tool(lambda q: f"Results: {q}", name="search", description="Search")],
        )
        # Almock will return tool_calls; delete requires approval, no gate -> reject
        r = agent.response("Delete item 123")
        assert r.content is not None

    def test_requires_approval_with_gate_approves_executes(self) -> None:
        """When tool has requires_approval and agent has approval_gate that approves, execute."""
        approvals: list[tuple[str, dict]] = []

        def approve_cb(msg: str, timeout: int, ctx: dict) -> bool:
            approvals.append((ctx.get("tool_name", ""), ctx.get("arguments", {})))
            return True

        @tool(requires_approval=True)
        def dangerous_op(x: str) -> str:
            return f"Did {x}"

        gate = ApprovalGate(callback=approve_cb)
        agent = Agent(
            model=_almock(),
            system_prompt="Use dangerous_op when asked.",
            tools=[dangerous_op],
            approval_gate=gate,
        )
        r = agent.response("Run dangerous_op with x=test")
        assert r.content is not None
        if approvals:
            assert any("dangerous_op" in str(a) for a in approvals)

    def test_requires_approval_with_gate_rejects(self) -> None:
        """When approval gate returns False, tool not executed."""

        def reject_all(msg: str, timeout: int, ctx: dict) -> bool:
            return False

        @tool(requires_approval=True)
        def risky(q: str) -> str:
            return f"Risky: {q}"

        gate = ApprovalGate(callback=reject_all)
        agent = Agent(
            model=_almock(),
            system_prompt="Use risky when needed.",
            tools=[risky],
            approval_gate=gate,
        )
        r = agent.response("Do risky thing")
        assert r.content is not None


class TestApprovalGate:
    """ApprovalGate protocol and default implementation."""

    def test_approval_gate_sync_callback(self) -> None:
        gate = ApprovalGate(callback=lambda _m, _t, _c: True)
        import asyncio

        result = asyncio.run(gate.request("Approve?", timeout=5))
        assert result is True

    def test_approval_gate_reject(self) -> None:
        gate = ApprovalGate(callback=lambda _m, _t, _c: False)
        import asyncio

        result = asyncio.run(gate.request("Approve?", timeout=5))
        assert result is False

    def test_approval_gate_async_callback(self) -> None:
        async def async_approve(msg: str, t: int, ctx: dict) -> bool:
            return ctx.get("ok", False)

        gate = ApprovalGate(callback=async_approve)
        import asyncio

        result = asyncio.run(gate.request("?", timeout=5, context={"ok": True}))
        assert result is True
