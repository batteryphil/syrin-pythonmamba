"""Approval gate protocol and default implementation."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Protocol


class ApprovalGateProtocol(Protocol):
    """Human-in-the-loop approval backend. Implement for Slack, webhook, custom UI.

    Use with HumanInTheLoop(approval_gate=MyGate()). Implement request() to
    block until human approves or rejects (or timeout).
    """

    async def request(
        self,
        message: str,
        timeout: int,
        context: dict[str, object] | None = None,
    ) -> bool:
        """Request human approval. Block until approved/rejected or timeout.

        Args:
            message: Human-readable description of what needs approval.
            timeout: Seconds to wait. On timeout, treat as rejection.
            context: Optional extra context (tool_name, arguments, etc.).

        Returns:
            True if approved, False if rejected or timeout.
        """
        ...


def _sync_to_async(fn: Callable[..., bool]) -> Callable[..., Awaitable[bool]]:  # type: ignore[explicit-any]
    """Wrap sync callback to async."""

    async def wrapper(message: str, timeout: int, context: dict[str, object] | None = None) -> bool:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: fn(message, timeout, context or {}))

    return wrapper


class ApprovalGate:
    """Callback-based approval gate. Use with HumanInTheLoop(approval_gate=...).

    Sync or async callback: (message, timeout, context) -> bool.
    On timeout, treat as rejection (False).

    Example:
        >>> gate = ApprovalGate(callback=lambda msg, timeout, ctx: input("Approve? [y/n]: ") == "y")
        >>> agent = Agent(
        ...     approval_gate=gate,
        ...     loop=HumanInTheLoop(approval_gate=gate, timeout=300),
        ... )
    """

    def __init__(
        self,
        callback: Callable[[str, int, dict[str, object]], bool]
        | Callable[[str, int, dict[str, object]], Awaitable[bool]],
    ) -> None:
        self._callback = callback

    async def request(
        self,
        message: str,
        timeout: int,
        context: dict[str, object] | None = None,
    ) -> bool:
        ctx = context or {}
        result = self._callback(message, timeout, ctx)
        if asyncio.iscoroutine(result):
            return await asyncio.wait_for(result, timeout=timeout)
        return bool(result)
