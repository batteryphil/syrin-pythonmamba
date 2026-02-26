"""Approval gate protocol and default implementation."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any, Protocol


class ApprovalGateProtocol(Protocol):
    """Human-in-the-loop approval backend. Implement for Slack, webhook, etc."""

    async def request(
        self,
        message: str,
        timeout: int,
        context: dict[str, Any] | None = None,
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


def _sync_to_async(fn: Callable[..., bool]) -> Callable[..., Awaitable[bool]]:
    """Wrap sync callback to async."""

    async def wrapper(message: str, timeout: int, context: dict[str, Any] | None = None) -> bool:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: fn(message, timeout, context or {}))

    return wrapper


class ApprovalGate:
    """Callback-based approval gate. Default for HITL.

    Example:
        >>> gate = ApprovalGate(callback=lambda msg, timeout, ctx: input("Approve? [y/n]: ") == "y")
        >>> approved = await gate.request("Execute delete?", timeout=60)
    """

    def __init__(
        self,
        callback: Callable[[str, int, dict[str, Any]], bool]
        | Callable[[str, int, dict[str, Any]], Awaitable[bool]],
    ) -> None:
        self._callback = callback

    async def request(
        self,
        message: str,
        timeout: int,
        context: dict[str, Any] | None = None,
    ) -> bool:
        ctx = context or {}
        result = self._callback(message, timeout, ctx)
        if asyncio.iscoroutine(result):
            return await asyncio.wait_for(result, timeout=timeout)
        return bool(result)
