"""Token counting for context management."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import tiktoken

try:
    import tiktoken
except ImportError:
    tiktoken = None  # type: ignore[assignment]

import contextlib
from dataclasses import dataclass

from syrin.context.snapshot import ContextBreakdown


@dataclass
class TokenCount:
    """Token count breakdown for messages and system prompt.

    Attributes:
        total: Total tokens.
        system: System prompt tokens.
        messages: Message tokens.
        tools: Tool definition tokens.
        memory: Memory injection tokens.
    """

    total: int
    system: int = 0
    messages: int = 0
    tools: int = 0
    memory: int = 0


class TokenCounter:
    """Token counter with support for multiple encodings.

    Uses tiktoken when available for accurate counting; falls back to
    ~4 chars per token when tiktoken is not installed. Used by context
    manager for budget and compaction.

    Methods:
        count: Count tokens in text.
        count_messages: Count tokens in message list with breakdown.
        count_tools: Count tokens in tool definitions.
    """

    def __init__(self, encoding: str = "cl100k_base"):
        self._encoding = encoding
        self._encoder = None

        if tiktoken is not None:
            with contextlib.suppress(Exception):
                self._encoder = tiktoken.get_encoding(encoding)

    def count(self, text: str) -> int:
        """Count tokens in text."""
        if self._encoder is not None:
            return len(self._encoder.encode(text, disallowed_special=()))
        return self._estimate(text)

    def count_messages(
        self,
        messages: list[dict[str, Any]],
        system_prompt: str = "",
    ) -> TokenCount:
        """Count tokens in a message list.

        Args:
            messages: List of message dicts with 'role' and 'content' keys.
            system_prompt: Optional system prompt to include.

        Returns:
            TokenCount with breakdown.
        """
        total = 0
        message_tokens = 0

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if isinstance(content, list):
                content_str = ""
                for part in content:
                    if isinstance(part, dict):
                        content_str += str(part.get("text", ""))
                        content_str += str(part.get("type", ""))
            else:
                content_str = str(content)

            tokens = self.count(content_str)
            tokens += self._role_overhead(role)
            message_tokens += tokens
            total += tokens

        system_tokens = 0
        if system_prompt:
            system_tokens = self.count(system_prompt)
            system_tokens += self._role_overhead("system")
            total += system_tokens

        return TokenCount(
            total=total,
            system=system_tokens,
            messages=message_tokens,
        )

    def count_tools(self, tools: list[dict[str, Any]]) -> int:
        """Count tokens in tool definitions.

        Args:
            tools: List of tool definition dicts.

        Returns:
            Token count for tools.
        """
        if not tools:
            return 0

        total = 0
        for tool in tools:
            tool_str = str(tool)
            total += self.count(tool_str)
            total += 12

        return total

    def count_breakdown(
        self,
        *,
        system_prompt: str,
        memory_context: str,
        tools: list[dict[str, Any]],
        tokens_used: int,
        injected_tokens: int = 0,
    ) -> ContextBreakdown:
        """Compute token breakdown by component (system, tools, memory, messages, injected).

        Used by the context manager to populate ContextSnapshot.breakdown and
        ContextStats.breakdown. messages_tokens is derived as the residual so
        that total_tokens equals tokens_used.

        Args:
            system_prompt: System prompt text (may be empty).
            memory_context: Recalled memory text injected as [Memory] block (may be empty).
            tools: Tool definition dicts.
            tokens_used: Total token count for this prepare (authoritative total).
            injected_tokens: Tokens in runtime-injected context (RAG, etc.). Default 0.

        Returns:
            ContextBreakdown with system_tokens, tools_tokens, memory_tokens,
            messages_tokens (residual), and injected_tokens.
        """
        system_tokens = 0
        if system_prompt:
            system_tokens = self.count(system_prompt) + self._role_overhead("system")
        tools_tokens = self.count_tools(tools)
        memory_tokens = 0
        if memory_context:
            memory_tokens = self.count(f"[Memory]\n{memory_context}") + self._role_overhead(
                "system"
            )
        messages_tokens = max(
            0,
            tokens_used - system_tokens - tools_tokens - memory_tokens - injected_tokens,
        )
        return ContextBreakdown(
            system_tokens=system_tokens,
            tools_tokens=tools_tokens,
            memory_tokens=memory_tokens,
            messages_tokens=messages_tokens,
            injected_tokens=injected_tokens,
        )

    def _role_overhead(self, role: str) -> int:
        """Get token overhead for a role."""
        overheads = {
            "system": 4,
            "user": 4,
            "assistant": 4,
            "tool": 5,
        }
        return overheads.get(role, 4)

    def _estimate(self, text: str) -> int:
        """Estimate token count when tiktoken unavailable."""
        return len(text) // 4


_default_counter: TokenCounter | None = None


def get_counter() -> TokenCounter:
    """Get the default token counter singleton.

    Returns:
        TokenCounter instance. Shared across context manager usage.
    """
    global _default_counter
    if _default_counter is None:
        _default_counter = TokenCounter()
    return _default_counter


__all__ = ["TokenCounter", "TokenCount", "get_counter"]
