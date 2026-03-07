"""Runtime context injection types.

PrepareInput and InjectPlacement for injecting additional context at prepare time
(e.g. RAG results, dynamic blocks).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    pass


class InjectPlacement(StrEnum):
    """Where to place injected messages in the context.

    Attributes:
        PREPEND_TO_SYSTEM: Injected messages go before the first system message.
        BEFORE_CURRENT_TURN: Injected messages go between conversation history
            and the current user message (default; good for RAG).
        AFTER_CURRENT_TURN: Injected messages go after the current user message.
    """

    PREPEND_TO_SYSTEM = "prepend_to_system"
    BEFORE_CURRENT_TURN = "before_current_turn"
    AFTER_CURRENT_TURN = "after_current_turn"


@dataclass
class PrepareInput:
    """Input passed to runtime_inject callable.

    Contains the current context state so the injector can decide what
    to add (e.g. RAG based on user_input).

    Attributes:
        messages: Current messages (system + conversation + current user).
        system_prompt: System prompt text.
        tools: Tool definitions (dicts).
        memory_context: Recalled memory block (may be empty).
        user_input: Current user message (for RAG queries).
    """

    messages: list[dict[str, Any]]
    system_prompt: str
    tools: list[dict[str, Any]]
    memory_context: str
    user_input: str


__all__ = ["InjectPlacement", "PrepareInput"]
