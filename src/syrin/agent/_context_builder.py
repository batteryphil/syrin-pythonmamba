"""Context builder: builds the message list for an LLM call.

This is the single place that builds the message list for the LLM. All logic for
"what messages go to the LLM" lives here. Agent._build_messages is a thin wrapper
that gathers agent state and calls build_messages().

Extracted from Agent so message/context construction has a single responsibility.
Agent passes in memory, context manager, and config; this module returns list[Message].
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from syrin.context import Context
from syrin.context.config import ContextWindowCapacity
from syrin.enums import MessageRole
from syrin.types import Message


def build_messages(
    user_input: str,
    *,
    system_prompt: str,
    tools: list[Any],
    conversation_memory: Any = None,
    memory_backend: Any = None,
    persistent_memory: Any = None,
    context_manager: Any = None,
    get_capacity: Callable[[], ContextWindowCapacity],
    call_context: Context | None = None,
    tracer: Any = None,
) -> list[Message]:
    """Build the message list for the next LLM call.

    Combines system prompt, persistent memory recall, conversation memory,
    and user input; then runs context management (prepare) and returns
    the final list of Message objects.

    Args:
        user_input: The user's message.
        system_prompt: System prompt text.
        tools: List of ToolSpec or tool-like (with to_tool_spec or dict).
        conversation_memory: Optional conversation memory (get_messages()).
        memory_backend: Optional persistent memory backend (search).
        persistent_memory: Optional Memory (top_k for recall).
        context_manager: Context manager with prepare(messages, system_prompt, tools, memory_context, capacity, context).
        get_capacity: Callable that returns ContextWindowCapacity for this call.
        call_context: Optional per-call Context override.
        tracer: Optional tracer for spans (memory.recall).

    Returns:
        List of Message ready for the LLM.
    """
    messages: list[Message] = []
    system_content = system_prompt or ""
    memory_context = ""

    # Persistent memory recall
    if memory_backend is not None and persistent_memory is not None:
        top_k = getattr(persistent_memory, "top_k", 10) or 10
        memories = _in_span(
            tracer,
            "memory.recall",
            {"memory.kind": "persistent"},
            lambda: memory_backend.search(user_input, None, top_k),
            result_attr=("MEMORY_RESULTS_COUNT", len),
        )
        if memories:
            memory_context = "## Relevant Memories:\n"
            for mem in memories:
                type_val = getattr(mem, "type", None)
                type_str = (
                    type_val.value
                    if type_val is not None and hasattr(type_val, "value")
                    else str(type_val)
                    if type_val is not None
                    else "unknown"
                )
                memory_context += f"- [{type_str}] {mem.content}\n"

    if system_content:
        messages.append(Message(role=MessageRole.SYSTEM, content=system_content))

    # Conversation memory
    if conversation_memory is not None:
        mem_messages = _in_span(
            tracer,
            "memory.recall",
            {"memory.kind": "conversation"},
            conversation_memory.get_messages,
            result_attr=("MEMORY_RESULTS_COUNT", len),
        )
        messages.extend(mem_messages)

    messages.append(Message(role=MessageRole.USER, content=user_input))

    # To dicts for context manager
    msg_dicts = []
    for msg in messages:
        role = msg.role
        role_str = role.value if hasattr(role, "value") else str(role)
        msg_dicts.append({"role": role_str, "content": msg.content})

    tool_dicts = []
    for tool in tools:
        if hasattr(tool, "to_tool_spec"):
            tool_dicts.append(tool.to_tool_spec())
        elif isinstance(tool, dict):
            tool_dicts.append(tool)

    if context_manager is None:
        return messages

    capacity = get_capacity()
    payload = context_manager.prepare(
        messages=msg_dicts,
        system_prompt=system_content,
        tools=tool_dicts,
        memory_context=memory_context,
        capacity=capacity,
        context=call_context,
    )

    final_messages = []
    for msg_dict in payload.messages:
        msg_data = msg_dict if isinstance(msg_dict, dict) else {}
        role_str = msg_data.get("role", "user")
        try:
            role_enum = MessageRole(role_str) if isinstance(role_str, str) else role_str
        except (ValueError, TypeError):
            role_enum = MessageRole.USER
        final_messages.append(Message(role=role_enum, content=msg_data.get("content", "")))

    return final_messages


def _in_span(
    tracer: Any,
    name: str,
    extra_attrs: dict[str, Any],
    fn: Callable[[], Any],
    result_attr: tuple[str, Callable[[Any], int]] | None = None,
) -> Any:
    """Run fn inside a span if tracer is available. Return fn()."""
    if tracer is None or not hasattr(tracer, "span"):
        return fn()
    try:
        from syrin.observability import SemanticAttributes, SpanKind

        attrs = {SemanticAttributes.MEMORY_OPERATION: "recall", **extra_attrs}
        with tracer.span(name, kind=SpanKind.MEMORY, attributes=attrs) as span:
            result = fn()
            if hasattr(span, "set_attribute") and result_attr is not None:
                attr_name, size_fn = result_attr
                if attr_name == "MEMORY_RESULTS_COUNT":
                    span.set_attribute(
                        getattr(SemanticAttributes, attr_name, attr_name),
                        size_fn(result),
                    )
            return result
    except Exception:
        return fn()
