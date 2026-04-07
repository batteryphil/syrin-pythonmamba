"""Hook System Documentation.

This module provides comprehensive documentation of all available hooks
and their EventContext schemas.

Usage:
    from syrin import Agent, Hook
    from syrin.hooks import HookContext

    # Register handler for a hook
    agent.events.on(Hook.AGENT_RUN_START, lambda ctx: print(f"Starting: {ctx.input}"))

Available Hooks:
================

AGENT LIFECYCLE
---------------
- AGENT_RUN_START: Agent starts processing input
  Context: {input: str, model: str, iteration: int}

- AGENT_RUN_END: Agent finishes processing
  Context: {content: str, cost: float, tokens: int, duration: float,
            stop_reason: str, iteration: int}

LLM REQUESTS
------------
- LLM_REQUEST_START: Before LLM API call
  Context: {messages: list, tools: list[str], iteration: int}

- LLM_REQUEST_END: After LLM API call
  Context: {content: str, iteration: int, tokens: TokenUsage}

TOOL CALLS
----------
- TOOL_CALL_START: Before tool execution
  Context: {tool_name: str, arguments: dict, iteration: int}

- TOOL_CALL_END: After tool execution
  Context: {tool_name: str, result: str, duration_ms: float}

- TOOL_ERROR: When tool execution fails
  Context: {tool_name: str, error: str, iteration: int}

GUARDRAILS
----------
- GUARDRAIL_INPUT: Input guardrail check starts
  Context: {text: str, stage: str, guardrail_count: int}

- GUARDRAIL_OUTPUT: Output guardrail check starts
  Context: {text: str, stage: str, guardrail_count: int}

- GUARDRAIL_BLOCKED: Guardrail blocked the request
  Context: {stage: str, reason: str, guardrail_names: list[str]}

MEMORY
------
- MEMORY_STORE: Memory stored successfully
  Context: {memory_id: str, content: str, memory_type: str, importance: float}

- MEMORY_RECALL: Memories recalled
  Context: {query: str, memory_type: str, results_count: int, limit: int}

- MEMORY_FORGET: Memories deleted
  Context: {memory_id: str, query: str, memory_type: str, deleted_count: int}

- MEMORY_CONSOLIDATE: Memory consolidation triggered
  Context: {memories_consolidated: int}

- MEMORY_EXTRACT: Semantic memory extraction
  Context: {extracted_count: int, source: str}

OUTPUT VALIDATION
-----------------
- OUTPUT_VALIDATION_START: Validation starts
  Context: {output_type: str, max_retries: int, raw_output: str}

- OUTPUT_VALIDATION_ATTEMPT: Validation attempt starts
  Context: {attempt: int, output_type: str}

- OUTPUT_VALIDATION_SUCCESS: Validation succeeded
  Context: {attempt: int, output_type: str, parsed_fields: list[str]}

- OUTPUT_VALIDATION_FAILED: Validation failed
  Context: {attempt: int, error: str, reason: str}

- OUTPUT_VALIDATION_RETRY: Retry attempt scheduled
  Context: {attempt: int, error: str, reason: str}

BUDGET
------
- BUDGET_CHECK: Budget check performed
  Context: {remaining: float, used: float, total: float}

- BUDGET_THRESHOLD: Budget threshold triggered
  Context: {threshold_percent: float, action: str}

- BUDGET_EXCEEDED: Budget exceeded
  Context: {used: float, limit: float, exceeded_by: float}

DISCOVERY
---------
- DISCOVERY_REQUEST: When /.well-known/agent-card.json is requested
  Context: {agent_name: str, path: str, user_agent: str (optional)}

CONTEXT
-------
- CONTEXT_COMPRESS: Context compression triggered
  Context: {initial_tokens: int, final_tokens: int, compression_ratio: float}

- CONTEXT_OFFLOAD: Context offloaded to memory
  Context: {offloaded_tokens: int, offload_target: str}

- CONTEXT_RESTORE: Context restored from memory
  Context: {restored_tokens: int, source: str}

CHECKPOINT
----------
- CHECKPOINT_SAVE: Checkpoint saved
  Context: {checkpoint_id: str, timestamp: float, state_size: int}

- CHECKPOINT_LOAD: Checkpoint loaded
  Context: {checkpoint_id: str, timestamp: float}

RATE LIMIT
----------
- RATE_LIMIT_CHECK: Rate limit checked
  Context: {rpm: int, tpm: int, current_rpm: int, current_tpm: int}

- RATE_LIMIT_EXCEEDED: Rate limit exceeded
  Context: {metric: str, limit: int, current: int}

Example:
    def on_validation_success(ctx):
        print(f"Validation succeeded after {ctx.attempt} attempts")
        print(f"Parsed fields: {ctx.parsed_fields}")

    agent.events.on(Hook.OUTPUT_VALIDATION_SUCCESS, on_validation_success)
"""

from __future__ import annotations

from dataclasses import dataclass

from syrin.enums import Hook


@dataclass
class HookContextSchema:
    """Schema definition for hook context fields."""

    hook: Hook
    description: str
    fields: dict[str, type]
    example: dict[str, object]


# Define schemas for all hooks
HOOK_SCHEMAS: dict[Hook, HookContextSchema] = {
    Hook.AGENT_RUN_START: HookContextSchema(
        hook=Hook.AGENT_RUN_START,
        description="Agent starts processing user input",
        fields={
            "input": str,
            "model": str,
            "iteration": int,
        },
        example={
            "input": "Hello, how are you?",
            "model": "gpt-4o-mini",
            "iteration": 0,
        },
    ),
    Hook.AGENT_RUN_END: HookContextSchema(
        hook=Hook.AGENT_RUN_END,
        description="Agent finishes processing",
        fields={
            "content": str,
            "cost": float,
            "tokens": int,
            "duration": float,
            "stop_reason": str,
            "iteration": int,
            "metadata": dict,
        },
        example={
            "content": "I'm doing well, thanks!",
            "cost": 0.0015,
            "tokens": 150,
            "duration": 1.2,
            "stop_reason": "end_turn",
            "iteration": 1,
            "metadata": {"loops": 2, "p_halt": 0.9},
        },
    ),
    Hook.LLM_REQUEST_START: HookContextSchema(
        hook=Hook.LLM_REQUEST_START,
        description="Before LLM API call",
        fields={
            "messages": list,
            "tools": list,
            "iteration": int,
        },
        example={
            "messages": [{"role": "user", "content": "Hello"}],
            "tools": ["search", "calculator"],
            "iteration": 1,
        },
    ),
    Hook.LLM_REQUEST_END: HookContextSchema(
        hook=Hook.LLM_REQUEST_END,
        description="After LLM API call",
        fields={
            "content": str,
            "iteration": int,
            "metadata": dict,
        },
        example={
            "content": "Hello! How can I help?",
            "iteration": 1,
            "metadata": {"loops": 12, "p_halt": 0.99},
        },
    ),
    Hook.TOOL_CALL_START: HookContextSchema(
        hook=Hook.TOOL_CALL_START,
        description="Before tool execution",
        fields={
            "tool_name": str,
            "arguments": dict,
            "iteration": int,
        },
        example={
            "tool_name": "search",
            "arguments": {"query": "Python tutorials"},
            "iteration": 2,
        },
    ),
    Hook.TOOL_CALL_END: HookContextSchema(
        hook=Hook.TOOL_CALL_END,
        description="After tool execution",
        fields={
            "tool_name": str,
            "result": str,
            "duration_ms": float,
        },
        example={
            "tool_name": "search",
            "result": "Found 5 results...",
            "duration_ms": 250.5,
        },
    ),
    Hook.TOOL_ERROR: HookContextSchema(
        hook=Hook.TOOL_ERROR,
        description="When tool execution fails",
        fields={
            "tool_name": str,
            "error": str,
            "iteration": int,
        },
        example={
            "tool_name": "calculator",
            "error": "Division by zero",
            "iteration": 2,
        },
    ),
    Hook.GUARDRAIL_INPUT: HookContextSchema(
        hook=Hook.GUARDRAIL_INPUT,
        description="Input guardrail check starts",
        fields={
            "text": str,
            "stage": str,
            "guardrail_count": int,
        },
        example={
            "text": "User input text...",
            "stage": "input",
            "guardrail_count": 2,
        },
    ),
    Hook.GUARDRAIL_OUTPUT: HookContextSchema(
        hook=Hook.GUARDRAIL_OUTPUT,
        description="Output guardrail check starts",
        fields={
            "text": str,
            "stage": str,
            "guardrail_count": int,
        },
        example={
            "text": "Model output text...",
            "stage": "output",
            "guardrail_count": 2,
        },
    ),
    Hook.GUARDRAIL_BLOCKED: HookContextSchema(
        hook=Hook.GUARDRAIL_BLOCKED,
        description="Guardrail blocked the request",
        fields={
            "stage": str,
            "reason": str,
            "guardrail_names": list,
        },
        example={
            "stage": "input",
            "reason": "Blocked word found: hack",
            "guardrail_names": ["content_filter"],
        },
    ),
    Hook.MEMORY_STORE: HookContextSchema(
        hook=Hook.MEMORY_STORE,
        description="Memory stored successfully",
        fields={
            "memory_id": str,
            "content": str,
            "memory_type": str,
            "importance": float,
        },
        example={
            "memory_id": "uuid-123",
            "content": "User likes Python",
            "memory_type": "knowledge",
            "importance": 0.8,
        },
    ),
    Hook.MEMORY_RECALL: HookContextSchema(
        hook=Hook.MEMORY_RECALL,
        description="Memories recalled",
        fields={
            "query": str,
            "memory_type": str,
            "results_count": int,
            "limit": int,
        },
        example={
            "query": "Python programming",
            "memory_type": "all",
            "results_count": 3,
            "limit": 10,
        },
    ),
    Hook.MEMORY_FORGET: HookContextSchema(
        hook=Hook.MEMORY_FORGET,
        description="Memories deleted",
        fields={
            "memory_id": str,
            "query": str,
            "memory_type": str,
            "deleted_count": int,
        },
        example={
            "memory_id": None,
            "query": "old data",
            "memory_type": "history",
            "deleted_count": 5,
        },
    ),
    Hook.OUTPUT_VALIDATION_START: HookContextSchema(
        hook=Hook.OUTPUT_VALIDATION_START,
        description="Output validation starts",
        fields={
            "output_type": str,
            "max_retries": int,
            "raw_output": str,
        },
        example={
            "output_type": "UserInfo",
            "max_retries": 3,
            "raw_output": '{"name": "John"}',
        },
    ),
    Hook.OUTPUT_VALIDATION_ATTEMPT: HookContextSchema(
        hook=Hook.OUTPUT_VALIDATION_ATTEMPT,
        description="Validation attempt starts",
        fields={
            "attempt": int,
            "output_type": str,
        },
        example={
            "attempt": 1,
            "output_type": "UserInfo",
        },
    ),
    Hook.OUTPUT_VALIDATION_SUCCESS: HookContextSchema(
        hook=Hook.OUTPUT_VALIDATION_SUCCESS,
        description="Validation succeeded",
        fields={
            "attempt": int,
            "output_type": str,
            "parsed_fields": list,
        },
        example={
            "attempt": 1,
            "output_type": "UserInfo",
            "parsed_fields": ["name", "email"],
        },
    ),
    Hook.OUTPUT_VALIDATION_FAILED: HookContextSchema(
        hook=Hook.OUTPUT_VALIDATION_FAILED,
        description="Validation failed",
        fields={
            "attempt": int,
            "error": str,
            "reason": str,
        },
        example={
            "attempt": 3,
            "error": "Invalid JSON",
            "reason": "json_parse_error",
        },
    ),
    Hook.OUTPUT_VALIDATION_RETRY: HookContextSchema(
        hook=Hook.OUTPUT_VALIDATION_RETRY,
        description="Retry attempt scheduled",
        fields={
            "attempt": int,
            "error": str,
            "reason": str,
        },
        example={
            "attempt": 1,
            "error": "Missing required field",
            "reason": "pydantic_validation_error",
        },
    ),
    Hook.BUDGET_CHECK: HookContextSchema(
        hook=Hook.BUDGET_CHECK,
        description="Budget check performed",
        fields={
            "remaining": float,
            "used": float,
            "total": float,
        },
        example={
            "remaining": 4.5,
            "used": 0.5,
            "total": 5.0,
        },
    ),
    Hook.BUDGET_THRESHOLD: HookContextSchema(
        hook=Hook.BUDGET_THRESHOLD,
        description="Budget threshold triggered",
        fields={
            "threshold_percent": float,
            "action": str,
        },
        example={
            "threshold_percent": 80.0,
            "action": "warn",
        },
    ),
    Hook.BUDGET_EXCEEDED: HookContextSchema(
        hook=Hook.BUDGET_EXCEEDED,
        description="Budget exceeded",
        fields={
            "used": float,
            "limit": float,
            "exceeded_by": float,
        },
        example={
            "used": 5.5,
            "limit": 5.0,
            "exceeded_by": 0.5,
        },
    ),
    Hook.DISCOVERY_REQUEST: HookContextSchema(
        hook=Hook.DISCOVERY_REQUEST,
        description="When /.well-known/agent-card.json is requested",
        fields={
            "agent_name": str,
            "path": str,
        },
        example={
            "agent_name": "product-agent",
            "path": "/.well-known/agent-card.json",
        },
    ),
    Hook.CHECKPOINT_SAVE: HookContextSchema(
        hook=Hook.CHECKPOINT_SAVE,
        description="Checkpoint saved",
        fields={
            "checkpoint_id": str,
            "timestamp": float,
            "state_size": int,
        },
        example={
            "checkpoint_id": "cp-123",
            "timestamp": 1234567890.0,
            "state_size": 1024,
        },
    ),
    Hook.CHECKPOINT_LOAD: HookContextSchema(
        hook=Hook.CHECKPOINT_LOAD,
        description="Checkpoint loaded",
        fields={
            "checkpoint_id": str,
            "timestamp": float,
        },
        example={
            "checkpoint_id": "cp-123",
            "timestamp": 1234567890.0,
        },
    ),
    Hook.CONTEXT_COMPRESS: HookContextSchema(
        hook=Hook.CONTEXT_COMPRESS,
        description="Context compression triggered",
        fields={
            "initial_tokens": int,
            "final_tokens": int,
            "compression_ratio": float,
        },
        example={
            "initial_tokens": 1000,
            "final_tokens": 500,
            "compression_ratio": 0.5,
        },
    ),
    Hook.CONTEXT_OFFLOAD: HookContextSchema(
        hook=Hook.CONTEXT_OFFLOAD,
        description="Context offloaded to memory",
        fields={
            "offloaded_tokens": int,
            "offload_target": str,
        },
        example={
            "offloaded_tokens": 200,
            "offload_target": "persistent_memory",
        },
    ),
    Hook.CONTEXT_RESTORE: HookContextSchema(
        hook=Hook.CONTEXT_RESTORE,
        description="Context restored from memory",
        fields={
            "restored_tokens": int,
            "source": str,
        },
        example={
            "restored_tokens": 200,
            "source": "persistent_memory",
        },
    ),
    Hook.SYSTEM_PROMPT_BEFORE_RESOLVE: HookContextSchema(
        hook=Hook.SYSTEM_PROMPT_BEFORE_RESOLVE,
        description="Before system prompt resolution (dynamic prompts)",
        fields={
            "template_variables": dict,
            "source": object,
        },
        example={
            "template_variables": {"user_name": "Alice", "date": "2025-02-28"},
            "source": "<Prompt or callable>",
        },
    ),
    Hook.SYSTEM_PROMPT_AFTER_RESOLVE: HookContextSchema(
        hook=Hook.SYSTEM_PROMPT_AFTER_RESOLVE,
        description="After system prompt resolved to string",
        fields={
            "resolved": str,
        },
        example={
            "resolved": "You assist Alice. Be professional.",
        },
    ),
}


def get_hook_schema(hook: Hook) -> HookContextSchema | None:
    """Get the schema for a specific hook.

    Args:
        hook: The hook enum value

    Returns:
        HookContextSchema if available, None otherwise
    """
    return HOOK_SCHEMAS.get(hook)


def list_all_hooks() -> list[Hook]:
    """List all available hooks."""
    return list(Hook)


__all__ = [
    "HOOK_SCHEMAS",
    "HookContextSchema",
    "get_hook_schema",
    "list_all_hooks",
]
