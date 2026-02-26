"""Syrin exception hierarchy."""

from __future__ import annotations

from typing import Any


class SyrinError(Exception):
    """Base exception for all Syrin errors."""

    pass


class BudgetExceededError(SyrinError):
    """Raised when a budget limit is exceeded.

    Attributes:
        message: Human-readable error message.
        current_cost: Current cost or token count that exceeded the limit.
        limit: The limit that was exceeded.
        budget_type: Which limit was exceeded (str, one of BudgetLimitType values).
    """

    def __init__(
        self,
        message: str,
        current_cost: float = 0.0,
        limit: float = 0.0,
        budget_type: str | Any = "run",
    ) -> None:
        super().__init__(message)
        self.current_cost = current_cost
        self.limit = limit
        _bt = budget_type
        self.budget_type: str = _bt.value if hasattr(_bt, "value") else _bt


class BudgetThresholdError(SyrinError):
    """Raised when a budget threshold triggers an action."""

    def __init__(
        self,
        message: str,
        threshold_percent: float = 0.0,
        action_taken: str = "",
    ) -> None:
        super().__init__(message)
        self.threshold_percent = threshold_percent
        self.action_taken = action_taken


class ModelNotFoundError(SyrinError):
    """Raised when a requested model is not found in the registry."""

    pass


class ToolExecutionError(SyrinError):
    """Raised when a tool execution fails."""

    pass


class TaskError(SyrinError):
    """Raised when a task execution fails."""

    pass


class ProviderError(SyrinError):
    """Raised when an LLM provider returns an error."""

    pass


class ProviderNotFoundError(SyrinError):
    """Raised when a requested provider name is not registered (e.g. typo in provider=)."""

    pass


class CodegenError(SyrinError):
    """Raised when DSL code generation fails."""

    pass


class ValidationError(SyrinError):
    """Raised when structured output validation fails.

    This exception includes information about validation attempts for debugging.

    Attributes:
        message: Error message
        attempts: List of validation attempts made
        last_error: Last error that occurred
    """

    def __init__(
        self,
        message: str,
        attempts: list[str] | None = None,
        last_error: Exception | None = None,
    ) -> None:
        super().__init__(message)
        self.attempts = attempts or []
        self.last_error = last_error


class HandoffBlockedError(SyrinError):
    """Raised when handoff is blocked by a before-handler or validation.

    Emitted as Hook.HANDOFF_BLOCKED when a handler blocks the transfer.

    Attributes:
        message: Reason for blocking
        source_agent: Source agent class or name
        target_agent: Target agent class or name
        task: Task that would have been passed
    """

    def __init__(
        self,
        message: str,
        source_agent: str = "",
        target_agent: str = "",
        task: str = "",
    ) -> None:
        super().__init__(message)
        self.source_agent = source_agent
        self.target_agent = target_agent
        self.task = task


class HandoffRetryRequested(SyrinError):
    """Target agent signals: data invalid, please retry with this hint.

    Raise this from the target (or a wrapper) to ask the caller to reformat
    and retry handoff. The caller implements the retry loop.

    Attributes:
        format_hint: Instructions for correct format (e.g. JSON schema, required fields)
    """

    def __init__(self, message: str, format_hint: str = "") -> None:
        super().__init__(message)
        self.format_hint = format_hint or message
