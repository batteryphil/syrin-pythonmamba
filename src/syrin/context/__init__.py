"""Context management for intelligent LLM context window handling.

Provides automatic context management with:
- Token counting and budget tracking
- Automatic compaction (middle-out truncation, summarization)
- Full observability via events and spans
- Protocol for custom implementations

Example:
    >>> from syrin import Agent, Model
    >>> from syrin.context import Context
    >>>
    >>> agent = Agent(
    ...     model=Model("openai/gpt-4o"),
    ...     context=Context(max_tokens=80000)
    ... )
    >>> result = agent.response("Hello")
    >>> print(agent.context.stats)
"""

from syrin.budget import TokenLimits
from syrin.context._manager import (
    ContextManager,
    ContextPayload,
    DefaultContextManager,
    create_context_manager,
)
from syrin.context.compactors import (
    CompactionResult,
    Compactor,
    ContextCompactor,
    ContextCompactorProtocol,
    MiddleOutTruncator,
    Summarizer,
)
from syrin.context.config import (
    Context,
    ContextStats,
    ContextWindowCapacity,
)
from syrin.context.counter import TokenCount, TokenCounter, get_counter
from syrin.context.injection import InjectPlacement, PrepareInput
from syrin.context.map import (
    ContextMap,
    ContextMapBackend,
    FileContextMapBackend,
)
from syrin.context.snapshot import (
    ContextBreakdown,
    ContextSegmentProvenance,
    ContextSegmentSource,
    ContextSnapshot,
    MessagePreview,
)
from syrin.context.store import (
    ContextSegment,
    ContextStore,
    InMemoryContextStore,
    RelevanceScorer,
    SimpleTextScorer,
)
from syrin.enums import CompactionMethod, ContextMode, FormationMode
from syrin.threshold import BudgetThreshold, ContextThreshold

__all__ = [
    # Config
    "Context",
    "ContextStats",
    "ContextWindowCapacity",
    "TokenLimits",
    "BudgetThreshold",
    "ContextThreshold",
    # Snapshot
    "InjectPlacement",
    "PrepareInput",
    "ContextBreakdown",
    "ContextSegmentProvenance",
    "ContextSegmentSource",
    "ContextSnapshot",
    "MessagePreview",
    # Counter
    "TokenCounter",
    "TokenCount",
    "get_counter",
    "ContextMode",
    # Compactors
    "CompactionMethod",
    "Compactor",
    "CompactionResult",
    "ContextCompactor",
    "ContextCompactorProtocol",
    "MiddleOutTruncator",
    "Summarizer",
    # Manager
    "ContextManager",
    "ContextPayload",
    "DefaultContextManager",
    "create_context_manager",
    # Map (persistent context map)
    "ContextMap",
    "ContextMapBackend",
    "FileContextMapBackend",
    # Store (pull-based context)
    "ContextSegment",
    "ContextStore",
    "InMemoryContextStore",
    "RelevanceScorer",
    "SimpleTextScorer",
    "FormationMode",
]
