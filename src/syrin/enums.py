from enum import StrEnum


class StopReason(StrEnum):
    """Why an agent run terminated."""

    END_TURN = "end_turn"
    BUDGET = "budget"
    MAX_ITERATIONS = "max_iterations"
    TIMEOUT = "timeout"
    TOOL_ERROR = "tool_error"
    HANDOFF = "handoff"
    GUARDRAIL = "guardrail"
    CANCELLED = "cancelled"


class LoopStrategy(StrEnum):
    """Agent execution loop strategy.

    Attributes:
        REACT: Reason → Act → Observe loop. Agent can call tools, get results,
            and decide next step. Use for tool-using agents (search, calculator, etc.).
        SINGLE_SHOT: One LLM call, no tools. Fast for simple Q&A. No tool loop.
    """

    REACT = "react"
    SINGLE_SHOT = "single_shot"


class ContextStrategy(StrEnum):
    """How to compress conversation context when it exceeds limits."""

    TRUNCATE = "truncate"
    SLIDING_WINDOW = "sliding_window"
    SUMMARIZE = "summarize"


class ContextMode(StrEnum):
    """How to select conversation history for context formation.

    Attributes:
        FULL: Full conversation history (default). Compaction when over capacity.
        FOCUSED: Keep only last N turns (user+assistant pairs). Reduces irrelevant history.
        INTELLIGENT: Relevance-filtered (requires scorer; not yet implemented).
    """

    FULL = "full"
    FOCUSED = "focused"
    INTELLIGENT = "intelligent"


class MemoryPreset(StrEnum):
    """Preset for agent memory. Use MemoryPreset.DEFAULT or MemoryPreset.DISABLED for clarity.

    Attributes:
        DISABLED: No memory. Stateless agent.
        DEFAULT: Core + episodic, top_k=10. Quick multi-turn chat.
    """

    DISABLED = "disabled"
    DEFAULT = "default"


class FormationMode(StrEnum):
    """How conversation history is fed into context.

    Attributes:
        PUSH: Use conversation memory directly (last N or full). Default.
        PULL: Query context store by relevance to current prompt; only matching segments.
    """

    PUSH = "push"
    PULL = "pull"


class OutputChunkStrategy(StrEnum):
    """How to split assistant content into chunks for retrieval by relevance.

    Attributes:
        PARAGRAPH: Split on blank lines (\\n\\n). Default.
        FIXED: Split by fixed character size (output_chunk_size).
    """

    PARAGRAPH = "paragraph"
    FIXED = "fixed"


class CompactionMethod(StrEnum):
    """Method used when context compaction runs. See ContextCompactor for when each is chosen.

    Use list(CompactionMethod) or CompactionMethod.__members__ to see all methods.
    stats.compact_method and CompactionResult.method use these string values.

    Attributes:
        NONE: No compaction; messages already fit in budget.
        MIDDLE_OUT_TRUNCATE: Kept start and end of conversation, truncated middle (overage < 1.5×).
        SUMMARIZE: Older messages summarized (e.g. via LLM or placeholder); used when overage ≥ 1.5×.
    """

    NONE = "none"
    MIDDLE_OUT_TRUNCATE = "middle_out_truncate"
    SUMMARIZE = "summarize"


class TracingBackend(StrEnum):
    """Built-in tracing output destinations."""

    CONSOLE = "console"
    FILE = "file"
    JSONL = "jsonl"
    OTLP = "otlp"


class TraceLevel(StrEnum):
    """Tracing verbosity levels."""

    MINIMAL = "minimal"
    STANDARD = "standard"
    VERBOSE = "verbose"


class MessageRole(StrEnum):
    """Conversation message roles. Use when building Message objects for model.complete().

    - SYSTEM: Instructions/context for the model (e.g., "You are helpful").
    - USER: Human input or prompt.
    - ASSISTANT: Model reply (or prior turn).
    - TOOL: Result of a tool/function call (used in function-calling loops).
    """

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class StepType(StrEnum):
    """Types of steps in an execution trace."""

    LLM_CALL = "llm_call"
    TOOL_CALL = "tool_call"
    MODEL_SWITCH = "model_switch"
    BUDGET_CHECK = "budget_check"
    HANDOFF = "handoff"
    GUARDRAIL = "guardrail"
    SPAWN = "spawn"


class GuardrailStage(StrEnum):
    """When a guardrail runs in the agent lifecycle."""

    INPUT = "input"
    ACTION = "action"
    OUTPUT = "output"


class DecisionAction(StrEnum):
    """Action to take after guardrail evaluation."""

    PASS = "pass"
    BLOCK = "block"
    WARN = "warn"
    REQUEST_APPROVAL = "request_approval"
    REDACT = "redact"


class SwitchReason(StrEnum):
    """Why the model was switched during execution."""

    BUDGET_THRESHOLD = "budget_threshold"
    FALLBACK = "fallback"
    MANUAL = "manual"


class RateWindow(StrEnum):
    """Time windows for rate limiting."""

    HOUR = "hour"
    DAY = "day"
    WEEK = "week"
    MONTH = "month"


class ThresholdWindow(StrEnum):
    """Window for thresholds: run (per execution), time-based (hour/day/week/month), or context (max_tokens).

    Reusable for budget thresholds, rate-limit thresholds, and context thresholds.
    Context thresholds use MAX_TOKENS only (current context window).
    """

    RUN = "run"
    HOUR = "hour"
    DAY = "day"
    WEEK = "week"
    MONTH = "month"
    MAX_TOKENS = "max_tokens"  # Context: current context window (no time window)


class BudgetLimitType(StrEnum):
    """Which budget limit was exceeded or is being reported.

    Used by CheckBudgetResult.exceeded_limit and BudgetExceededContext.budget_type.
    Exhaustive: run, run_tokens, and cost/token rate limits per window.
    """

    RUN = "run"
    RUN_TOKENS = "run_tokens"
    HOUR = "hour"
    DAY = "day"
    WEEK = "week"
    MONTH = "month"
    HOUR_TOKENS = "hour_tokens"
    DAY_TOKENS = "day_tokens"
    WEEK_TOKENS = "week_tokens"
    MONTH_TOKENS = "month_tokens"
    MEMORY = "memory"  # custom limit for memory/store extraction budget


class AuditBackend(StrEnum):
    """Built-in audit log destinations.

    FILE and JSONL are equivalent (both write JSONL to file).
    OTLP for tracing backends (future).
    """

    FILE = "file"  # Same as JSONL
    JSONL = "jsonl"
    OTLP = "otlp"


class AuditEventType(StrEnum):
    """Canonical audit event types. Maps from Hook to audit event."""

    # Agent
    AGENT_RUN_START = "agent_run_start"
    AGENT_RUN_END = "agent_run_end"
    AGENT_INIT = "agent_init"
    AGENT_RESET = "agent_reset"

    # LLM
    LLM_CALL = "llm_call"
    LLM_RETRY = "llm_retry"
    LLM_FALLBACK = "llm_fallback"

    # Tools
    TOOL_CALL = "tool_call"
    TOOL_ERROR = "tool_error"

    # Handoff & Spawn
    HANDOFF_START = "handoff_start"
    HANDOFF_END = "handoff_end"
    HANDOFF_BLOCKED = "handoff_blocked"
    SPAWN_START = "spawn_start"
    SPAWN_END = "spawn_end"

    # Budget
    BUDGET_CHECK = "budget_check"
    BUDGET_THRESHOLD = "budget_threshold"
    BUDGET_EXCEEDED = "budget_exceeded"

    # Guardrails
    GUARDRAIL_INPUT = "guardrail_input"
    GUARDRAIL_OUTPUT = "guardrail_output"
    GUARDRAIL_BLOCKED = "guardrail_blocked"

    # Memory
    MEMORY_STORE = "memory_store"
    MEMORY_RECALL = "memory_recall"
    MEMORY_FORGET = "memory_forget"

    # Pipeline (static) — values match Hook for single source of truth
    PIPELINE_START = "pipeline.start"
    PIPELINE_END = "pipeline.end"
    PIPELINE_AGENT_START = "pipeline.agent.start"
    PIPELINE_AGENT_COMPLETE = "pipeline.agent.complete"

    # Dynamic Pipeline — values match Hook
    DYNAMIC_PIPELINE_START = "dynamic.pipeline.start"
    DYNAMIC_PIPELINE_PLAN = "dynamic.pipeline.plan"
    DYNAMIC_PIPELINE_EXECUTE = "dynamic.pipeline.execute"
    DYNAMIC_PIPELINE_AGENT_SPAWN = "dynamic.pipeline.agent.spawn"
    DYNAMIC_PIPELINE_AGENT_COMPLETE = "dynamic.pipeline.agent.complete"
    DYNAMIC_PIPELINE_END = "dynamic.pipeline.end"
    DYNAMIC_PIPELINE_ERROR = "dynamic.pipeline.error"

    # Serve
    SERVE_REQUEST_START = "serve_request_start"
    SERVE_REQUEST_END = "serve_request_end"


class AlmockPricing(StrEnum):
    """Pricing tier for Almock (An LLM Mock). Use to test costing without real API calls.

    LOW, MEDIUM, HIGH, ULTRA_HIGH map to increasing USD-per-1M-tokens for budget testing.
    """

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    ULTRA_HIGH = "ultra_high"


class Media(StrEnum):
    """Single canonical enum for content and model capabilities.

    Use for: message content type, agent input/output capabilities, router profile
    support, and file/media detection. One enum everywhere — no Modality/ContentType split.

    Attributes:
        TEXT: Plain text.
        IMAGE: Image input/output.
        VIDEO: Video input/output.
        AUDIO: Audio input/output.
        FILE: Generic file attachment (e.g. PDF); use with InputFileRules for allowed types.
    """

    TEXT = "text"
    IMAGE = "image"
    VIDEO = "video"
    AUDIO = "audio"
    FILE = "file"


class SandboxRuntime(StrEnum):
    """Supported sandbox execution runtimes."""

    DOCKER = "docker"
    E2B = "e2b"
    LOCAL = "local"
    SUBPROCESS = "subprocess"


class Hook(StrEnum):
    """Lifecycle hooks — the primary observability mechanism."""

    AGENT_INIT = "agent.init"
    AGENT_RUN_START = "agent.run.start"
    AGENT_RUN_END = "agent.run.end"
    AGENT_RESET = "agent.reset"

    SERVE_REQUEST_START = "serve.request.start"
    SERVE_REQUEST_END = "serve.request.end"
    DISCOVERY_REQUEST = "discovery.request"

    MCP_CONNECTED = "mcp.connected"
    MCP_DISCONNECTED = "mcp.disconnected"
    MCP_TOOL_CALL_START = "mcp.tool.call.start"
    MCP_TOOL_CALL_END = "mcp.tool.call.end"

    LLM_REQUEST_START = "llm.request.start"
    LLM_REQUEST_END = "llm.request.end"
    LLM_STREAM_CHUNK = "llm.stream.chunk"
    LLM_RETRY = "llm.retry"
    LLM_FALLBACK = "llm.fallback"

    TOOL_CALL_START = "tool.call.start"
    TOOL_CALL_END = "tool.call.end"
    TOOL_ERROR = "tool.error"

    BUDGET_CHECK = "budget.check"
    BUDGET_THRESHOLD = "budget.threshold"
    BUDGET_EXCEEDED = "budget.exceeded"

    MODEL_SWITCH = "model.switch"

    ROUTING_DECISION = "routing.decision"

    GENERATION_IMAGE_START = "generation.image.start"
    GENERATION_IMAGE_END = "generation.image.end"
    GENERATION_IMAGE_ERROR = "generation.image.error"
    GENERATION_VIDEO_START = "generation.video.start"
    GENERATION_VIDEO_END = "generation.video.end"
    GENERATION_VIDEO_ERROR = "generation.video.error"

    HANDOFF_START = "handoff.start"
    HANDOFF_END = "handoff.end"
    HANDOFF_BLOCKED = "handoff.blocked"
    SPAWN_START = "spawn.start"
    SPAWN_END = "spawn.end"

    GUARDRAIL_INPUT = "guardrail.input"
    GUARDRAIL_OUTPUT = "guardrail.output"
    GUARDRAIL_BLOCKED = "guardrail.blocked"

    MEMORY_RECALL = "memory.recall"
    MEMORY_STORE = "memory.store"
    MEMORY_FORGET = "memory.forget"
    MEMORY_CONSOLIDATE = "memory.consolidate"
    MEMORY_EXTRACT = "memory.extract"

    KNOWLEDGE_INGEST_START = "knowledge.ingest.start"
    KNOWLEDGE_INGEST_END = "knowledge.ingest.end"
    KNOWLEDGE_SEARCH_START = "knowledge.search.start"
    KNOWLEDGE_SEARCH_END = "knowledge.search.end"
    KNOWLEDGE_SYNC = "knowledge.sync"
    KNOWLEDGE_SOURCE_ADDED = "knowledge.source.added"
    KNOWLEDGE_SOURCE_REMOVED = "knowledge.source.removed"

    # Agentic RAG (Step 6)
    KNOWLEDGE_AGENTIC_DECOMPOSE = "knowledge.agentic.decompose"
    KNOWLEDGE_AGENTIC_GRADE = "knowledge.agentic.grade"
    KNOWLEDGE_AGENTIC_REFINE = "knowledge.agentic.refine"
    KNOWLEDGE_AGENTIC_VERIFY = "knowledge.agentic.verify"

    CHECKPOINT_SAVE = "checkpoint.save"
    CHECKPOINT_LOAD = "checkpoint.load"

    REMOTE_CONFIG_UPDATE = "remote.config.update"
    REMOTE_CONFIG_ERROR = "remote.config.error"

    CONTEXT_COMPRESS = "context.compress"
    CONTEXT_COMPACT = "context.compact"
    CONTEXT_THRESHOLD = "context.threshold"
    CONTEXT_SNAPSHOT = "context.snapshot"
    CONTEXT_OFFLOAD = "context.offload"
    CONTEXT_RESTORE = "context.restore"

    RATELIMIT_CHECK = "ratelimit.check"
    RATELIMIT_THRESHOLD = "ratelimit.threshold"
    RATELIMIT_EXCEEDED = "ratelimit.exceeded"

    OUTPUT_VALIDATION_START = "output.validation.start"
    OUTPUT_VALIDATION_ATTEMPT = "output.validation.attempt"
    OUTPUT_VALIDATION_SUCCESS = "output.validation.success"
    OUTPUT_VALIDATION_FAILED = "output.validation.failed"
    OUTPUT_VALIDATION_RETRY = "output.validation.retry"

    HARNESS_SESSION_START = "harness.session.start"
    HARNESS_SESSION_END = "harness.session.end"
    HARNESS_PROGRESS = "harness.progress"
    HARNESS_CIRCUIT_TRIP = "harness.circuit.trip"
    HARNESS_CIRCUIT_RESET = "harness.circuit.reset"

    CIRCUIT_TRIP = "circuit.trip"
    CIRCUIT_RESET = "circuit.reset"

    HITL_PENDING = "hitl.pending"
    HITL_APPROVED = "hitl.approved"
    HITL_REJECTED = "hitl.rejected"

    SYSTEM_PROMPT_BEFORE_RESOLVE = "system_prompt.before_resolve"
    SYSTEM_PROMPT_AFTER_RESOLVE = "system_prompt.after_resolve"

    DYNAMIC_PIPELINE_START = "dynamic.pipeline.start"
    DYNAMIC_PIPELINE_PLAN = "dynamic.pipeline.plan"
    DYNAMIC_PIPELINE_EXECUTE = "dynamic.pipeline.execute"
    DYNAMIC_PIPELINE_AGENT_SPAWN = "dynamic.pipeline.agent.spawn"
    DYNAMIC_PIPELINE_AGENT_COMPLETE = "dynamic.pipeline.agent.complete"
    DYNAMIC_PIPELINE_END = "dynamic.pipeline.end"
    DYNAMIC_PIPELINE_ERROR = "dynamic.pipeline.error"
    HARNESS_RETRY = "harness.retry"

    # Static Pipeline (audit)
    PIPELINE_START = "pipeline.start"
    PIPELINE_END = "pipeline.end"
    PIPELINE_AGENT_START = "pipeline.agent.start"
    PIPELINE_AGENT_COMPLETE = "pipeline.agent.complete"


class AspectRatio(StrEnum):
    """Aspect ratio for generated images or videos.

    Attributes:
        ONE_TO_ONE: 1:1 (square).
        THREE_FOUR: 3:4 (portrait).
        FOUR_THREE: 4:3 (landscape).
        NINE_SIXTEEN: 9:16 (portrait).
        SIXTEEN_NINE: 16:9 (landscape).
    """

    ONE_TO_ONE = "1:1"
    THREE_FOUR = "3:4"
    FOUR_THREE = "4:3"
    NINE_SIXTEEN = "9:16"
    SIXTEEN_NINE = "16:9"


class OutputMimeType(StrEnum):
    """Output MIME type for generated images."""

    IMAGE_PNG = "image/png"
    IMAGE_JPEG = "image/jpeg"


class DocFormat(StrEnum):
    """Format for tool documentation sent to LLMs."""

    TOON = "toon"
    JSON = "json"
    YAML = "yaml"


class MemoryType(StrEnum):
    """Types of memory an agent can store and retrieve.

    Based on cognitive science: different types for different use cases.
    Use with Memory.types, remember(), and recall(memory_type=...).
    """

    CORE = "core"
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    PROCEDURAL = "procedural"


# Per-member docstrings for IDE hover (StrEnum doesn't support inline docstrings)
MemoryType.CORE.__doc__ = (
    "Identity and preferences. Use for: user name, role, language, persistent "
    "facts that rarely change. Survives across sessions."
)
MemoryType.EPISODIC.__doc__ = (
    "Past events and conversations. Use for: 'last discussed X', 'user asked about Y', "
    "'we talked about Z yesterday'. Chronological, context-dependent."
)
MemoryType.SEMANTIC.__doc__ = (
    "General knowledge and concepts. Use for: facts, definitions, extracted insights "
    "from turns. Ideal for vector/semantic search (Qdrant, Chroma)."
)
MemoryType.PROCEDURAL.__doc__ = (
    "How-to knowledge and skills. Use for: workflows, steps, preferences on how "
    "to do things (e.g. 'user prefers async over sync')."
)


class ServeProtocol(StrEnum):
    """Transport protocol for serving agents.

    Use when calling ``agent.serve(protocol=ServeProtocol.HTTP)`` or via
    ServeConfig.protocol.

    Attributes:
        HTTP: FastAPI server. Exposes /chat, /stream, /playground, etc. Default.
        CLI: Interactive REPL in terminal. Prompt → run → show cost.
        STDIO: JSON lines over stdin/stdout. For process spawning, background tasks.
    """

    CLI = "cli"
    HTTP = "http"
    STDIO = "stdio"


class WriteMode(StrEnum):
    """How remember/forget ops behave: SYNC blocks until complete; ASYNC fire-and-forget."""

    SYNC = "sync"
    ASYNC = "async"


class MemoryBackend(StrEnum):
    """Built-in memory storage backends.

    Available:
    - MEMORY: In-memory (default, ephemeral, fast)
    - SQLITE: File-based SQLite (persistent, stored at path or ~/.syrin/memory.db)
    - QDRANT: Vector database for semantic search
    - CHROMA: Lightweight vector database
    - REDIS: Fast in-memory cache with persistence options
    - POSTGRES: PostgreSQL for production (with pgvector for embeddings)
    """

    MEMORY = "memory"
    SQLITE = "sqlite"
    QDRANT = "qdrant"
    CHROMA = "chroma"
    REDIS = "redis"
    POSTGRES = "postgres"


class KnowledgeBackend(StrEnum):
    """Knowledge store vector backend selection.

    Available:
    - MEMORY: In-memory (testing, ephemeral, no deps)
    - SQLITE: Single-file, zero-config (sqlite-vec)
    - POSTGRES: Production, pgvector, ACID
    - QDRANT: High-performance vector search, cloud-ready
    - CHROMA: Local dev, lightweight
    """

    MEMORY = "memory"
    SQLITE = "sqlite"
    POSTGRES = "postgres"
    QDRANT = "qdrant"
    CHROMA = "chroma"


class MemoryScope(StrEnum):
    """Scope boundary for memory isolation. USER (default): per-user; SESSION: per conversation; AGENT: per agent; GLOBAL: shared across all."""

    SESSION = "session"
    AGENT = "agent"
    USER = "user"
    GLOBAL = "global"


class DecayStrategy(StrEnum):
    """How memory importance decays over time."""

    EXPONENTIAL = "exponential"
    LINEAR = "linear"
    LOGARITHMIC = "logarithmic"
    STEP = "step"
    NONE = "none"


class InjectionStrategy(StrEnum):
    """How recalled memories are ordered when injected into context.

    CHRONOLOGICAL: Oldest first. RELEVANCE: By relevance score (highest first).
    ATTENTION_OPTIMIZED (default): Order optimized for model attention (e.g. most relevant near current turn).
    """

    CHRONOLOGICAL = "chronological"
    RELEVANCE = "relevance"
    ATTENTION_OPTIMIZED = "attention_optimized"


class CheckpointStrategy(StrEnum):
    """How agent state is checkpointed for long-running tasks."""

    FULL = "full"
    INCREMENTAL = "incremental"
    EVENT_SOURCED = "event_sourced"
    HYBRID = "hybrid"


class CheckpointBackend(StrEnum):
    """Built-in checkpoint storage backends."""

    MEMORY = "memory"
    SQLITE = "sqlite"
    POSTGRES = "postgres"
    FILESYSTEM = "filesystem"


class OffloadBackend(StrEnum):
    """Where to store offloaded context data."""

    MEMORY = "memory"
    FILESYSTEM = "filesystem"
    SQLITE = "sqlite"
    REDIS = "redis"


class RetryBackoff(StrEnum):
    """Retry backoff strategies for provider failures."""

    EXPONENTIAL = "exponential"
    LINEAR = "linear"
    CONSTANT = "constant"


class CircuitState(StrEnum):
    """Circuit breaker states."""

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class ProgressStatus(StrEnum):
    """Status of a tracked progress item."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


class ThresholdMetric(StrEnum):
    """Metrics that can be tracked with thresholds."""

    COST = "cost"  # Budget cost (USD)
    TOKENS = "tokens"  # Context tokens
    RPM = "rpm"  # Requests per minute
    TPM = "tpm"  # Tokens per minute
    RPD = "rpd"  # Requests per day


class RateLimitAction(StrEnum):
    """Actions triggered at rate limit thresholds."""

    WARN = "warn"
    WAIT = "wait"
    SWITCH_MODEL = "switch_model"
    STOP = "stop"
    ERROR = "error"
    CUSTOM = "custom"
