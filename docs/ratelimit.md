# Rate Limiting

Syrin provides a comprehensive API rate limiting system that proactively manages provider rate limits with threshold actions. It tracks RPM (requests per minute), TPM (tokens per minute), and RPD (requests per day), and can automatically switch models, warn, or stop when limits are approached.

> **Agent integration:** For the agent `rate_limit=` param and `rate_limit_stats`, see [Agent: Rate Limiting](agent/rate-limiting.md).

## Installation

No additional dependencies required. The rate limiting system is built into Syrin.

```python
# Optional: for Redis backend
pip install redis
```

## Quick Start

```python
from syrin import Agent, Model
from syrin.ratelimit import APIRateLimit
from syrin.enums import RateLimitAction, ThresholdMetric
from syrin.ratelimit import RateLimitThreshold

# Auto-detect limits (recommended)
agent = Agent(
    model=Model("openai/gpt-4o"),
    rate_limit=APIRateLimit.auto_detect_for_model("openai/gpt-4o"),
)

# Or with proactive thresholds
agent = Agent(
    model=Model("gpt-4o"),
    rate_limit=APIRateLimit(
        rpm=500,
        tpm=150000,
        thresholds=[
            RateLimitThreshold(at=80, metric=ThresholdMetric.RPM, action=RateLimitAction.WARN),
            RateLimitThreshold(at=100, metric=ThresholdMetric.RPM, action=RateLimitAction.SWITCH_MODEL, switch_to_model="gpt-4o-mini"),
        ]
    )
)
```

That's it! The rate limiter:
- Auto-detects provider limits (or uses your custom limits)
- Tracks RPM, TPM, RPD with rolling windows
- Executes threshold actions automatically
- Emits events for observability

---

## Why Rate Limiting Matters

LLM providers enforce rate limits to prevent abuse. When you hit these limits:

1. **Requests fail** - 429 errors crash your application
2. **Costs spike** - Retries without backoff waste money
3. **UX suffers** - Users see errors or slow responses

Syrin handles this automatically so you don't have to think about it.

---

## Basic Usage

### Auto-Detect Limits

The easiest way to get started:

```python
from syrin import Agent, Model
from syrin.ratelimit import APIRateLimit

agent = Agent(
    model=Model("openai/gpt-4o"),
    rate_limit=APIRateLimit.auto_detect_for_model("openai/gpt-4o"),
)

# Access rate limit configuration
print(agent.rate_limit.rpm)   # 500 (detected from presets)
print(agent.rate_limit.tpm)   # 150000
```

### Manual Configuration

```python
from syrin import Agent, Model
from syrin.ratelimit import APIRateLimit

agent = Agent(
    model=Model("openai/gpt-4o"),
    rate_limit=APIRateLimit(
        rpm=500,      # Requests per minute
        tpm=150000,   # Tokens per minute
        rpd=10000,    # Requests per day (optional)
    ),
)
```

---

## Threshold Actions

Define actions at specific usage percentages:

```python
from syrin.ratelimit import APIRateLimit, RateLimitThreshold
from syrin.enums import RateLimitAction, ThresholdMetric

rate_limit = APIRateLimit(
    rpm=500,
    thresholds=[
        # At 50% - warn developers
        RateLimitThreshold(
            at=50,
            metric=ThresholdMetric.RPM,
            action=RateLimitAction.WARN,
        ),
        
        # At 80% - wait before next request
        RateLimitThreshold(
            at=80,
            metric=ThresholdMetric.RPM,
            action=RateLimitAction.WAIT,
            wait_seconds=2.0,
        ),
        
        # At 100% - switch to cheaper model
        RateLimitThreshold(
            at=100,
            metric=ThresholdMetric.RPM,
            action=RateLimitAction.SWITCH_MODEL,
            switch_to_model="gpt-4o-mini",
        ),
    ],
)
```

### RateLimitAction Options

| Action | Description |
|--------|-------------|
| `WARN` | Emit warning event, continue |
| `WAIT` | Wait specified seconds before proceeding |
| `SWITCH_MODEL` | Switch to fallback model |
| `STOP` | Stop execution |
| `ERROR` | Raise an error |
| `CUSTOM` | Call custom handler function |

### Custom Handler

```python
from syrin.ratelimit import APIRateLimit, RateLimitThreshold, RateLimitThresholdContext

def my_handler(ctx: RateLimitThresholdContext):
    print(f"Approaching {ctx.metric} limit: {ctx.percentage}% ({ctx.used}/{ctx.limit})")
    # Send alert, log, etc.

rate_limit = APIRateLimit(
    rpm=500,
    thresholds=[
        RateLimitThreshold(
            at=80,
            metric=ThresholdMetric.RPM,
            action=RateLimitAction.CUSTOM,
            handler=my_handler,
        ),
    ],
)
```

---

## ThresholdMetric Options

| Metric | Description |
|--------|-------------|
| `RPM` | Requests per minute |
| `TPM` | Tokens per minute |
| `RPD` | Requests per day |

---

## Configuration Options

### APIRateLimit

```python
from syrin.ratelimit import APIRateLimit

rate_limit = APIRateLimit(
    rpm=500,              # Requests per minute (None = unlimited)
    tpm=150000,          # Tokens per minute (None = unlimited)
    rpd=None,            # Requests per day (None = unlimited)
    thresholds=[],       # List of RateLimitThreshold
    wait_backoff=1.0,    # Default wait time for WAIT action
    auto_switch=True,    # Auto-switch model on threshold
    auto_detect=False,   # Auto-detect from provider (use auto_detect_for_model)
    retry_on_429=True,   # Retry on 429 errors
    max_retries=3,       # Max retries for 429
)
```

### RateLimitThreshold

```python
from syrin.ratelimit import RateLimitThreshold
from syrin.enums import RateLimitAction, ThresholdMetric

threshold = RateLimitThreshold(
    at=80,                           # Percentage (0-100)
    metric=ThresholdMetric.RPM,      # Which metric to monitor
    action=RateLimitAction.WARN,      # What to do
    handler=None,                     # For CUSTOM action
    switch_to_model=None,             # For SWITCH_MODEL action
    wait_seconds=1.0,                 # For WAIT action
    message=None,                     # Optional message
)
```

---

## Rate Limit Properties

### agent.rate_limit

Returns the `APIRateLimit` configuration:

```python
agent = Agent(
    model=Model("openai/gpt-4o"),
    rate_limit=APIRateLimit(rpm=500, tpm=150000),
)

print(agent.rate_limit.rpm)   # 500
print(agent.rate_limit.tpm)   # 150000
```

### agent.rate_limit_stats

Returns `RateLimitStats` from tracking:

```python
agent = Agent(model=Model("openai/gpt-4o"), rate_limit=APIRateLimit(rpm=500))

# After some requests
stats = agent.rate_limit_stats
print(stats.rpm_used)          # Current RPM usage
print(stats.rpm_limit)         # RPM limit
print(stats.tpm_used)         # Current TPM usage
print(stats.tpm_limit)        # TPM limit
print(stats.thresholds_triggered)  # List of triggered actions
```

---

## Events

The rate limiter emits events you can listen to:

### ratelimit.threshold

Emitted when a threshold is crossed:

```python
from syrin import Agent, Model
from syrin.ratelimit import APIRateLimit, RateLimitThreshold
from syrin.enums import RateLimitAction, ThresholdMetric

agent = Agent(
    model=Model("openai/gpt-4o"),
    rate_limit=APIRateLimit(
        rpm=100,
        thresholds=[
            RateLimitThreshold(at=80, metric=ThresholdMetric.RPM, action=RateLimitAction.WARN),
        ],
    ),
)

def on_threshold(ctx):
    print(f"Threshold reached: {ctx['at']}%")
    print(f"  Metric: {ctx['metric']}")
    print(f"  Action: {ctx['action']}")
    print(f"  Usage: {ctx['used']}/{ctx['limit']}")

agent.events.on("ratelimit.threshold", on_threshold)
```

### ratelimit.exceeded

Emitted when a rate limit is exceeded:

```python
def on_exceeded(ctx):
    print(f"Rate limit exceeded!")
    print(f"  Metric: {ctx['metric']}")
    print(f"  Usage: {ctx['used']}/{ctx['limit']}")

agent.events.on("ratelimit.exceeded", on_exceeded)
```

---

## Provider Auto-Detection

Syrin includes presets for known providers:

```python
from syrin.ratelimit import auto_detect_limits, suggest_limits

# Get limits for a model
limits = auto_detect_limits("openai/gpt-4o")
# {'rpm': 500, 'tpm': 150000, 'rpd': None}

# Get limits for different tiers
limits = suggest_limits("openai/gpt-4o", "tier2")
# {'rpm': 1000, 'tpm': 300000, 'rpd': None}
```

### Supported Providers

- OpenAI (gpt-4*, gpt-3.5*, o1*)
- Anthropic (claude-3*, claude-2*)
- Google (gemini-1.5*, gemini-2*)
- Groq
- Azure OpenAI
- Cohere
- Mistral
- Together AI
- Fireworks AI
- Perplexity
- Ollama (no limits)
- LiteLLM

---

## Persistence Backends

For multi-instance deployments, use persistent storage:

### Memory (Default)

```python
from syrin.ratelimit import get_rate_limit_backend

backend = get_rate_limit_backend("memory")
```

### SQLite

```python
# File-based, single machine
backend = get_rate_limit_backend("sqlite", path="/tmp/ratelimit.db")
```

### Redis

```python
# Distributed, multi-instance
backend = get_rate_limit_backend(
    "redis",
    host="redis.example.com",
    port=6379,
    db=0,
)
```

### Using with Agent

```python
from syrin import Agent, Model
from syrin.ratelimit import APIRateLimit, create_rate_limit_manager, get_rate_limit_backend

# Create rate limit manager
rate_limit = APIRateLimit(rpm=500)
manager = create_rate_limit_manager(rate_limit)

# Add persistence
backend = get_rate_limit_backend("redis", host="localhost")
manager.set_backend(backend, key="my-agent")

# Note: For full integration, rate_limit parameter handles this automatically
# This is for advanced custom usage
```

---

## Retry with Backoff

Automatic retry on 429 errors:

```python
from syrin.ratelimit import RetryConfig, create_retry_handler
from syrin.enums import RetryBackoff

# Configure retry behavior
config = RetryConfig(
    max_retries=3,
    base_delay=1.0,           # Initial delay
    max_delay=60.0,           # Maximum delay
    backoff_strategy=RetryBackoff.EXPONENTIAL,
    jitter=True,              # Add randomness to prevent thundering herd
)

handler = create_retry_handler(max_retries=3, base_delay=1.0)
```

---

## Observability

Rate limit operations are automatically traced:

```python
agent = Agent(
    model=Model("openai/gpt-4o"),
    rate_limit=APIRateLimit(rpm=500),
    debug=True,  # Enables trace output
)
```

Output shows:
- `ratelimit.check` spans with usage stats
- Threshold trigger events
- Model switch events

---

## Custom Rate Limit Manager

For advanced use cases, implement the `RateLimitManager` Protocol:

```python
from syrin.ratelimit import RateLimitManager, RateLimitStats, APIRateLimit

class MyRateLimitManager(RateLimitManager):
    """Custom rate limit strategy."""
    
    def __init__(self, config: APIRateLimit):
        self.config = config
        self._usage = {"rpm": 0, "tpm": 0, "rpd": 0}
    
    def check(self, tokens_used: int = 0) -> tuple[bool, str]:
        if self._usage["rpm"] >= (self.config.rpm or float('inf')):
            return False, "RPM exceeded"
        return True, "OK"
    
    def record(self, tokens_used: int = 0) -> None:
        self._usage["rpm"] += 1
        self._usage["tpm"] += tokens_used
    
    def get_stats(self) -> RateLimitStats:
        return RateLimitStats(
            rpm_used=self._usage["rpm"],
            rpm_limit=self.config.rpm or 0,
        )

# Use custom manager
agent = Agent(
    model=Model("openai/gpt-4o"),
    rate_limit=MyRateLimitManager(APIRateLimit(rpm=500)),
)
```

---

## Complete Example

```python
from syrin import Agent, Model
from syrin.ratelimit import APIRateLimit, RateLimitThreshold
from syrin.enums import RateLimitAction, ThresholdMetric

# Create agent with full rate limit configuration
agent = Agent(
    model=Model("openai/gpt-4o"),
    rate_limit=APIRateLimit(
        rpm=500,
        tpm=150000,
        rpd=10000,
        thresholds=[
            RateLimitThreshold(at=50, metric=ThresholdMetric.RPM, action=RateLimitAction.WARN),
            RateLimitThreshold(at=80, metric=ThresholdMetric.RPM, action=RateLimitAction.WAIT, wait_seconds=2.0),
            RateLimitThreshold(at=100, metric=ThresholdMetric.RPM, action=RateLimitAction.SWITCH_MODEL, switch_to_model="openai/gpt-4o-mini"),
        ],
        retry_on_429=True,
        max_retries=3,
    ),
)

# Listen to events
def on_threshold(ctx):
    print(f"Threshold: {ctx['at']}% -> {ctx['action']}")

def on_exceeded(ctx):
    print(f"Rate limit exceeded: {ctx['metric']}")

agent.events.on("ratelimit.threshold", on_threshold)
agent.events.on("ratelimit.exceeded", on_exceeded)

# Run conversation - rate limits checked automatically
result = agent.response("Hello!")
print(result.content)

# Check stats
print(f"RPM: {agent.rate_limit_stats.rpm_used}/{agent.rate_limit_stats.rpm_limit}")
print(f"TPM: {agent.rate_limit_stats.tpm_used}/{agent.rate_limit_stats.tpm_limit}")
```

---

## API Reference

### Classes

| Class | Description |
|-------|-------------|
| `APIRateLimit` | Main configuration for API rate limiting |
| `RateLimitStats` | Statistics from rate limit tracking |
| `RateLimitThreshold` | Threshold + action definition |
| `RateLimitThresholdContext` | Context passed to custom handlers |
| `DefaultRateLimitManager` | Default implementation |
| `RateLimitRetryHandler` | Retry logic for 429 errors |

### Backends

| Class | Description |
|-------|-------------|
| `MemoryRateLimitBackend` | In-memory (default) |
| `SQLiteRateLimitBackend` | File-based persistence |
| `RedisRateLimitBackend` | Distributed storage |

### Functions

| Function | Description |
|----------|-------------|
| `auto_detect_limits(model_id)` | Get limits from presets |
| `suggest_limits(model_id, tier)` | Get limits with tier scaling |
| `get_rate_limit_backend(type)` | Create a backend |
| `create_retry_handler()` | Create retry handler |

### RateLimitManager Protocol

```python
class RateLimitManager(Protocol):
    def check(self, tokens_used: int = 0) -> tuple[bool, str]:
        """Check if request is allowed."""
        ...
    
    def record(self, tokens_used: int = 0) -> None:
        """Record a request."""
        ...
    
    def get_stats(self) -> RateLimitStats:
        """Get current statistics."""
        ...
```

---

## Troubleshooting

### Rate limits not being tracked

- Ensure `rate_limit` is configured on Agent
- Check `agent.rate_limit_stats` after requests

### Threshold actions not executing

- Verify threshold percentage is being reached
- Ensure action is not STOP/ERROR (which would end execution)

### Model not switching

- Check that `switch_to_model` is set for SWITCH_MODEL action
- Verify the fallback model name is valid

### 429 errors still occurring

- Increase rate limits if below provider limits
- Add retry configuration: `retry_on_429=True`, `max_retries=5`

### Multi-instance not working

- Use Redis backend for distributed deployments
- Ensure Redis is accessible from all instances
- Check that `key` is unique per deployment
