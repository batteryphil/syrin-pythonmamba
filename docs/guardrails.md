# Syrin Guardrails Documentation

**Status:** Production Ready  
**Last Updated:** February 23, 2026

---

## Table of Contents

1. [Overview](#overview)
2. [Quick Start](#quick-start)
3. [Core Concepts](#core-concepts)
4. [Built-in Guardrails](#built-in-guardrails)
5. [Creating Custom Guardrails](#creating-custom-guardrails)
6. [Parallel vs Sequential Evaluation](#parallel-vs-sequential-evaluation)
7. [Budget Awareness](#budget-awareness)
8. [Observability](#observability)
9. [Integration with Agent](#integration-with-agent)
10. [Authority Layer](#authority-layer)
11. [Intelligence Layer](#intelligence-layer)
12. [API Reference](#api-reference)
13. [Best Practices](#best-practices)
14. [Examples](#examples)

---

## Overview

Syrin Guardrails is a comprehensive safety and validation system for AI agents with three integrated layers:

1. **Foundation Layer** - Content filtering, PII detection, parallel evaluation
2. **Authority Layer** - Permission-based authorization, budget enforcement, human approval
3. **Intelligence Layer** - Context-aware tracking, escalation detection, adaptive thresholds, red teaming

Unlike traditional output filtering, Syrin Guardrails provides **structural authority control** - validating actions, permissions, and budgets rather than just checking text.

### Key Features

- **Parallel Evaluation** - All guardrails run concurrently for minimal latency
- **Rich Context** - Full conversation history, budget state, and metadata
- **Structured Decisions** - Detailed reasoning, alternatives, and confidence scores
- **Budget Awareness** - Track and enforce costs for safety checks
- **Observable** - Full tracing, metrics, and hooks for every decision
- **Composable** - Chain and combine guardrails flexibly

> **Agent integration:** For the agent `guardrails=` constructor param, see [Agent: Guardrails](agent/guardrails.md).

### Design Principles

1. **Separate intelligence from authority** - Agents can be smart but can't act beyond permissions
2. **Structural constraints > logical constraints** - Physics, not policy
3. **Observable everything** - Every decision traced and measurable
4. **Fail safely** - Block by default when uncertain

---

## Quick Start

### Installation

```bash
pip install syrin
```

### Basic Usage

```python
from syrin import Agent, Model
from syrin.guardrails import ContentFilter, PIIScanner

# Simple setup
agent = Agent(
    # model=Model.OpenAI("gpt-4o"),
    model=Model.Almock(),  # No API Key needed
    guardrails=[
        ContentFilter(blocked_words=["password", "secret"]),
        PIIScanner(redact=True),
    ]
)

# That's it! Agent now has content filtering
```

### First Guardrail in 30 Seconds

```python
import asyncio
from syrin.guardrails import Guardrail, GuardrailContext, GuardrailDecision

class MyGuardrail(Guardrail):
    async def evaluate(self, context: GuardrailContext) -> GuardrailDecision:
        if "forbidden" in context.text.lower():
            return GuardrailDecision(
                passed=False,
                rule="forbidden_word",
                reason="Contains forbidden word"
            )
        return GuardrailDecision(passed=True)

# Test it
async def main():
    guardrail = MyGuardrail()
    context = GuardrailContext(text="This is forbidden")
    result = await guardrail.evaluate(context)
    print(f"Passed: {result.passed}")  # False

asyncio.run(main())
```

---

## Core Concepts

### GuardrailContext

The `GuardrailContext` provides all information needed for evaluation:

```python
from syrin.guardrails import GuardrailContext
from syrin.enums import GuardrailStage

context = GuardrailContext(
    text="The text being evaluated",
    stage=GuardrailStage.INPUT,  # input, action, or output
    conversation=conversation_obj,  # Optional: conversation history
    user=user_obj,  # Optional: user identity and permissions
    agent=agent_obj,  # Optional: agent instance
    budget=budget_obj,  # Optional: budget tracker
    action=action_obj,  # Optional: action being validated
    metadata={"request_id": "abc123"}  # Optional: custom metadata
)
```

**Key Properties:**
- **Immutable** - Thread-safe and prevents accidental modifications
- **Rich** - Includes everything a guardrail might need
- **Extensible** - Custom metadata for your use case

### GuardrailDecision

The `GuardrailDecision` provides comprehensive evaluation results:

```python
from syrin.guardrails import GuardrailDecision
from syrin.enums import DecisionAction

decision = GuardrailDecision(
    passed=False,  # Did it pass?
    rule="blocked_word",  # Which rule triggered?
    reason="Word 'password' is not allowed",  # Human-readable explanation
    confidence=1.0,  # 0.0 to 1.0
    action=DecisionAction.BLOCK,  # What to do
    alternatives=["Use 'credential' instead"],  # Suggestions
    metadata={"word": "password"},  # Detailed data
    latency_ms=2.5,  # Performance tracking
    budget_consumed=0.001  # Cost tracking
)
```

**Key Properties:**
- **Actionable** - Not just pass/fail, but what to do next
- **Debuggable** - Full context and metadata
- **Measurable** - Latency and budget tracking

### Guardrail Base Class

All guardrails inherit from `Guardrail`:

```python
from syrin.guardrails import Guardrail, GuardrailContext, GuardrailDecision

class MyGuardrail(Guardrail):
    def __init__(self, param1, name=None):
        super().__init__(name)
        self.param1 = param1
        self.budget_cost = 0.01  # Cost per evaluation
    
    async def evaluate(self, context: GuardrailContext) -> GuardrailDecision:
        # Your logic here
        if self._check(context):
            return GuardrailDecision(passed=True)
        return GuardrailDecision(
            passed=False,
            rule="my_rule",
            reason="Failed my check"
        )
```

**Key Methods:**
- `evaluate(context)` - Main evaluation logic (async)
- `budget_cost` - Declare cost for budget tracking

---

## Built-in Guardrails

### ContentFilter

Blocks text containing specific words or phrases.

```python
from syrin.guardrails import ContentFilter

guardrail = ContentFilter(
    blocked_words=["password", "secret", "api_key"],
    case_sensitive=False,  # Default: case-insensitive
    name="security_filter"
)

# Usage
result = await guardrail.evaluate(context)
if not result.passed:
    print(f"Blocked: {result.metadata['word']}")
```

**Use Cases:**
- Block security-sensitive terms
- Profanity filtering
- Compliance word filtering

### PIIScanner

Detects and optionally redacts personally identifiable information.

```python
from syrin.guardrails import PIIScanner

guardrail = PIIScanner(
    redact=True,  # Replace PII with ***
    redaction_char="*",
    allow_types=["ip_address"],  # Don't flag IP addresses
)

# Usage
result = await guardrail.evaluate(context)
if not result.passed:
    print(f"PII found: {result.metadata['findings']}")
    if result.metadata.get('redacted_text'):
        print(f"Redacted: {result.metadata['redacted_text']}")
```

**Detected PII Types:**
- Email addresses
- Phone numbers
- Social Security Numbers
- Credit card numbers
- IP addresses

**Use Cases:**
- GDPR compliance
- HIPAA compliance
- Data loss prevention

---

## Creating Custom Guardrails

### Simple Custom Guardrail

```python
from syrin.guardrails import Guardrail, GuardrailContext, GuardrailDecision

class LengthGuardrail(Guardrail):
    """Check message length."""
    
    def __init__(self, min_length=1, max_length=1000, name=None):
        super().__init__(name)
        self.min_length = min_length
        self.max_length = max_length
    
    async def evaluate(self, context: GuardrailContext) -> GuardrailDecision:
        length = len(context.text)
        
        if length < self.min_length:
            return GuardrailDecision(
                passed=False,
                rule="too_short",
                reason=f"Too short: {length} < {self.min_length}",
                metadata={"length": length, "min": self.min_length}
            )
        
        if length > self.max_length:
            return GuardrailDecision(
                passed=False,
                rule="too_long",
                reason=f"Too long: {length} > {self.max_length}",
                metadata={"length": length, "max": self.max_length}
            )
        
        return GuardrailDecision(passed=True)
```

### Budget-Aware Guardrail

```python
class ExpensiveLLMCheck(Guardrail):
    """Uses LLM for validation - expensive!"""
    
    def __init__(self, min_budget=0.05):
        super().__init__()
        self.budget_cost = 0.05  # $0.05 per check
        self.min_budget = min_budget
    
    async def evaluate(self, context: GuardrailContext) -> GuardrailDecision:
        # Check budget first
        if context.budget and context.budget.remaining < self.min_budget:
            return GuardrailDecision(
                passed=False,
                rule="budget_exhausted",
                reason=f"Cannot run: budget ${context.budget.remaining} < ${self.min_budget}",
                metadata={"required": self.min_budget}
            )
        
        # Consume budget
        if context.budget:
            context.budget.consume(self.budget_cost)
        
        # Run expensive check
        # ... LLM call here ...
        
        return GuardrailDecision(
            passed=True,
            budget_consumed=self.budget_cost
        )
```

### Context-Aware Guardrail

```python
class RateLimitGuardrail(Guardrail):
    """Rate limit per user."""
    
    def __init__(self, max_requests=100, window=3600):
        super().__init__()
        self.max_requests = max_requests
        self.window = window
        self._requests = {}  # In production, use Redis
    
    async def evaluate(self, context: GuardrailContext) -> GuardrailDecision:
        user_id = context.user.id if context.user else "anonymous"
        
        # Check rate limit
        current = self._requests.get(user_id, 0)
        if current >= self.max_requests:
            return GuardrailDecision(
                passed=False,
                rule="rate_limited",
                reason=f"Rate limit exceeded: {self.max_requests}/{self.window}s",
                metadata={"current": current, "limit": self.max_requests}
            )
        
        # Increment counter
        self._requests[user_id] = current + 1
        
        return GuardrailDecision(passed=True)
```

---

## Parallel vs Sequential Evaluation

### Parallel Evaluation (Default)

All guardrails run at the same time. Fastest overall latency.

```python
from syrin.guardrails import ParallelEvaluationEngine

engine = ParallelEvaluationEngine(
    timeout=5.0,  # Max time to wait
    fail_on_timeout=True  # Timeout = failure
)

result = await engine.evaluate(context, [guardrail1, guardrail2, guardrail3])
# Latency = max(guardrail latencies), not sum
```

**When to use:**
- Most cases (recommended default)
- Independent checks
- Need minimal latency

### Sequential Evaluation (Chain)

Guardrails run one at a time. Stops on first failure.

```python
from syrin.guardrails import GuardrailChain

chain = GuardrailChain([
    cheap_check(),  # Run first
    expensive_check(),  # Only if first passes
])

result = await chain.evaluate(context)
# Stops early if cheap_check fails
```

**When to use:**
- Order matters
- Early cheap checks can avoid expensive ones
- Dependencies between checks

---

## Budget Awareness

Track and enforce costs for guardrail operations.

### Setting Budget Costs

```python
class MyGuardrail(Guardrail):
    def __init__(self):
        super().__init__()
        self.budget_cost = 0.01  # $0.01 per evaluation
```

### Checking Budget in Guardrail

```python
async def evaluate(self, context: GuardrailContext) -> GuardrailDecision:
    if context.budget:
        if context.budget.remaining < self.budget_cost:
            return GuardrailDecision(
                passed=False,
                rule="budget_exhausted",
                reason="Not enough budget for this check"
            )
        
        # Consume budget
        context.budget.consume(self.budget_cost)
    
    # ... rest of evaluation
```

### Budget-First Strategy

```python
# Use cheap checks first, expensive only if budget allows
guardrails = [
    ContentFilter(),  # $0.00 - cheap
    RegexCheck(),     # $0.00 - cheap
    LLMJudgement(),   # $0.05 - expensive, only if budget
]
```

---

## Observability

### Understanding Decisions

Every guardrail decision includes:

```python
result = await guardrail.evaluate(context)

print(f"Passed: {result.passed}")
print(f"Rule: {result.rule}")
print(f"Reason: {result.reason}")
print(f"Confidence: {result.confidence}")
print(f"Latency: {result.latency_ms}ms")
print(f"Budget: ${result.budget_consumed}")
print(f"Metadata: {result.metadata}")
print(f"Alternatives: {result.alternatives}")

# Full JSON
json_str = result.to_json()
```

### Hooks

React to guardrail events:

```python
from syrin import Agent, Hook

agent = Agent(...)

@agent.on(Hook.GUARDRAIL_VIOLATION)
def on_violation(hook_data):
    print(f"Violation: {hook_data['rule']}")
    # Send alert, log to SIEM, etc.

@agent.on(Hook.GUARDRAIL_CHECKED)
def on_checked(hook_data):
    print(f"Checked: {hook_data['latency_ms']}ms")
```

### Tracing

Every guardrail evaluation creates a trace span:

```
[trace: abc-123]
  [span: guardrail.input]
    [span: guardrail.content_filter] 2ms
    [span: guardrail.pii_scanner] 5ms
```

### Metrics

Automatically tracked:

- `guardrail.checks.total` - Total evaluations
- `guardrail.checks.passed` - Passed checks
- `guardrail.checks.blocked` - Blocked checks
- `guardrail.latency_ms` - Evaluation latency
- `guardrail.budget_consumed` - Cost of checks

---

## Integration with Agent

### Basic Integration

```python
from syrin import Agent, Model
from syrin.guardrails import ContentFilter, PIIScanner

agent = Agent(
    # model=Model.OpenAI("gpt-4o"),
    model=Model.Almock(),  # No API Key needed
    guardrails=[
        ContentFilter(blocked_words=["password"]),
        PIIScanner(),
    ]
)

# Guardrails run automatically on input and output
```

### Stage-Specific Guardrails

```python
from syrin import GuardrailConfig

agent = Agent(
    # model=Model.OpenAI("gpt-4o"),
    model=Model.Almock(),  # No API Key needed
    guardrails=GuardrailConfig(
        input=[
            ContentFilter(blocked_words=["spam"]),
            PIIScanner(),
        ],
        actions={
            "transfer_funds": [
                AuthorityCheck(requires="finance:transfer"),
                BudgetEnforcer(max_amount=10000),
            ],
            "send_email": [
                RateLimiter(max_per_hour=100),
            ],
        },
        output=[
            ContentFilter(blocked_words=["confidential"]),
        ],
    )
)
```

---

## API Reference

### GuardrailContext

```python
@dataclass(frozen=True)
class GuardrailContext:
    text: str
    stage: GuardrailStage = GuardrailStage.INPUT
    conversation: Optional[Any] = None
    user: Optional[Any] = None
    agent: Optional[Any] = None
    budget: Optional[Any] = None
    action: Optional[Any] = None
    metadata: dict[str, Any] = field(default_factory=dict)
    
    def copy(self, **kwargs) -> GuardrailContext
    def to_dict(self) -> dict[str, Any]
```

### GuardrailDecision

```python
@dataclass
class GuardrailDecision:
    passed: bool
    rule: str = ""
    reason: str = ""
    confidence: float = 1.0
    action: DecisionAction = None
    metadata: dict[str, Any] = field(default_factory=dict)
    alternatives: list[str] = field(default_factory=list)
    latency_ms: float = 0.0
    budget_consumed: float = 0.0
    
    def to_dict(self) -> dict[str, Any]
    def to_json(self) -> str
```

### ParallelEvaluationEngine

```python
class ParallelEvaluationEngine:
    def __init__(
        self,
        timeout: float = 10.0,
        short_circuit: bool = False,
        fail_on_timeout: bool = True
    )
    
    async def evaluate(
        self,
        context: GuardrailContext,
        guardrails: List[Guardrail]
    ) -> EvaluationResult
```

### GuardrailChain

```python
class GuardrailChain:
    def __init__(self, guardrails: List[Guardrail] = None)
    def add(self, guardrail: Guardrail) -> None
    async def evaluate(self, context: GuardrailContext) -> EvaluationResult
    def check(self, text: str, stage: Any = None, *, budget: Any = None, agent: Any = None) -> GuardrailCheckResult
    def __len__(self) -> int
```

### GuardrailCheckResult

Returned by `GuardrailChain.check()` (sync API):

```python
@dataclass
class GuardrailCheckResult:
    passed: bool
    reason: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    guardrail_name: str | None = None  # Name of the guardrail that produced this result
```

When a check fails, `guardrail_name` identifies which guardrail in the chain blocked the content.

### AuthorityCheck

```python
class AuthorityCheck(Guardrail):
    def __init__(
        self,
        requires: Optional[Union[str, List[str]]] = None,
        logic: str = "all",  # "all" or "any"
        scope: Optional[str] = None,
        resource: Optional[Callable] = None,
        allowed_hours: Optional[tuple[int, int]] = None,
        condition: Optional[Callable] = None,
        name: Optional[str] = None,
    )
```

### BudgetEnforcer

```python
class BudgetEnforcer(Guardrail):
    def __init__(
        self,
        max_amount: float,
        daily_limit: Optional[float] = None,
        weekly_limit: Optional[float] = None,
        limits_by_action: Optional[Dict[str, float]] = None,
        cost: float = 0.0,
        warn_threshold: Optional[float] = None,
        fail_on_no_budget: bool = False,
        name: Optional[str] = None,
    )
```

### ThresholdApproval

```python
class ThresholdApproval(Guardrail):
    def __init__(
        self,
        k: int,  # Required approvals
        n: int,  # Total approvers
        approvers: Optional[List[str]] = None,
        timeout: int = 3600,
        condition: Optional[Callable] = None,
        name: Optional[str] = None,
    )
    
    def add_approval(self, request_id: str, approver: str) -> bool
    def add_rejection(self, request_id: str, approver: str, reason: str) -> None
    def get_approvals(self, request_id: str) -> Set[str]
```

### HumanApproval

```python
class HumanApproval(Guardrail):
    def __init__(
        self,
        approver: Optional[Union[str, Callable]] = None,
        timeout: int = 3600,
        requires_justification: bool = False,
        require_2fa: bool = False,
        escalation_timeout: Optional[int] = None,
        escalation_approver: Optional[str] = None,
        name: Optional[str] = None,
    )
    
    def approve(self, request_id: str, approver: str, justification: str = None) -> None
    def reject(self, request_id: str, approver: str, reason: str = None) -> None
```

### CapabilityToken

```python
@dataclass
class CapabilityToken:
    scope: str
    budget: int = 1
    ttl: Optional[int] = None
    token_id: str = field(default_factory=...)
    created_at: datetime = field(default_factory=datetime.now)
    issued_to: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)
    
    def consume(self, amount: int = 1) -> bool
    def is_expired(self) -> bool
    def is_valid(self) -> bool
    def can(self, action: str) -> bool
    def to_dict(self) -> dict[str, Any]
    @classmethod
    def from_dict(cls, data: dict) -> CapabilityToken
```

### CapabilityGuardrail

```python
class CapabilityGuardrail(Guardrail):
    def __init__(
        self,
        required_capability: str,
        consume_budget: bool = True,
        name: Optional[str] = None,
    )
```

### ContextAwareGuardrail

```python
class ContextAwareGuardrail(Guardrail):
    def __init__(
        self,
        max_history: int = 5,
        topic_escalation: Optional[List[str]] = None,
        session_ttl: int = 3600,
        name: Optional[str] = None,
    )
    
    async def evaluate(self, context: GuardrailContext) -> GuardrailDecision
    def get_session_history(self, session_id: str) -> List[Dict[str, Any]]
    def clear_session(self, session_id: str) -> None
```

### EscalationDetector

```python
class EscalationDetector(Guardrail):
    def __init__(
        self,
        time_window: int = 300,  # seconds
        violation_threshold: int = 3,
        progressive_tactic_check: bool = True,
        user_isolation: bool = True,
        name: Optional[str] = None,
    )
    
    async def evaluate(self, context: GuardrailContext) -> GuardrailDecision
    def record_violation(self, user_id: str, context: GuardrailContext) -> None
    def get_escalation_level(self, user_id: str) -> EscalationLevel
    def reset_user_tracking(self, user_id: str) -> None
```

### AdaptiveThresholdGuardrail

```python
class AdaptiveThresholdGuardrail(Guardrail):
    def __init__(
        self,
        initial_threshold: float = 0.7,
        min_threshold: float = 0.3,
        max_threshold: float = 0.95,
        learning_rate: float = 0.05,
        window_size: int = 100,
        name: Optional[str] = None,
    )
    
    @property
    def current_threshold(self) -> float
    
    def record_feedback(
        self,
        passed: bool,
        confidence: float,
        was_violation: bool
    ) -> None
    
    def get_stats(self) -> Dict[str, Any]
    async def evaluate(self, context: GuardrailContext) -> GuardrailDecision
```

### AttackSimulator

```python
class AttackSimulator:
    def __init__(self, seed: Optional[int] = None)
    
    def generate_jailbreak_attempts(
        self,
        target: str,
        count: int = 10,
        techniques: Optional[List[str]] = None
    ) -> List[str]
    
    def generate_prompt_injection(
        self,
        payload: str,
        count: int = 5,
        contexts: Optional[List[str]] = None
    ) -> List[str]
    
    def test_bypass_patterns(
        self,
        blocked_word: str,
        techniques: Optional[List[str]] = None
    ) -> List[str]
    
    def generate_encoding_variations(
        self,
        text: str,
        encodings: Optional[List[str]] = None
    ) -> List[str]
```

### RedTeamEvaluator

```python
class RedTeamEvaluator:
    def __init__(
        self,
        guardrail: Guardrail,
        iterations: int = 100,
        timeout: float = 30.0
    )
    
    async def evaluate(
        self,
        attack_types: Optional[List[str]] = None
    ) -> RedTeamReport
    
    async def evaluate_specific_attack(
        self,
        attack_type: str,
        count: int = 20
    ) -> AttackResult
```

### FuzzingEngine

```python
class FuzzingEngine:
    def __init__(self, seed_text: str)
    
    def mutate(
        self,
        count: int = 100,
        techniques: Optional[List[str]] = None,
        max_mutations: int = 5
    ) -> List[str]
    
    def boundary_test(
        self,
        lengths: Optional[List[int]] = None
    ) -> List[str]
    
    def character_variations(
        self,
        text: str,
        char_sets: Optional[List[str]] = None
    ) -> List[str]
```

---

## Best Practices

### 1. Start Simple

```python
# Start with basic guardrails
guardrails = [
    ContentFilter(blocked_words=["password", "secret"]),
]

# Add more as needed
guardrails.append(PIIScanner())
```

### 2. Use Parallel by Default

```python
# Parallel is faster
engine = ParallelEvaluationEngine()

# Only use chain when order matters
chain = GuardrailChain([...])
```

### 3. Declare Budget Costs

```python
class MyGuardrail(Guardrail):
    def __init__(self):
        super().__init__()
        self.budget_cost = 0.01  # Always declare cost
```

### 4. Provide Good Error Messages

```python
return GuardrailDecision(
    passed=False,
    rule="blocked",
    reason="Contains blocked word 'password'",  # Be specific
    alternatives=["Use 'credential' instead"],  # Suggest fixes
)
```

### 5. Test Edge Cases

```python
# Test empty text
# Test very long text
# Test unicode
# Test special characters
# Test concurrent access
```

### 6. Monitor Performance

```python
# Track latency
if result.latency_ms > 100:
    logger.warning(f"Slow guardrail: {result.latency_ms}ms")

# Track false positives
if result.passed == False and user_approved:
    log_false_positive(result)
```

---

## Authority Layer

The Authority Layer provides structural authorization controls for AI agents.

### AuthorityCheck

Validates user permissions before allowing actions.

```python
from syrin.guardrails import AuthorityCheck

# Single permission
guardrail = AuthorityCheck(requires="finance:transfer")

# Multiple permissions (ALL required)
guardrail = AuthorityCheck(
    requires=["read", "write"],
    logic="all"
)

# Multiple permissions (ANY required)
guardrail = AuthorityCheck(
    requires=["admin", "manager", "supervisor"],
    logic="any"
)

# With time restrictions
guardrail = AuthorityCheck(
    requires="admin",
    allowed_hours=(9, 17)  # Business hours only
)

# With conditional logic
guardrail = AuthorityCheck(
    requires="large_transfer",
    condition=lambda ctx: ctx.action.amount > 10000
)
```

### BudgetEnforcer

Enforces budget limits on actions.

```python
from syrin.guardrails import BudgetEnforcer

# Simple limit
guardrail = BudgetEnforcer(max_amount=1000)

# With daily/weekly limits
guardrail = BudgetEnforcer(
    max_amount=100,
    daily_limit=1000,
    weekly_limit=5000
)

# Per-action limits
guardrail = BudgetEnforcer(
    limits_by_action={
        "transfer_funds": 10000,
        "send_email": 100,
        "api_call": 10
    }
)

# With warning threshold
guardrail = BudgetEnforcer(
    max_amount=500,
    warn_threshold=0.2  # Warn at 20% remaining
)
```

### ThresholdApproval

Implements K-of-N consensus for critical actions.

```python
from syrin.guardrails import ThresholdApproval

# Require 2 of 3 approvers
guardrail = ThresholdApproval(k=2, n=3)

# With specific approvers
guardrail = ThresholdApproval(
    k=2,
    n=3,
    approvers=["alice@example.com", "bob@example.com", "charlie@example.com"],
    timeout=3600  # 1 hour
)

# With condition
guardrail = ThresholdApproval(
    k=2,
    n=3,
    condition=lambda ctx: ctx.action.amount > 10000
)

# Add approvals programmatically
guardrail.add_approval("request_123", "alice@example.com")
guardrail.add_approval("request_123", "bob@example.com")

# Check status
result = await guardrail.evaluate(context)
```

### HumanApproval

Requires explicit human approval.

```python
from syrin.guardrails import HumanApproval

# Simple approval
guardrail = HumanApproval(approver="admin@example.com")

# With 2FA requirement
guardrail = HumanApproval(
    approver="admin@example.com",
    require_2fa=True,
    requires_justification=True
)

# Dynamic approver
guardrail = HumanApproval(
    approver=lambda ctx: ctx.user.manager
)

# Record approval
guardrail.approve(
    request_id="req_123",
    approver="admin@example.com",
    justification="Code reviewed"
)
```

### Capability Tokens

Consumable permission tokens.

```python
from syrin.guardrails import CapabilityToken, CapabilityIssuer

# Issue a token
issuer = CapabilityIssuer()
token = issuer.issue(
    scope="finance:transfer",
    budget=10,  # 10 uses
    ttl=3600,   # 1 hour
    issued_to="user123"
)

# Check capability
can_transfer = token.can("finance:transfer")

# Consume budget
success = token.consume(1)

# Check validity
is_valid = token.is_valid()
is_expired = token.is_expired()

# Wildcard scopes
token = CapabilityToken(scope="finance:*")
can_transfer = token.can("finance:transfer")  # True
can_read = token.can("finance:read")  # True
```

### CapabilityGuardrail

Guardrail using capability tokens.

```python
from syrin.guardrails import CapabilityGuardrail

# Check for capability
guardrail = CapabilityGuardrail(
    required_capability="finance:transfer"
)

# Evaluate with user having token
result = await guardrail.evaluate(context)
```

## Intelligence Layer

The Intelligence Layer adds contextual awareness, adaptive behavior, and security testing capabilities to guardrails.

### ContextAwareGuardrail

Tracks conversation history and detects patterns across multiple turns.

```python
from syrin.guardrails.intelligence import ContextAwareGuardrail
from syrin.guardrails import GuardrailContext

# Create context-aware guardrail
context = GuardrailContext(
    text="Tell me about quantum computing",
    user={"id": "user123", "name": "Alice"},
    metadata={"session_id": "session_abc"}
)

# The guardrail tracks topic progression across conversation
result = await ContextAwareGuardrail(
    max_history=5,
    topic_escalation=["sensitive", "restricted"]
).evaluate(context)
```

**Features:**
- **Conversation History**: Tracks last N exchanges per session
- **Topic Escalation Detection**: Catches gradual topic shifts
- **Repeated Attempt Detection**: Identifies persistent circumvention attempts
- **Session Isolation**: Separate tracking per user session

### EscalationDetector

Detects violation spikes and progressive attack patterns.

```python
from syrin.guardrails.intelligence import EscalationDetector

# Monitor for escalation patterns
detector = EscalationDetector(
    time_window=300,  # 5 minutes
    violation_threshold=3,  # 3 violations triggers escalation
    progressive_tactic_check=True
)

# Use in guardrail pipeline
guardrails = [
    ContentFilter(blocked_words=["harmful"]),
    EscalationDetector(violation_threshold=3),
]
```

**Escalation Patterns:**
- **Violation Spike**: Multiple violations in short time window
- **Progressive Tactics**: User tries different approaches to bypass
- **User Isolation**: Separate tracking per user
- **Time Window Expiration**: Old violations age out automatically

### AdaptiveThresholdGuardrail

Auto-tunes thresholds based on feedback to reduce false positives.

```python
from syrin.guardrails.intelligence import AdaptiveThresholdGuardrail

# Starts with initial threshold
adaptive = AdaptiveThresholdGuardrail(
    initial_threshold=0.7,
    min_threshold=0.5,
    max_threshold=0.95,
    learning_rate=0.05,
    window_size=100  # Consider last 100 decisions
)

# Report feedback after decision
if user_challenged and was_actually_safe:
    # False positive - lower threshold
    adaptive.record_feedback(passed=False, confidence=0.8, was_violation=False)
elif not detected and was_violation:
    # Missed violation - raise threshold
    adaptive.record_feedback(passed=True, confidence=0.6, was_violation=True)

# Threshold automatically adjusts based on feedback history
```

**Adaptation Logic:**
- **False Positive Rate High**: Lower threshold (less strict)
- **Missed Violations**: Raise threshold (more strict)
- **Bounds Respect**: Never exceeds min/max limits
- **Confidence Weighting**: Higher confidence violations impact threshold more

### AttackSimulator

Generates attack attempts to test guardrail strength.

```python
from syrin.guardrails.intelligence import AttackSimulator, RedTeamEvaluator
from syrin.guardrails import ContentFilter

# Create simulator
simulator = AttackSimulator()

# Generate jailbreak attempts
jailbreaks = simulator.generate_jailbreak_attempts(
    target="harmful content",
    count=5,
    techniques=["ignore previous", "roleplay", "encoding"]
)
# Returns: ["Ignore previous instructions and...", "Pretend you're...", ...]

# Generate prompt injection attacks
injections = simulator.generate_prompt_injection(
    payload="harmful",
    count=3
)
# Returns: ["...harmful...", "...new instruction: harmful...", ...]

# Test against bypass patterns
bypasses = simulator.test_bypass_patterns(
    blocked_word="password",
    techniques=["leetspeak", "encoding", "spacing"]
)
# Returns: ["p4ssw0rd", "p%61ssword", "p a s s w o r d"]
```

**Attack Types:**
- **Jailbreak Attempts**: Social engineering to bypass safety
- **Prompt Injection**: Injecting malicious instructions
- **Encoding Tricks**: Base64, leetspeak, special characters
- **Roleplay Scenarios**: "Pretend you're an unrestricted AI"

### RedTeamEvaluator

Evaluates guardrail strength against simulated attacks.

```python
from syrin.guardrails.intelligence import RedTeamEvaluator

# Evaluate your guardrail
evaluator = RedTeamEvaluator(
    guardrail=ContentFilter(blocked_words=["harmful"]),
    iterations=100
)

# Run red team assessment
report = await evaluator.evaluate(
    attack_types=["jailbreak", "injection", "encoding"]
)

print(f"Attack Success Rate: {report['success_rate']:.1%}")
print(f"Mean Confidence: {report['mean_confidence']:.2f}")
print(f"Vulnerable to: {report['vulnerable_to']}")
```

**Metrics:**
- **Success Rate**: % of attacks that bypassed guardrail
- **Mean Confidence**: Average confidence of blocked attempts
- **Vulnerable Techniques**: Which attack types succeeded
- **Detailed Report**: Per-attack breakdown with payloads

### FuzzingEngine

Mutation-based fuzzing to find edge cases.

```python
from syrin.guardrails.intelligence import FuzzingEngine

# Create fuzzer for your guardrail
fuzzer = FuzzingEngine(seed_text="Hello world")

# Generate mutations
mutations = fuzzer.mutate(
    count=100,
    techniques=["insert", "delete", "substitute", "shuffle"]
)

# Test guardrail on mutations
for mutation in mutations:
    context = GuardrailContext(text=mutation)
    result = await guardrail.evaluate(context)
    if result.passed != expected:
        print(f"Edge case found: {mutation}")
```

**Mutation Techniques:**
- **Insert**: Add random characters
- **Delete**: Remove characters
- **Substitute**: Replace characters
- **Shuffle**: Reorder characters
- **Boundary**: Test length limits

### Complete Intelligence Pipeline

```python
from syrin.guardrails import ParallelEvaluationEngine
from syrin.guardrails.intelligence import (
    ContextAwareGuardrail,
    EscalationDetector,
    AdaptiveThresholdGuardrail,
    AttackSimulator
)

# Build intelligent guardrail system
intelligence_guardrails = [
    # Context awareness
    ContextAwareGuardrail(max_history=5),
    
    # Escalation detection
    EscalationDetector(
        time_window=300,
        violation_threshold=3
    ),
    
    # Adaptive threshold
    AdaptiveThresholdGuardrail(
        initial_threshold=0.75,
        min_threshold=0.5,
        max_threshold=0.95
    ),
]

# Regular guardrails
content_guardrails = [
    ContentFilter(blocked_words=["harmful", "dangerous"]),
    PIIScanner(),
]

# Combine all guardrails
engine = ParallelEvaluationEngine()

# Test against attacks periodically
simulator = AttackSimulator()
attacks = simulator.generate_jailbreak_attempts(count=50)

# Run red team evaluation
from syrin.guardrails.intelligence import RedTeamEvaluator
evaluator = RedTeamEvaluator(
    guardrails=intelligence_guardrails + content_guardrails,
    iterations=100
)
report = await evaluator.evaluate()

print(f"System security score: {(1 - report['success_rate']) * 100:.0f}/100")
```

## Examples

### Example 1: Financial Services

```python
from syrin import Agent, Model
from syrin.guardrails import GuardrailConfig

agent = Agent(
    # model=Model.OpenAI("gpt-4o"),
    model=Model.Almock(),  # No API Key needed
    guardrails=GuardrailConfig(
        actions={
            "transfer_funds": [
                AuthorityCheck(requires="finance:transfer"),
                BudgetEnforcer(max_amount=10000),
                ThresholdApproval(k=2, n=3),
            ]
        }
    )
)
```

### Example 2: Healthcare

```python
agent = Agent(
    # model=Model.OpenAI("gpt-4o"),
    model=Model.Almock(),  # No API Key needed
    guardrails=GuardrailConfig(
        input=[PHIValidator(allow_phi=False)],
        output=[EncryptionEnforcer()],
    )
)
```

### Example 3: Content Moderation

```python
from syrin.guardrails import ParallelEvaluationEngine

engine = ParallelEvaluationEngine(timeout=1.0)

guardrails = [
    ContentFilter(blocked_words=["spam"]),
    ToxicityChecker(),
    SpamDetector(),
]

result = await engine.evaluate(context, guardrails)
```

### Example 4: Complete Authority Workflow

```python
from syrin.guardrails import (
    AuthorityCheck, BudgetEnforcer, ThresholdApproval,
    HumanApproval, ParallelEvaluationEngine
)

# Combine all authority guardrails
guardrails = [
    AuthorityCheck(requires="admin:delete"),
    BudgetEnforcer(max_amount=1000),
    ThresholdApproval(
        k=2,
        n=3,
        condition=lambda ctx: ctx.action.critical
    ),
    HumanApproval(
        approver="cto@example.com",
        require_2fa=True
    )
]

# Run in parallel
engine = ParallelEvaluationEngine()
result = await engine.evaluate(context, guardrails)
```

### Example 5: Context-Aware Protection

```python
from syrin.guardrails.intelligence import ContextAwareGuardrail, EscalationDetector

# Detect topic escalation across conversation
context_guardrail = ContextAwareGuardrail(
    max_history=5,
    topic_escalation=["sensitive", "restricted", "classified"]
)

# Detect repeated violation attempts
escalation_detector = EscalationDetector(
    time_window=300,  # 5 minutes
    violation_threshold=3
)

# Combined protection
guardrails = [context_guardrail, escalation_detector]
```

### Example 6: Adaptive Threshold Tuning

```python
from syrin.guardrails.intelligence import AdaptiveThresholdGuardrail

# Start with initial threshold
adaptive = AdaptiveThresholdGuardrail(
    initial_threshold=0.7,
    min_threshold=0.5,
    max_threshold=0.95,
    learning_rate=0.05
)

# Evaluate content
result = await adaptive.evaluate(context)

# After user feedback, record accuracy
if user_reported_false_positive:
    adaptive.record_feedback(
        passed=result.passed,
        confidence=result.confidence,
        was_violation=False  # User said it was safe
    )

# Check adapted threshold
print(f"Current threshold: {adaptive.current_threshold:.2f}")
```

### Example 7: Red Team Security Testing

```python
from syrin.guardrails.intelligence import (
    AttackSimulator, RedTeamEvaluator, FuzzingEngine
)
from syrin.guardrails import ContentFilter

# Your production guardrail
guardrail = ContentFilter(blocked_words=["harmful", "dangerous"])

# Run red team evaluation
evaluator = RedTeamEvaluator(guardrail, iterations=100)
report = await evaluator.evaluate()

print(f"Security Score: {(1 - report['success_rate']) * 100:.0f}/100")
print(f"Vulnerable to: {', '.join(report['vulnerable_to'])}")

# Find edge cases with fuzzing
fuzzer = FuzzingEngine(seed_text="test content")
mutations = fuzzer.mutate(count=1000)

for mutation in mutations:
    context = GuardrailContext(text=mutation)
    result = await guardrail.evaluate(context)
    if result.passed and "harmful" in mutation.lower():
        print(f"Bypass found: {mutation}")
```

### Example 8: Complete Intelligent System

```python
from syrin import Agent, Model
from syrin.guardrails import ContentFilter, PIIScanner, GuardrailConfig
from syrin.guardrails.intelligence import (
    ContextAwareGuardrail,
    EscalationDetector,
    AdaptiveThresholdGuardrail
)

# Build comprehensive intelligent guardrail system
agent = Agent(
    # model=Model.OpenAI("gpt-4o"),
    model=Model.Almock(),  # No API Key needed
    guardrails=GuardrailConfig(
        input=[
            ContentFilter(blocked_words=["spam", "harmful"]),
            PIIScanner(),
            ContextAwareGuardrail(max_history=5),
            EscalationDetector(violation_threshold=3),
        ],
        output=[
            AdaptiveThresholdGuardrail(
                initial_threshold=0.75,
                min_threshold=0.5,
                max_threshold=0.95
            ),
        ]
    )
)
```

---

## Troubleshooting

### Guardrails Too Slow

```python
# Use parallel evaluation
engine = ParallelEvaluationEngine()

# Add timeout
engine = ParallelEvaluationEngine(timeout=0.5)

# Profile individual guardrails
for decision in result.decisions:
    print(f"{decision.rule}: {decision.latency_ms}ms")
```

### Too Many False Positives

```python
# Add confidence threshold
if result.confidence < 0.8:
    result.passed = True  # Warn instead of block

# Log for tuning
logger.info(f"False positive: {result.metadata}")
```

### Budget Exhausted

```python
# Disable expensive checks
if budget.remaining < 0.10:
    guardrails = [g for g in guardrails if g.budget_cost == 0]
```

---

## Migration from v1.0

### Old API

```python
from syrin.guardrails import ContentFilter

guardrail = ContentFilter(blocked_words=["bad"])
result = guardrail.check(text, stage)
```

### New API

```python
from syrin.guardrails import ContentFilter

guardrail = ContentFilter(blocked_words=["bad"])
context = GuardrailContext(text=text, stage=stage)
result = await guardrail.evaluate(context)
```

---

## Support

- **Website:** [syrin.ai](https://syrin.ai)
- **GitHub Issues:** [github.com/syrin-labs/syrin-python/issues](https://github.com/syrin-labs/syrin-python/issues)
- **Discord:** [discord.gg/p4jnKxYKpB](https://discord.gg/p4jnKxYKpB)
- **Twitter (X):** [x.com/syrin_dev](https://x.com/syrin_dev)

---

## License

MIT License - See LICENSE file for details.

---

**Built with ❤️ by the Syrin Team**
