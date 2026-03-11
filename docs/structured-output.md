# Structured Output Validation

Syrin provides comprehensive structured output validation that ensures LLM responses conform to your expected schema, with automatic retries, error visibility, and custom validation logic.

> **Agent integration:** For the agent `output=` param and `response.structured`, see [Agent: Structured Output](agent/structured-output.md).

## Installation

No additional dependencies required. Structured output validation is built into Syrin.

## Quick Start

```python
from syrin import Agent, Model, Output
from syrin.model import structured

# Define your expected output schema using @structured decorator
@structured
class UserInfo:
    name: str
    email: str
    age: int

# Clean API with Output configuration object
agent = Agent(
    model=Model.OpenAI("gpt-4o"),
    output=Output(
        type=UserInfo,
        validation_retries=3,
        context={"key": "value"},
    ),
)

# Response is automatically validated
result = agent.response("Extract user info from: John Doe, john@example.com, age 30")
print(result.structured.parsed.name)  # "John Doe"
print(result.structured.is_valid)      # True
```

### Shorthand

For simple cases, you can use shorthand:

```python
# Just the type (uses defaults)
agent = Agent(model=Model.OpenAI("gpt-4o"), output=Output(UserInfo))

# With just validation_retries
agent = Agent(model=Model.OpenAI("gpt-4o"), output=Output(UserInfo, validation_retries=5))
```

### Advanced: Using Pydantic

Use Pydantic `BaseModel` with `Output` when you need advanced validation features like field validators with context:

```python
from pydantic import BaseModel, field_validator
from syrin import Agent, Model, Output

class RestrictedUser(BaseModel):
    name: str
    email: str
    
    @field_validator('email')
    @classmethod
    def email_must_be_allowed(cls, v: str, info) -> str:
        context = getattr(info, 'context', None)
        allowed = context.get("allowed_domains", []) if context else []
        if allowed and not any(v.endswith(f"@{d}") for d in allowed):
            raise ValueError(f"Email must be from allowed domains")
        return v

agent = Agent(
    model=Model.OpenAI("gpt-4o"),
    output=Output(
        type=RestrictedUser,
        validation_retries=3,
        context={"allowed_domains": ["company.com"]},
    ),
)
```

## Basic Usage

### Using @structured (Recommended)

Define your output schema using Syrin's `@structured` decorator:

```python
from syrin import Agent, Model, Output
from syrin.model import structured

@structured
class UserInfo:
    name: str
    email: str
    age: int

# Everything in one place - clean and extensible!
agent = Agent(
    model=Model.OpenAI("gpt-4o"),
    output=Output(
        type=UserInfo,                    # Output schema
        validation_retries=3,             # Retry on failure
        context={"key": "val"},          # Dynamic validation data
    ),
)

result = agent.response("Extract user info")
print(result.structured.parsed.name)
```

### Using Model's output (Also Works)

Set output on model, Agent inherits it:

```python
# Set output on model
model = Model.OpenAI("gpt-4o", output=UserInfo)

# Agent uses model's output
agent = Agent(model=model)
```

### Shorthand Syntax

```python
# Just the type - uses all defaults
agent = Agent(model=Model.OpenAI("gpt-4o"), output=Output(UserInfo))

# With custom validation_retries
agent = Agent(model=Model.OpenAI("gpt-4o"), output=Output(UserInfo, validation_retries=5))
```

### Advanced: Pydantic with Context

When you need advanced validation with dynamic context, use Pydantic:

```python
from pydantic import BaseModel, field_validator
from syrin import Agent, Model, Output

class RestrictedUser(BaseModel):
    name: str
    email: str
    
    @field_validator('email')
    @classmethod
    def email_must_be_allowed(cls, v: str, info) -> str:
        context = getattr(info, 'context', None)
        allowed = context.get("allowed_domains", []) if context else []
        if allowed and not any(v.endswith(f"@{d}") for d in allowed):
            raise ValueError(f"Email must be from allowed domains")
        return v

agent = Agent(
    model=Model.OpenAI("gpt-4o"),
    output=Output(
        type=RestrictedUser,
        validation_retries=3,
        context={"allowed_domains": ["company.com", "partner.com"]},
    ),
)
```

### Using Model's output (Also Works)

Set output on model, Agent inherits it:

```python
# Set output on model
model = Model.OpenAI("gpt-4o", output=UserInfo)

# Agent uses model's output
agent = Agent(model=model)
```

### Shorthand Syntax

```python
# Just the type - uses all defaults
agent = Agent(model=Model.OpenAI("gpt-4o"), output=Output(UserInfo))

# With custom retries
agent = Agent(model=Model.OpenAI("gpt-4o"), output=Output(UserInfo, retries=5))
```

## ValidationPipeline API

### Constructor

```python
ValidationPipeline(
    output_type: Type,              # Pydantic model to validate against
    max_retries: int = 3,          # Maximum validation attempts
    backoff_factor: float = 1.0,   # Exponential backoff between retries
    validator: OutputValidator | None = None,  # Custom validator
    context: dict | None = None,   # Validation context
)
```

### Methods

#### validate(raw_output: str) -> tuple[parsed, attempts, error]

Validates raw LLM output against the schema:

```python
pipeline = ValidationPipeline(UserInfo, max_retries=3)
parsed, attempts, error = pipeline.validate('{"name": "John", "age": 30}')

# parsed: Validated Pydantic model
# attempts: List[ValidationAttempt] - all validation attempts
# error: Exception if validation failed
```

## Validation Types

### ValidationAttempt

Tracks a single validation attempt:

```python
@dataclass
class ValidationAttempt:
    attempt: int           # Attempt number (1, 2, 3...)
    raw_output: str       # Raw LLM output for this attempt
    parsed: Any           # Parsed object (if successful)
    error: str | None     # Error message (if failed)
    timestamp: float       # When the attempt was made
```

### ValidationResult

Result from custom validators:

```python
# Valid result
ValidationResult.valid(parsed_object)

# Invalid with retry
ValidationResult.invalid(
    message="Error description",
    action=ValidationAction.RETRY,   # Will retry with hint
    hint="How to fix"
)

# Invalid - stop
ValidationResult.invalid(
    message="Critical error",
    action=ValidationAction.STOP,    # Stop immediately
)
```

### ValidationAction

What to do when validation fails:

| Action | Description |
|--------|-------------|
| `RETRY` | Retry with the hint message |
| `STOP` | Stop validation immediately |
| `FALLBACK` | Use fallback value |

### ValidationContext

Context passed to custom validators:

```python
@dataclass
class ValidationContext:
    raw_output: str                    # Raw LLM output
    attempt: int                       # Current attempt number
    max_attempts: int                  # Maximum attempts allowed
    user_context: dict[str, Any]      # Your custom context
    llm_messages: list[dict]           # Message history for debugging
```

## StructuredOutput (Response)

The `Response.structured` property provides full access to validation results:

```python
result = agent.response("Extract user info")

# Check validity
result.structured.is_valid        # True/False

# Access parsed data
result.structured.parsed          # Pydantic model
result.structured.raw             # Raw JSON string
result.structured._data           # Dict of fields

# Debug validation
result.structured.validation_attempts  # List of all attempts
result.structured.final_error          # Final error if failed
result.structured.last_error           # Last error message
result.structured.all_errors          # All error messages
```

## Error Visibility

One of the key features is complete visibility into what happened during validation:

```python
result = agent.response("Extract user info")

# See every attempt
for attempt in result.structured.validation_attempts:
    print(f"Attempt {attempt.attempt}")
    print(f"  Raw: {attempt.raw_output[:100]}...")
    print(f"  Error: {attempt.error}")

# Check specific errors
if not result.structured.is_valid:
    print(f"Final error: {result.structured.final_error}")
    print(f"Last error: {result.structured.last_error}")
    print(f"All errors: {result.structured.all_errors}")
```

## Custom Validators

Create custom validation logic with the `OutputValidator` protocol:

```python
from syrin.types.validation import OutputValidator, ValidationResult, ValidationContext
from syrin.enums import ValidationAction

class BusinessValidator(OutputValidator):
    max_retries = 3
    backoff_factor = 1.0
    
    def validate(self, output: Any, context: ValidationContext) -> ValidationResult:
        # Custom business logic
        if hasattr(output, 'status') and output.status not in ['active', 'pending']:
            return ValidationResult.invalid(
                message=f"Invalid status: {output.status}",
                action=ValidationAction.RETRY,
                hint="Status must be 'active' or 'pending'"
            )
        return ValidationResult.valid(output)
    
    def on_retry(self, error: Exception, attempt: int) -> str:
        return f"Error: {error}. Please fix and retry."

# Use custom validator
pipeline = ValidationPipeline(
    UserInfo,
    validator=BusinessValidator(),
)
```

## Markdown Handling

The pipeline automatically handles common LLM output formats:

```python
# Plain JSON
'{"name": "John", "age": 30}'

# Markdown code block
'```json\n{"name": "John", "age": 30}\n```'

# Text before/after
'Here is the JSON:\n{"name": "John", "age": 30}\n\nHope this helps!'

# JavaScript code block
'```javascript\n{"name": "John", "age": 30}\n```'
```

## Validation Context

Pass dynamic data to validators using context:

```python
class RestrictedUser(BaseModel):
    name: str
    email: str
    
    @field_validator('email')
    @classmethod
    def email_must_be_allowed(cls, v: str, info) -> str:
        # Access context via info.context
        context = getattr(info, 'context', None)
        allowed = context.get("allowed_domains", []) if context else []
        if allowed and not any(v.endswith(f"@{d}") for d in allowed):
            raise ValueError(f"Email must be from allowed domains")
        return v

# Pass context
pipeline = ValidationPipeline(
    RestrictedUser,
    context={"allowed_domains": ["company.com", "partner.com"]},
)
```

## Examples

### Simple Validation with @structured

```python
from syrin.model import structured
from syrin.validation import ValidationPipeline

@structured
class User:
    name: str
    age: int

pipeline = ValidationPipeline(User._structured_pydantic, max_retries=3)
parsed, attempts, error = pipeline.validate('{"name": "John", "age": 30}')
```

### With Agent

```python
from syrin import Agent, Model, Output
from syrin.model import structured

@structured
class UserInfo:
    name: str
    email: str
    age: int

agent = Agent(
    model=Model.OpenAI("gpt-4o"),
    output=Output(UserInfo, validation_retries=3),
)

result = agent.response("John Doe, john@example.com, age 30")
print(result.structured.parsed.name)  # "John Doe"
```

### Debugging Failures

```python
result = agent.response("Some input")

if not result.structured.is_valid:
    print("Validation failed!")
    print(f"Attempts: {len(result.structured.validation_attempts)}")
    
    for attempt in result.structured.validation_attempts:
        print(f"Attempt {attempt.attempt}: {attempt.error}")
```

### With Observability (Hooks)

Monitor validation lifecycle with hooks:

```python
from syrin import Agent, Model, Output
from syrin.enums import Hook

agent = Agent(
    model=Model.OpenAI("gpt-4o"),
    output=Output(UserInfo, retries=3),
)

# Register hooks
agent.events.on(Hook.OUTPUT_VALIDATION_START, lambda ctx: print(f"Starting: {ctx.output_type}"))
agent.events.on(Hook.OUTPUT_VALIDATION_ATTEMPT, lambda ctx: print(f"Attempt {ctx.attempt}"))
agent.events.on(Hook.OUTPUT_VALIDATION_RETRY, lambda ctx: print(f"Retry: {ctx.error}"))
agent.events.on(Hook.OUTPUT_VALIDATION_SUCCESS, lambda ctx: print(f"Success!"))
agent.events.on(Hook.OUTPUT_VALIDATION_FAILED, lambda ctx: print(f"Failed: {ctx.error}"))
```

### Hooks Available

| Hook | Description | Context Fields |
|------|-------------|----------------|
| `OUTPUT_VALIDATION_START` | Validation started | `output_type`, `max_retries`, `raw_output` |
| `OUTPUT_VALIDATION_ATTEMPT` | Each attempt | `attempt`, `output_type` |
| `OUTPUT_VALIDATION_RETRY` | Retry triggered | `attempt`, `error`, `reason` |
| `OUTPUT_VALIDATION_SUCCESS` | Validation succeeded | `attempt`, `output_type`, `parsed_fields` |
| `OUTPUT_VALIDATION_FAILED` | Validation failed | `attempt`, `error`, `reason`, `total_attempts` |
```

## API Reference

### Classes

| Class | Description |
|-------|-------------|
| `ValidationPipeline` | Main validation orchestrator |
| `ValidationAttempt` | Single validation attempt |
| `ValidationResult` | Result from custom validator |
| `ValidationContext` | Context passed to validators |
| `StructuredOutput` | Enhanced response wrapper |
| `OutputValidator` | Protocol for custom validators |
| `ToolOutput` | Multiple output types |
| `ValidationAction` | RETRY, STOP, FALLBACK |

### Functions

| Function | Description |
|----------|-------------|
| `validate_output()` | Convenience function for simple validation |

### Exceptions

| Exception | Description |
|-----------|-------------|
| `ValidationError` | Raised when validation fails after all retries |
