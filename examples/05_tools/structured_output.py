"""Structured Output Example -- Getting typed, validated responses from agents.

Demonstrates:
- @structured decorator for output schemas
- Output(output_type, validation_retries) configuration
- Pydantic models as output types
- Custom OutputValidator with ValidationResult
- Validation hooks (OUTPUT_VALIDATION_START, etc.)
- Pydantic field validators for restricted values

Run: python examples/05_tools/structured_output.py
"""

from __future__ import annotations

from pydantic import BaseModel, field_validator

from syrin import Agent, Model, Output
from syrin.enums import Hook
from syrin.model import structured
from syrin.types.validation import (
    OutputValidator,
    ValidationAction,
    ValidationContext,
    ValidationResult,
)


# --- 1. Basic @structured class ---

print("=" * 50)
print("1. Basic @structured class")
print("=" * 50)

@structured
class UserInfo:
    name: str
    email: str
    age: int
    city: str


agent = Agent(model=Model.Almock(), output=Output(UserInfo, validation_retries=3))
result = agent.response("Extract: John Doe, 35, john@example.com, San Francisco")
print(f"  is_valid: {result.structured.is_valid}")
if result.structured.parsed:
    print(f"  parsed.name: {result.structured.parsed.name}")


# --- 2. Pydantic model as output ---

print("\n" + "=" * 50)
print("2. Pydantic model as output")
print("=" * 50)

class ProductInfo(BaseModel):
    name: str
    price: float
    in_stock: bool
    category: str


agent = Agent(model=Model.Almock(), output=Output(ProductInfo, validation_retries=3))
result = agent.response("Product: Widget, $29.99, in stock, electronics")
print(f"  is_valid: {result.structured.is_valid}")
if result.structured.parsed:
    print(f"  parsed: {result.structured.parsed}")


# --- 3. Validation hooks ---

print("\n" + "=" * 50)
print("3. Validation hooks")
print("=" * 50)

@structured
class SentimentResult:
    sentiment: str
    confidence: float
    explanation: str


agent = Agent(model=Model.Almock(), output=Output(SentimentResult, validation_retries=3))


def on_start(ctx: object) -> None:
    print(f"  VALIDATION START: {ctx.output_type}")


def on_success(ctx: object) -> None:
    print(f"  VALIDATION SUCCESS at attempt {ctx.attempt}")


def on_failed(ctx: object) -> None:
    print(f"  VALIDATION FAILED: {ctx.reason}")


agent.events.on(Hook.OUTPUT_VALIDATION_START, on_start)
agent.events.on(Hook.OUTPUT_VALIDATION_SUCCESS, on_success)
agent.events.on(Hook.OUTPUT_VALIDATION_FAILED, on_failed)
result = agent.response("Analyze: 'This product is amazing!'")
print(f"  is_valid: {result.structured.is_valid}")


# --- 4. Custom validator ---

print("\n" + "=" * 50)
print("4. Custom validator")
print("=" * 50)

class ReviewResult(BaseModel):
    rating: int
    sentiment: str
    summary: str


class RatingValidator(OutputValidator):
    max_retries = 3

    def validate(self, output: object, context: ValidationContext) -> ValidationResult:
        data = (
            output
            if isinstance(output, dict)
            else output.model_dump()
            if hasattr(output, "model_dump")
            else {}
        )
        rating = data.get("rating", 0)
        if rating < 1 or rating > 5:
            return ValidationResult.invalid(
                message=f"Rating {rating} out of range 1-5",
                action=ValidationAction.RETRY,
                hint="Rating must be between 1 and 5",
            )
        sentiment = data.get("sentiment", "").lower()
        if sentiment not in ["positive", "negative", "neutral"]:
            return ValidationResult.invalid(
                message=f"Invalid sentiment: {sentiment}",
                action=ValidationAction.RETRY,
            )
        return ValidationResult.valid(output)

    def on_retry(self, error: str, attempt: int) -> str:
        return f"Error: {error}. Please fix and retry."


agent = Agent(
    model=Model.Almock(),
    output=Output(ReviewResult, validator=RatingValidator(), validation_retries=3),
)
result = agent.response("Review: 'Terrible product.' rating 1, negative")
print(f"  is_valid: {result.structured.is_valid}")


# --- 5. Pydantic field validator for restricted values ---

print("\n" + "=" * 50)
print("5. Pydantic field validator")
print("=" * 50)

class RestrictedUser(BaseModel):
    name: str
    email: str
    role: str

    @field_validator("role")
    @classmethod
    def role_must_be_valid(cls, v: str) -> str:
        allowed = ["admin", "user", "guest"]
        if v.lower() not in allowed:
            raise ValueError(f"Role must be one of: {allowed}")
        return v.lower()


agent = Agent(
    model=Model.Almock(),
    output=Output(
        RestrictedUser,
        validation_retries=3,
        context={"allowed_domains": ["company.com"]},
    ),
)
result = agent.response("Create user: John, john@company.com, admin")
print(f"  is_valid: {result.structured.is_valid}")


# --- Optional: serve as a class-based agent ---

class StructuredOutputAgent(Agent):
    _agent_name = "structured-output"
    _agent_description = "Agent with structured output (UserInfo extraction)"
    model = Model.Almock()
    system_prompt = "You extract user information from text. Return valid UserInfo."
    output = Output(UserInfo, validation_retries=3)


if __name__ == "__main__":
    pass
    # To serve with playground UI:
    # agent = StructuredOutputAgent()
    # agent.serve(port=8000, enable_playground=True, debug=True)
