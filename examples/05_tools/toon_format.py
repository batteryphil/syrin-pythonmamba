"""TOON Format Example -- Token-efficient tool schemas.

Demonstrates:
- TOON (Token-Oriented Object Notation) tool schema format
- Why TOON uses ~40% fewer tokens than JSON for tool definitions
- ToolSpec.schema_to_toon() and ToolSpec.to_format()
- Side-by-side comparison of TOON vs JSON efficiency

Run: python examples/05_tools/toon_format.py
"""

from __future__ import annotations

import json

from syrin import Agent, Model, tool
from syrin.enums import DocFormat

# --- Define some tools ---


@tool
def calculate(a: float, b: float, operation: str = "add") -> str:
    """Perform basic arithmetic operations.

    Args:
        a: First number
        b: Second number
        operation: One of add, subtract, multiply, divide
    """
    ops = {"add": a + b, "subtract": a - b, "multiply": a * b, "divide": a / b if b else 0}
    return str(ops.get(operation, "Unknown"))


@tool
def search_web(query: str, max_results: int = 5) -> str:
    """Search the web for information.

    Args:
        query: The search query to execute
        max_results: Maximum number of results (1-10)
    """
    return f"Found {max_results} results for: {query}"


@tool
def send_email(to: str, subject: str, body: str, priority: str = "normal") -> str:
    """Send an email to a recipient.

    Args:
        to: Email address of recipient
        subject: Email subject line
        body: Email body content
        priority: Priority level (low, normal, high)
    """
    return f"Email sent to {to}"


# --- 1. TOON vs JSON comparison for a single tool ---

print("=" * 50)
print("1. TOON vs JSON -- single tool")
print("=" * 50)

json_schema = json.dumps(calculate.parameters_schema, indent=2)
toon_schema = calculate.schema_to_toon()
savings = ((len(json_schema) - len(toon_schema)) / len(json_schema)) * 100

print(f"  Tool: {calculate.name}")
print(f"  JSON: {len(json_schema)} chars")
print(f"  TOON: {len(toon_schema)} chars")
print(f"  Savings: {savings:.1f}%")


# --- 2. Format conversion ---

print("\n" + "=" * 50)
print("2. Format conversion (TOON vs JSON)")
print("=" * 50)

for fmt in [DocFormat.TOON, DocFormat.JSON]:
    schema = search_web.to_format(fmt)
    print(f"  {fmt.value}: {json.dumps(schema)[:80]}...")


# --- 3. Multi-tool efficiency ---

print("\n" + "=" * 50)
print("3. Multi-tool efficiency comparison")
print("=" * 50)

tools = [calculate, search_web, send_email]
total_json = sum(len(json.dumps(t.parameters_schema)) for t in tools)
total_toon = sum(len(t.schema_to_toon()) for t in tools)
total_savings = ((total_json - total_toon) / total_json) * 100

print("  3 tools combined:")
print(f"  JSON: {total_json} chars")
print(f"  TOON: {total_toon} chars")
print(f"  Savings: {total_savings:.1f}%")


# --- 4. Use the tools with an agent ---

print("\n" + "=" * 50)
print("4. Agent with TOON-format tools")
print("=" * 50)

agent = Agent(
    model=Model.Almock(),
    system_prompt="You are a helpful assistant. Use tools when needed.",
    tools=[calculate, search_web, send_email],
)
result = agent.response("What is 2 + 3?")
print(f"  Response: {result.content[:120]}")

if __name__ == "__main__":
    pass
    # Optional: serve with playground UI
    # agent.serve(port=8000, enable_playground=True, debug=True)
