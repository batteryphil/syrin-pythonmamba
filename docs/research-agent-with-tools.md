# Use Case 2: Research Agent with Tools

## What You'll Learn

How to give your agent special **Tools** so it can:
- Search the web
- Do calculations
- Look up information
- Send emails
- Or anything else!

A tool is just a Python function that your agent can call when it needs to do something.

## The Idea

Instead of the agent just talking, it can actually **DO** things:

```
User: "What's the current Bitcoin price and convert to EUR?"
  ↓
Agent: "I need to search for Bitcoin price and convert currency"
  ↓
Agent calls: search_bitcoin_price() → gets "$48,500"
  ↓
Agent calls: convert_currency("USD", "EUR", 48500) → gets "€45,000"
  ↓
Agent answers: "Bitcoin is $48,500 USD, which is €45,000 EUR"
```

## Complete Example: Copy & Paste This!

```python
"""
Research Agent with Tools
Copy this code and run it!
"""

from syrin import Agent
from syrin.model import Model
from syrin.tool import tool


# Define your tools (special functions the agent can call)
@tool
def get_stock_price(symbol: str) -> dict:
    """Get the current stock price for a symbol (like AAPL, GOOGL)."""
    # In real life, you'd call an API here
    # For now, we'll return fake data
    prices = {
        "AAPL": 195.50,
        "GOOGL": 140.25,
        "MSFT": 425.75,
    }
    price = prices.get(symbol.upper(), "Unknown")
    return {
        "symbol": symbol.upper(),
        "price": price,
        "currency": "USD"
    }


@tool
def calculate(expression: str) -> dict:
    """Calculate a math expression (like '2 + 2' or '5 * 10')."""
    try:
        result = eval(expression)
        return {
            "expression": expression,
            "result": result,
            "success": True
        }
    except Exception as e:
        return {
            "expression": expression,
            "error": str(e),
            "success": False
        }


@tool
def get_weather(city: str) -> dict:
    """Get weather information for a city."""
    # Fake weather data
    weather_data = {
        "New York": {"temp": 72, "condition": "Sunny"},
        "London": {"temp": 65, "condition": "Rainy"},
        "Tokyo": {"temp": 75, "condition": "Cloudy"},
    }
    data = weather_data.get(city, None)
    if data:
        return {
            "city": city,
            "temperature_f": data["temp"],
            "condition": data["condition"],
            "success": True
        }
    else:
        return {
            "city": city,
            "error": "City not found",
            "success": False
        }


# Create your agent with tools
class ResearchAgent(Agent):
    """An agent that can search, calculate, and look up information."""
    
    model = Model.OpenAI("gpt-4o-mini")
    
    system_prompt = """
    You are a helpful research assistant.
    You have access to tools to search information and do calculations.
    Always use the available tools to answer questions accurately.
    """
    
    # Register your tools
    tools = [get_stock_price, calculate, get_weather]


def main():
    agent = ResearchAgent()
    
    # Ask the agent to use its tools
    questions = [
        "What's the price of Apple stock?",
        "Calculate 25 * 4",
        "What's the weather in Tokyo?",
        "What's Google stock price times 2?",
    ]
    
    for question in questions:
        print(f"\n❓ Question: {question}")
        response = agent.response(question)
        print(f"✓ Answer: {response.content}")
        
        # See which tools were called
        if hasattr(response, 'tool_calls') and response.tool_calls:
            print(f"🔧 Tools used: {len(response.tool_calls)}")
            for call in response.tool_calls:
                print(f"  - {call}")
        
        print(f"💰 Cost: ${response.cost_usd:.4f}")


if __name__ == "__main__":
    main()
```

## Understanding Tools

### What is a Tool?

A tool is a Python function with a `@tool` decorator:

```python
from syrin.tool import tool

@tool
def my_tool(parameter: str) -> dict:
    """Tool description - explain what it does."""
    # Do something
    return {"result": "something"}
```

### Tool Parts Explained

```python
@tool                                    # This makes it a tool
def search_google(                       # Function name
    query: str,                          # Parameter with type hint
    max_results: int = 10                # Optional parameter with default
) -> dict:                               # Returns a dictionary
    """Search Google for information.
    
    This text is shown to the AI so it knows when to use this tool.
    """
    # Your actual code here
    results = ["result1", "result2"]
    return {
        "query": query,
        "results": results,
        "count": len(results)
    }
```

## Real-World Examples

### Example 1: Search Tool

```python
from syrin.tool import tool
import requests

@tool
def search_web(query: str, num_results: int = 5) -> dict:
    """Search the web for information about a topic."""
    # In real life, you'd use a search API like Google Custom Search
    # For this example, we'll return mock data
    return {
        "query": query,
        "results": [
            {"title": f"Result {i}", "url": f"https://example.com/{i}"}
            for i in range(num_results)
        ]
    }


class SearchAgent(Agent):
    model = Model.OpenAI("gpt-4o-mini")
    system_prompt = "You are a helpful research assistant."
    tools = [search_web]


agent = SearchAgent()
response = agent.response("What are the latest developments in AI?")
print(response.content)
```

### Example 2: Data Lookup Tool

```python
from syrin.tool import tool

@tool
def lookup_customer(customer_id: str) -> dict:
    """Look up customer information by ID."""
    # Pretend database
    customers = {
        "C001": {"name": "Alice", "email": "alice@example.com", "account": "Premium"},
        "C002": {"name": "Bob", "email": "bob@example.com", "account": "Free"},
        "C003": {"name": "Carol", "email": "carol@example.com", "account": "Premium"},
    }
    
    if customer_id in customers:
        return {"success": True, "customer": customers[customer_id]}
    else:
        return {"success": False, "error": f"Customer {customer_id} not found"}


@tool
def update_customer_note(customer_id: str, note: str) -> dict:
    """Add a note to a customer's account."""
    # In real life, this would update a database
    return {
        "success": True,
        "customer_id": customer_id,
        "note_added": note
    }


class CustomerServiceAgent(Agent):
    model = Model.OpenAI("gpt-4o-mini")
    system_prompt = """
    You are a customer service representative.
    Help customers with their accounts using the available tools.
    """
    tools = [lookup_customer, update_customer_note]


agent = CustomerServiceAgent()
response = agent.response("Can you look up customer C001 and add a note that they asked about bulk discounts?")
print(response.content)
```

### Example 3: API Integration Tool

```python
from syrin.tool import tool

@tool
def convert_currency(from_currency: str, to_currency: str, amount: float) -> dict:
    """Convert amount from one currency to another."""
    # Mock exchange rates
    rates = {
        ("USD", "EUR"): 0.92,
        ("EUR", "USD"): 1.09,
        ("USD", "GBP"): 0.79,
        ("GBP", "USD"): 1.27,
    }
    
    rate = rates.get((from_currency.upper(), to_currency.upper()))
    
    if rate:
        converted = amount * rate
        return {
            "original": f"{amount} {from_currency.upper()}",
            "converted": f"{converted:.2f} {to_currency.upper()}",
            "rate": rate
        }
    else:
        return {
            "error": f"Cannot convert {from_currency} to {to_currency}"
        }


@tool
def check_bitcoin_price() -> dict:
    """Get current Bitcoin price."""
    # Mock data
    return {
        "currency": "Bitcoin",
        "price_usd": 48500,
        "change_24h": "+5.2%"
    }


class CryptoAgent(Agent):
    model = Model.OpenAI("gpt-4o-mini")
    system_prompt = "You are a crypto price assistant."
    tools = [convert_currency, check_bitcoin_price]


agent = CryptoAgent()
response = agent.response("What's the Bitcoin price in Euros?")
print(response.content)
```

## Tool Parameters: Type Support

Your tools can accept different types:

```python
from syrin.tool import tool
from typing import Optional, Union

@tool
def complex_tool(
    name: str,                           # Text
    age: int,                            # Number (whole)
    salary: float,                       # Number (decimal)
    is_manager: bool,                    # True/False
    tags: list,                          # List of items
    metadata: dict,                      # Dictionary/object
    status: Optional[str] = None,        # Optional (might be None)
) -> dict:
    """A tool with many parameter types."""
    return {
        "received": {
            "name": name,
            "age": age,
            "salary": salary,
            "is_manager": is_manager,
            "tags": tags,
            "metadata": metadata,
            "status": status
        }
    }
```

## Handling Tool Errors

Tools might fail sometimes. Here's how to handle it:

```python
from syrin.tool import tool

@tool
def divide(a: float, b: float) -> dict:
    """Divide a by b."""
    if b == 0:
        return {
            "error": "Cannot divide by zero",
            "success": False
        }
    
    return {
        "result": a / b,
        "success": True
    }


@tool
def risky_operation(data: str) -> dict:
    """Do something risky."""
    try:
        # Try to do something
        processed = data.upper()
        return {
            "result": processed,
            "success": True
        }
    except Exception as e:
        return {
            "error": str(e),
            "success": False
        }
```

## Checking What Tools Were Called

After running your agent, you can see which tools it used:

```python
agent = ResearchAgent()
response = agent.response("Calculate 100 * 5 and get Apple stock price")

# Check if tools were called
if hasattr(response, 'tool_calls'):
    print(f"Tools called: {response.tool_calls}")
    for tool_call in response.tool_calls:
        print(f"  - Tool: {tool_call}")
```

## Tool Best Practices

### 1. Clear Descriptions

```python
# ❌ Bad description
@tool
def get_data(x: str) -> dict:
    """Get some data."""
    pass

# ✓ Good description
@tool
def get_customer_data(customer_id: str) -> dict:
    """
    Retrieve customer information by customer ID.
    Returns name, email, account type, and status.
    """
    pass
```

### 2. Clear Parameter Names

```python
# ❌ Unclear
@tool
def process(x: str, y: int) -> dict:
    pass

# ✓ Clear
@tool
def search_products(query: str, max_results: int = 10) -> dict:
    pass
```

### 3. Always Return Dictionaries

```python
@tool
def my_tool(param: str) -> dict:
    return {
        "status": "success",
        "data": "result"
    }
```

### 4. Include Error Information

```python
@tool
def my_tool(param: str) -> dict:
    try:
        result = do_something(param)
        return {
            "success": True,
            "result": result
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }
```

## Multiple Agents with Different Tools

```python
from syrin import Agent
from syrin.model import Model
from syrin.tool import tool


@tool
def search_books(query: str) -> dict:
    """Search for books."""
    return {"results": ["Book 1", "Book 2"]}


@tool
def search_movies(query: str) -> dict:
    """Search for movies."""
    return {"results": ["Movie 1", "Movie 2"]}


@tool
def search_recipes(query: str) -> dict:
    """Search for recipes."""
    return {"results": ["Recipe 1", "Recipe 2"]}


class BookAgent(Agent):
    model = Model.OpenAI("gpt-4o-mini")
    system_prompt = "You are a book expert."
    tools = [search_books]


class MovieAgent(Agent):
    model = Model.OpenAI("gpt-4o-mini")
    system_prompt = "You are a movie expert."
    tools = [search_movies]


class ChefAgent(Agent):
    model = Model.OpenAI("gpt-4o-mini")
    system_prompt = "You are a helpful chef."
    tools = [search_recipes]


# Use them
book_agent = BookAgent()
movie_agent = MovieAgent()
chef_agent = ChefAgent()

print(book_agent.response("Find me a good science fiction book"))
print(movie_agent.response("Recommend an action movie"))
print(chef_agent.response("What's a good Italian recipe?"))
```

## Grouping Tools in MCP and Using MCP in Agents

You can **group related tools in an MCP** and **use the MCP inside your agent's tools**. Define an MCP with `@tool` (same as Agent), then add the MCP to `tools=[]`:

```python
from syrin import MCP, Agent, tool

class ProductMCP(MCP):
    """MCP that groups product catalog tools."""

    @tool
    def search_products(self, query: str, limit: int = 10) -> str:
        """Search the product catalog."""
        return f"Results for: {query}"

    @tool
    def get_product(self, product_id: str) -> str:
        """Get product by ID."""
        return f"Product {product_id} details"

class ProductAgent(Agent):
    model = Model.OpenAI("gpt-4o-mini")
    tools = [ProductMCP()]  # MCP tools become agent tools
```

When serving, if MCP is in `tools`, `/mcp` is auto-mounted alongside `/chat`. See [MCP](mcp.md).

## Next Steps

- **Add Memory** → Learn [Use Case 3: Agent with Memory](agent-with-memory.md)
- **Control Costs** → Learn [Use Case 4: Budget Control](budget-control.md)
- **Build Teams** → Learn [Use Case 5: Multi-Agent Orchestration](multi-agent.md)

---

Questions? See the [FAQ in Getting Started](getting-started.md#common-questions)
