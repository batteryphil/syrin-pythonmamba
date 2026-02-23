# Use Case 4: Budget Control & Cost Management

## What You'll Learn

How to **keep your AI costs under control**. Perfect for:
- Production applications
- Limiting spending
- Testing without surprises
- Enterprise deployments

AI API calls cost money. Syrin helps you manage and monitor every dollar.

## The Idea

```
You: "Run this agent but don't spend more than $1"
  ↓
Agent: "I'll do everything within budget"
  ↓
Agent runs... costs $0.50
  ↓
Agent: ✓ "Stayed within budget!"

OR

Agent runs... costs would be $2
  ↓
Agent: ❌ "STOP! Would exceed budget!"
```

## Complete Example: Copy & Paste This!

```python
"""
Budget Control Example
Keep your AI spending under control!
"""

from Syrin import Agent, Budget, OnExceeded
from Syrin.model import Model
from Syrin.threshold import BudgetThreshold


class BudgetAwareAgent(Agent):
    """An agent with a spending budget."""
    
    model = Model.OpenAI("gpt-4o-mini")
    system_prompt = "You are a helpful assistant."
    
    def __init__(self):
        super().__init__()
        # Set a budget of $1.00 per run
        self.budget = Budget(
            run=1.00,  # Max $1 per single request
            on_exceeded=OnExceeded.WARN,  # Warn if we hit 80% or 95%
            thresholds=[
                BudgetThreshold(at=80, action=lambda ctx: print(f"80% budget used: ${ctx.current_value:.2f}")),
                BudgetThreshold(at=95, action=lambda ctx: print(f"95% budget used: ${ctx.current_value:.2f}")),
            ]
        )


def main():
    agent = BudgetAwareAgent()
    
    questions = [
        "What is Python?",
        "Explain machine learning",
        "Tell me a joke",
    ]
    
    total_spent = 0.0
    
    for question in questions:
        print(f"\n❓ {question}")
        
        response = agent.response(question)
        cost = response.cost_usd
        total_spent += cost
        
        print(f"✓ Answer: {response.content[:100]}...")
        print(f"💰 Cost: ${cost:.4f}")
        print(f"📊 Total spent: ${total_spent:.4f}")
        print(f"📈 Budget remaining: ${1.00 - total_spent:.4f}")


if __name__ == "__main__":
    main()
```

## Budget Types

You can set different budget limits:

### 1. Per-Run Budget

Max spending for a single request:

```python
from Syrin import Agent, Budget, OnExceeded
from Syrin.model import Model

class Agent1(Agent):
    model = Model.OpenAI("gpt-4o-mini")
    
    def __init__(self):
        super().__init__()
        self.budget = Budget(
            run=0.10  # Max $0.10 per single request
        )
```

### 2. Hourly Budget

Max spending per hour:

```python
class Agent2(Agent):
    model = Model.OpenAI("gpt-4o-mini")
    
    def __init__(self):
        super().__init__()
        self.budget = Budget(
            hourly=5.00  # Max $5 per hour
        )
```

### 3. Daily Budget

Max spending per day:

```python
class Agent3(Agent):
    model = Model.OpenAI("gpt-4o-mini")
    
    def __init__(self):
        super().__init__()
        self.budget = Budget(
            daily=50.00  # Max $50 per day
        )
```

### 4. Monthly Budget

Max spending per month:

```python
class Agent4(Agent):
    model = Model.OpenAI("gpt-4o-mini")
    
    def __init__(self):
        super().__init__()
        self.budget = Budget(
            monthly=500.00  # Max $500 per month
        )
```

### 5. All Combined

You can combine multiple budgets:

```python
from Syrin import Agent, Budget, OnExceeded
from Syrin.model import Model

class MultibudgetAgent(Agent):
    model = Model.OpenAI("gpt-4o-mini")
    
    def __init__(self):
        super().__init__()
        self.budget = Budget(
            run=0.10,          # Each request max $0.10
            hourly=5.00,       # Hour max $5
            daily=50.00,       # Day max $50
            monthly=500.00,    # Month max $500
            on_exceeded=OnExceeded.ERROR
        )
```

## What Happens When Budget is Exceeded

You can control what happens using `on_exceeded`:

### 1. OnExceeded.WARN

Just warn (but keep going):

```python
class WarnAgent(Agent):
    model = Model.OpenAI("gpt-4o-mini")
    
    def __init__(self):
        super().__init__()
        self.budget = Budget(
            run=0.10,
            on_exceeded=OnExceeded.WARN  # Just print a warning
        )


# Result: Agent will still work, but warns you
```

### 2. OnExceeded.ERROR

Stop and raise an error:

```python
class StrictAgent(Agent):
    model = Model.OpenAI("gpt-4o-mini")
    
    def __init__(self):
        super().__init__()
        self.budget = Budget(
            run=0.10,
            on_exceeded=OnExceeded.ERROR  # Stop immediately
        )


# Result: If budget would be exceeded, agent raises an error
agent = StrictAgent()

try:
    response = agent.response("Answer 100 questions")
except Exception as e:
    print(f"Budget exceeded: {e}")
```

## Budget Thresholds

Thresholds let you do something when you hit a certain percentage:

```python
from Syrin import Agent, Budget, OnExceeded
from Syrin.threshold import BudgetThreshold
from Syrin.model import Model


class ThresholdAgent(Agent):
    model = Model.OpenAI("gpt-4o-mini")
    
    def __init__(self):
        super().__init__()
        self.budget = Budget(
            run=1.00,
            thresholds=[
                BudgetThreshold(at=50, action=lambda ctx: print(f"Halfway there! Budget at {ctx.percentage}%")),
                BudgetThreshold(at=80, action=lambda ctx: print(f"Getting close! Budget at {ctx.percentage}%")),
                BudgetThreshold(at=95, action=lambda ctx: ctx.parent.switch_model(Model("openai/gpt-4o-mini"))),
            ]
        )
```

## Cost Tracking

Check how much you've spent:

```python
agent = BudgetAwareAgent()

response1 = agent.response("Question 1")
print(f"Cost: ${response1.cost_usd:.4f}")

response2 = agent.response("Question 2")
print(f"Cost: ${response2.cost_usd:.4f}")

# Check budget status
print(f"Budget: {agent.budget}")
print(f"Remaining: {response2.budget_remaining}")
print(f"Used: {response2.budget_used}")
```

## Real-World Example: Production Agent

```python
"""
Production-Grade Agent with Budget Management
Safe to deploy with cost controls
"""

from Syrin import Agent, Budget, OnExceeded
from Syrin.model import Model
from Syrin.threshold import BudgetThreshold
from Syrin.tool import tool


@tool
def search_database(query: str) -> dict:
    """Search internal database."""
    return {"results": ["result1", "result2"]}


class ProductionAgent(Agent):
    """
    A production agent with careful budget management.
    
    Budget breakdown:
    - $0.01 per request (gpt-4o-mini is very cheap)
    - $10 per hour maximum
    - $100 per day maximum
    - $1000 per month maximum
    """
    
    model = Model.OpenAI("gpt-4o-mini")  # Cheapest reliable model
    
    system_prompt = """
    You are a production assistant.
    Be efficient and concise.
    Help users quickly without unnecessary details.
    """
    
    tools = [search_database]
    
    def __init__(self):
        super().__init__()
        
        # Production budget with multiple safeguards
        self.budget = Budget(
            run=0.01,           # Each request max $0.01
            hourly=10.00,       # Hour max $10
            daily=100.00,       # Day max $100
            monthly=1000.00,    # Month max $1000
            on_exceeded=OnExceeded.ERROR,  # Fail hard if exceeded
            thresholds=[
                BudgetThreshold(at=75, action=lambda ctx: print(f"Run budget at {ctx.percentage}%")),
                BudgetThreshold(at=90, action=lambda ctx: print(f"Run budget at {ctx.percentage}%")),
            ]
        )


def main():
    agent = ProductionAgent()
    
    # Simulate production usage
    for i in range(5):
        try:
            response = agent.response(f"Help with request {i+1}")
            print(f"Request {i+1}: ${response.cost_usd:.4f}")
        except Exception as e:
            print(f"Request {i+1}: Budget exceeded - {e}")
            break


if __name__ == "__main__":
    main()
```

## Using Different Models by Cost

Cheaper models:
```python
# ~$0.00005 per 1K tokens
Model.OpenAI("gpt-4o-mini")

# ~$0.0001 per 1K tokens
Model.Anthropic("claude-3-haiku")
```

Mid-range models:
```python
# ~$0.0015 per 1K tokens
Model.OpenAI("gpt-4o")

# ~$0.003 per 1K tokens
Model.Anthropic("claude-3-sonnet")
```

Expensive (but better) models:
```python
# ~$0.015 per 1K tokens
Model.OpenAI("gpt-4")

# ~$0.02 per 1K tokens
Model.Anthropic("claude-3-opus")
```

## Cost Monitoring Dashboard

Track costs over time:

```python
"""
Simple Cost Dashboard
Monitor spending trends
"""

from Syrin import Agent, Budget, OnExceeded
from Syrin.model import Model
import json
from datetime import datetime


class MonitoredAgent(Agent):
    model = Model.OpenAI("gpt-4o-mini")
    
    def __init__(self):
        super().__init__()
        self.budget = Budget(run=0.10, hourly=10.00)
        self.costs_log = []
    
    def response(self, prompt: str):
        response = super().response(prompt)
        
        # Log the cost
        self.costs_log.append({
            "timestamp": datetime.now().isoformat(),
            "cost": response.cost_usd,
            "tokens": response.tokens,
            "model": str(response.model)
        })
        
        return response
    
    def print_stats(self):
        if not self.costs_log:
            print("No requests made yet")
            return
        
        total_cost = sum(log["cost"] for log in self.costs_log)
        total_tokens = sum(log["tokens"] for log in self.costs_log)
        avg_cost = total_cost / len(self.costs_log)
        
        print("\n📊 Cost Statistics")
        print(f"Total requests: {len(self.costs_log)}")
        print(f"Total cost: ${total_cost:.4f}")
        print(f"Total tokens: {total_tokens}")
        print(f"Average cost per request: ${avg_cost:.4f}")
        print(f"Requests: {json.dumps(self.costs_log, indent=2)}")


def main():
    agent = MonitoredAgent()
    
    # Make some requests
    for i in range(3):
        response = agent.response(f"Answer question {i+1}")
        print(f"Request {i+1}: Completed")
    
    # Show statistics
    agent.print_stats()


if __name__ == "__main__":
    main()
```

## Budget Best Practices

### 1. Start Conservative

```python
# ✓ Good - start small
budget = Budget(run=0.05, hourly=5.00, daily=50.00)

# ❌ Not good - too generous
budget = Budget(run=1.00, hourly=100.00, daily=1000.00)
```

### 2. Use Cheap Models for Testing

```python
# ✓ Good - cheap for testing
test_agent = Agent()
test_agent.model = Model.OpenAI("gpt-4o-mini")

# ❌ Not good - expensive for testing
test_agent = Agent()
test_agent.model = Model.OpenAI("gpt-4")
```

### 3. Monitor Costs Regularly

```python
# ✓ Good - track spending
for i in range(10):
    response = agent.response(prompt)
    print(f"Cost so far: ${total_cost}")

# ❌ Not good - don't check
for i in range(10):
    response = agent.response(prompt)
```

### 4. Set Multiple Budget Levels

```python
# ✓ Good - multiple safeguards
budget = Budget(
    run=0.01,
    hourly=10,
    daily=100,
    monthly=1000,
    on_exceeded=OnExceeded.ERROR
)

# ❌ Not good - only one limit
budget = Budget(run=100.00)
```

## Common Questions

**Q: How accurate is the cost estimate?**
A: Very accurate! It's based on actual token counts from the API.

**Q: What if the budget is exceeded?**
A: It depends on your `on_exceeded` setting:
- `WARN`: Shows a warning but continues
- `ERROR`: Raises an exception and stops

**Q: Can I see all my costs?**
A: Yes! Check `response.cost_usd` and track them over time.

**Q: What's the cheapest model?**
A: `gpt-4o-mini` (~$0.00005 per 1K tokens)

## Next Steps

- **Build Teams** → Learn [Use Case 5: Multi-Agent Orchestration](multi-agent.md)
- **Get Real-Time Updates** → Learn [Use Case 6: Streaming](streaming.md)
- **Advanced Features** → See [Feature Reference Guide](reference.md)

---

Questions? See the [FAQ in Getting Started](getting-started.md#common-questions)
