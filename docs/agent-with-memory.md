# Use Case 3: Agent with Memory

## What You'll Learn

How to make your agent **remember** conversations. Perfect for:
- Chatbots that remember users
- Customer service agents
- Personal assistants
- Learning systems

Without memory, your agent forgets everything after each conversation. With memory, it can learn and grow!

## The Idea

```
Conversation 1:
User: "My name is Alice and I like pizza"
Agent: "Nice to meet you Alice!"
Agent: ✅ REMEMBERS: Alice likes pizza

Conversation 2 (days later):
User: "What's my favorite food?"
Agent: "You like pizza, Alice!"
Agent: ✓ It remembered!
```

## Complete Example: Copy & Paste This!

```python
"""
Agent with Memory
Copy this code and run it!
"""

from syrin import Agent, Memory
from syrin.model import Model


class MemoryfulAgent(Agent):
    """An agent that remembers conversations."""
    
    model = Model.OpenAI("gpt-4o-mini")
    
    system_prompt = """
    You are a helpful assistant that remembers information about users.
    Use what you know about them to provide personalized responses.
    """
    
    def __init__(self):
        super().__init__()
        # Give the agent memory
        self.memory = Memory()


def main():
    agent = MemoryfulAgent()
    
    print("🤖 Agent with Memory")
    print("Type 'quit' to exit\n")
    
    # Simulate a conversation
    conversation = [
        "My name is Sarah",
        "I work as a software engineer",
        "I love Python and coffee",
        "What's my name?",
        "What do I like?",
        "What do I do for work?",
    ]
    
    for user_input in conversation:
        print(f"You: {user_input}")
        
        response = agent.response(user_input)
        print(f"Agent: {response.content}\n")
    
    print("✓ Notice how the agent remembered your information!")


if __name__ == "__main__":
    main()
```

## Types of Memory

Your agent can use 4 different types of memory:

### 1. **Core Memory** (Identity)
Things that define who the user is:
```python
@tool
def remember_user_profile(name: str, profession: str, interests: list) -> dict:
    """Store basic user information."""
    return {
        "stored": {
            "name": name,
            "profession": profession,
            "interests": interests
        }
    }
```

### 2. **Episodic Memory** (Events)
Specific things that happened:
```python
# "User told me about their vacation to Paris on Feb 21"
# "User complained about slow service yesterday"
```

### 3. **Semantic Memory** (Facts)
General knowledge learned:
```python
# "User likes Python"
# "User is allergic to peanuts"
# "User prefers email communication"
```

### 4. **Procedural Memory** (How-to)
Things the agent learned to do:
```python
# "User prefers me to summarize in 3 points"
# "User wants formal communication"
```

## Complete Example with All Memory Types

```python
"""
Agent with All Memory Types
This shows how to use all 4 types of memory
"""

from syrin import Agent, Memory, MemoryType
from syrin.model import Model
from syrin.tool import tool


# Tools to store different memory types
@tool
def learn_about_user(name: str, age: int, job: str) -> dict:
    """Store basic information about the user (Core Memory)."""
    return {
        "type": "core",
        "data": {"name": name, "age": age, "job": job},
        "stored": True
    }


@tool
def remember_event(event: str, date: str) -> dict:
    """Remember something that happened (Episodic Memory)."""
    return {
        "type": "episodic",
        "event": event,
        "date": date,
        "stored": True
    }


@tool
def learn_fact(fact: str, category: str) -> dict:
    """Learn a general fact (Semantic Memory)."""
    return {
        "type": "semantic",
        "fact": fact,
        "category": category,
        "stored": True
    }


@tool
def learn_preference(preference: str) -> dict:
    """Remember how the user likes to interact (Procedural Memory)."""
    return {
        "type": "procedural",
        "preference": preference,
        "stored": True
    }


class SmartAgent(Agent):
    """An agent with full memory capabilities."""
    
    model = Model.OpenAI("gpt-4o-mini")
    
    system_prompt = """
    You are a personal assistant with perfect memory.
    Remember everything the user tells you.
    Use what you know to be helpful and personal.
    When the user provides information, acknowledge that you'll remember it.
    """
    
    tools = [
        learn_about_user,
        remember_event,
        learn_fact,
        learn_preference
    ]
    
    def __init__(self):
        super().__init__()
        self.memory = Memory()


def main():
    agent = SmartAgent()
    
    # Test different types of memory
    queries = [
        "Hi! I'm John, I'm 30 years old and I'm a doctor",
        "I just got back from a trip to Japan on Feb 20",
        "I hate spam emails - I prefer phone calls",
        "I'm allergic to shellfish",
        "What do you know about me?",
    ]
    
    for query in queries:
        print(f"You: {query}")
        response = agent.response(query)
        print(f"Agent: {response.content}\n")


if __name__ == "__main__":
    main()
```

## Interactive Memory Example

Make an agent that learns as you chat:

```python
"""
Interactive Memory Agent
Chat with an agent that learns about you over time
"""

from syrin import Agent, Memory
from syrin.model import Model


class InteractiveMemoryAgent(Agent):
    """An agent that remembers your preferences."""
    
    model = Model.OpenAI("gpt-4o-mini")
    
    system_prompt = """
    You are a personal assistant that remembers user preferences.
    When users tell you things about themselves:
    1. Acknowledge what you learned
    2. Store it in your memory
    3. Use it in future conversations
    
    Be warm and personable.
    """
    
    def __init__(self):
        super().__init__()
        self.memory = Memory()


def main():
    agent = InteractiveMemoryAgent()
    
    print("💭 Memory-Enabled Agent")
    print("Tell me about yourself and I'll remember!")
    print("Type 'quit' to exit\n")
    
    while True:
        user_input = input("You: ").strip()
        
        if user_input.lower() == "quit":
            print("Goodbye! I'll remember our conversation!")
            break
        
        if not user_input:
            continue
        
        response = agent.response(user_input)
        print(f"Agent: {response.content}\n")


if __name__ == "__main__":
    main()
```

## Customer Support Agent with Memory

Here's a real-world example:

```python
"""
Customer Support Agent with Memory
A support bot that remembers customer issues
"""

from syrin import Agent, Memory
from syrin.model import Model
from syrin.tool import tool


@tool
def get_customer_history(customer_id: str) -> dict:
    """Get past support tickets for a customer."""
    # Mock data
    histories = {
        "C001": [
            "Issue with login on Jan 15",
            "Billing question on Feb 1",
            "Feature request on Feb 10"
        ],
        "C002": [
            "Password reset on Jan 20"
        ]
    }
    return {
        "customer_id": customer_id,
        "history": histories.get(customer_id, [])
    }


@tool
def create_support_ticket(customer_id: str, issue: str) -> dict:
    """Create a new support ticket."""
    return {
        "ticket_id": "T001",
        "customer_id": customer_id,
        "issue": issue,
        "status": "created"
    }


class CustomerSupportAgent(Agent):
    """A support agent that remembers customer issues."""
    
    model = Model.OpenAI("gpt-4o-mini")
    
    system_prompt = """
    You are a professional customer support agent.
    Be empathetic and helpful.
    Remember previous issues and reference them to show you care.
    Use the tools to access customer history and create tickets.
    """
    
    tools = [get_customer_history, create_support_ticket]
    
    def __init__(self):
        super().__init__()
        self.memory = Memory()


def main():
    agent = CustomerSupportAgent()
    
    # Simulate support interactions
    support_interactions = [
        "Hello, I'm customer C001. I'm having trouble logging in again.",
        "Can you check my history?",
        "I've had this issue before. Can you help faster this time?",
        "Thank you for remembering my previous issues!",
    ]
    
    for interaction in support_interactions:
        print(f"\nCustomer: {interaction}")
        response = agent.response(interaction)
        print(f"Agent: {response.content}")


if __name__ == "__main__":
    main()
```

## Forgetting & Forgetting Strategy

Agents can also "forget" old information:

```python
from syrin import Agent, Memory, DecayStrategy
from syrin.model import Model


class ForgetfulAgent(Agent):
    """An agent that forgets old memories gradually."""
    
    model = Model.OpenAI("gpt-4o-mini")
    system_prompt = "You are an assistant."
    
    def __init__(self):
        super().__init__()
        # Memory that forgets old items (exponential decay)
        self.memory = Memory(
            decay_strategy="EXPONENTIAL"  # Old memories fade over time
        )


# Decay strategies:
# "EXPONENTIAL"  - Older memories fade fast (most realistic)
# "LINEAR"       - Memories fade at constant rate
# "LOGARITHMIC"  - Memories fade slowly
# "STEP"         - Memories disappear suddenly after time limit
# "NONE"         - Memories never fade (keep forever)
```

## Memory Best Practices

### 1. Tell the Agent to Remember

```python
# ✓ Good - tells agent to remember explicitly
system_prompt = """
You are a personal assistant.
Remember important information about the user.
When they tell you something personal, acknowledge that you'll remember it.
"""

# ❌ Not good - doesn't mention memory
system_prompt = "You are a helpful assistant."
```

### 2. Use Tools to Store Important Info

```python
from syrin.tool import tool

@tool
def save_user_preference(preference: str, category: str) -> dict:
    """Save something important to remember."""
    return {"saved": True, "preference": preference}


class SmartAgent(Agent):
    model = Model.OpenAI("gpt-4o-mini")
    system_prompt = "Remember user preferences using the save tool."
    tools = [save_user_preference]
```

### 3. Reference Memory in Responses

```python
# ✓ Good
agent_response = """
I remember that you prefer emails over phone calls.
I'll send you everything via email from now on.
"""

# ❌ Not good - doesn't reference what was learned
agent_response = "OK, I'll do that."
```

## Privacy Considerations

Be careful with memory! Consider:

```python
class PrivateAgent(Agent):
    """An agent that takes privacy seriously."""
    
    model = Model.OpenAI("gpt-4o-mini")
    
    system_prompt = """
    You are a private assistant.
    Never store or remember sensitive information like:
    - Credit card numbers
    - Social security numbers
    - Medical information
    - Passwords
    
    Always ask before storing personal information.
    """
    
    def __init__(self):
        super().__init__()
        # Use ephemeral memory (forgets when program ends)
        self.memory = Memory()  # In-memory storage only
```

## Debugging Memory

See what your agent remembers:

```python
agent = InteractiveMemoryAgent()
response = agent.response("I like coffee and Python")

# Check what was stored
print(agent.memory)  # See all memories
```

## Next Steps

- **Control Costs** → Learn [Use Case 4: Budget Control](budget-control.md)
- **Build Teams** → Learn [Use Case 5: Multi-Agent Orchestration](multi-agent.md)
- **Get Real-Time Updates** → Learn [Use Case 6: Streaming](streaming.md)

---

Questions? See the [FAQ in Getting Started](getting-started.md#common-questions)
