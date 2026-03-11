# Use Case 5: Multi-Agent Orchestration

## What You'll Learn

How to create **multiple agents that work together**. Perfect for:
- Complex workflows
- Specialized teams
- Parallel processing
- Handoffs between agents

Instead of one agent doing everything, use specialized agents working as a team!

## The Idea

```
Task: "Write and proofread a blog post"
  ↓
Writer Agent: Creates the blog post
  ↓
Editor Agent: Proofreads and improves it
  ↓
Final Article: Ready to publish!
```

## Complete Example: Copy & Paste This!

```python
"""
Multi-Agent Team Example
Copy this code and run it!
"""

from syrin import Agent
from syrin.model import Model
from syrin.tool import tool


# Define shared tools
@tool
def save_text(filename: str, content: str) -> dict:
    """Save text to a file."""
    return {
        "filename": filename,
        "saved": True,
        "length": len(content)
    }


# Agent 1: Writer
class WriterAgent(Agent):
    """Writes blog posts."""
    
    # model = Model.OpenAI("gpt-4o-mini")
    model = Model.Almock()  # No API Key needed
    
    system_prompt = """
    You are a professional blog writer.
    Write engaging, well-structured blog posts.
    Use clear language and good examples.
    """
    
    tools = [save_text]


# Agent 2: Editor
class EditorAgent(Agent):
    """Edits and improves text."""
    
    # model = Model.OpenAI("gpt-4o-mini")
    model = Model.Almock()  # No API Key needed
    
    system_prompt = """
    You are a professional editor.
    Review text for grammar, clarity, and flow.
    Suggest improvements without changing the voice.
    """
    
    tools = [save_text]


# Agent 3: Reviewer
class ReviewerAgent(Agent):
    """Final quality check."""
    
    # model = Model.OpenAI("gpt-4o-mini")
    model = Model.Almock()  # No API Key needed
    
    system_prompt = """
    You are a quality reviewer.
    Check if content is accurate, well-written, and ready to publish.
    Give a final approval or suggestions.
    """
    
    tools = [save_text]


def main():
    # Create the team
    writer = WriterAgent()
    editor = EditorAgent()
    reviewer = ReviewerAgent()
    
    topic = "Why Python is Great for Beginners"
    
    print("📝 Stage 1: Writing")
    print("-" * 50)
    draft = writer.response(f"Write a blog post about: {topic}")
    print(f"Draft created:\n{draft.content[:200]}...\n")
    
    print("✏️  Stage 2: Editing")
    print("-" * 50)
    edited = editor.response(f"Edit this post for quality:\n\n{draft.content}")
    print(f"Edited version:\n{edited.content[:200]}...\n")
    
    print("✅ Stage 3: Review")
    print("-" * 50)
    reviewed = reviewer.response(f"Review this post:\n\n{edited.content}")
    print(f"Review:\n{reviewed.content}\n")
    
    print("📊 Summary")
    print("-" * 50)
    total_cost = draft.cost_usd + edited.cost_usd + reviewed.cost_usd
    print(f"Total cost: ${total_cost:.4f}")


if __name__ == "__main__":
    main()
```

## Sequential Processing (Pipeline)

Process tasks one after another:

```python
"""
Sequential Pipeline: One Agent After Another
"""

from syrin import Agent
from syrin.model import Model


class ResearcherAgent(Agent):
    """Researches topics."""
    # model = Model.OpenAI("gpt-4o-mini")
    model = Model.Almock()  # No API Key needed
    system_prompt = "You are a researcher. Find key points about the topic."


class AnalystAgent(Agent):
    """Analyzes findings."""
    # model = Model.OpenAI("gpt-4o-mini")
    model = Model.Almock()  # No API Key needed
    system_prompt = "You are an analyst. Analyze and summarize the key findings."


class ReportAgent(Agent):
    """Writes reports."""
    # model = Model.OpenAI("gpt-4o-mini")
    model = Model.Almock()  # No API Key needed
    system_prompt = "You are a report writer. Create a professional report."


def sequential_pipeline(topic: str):
    """Run agents sequentially."""
    
    researcher = ResearcherAgent()
    analyst = AnalystAgent()
    reporter = ReportAgent()
    
    # Step 1: Research
    print("🔍 Step 1: Research")
    research_result = researcher.response(f"Research {topic}")
    print(f"Research: {research_result.content[:100]}...\n")
    
    # Step 2: Analyze
    print("📊 Step 2: Analyze")
    analysis_result = analyst.response(f"Analyze this: {research_result.content}")
    print(f"Analysis: {analysis_result.content[:100]}...\n")
    
    # Step 3: Report
    print("📝 Step 3: Report")
    report_result = reporter.response(f"Write a report based on: {analysis_result.content}")
    print(f"Report: {report_result.content[:100]}...\n")
    
    # Return final result
    return report_result


# Run it
result = sequential_pipeline("Renewable Energy")
print(f"Total cost: ${result.cost_usd:.4f}")
```

## Parallel Processing

Run agents at the same time:

```python
"""
Parallel Processing: Multiple Agents at Once
"""

from syrin import Agent
from syrin.model import Model
import asyncio


class SocialMediaAgent(Agent):
    """Create social media posts."""
    # model = Model.OpenAI("gpt-4o-mini")
    model = Model.Almock()  # No API Key needed
    system_prompt = "Create catchy social media posts."


class EmailAgentAgent(Agent):
    """Write marketing emails."""
    # model = Model.OpenAI("gpt-4o-mini")
    model = Model.Almock()  # No API Key needed
    system_prompt = "Write professional marketing emails."


class NewsletterAgent(Agent):
    """Write newsletters."""
    # model = Model.OpenAI("gpt-4o-mini")
    model = Model.Almock()  # No API Key needed
    system_prompt = "Write engaging newsletters."


async def parallel_marketing(product: str):
    """Create marketing content in parallel."""
    
    social = SocialMediaAgent()
    email = EmailAgentAgent()
    newsletter = NewsletterAgent()
    
    # Create all at once
    prompt = f"Create marketing content for {product}"
    
    # Using async for parallel execution
    results = await asyncio.gather(
        run_agent_async(social, prompt),
        run_agent_async(email, prompt),
        run_agent_async(newsletter, prompt)
    )
    
    return results


async def run_agent_async(agent, prompt):
    """Run agent asynchronously."""
    return agent.response(prompt)


# Run it
# results = asyncio.run(parallel_marketing("Cool App"))
# print("All marketing content created!")
```

## Agent Specialization

Each agent focuses on one thing:

```python
"""
Specialized Agents: Each Expert in Their Field
"""

from syrin import Agent
from syrin.model import Model


class CodeReviewerAgent(Agent):
    """Reviews code quality."""
    # model = Model.OpenAI("gpt-4o-mini")
    model = Model.Almock()  # No API Key needed
    system_prompt = """
    You are a senior code reviewer.
    Check code for:
    - Performance issues
    - Security vulnerabilities
    - Code style
    - Best practices
    """


class SecurityAuditAgent(Agent):
    """Audits security."""
    # model = Model.OpenAI("gpt-4o-mini")
    model = Model.Almock()  # No API Key needed
    system_prompt = """
    You are a security expert.
    Audit code for security issues:
    - SQL injection
    - XSS attacks
    - Authentication flaws
    - Data exposure
    """


class DocumentationAgent(Agent):
    """Writes documentation."""
    # model = Model.OpenAI("gpt-4o-mini")
    model = Model.Almock()  # No API Key needed
    system_prompt = """
    You are a technical writer.
    Create clear documentation:
    - Function descriptions
    - Usage examples
    - Error handling
    - Edge cases
    """


def review_code(code_snippet: str):
    """Get reviews from all specialists."""
    
    reviewer = CodeReviewerAgent()
    security = SecurityAuditAgent()
    docs = DocumentationAgent()
    
    print("🔍 Code Review")
    review = reviewer.response(f"Review this code:\n{code_snippet}")
    print(review.content[:200])
    
    print("\n🔐 Security Audit")
    audit = security.response(f"Audit this code:\n{code_snippet}")
    print(audit.content[:200])
    
    print("\n📚 Documentation")
    documentation = docs.response(f"Document this code:\n{code_snippet}")
    print(documentation.content[:200])


# Example code to review
example_code = """
def login(username, password):
    user = database.find(username)
    if user and user.password == password:
        return True
    return False
"""

review_code(example_code)
```

## Agent with Delegation

One agent delegating to others:

```python
"""
Delegating Agent: Asks Other Agents for Help
"""

from syrin import Agent
from syrin.model import Model
from syrin.tool import tool


# Helper agents
class MathAgent(Agent):
    # model = Model.OpenAI("gpt-4o-mini")
    model = Model.Almock()  # No API Key needed
    system_prompt = "You are a math expert. Solve math problems."


class HistoryAgent(Agent):
    # model = Model.OpenAI("gpt-4o-mini")
    model = Model.Almock()  # No API Key needed
    system_prompt = "You are a history expert. Answer history questions."


class ScienceAgent(Agent):
    # model = Model.OpenAI("gpt-4o-mini")
    model = Model.Almock()  # No API Key needed
    system_prompt = "You are a science expert. Answer science questions."


# Main agent that delegates
class TeacherAgent(Agent):
    # model = Model.OpenAI("gpt-4o-mini")
    model = Model.Almock()  # No API Key needed
    
    system_prompt = """
    You are a teacher assistant.
    You can ask specialist agents for help.
    Combine their answers to give complete responses.
    """


def answer_complex_question(question: str):
    """Answer a complex question by delegating."""
    
    teacher = TeacherAgent()
    
    # Teacher decides what to do
    print(f"Question: {question}\n")
    
    response = teacher.response(question)
    print(f"Answer: {response.content}")
    
    return response


# Examples
answer_complex_question("What is the Pythagorean theorem and why is it important?")
```

## Real-World Example: Customer Support Team

```python
"""
Customer Support Multi-Agent System
Different agents handle different types of requests
"""

from syrin import Agent
from syrin.model import Model
from syrin.tool import tool


@tool
def check_order_status(order_id: str) -> dict:
    """Check status of an order."""
    return {"order_id": order_id, "status": "shipped"}


@tool
def process_refund(order_id: str, reason: str) -> dict:
    """Process a refund."""
    return {"order_id": order_id, "refund": "processed"}


@tool
def send_message(customer_id: str, message: str) -> dict:
    """Send message to customer."""
    return {"customer_id": customer_id, "sent": True}


class OrderSupportAgent(Agent):
    """Handles order-related issues."""
    # model = Model.OpenAI("gpt-4o-mini")
    model = Model.Almock()  # No API Key needed
    system_prompt = "You handle order inquiries and shipping issues."
    tools = [check_order_status, send_message]


class RefundAgent(Agent):
    """Handles refunds and returns."""
    # model = Model.OpenAI("gpt-4o-mini")
    model = Model.Almock()  # No API Key needed
    system_prompt = "You process refunds and handle returns."
    tools = [process_refund, send_message]


class TechnicalSupportAgent(Agent):
    """Handles technical issues."""
    # model = Model.OpenAI("gpt-4o-mini")
    model = Model.Almock()  # No API Key needed
    system_prompt = "You help with product technical issues."
    tools = [send_message]


class SupportRouter(Agent):
    """Routes requests to the right agent."""
    
    # model = Model.OpenAI("gpt-4o-mini")
    model = Model.Almock()  # No API Key needed
    
    system_prompt = """
    You are a support router.
    Determine if the issue is:
    - ORDER: About orders/shipping
    - REFUND: About refunds/returns
    - TECH: About product usage
    """


def handle_support_request(customer_message: str):
    """Route support request to right agent."""
    
    router = SupportRouter()
    order_agent = OrderSupportAgent()
    refund_agent = RefundAgent()
    tech_agent = TechnicalSupportAgent()
    
    # Determine type
    print(f"Customer: {customer_message}\n")
    
    routing = router.response(customer_message)
    print(f"Routing decision: {routing.content}\n")
    
    # Route to appropriate agent
    if "ORDER" in routing.content.upper():
        response = order_agent.response(customer_message)
    elif "REFUND" in routing.content.upper():
        response = refund_agent.response(customer_message)
    else:
        response = tech_agent.response(customer_message)
    
    print(f"Support Response: {response.content}")


# Examples
handle_support_request("Where's my order?")
handle_support_request("I want to return my product")
handle_support_request("How do I use feature X?")
```

## Budget Sharing Across Team

Control team spending:

```python
"""
Team Budget: All agents share one budget
"""

from syrin import Agent, Budget, raise_on_exceeded
from syrin.model import Model
from syrin.threshold import BudgetThreshold


shared_budget = Budget(
    hourly=50.00,  # Team max $50/hour
    on_exceeded=raise_on_exceeded,
    thresholds=[
        BudgetThreshold(at=80, action={"type": "warn"})
    ]
)


class TeamAgent1(Agent):
    # model = Model.OpenAI("gpt-4o-mini")
    model = Model.Almock()  # No API Key needed
    
    def __init__(self):
        super().__init__()
        self.budget = shared_budget  # Share budget


class TeamAgent2(Agent):
    # model = Model.OpenAI("gpt-4o-mini")
    model = Model.Almock()  # No API Key needed
    
    def __init__(self):
        super().__init__()
        self.budget = shared_budget  # Share budget


class TeamAgent3(Agent):
    # model = Model.OpenAI("gpt-4o-mini")
    model = Model.Almock()  # No API Key needed
    
    def __init__(self):
        super().__init__()
        self.budget = shared_budget  # Share budget


def main():
    """All agents share the same budget."""
    
    agent1 = TeamAgent1()
    agent2 = TeamAgent2()
    agent3 = TeamAgent3()
    
    total = 0
    
    for agent in [agent1, agent2, agent3]:
        response = agent.response("Do something")
        total += response.cost_usd
        print(f"Cost: ${response.cost_usd:.4f}, Team total: ${total:.4f}")
    
    print(f"Total team spending: ${total:.4f}")


if __name__ == "__main__":
    main()
```

## Best Practices

### 1. Each Agent One Job

```python
# ✓ Good - focused responsibility
class WriterAgent(Agent):
    system_prompt = "Write content only."

# ❌ Not good - too many responsibilities
class JackOfAllTradesAgent(Agent):
    system_prompt = "Write, edit, review, publish everything."
```

### 2. Clear Communication

```python
# ✓ Good - structured output
response = agent.response("Write a 3-paragraph article about Python")

# ❌ Not good - ambiguous
response = agent.response("Write about Python")
```

### 3. Appropriate Tools

```python
# ✓ Good - agents have right tools
writer_agent.tools = [save_text, format_text]
editor_agent.tools = [save_text, check_grammar]

# ❌ Not good - wrong tools
writer_agent.tools = [delete_database, send_email]
```

### 4. Handle Failures

```python
try:
    response = agent.response(prompt)
except Exception as e:
    print(f"Agent failed: {e}")
    # Use backup agent or retry
```

## Next Steps

- **Get Real-Time Updates** → Learn [Use Case 6: Streaming](streaming.md)
- **Advanced Features** → See [Feature Reference Guide](reference.md)

---

Questions? See the [FAQ in Getting Started](getting-started.md#common-questions)
