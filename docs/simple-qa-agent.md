# Use Case 1: Simple Q&A Agent

## What You'll Learn

How to create an AI agent that answers questions. Perfect for:
- Chatbots
- Q&A systems
- Information lookups
- Writing assistants

## Complete Example: Copy & Paste This!

```python
"""
Simple Q&A Agent Example
Copy this code and run it!
"""

from syrin import Agent
from syrin.model import Model


class MyQAAgent(Agent):
    """An agent that answers questions."""
    
    # Tell it which AI to use
    model = Model.OpenAI("gpt-4o-mini")
    
    # Give it personality/instructions
    system_prompt = """
    You are a helpful Q&A assistant.
    Answer questions accurately and concisely.
    If you don't know, say so instead of guessing.
    """


def main():
    # Create the agent
    agent = MyQAAgent()
    
    # Ask it questions
    questions = [
        "What is Python?",
        "How do I make pasta?",
        "Explain quantum computing in simple terms"
    ]
    
    for question in questions:
        print(f"\n❓ Question: {question}")
        response = agent.response(question)
        print(f"✓ Answer: {response.content}")
        print(f"💰 Cost: ${response.cost_usd:.4f}")
        print(f"⏱️  Time: {response.duration:.2f}s")


if __name__ == "__main__":
    main()
```

**To run this:**
```bash
python my_qa_agent.py
```

**Want to serve your agent?** Use **HTTP**, **CLI**, or the **playground**:

```python
from syrin.enums import ServeProtocol

agent = MyQAAgent()
agent.serve(port=8000)  # HTTP: POST /chat, /stream
agent.serve(protocol=ServeProtocol.CLI)  # CLI: terminal REPL
agent.serve(port=8000, enable_playground=True)  # Web playground at /playground
```

See [Serving](serving.md). Use `debug=True` or `--trace` to see traces.

## How to Customize

### Change the AI Model

The example uses `gpt-4o-mini` (cheap, fast, good). Here are other options:

```python
class MyQAAgent(Agent):
    # Use OpenAI's advanced model
    model = Model.OpenAI("gpt-4o")
    
    # Or use Anthropic's Claude
    # model = Model.Anthropic("claude-3-5-sonnet")
    
    # Or use Google's model
    # model = Model.Google("gemini-pro")
    
    # Or use local model (free!)
    # model = Model.Ollama("llama2")
    
    system_prompt = "You are a helpful assistant."
```

### Change the Personality

Just change the `system_prompt`:

```python
class TranslatorAgent(Agent):
    model = Model.OpenAI("gpt-4o-mini")
    system_prompt = """
    You are a professional translator.
    Translate all responses to Spanish.
    Be formal and accurate.
    """


class FunnyAgent(Agent):
    model = Model.OpenAI("gpt-4o-mini")
    system_prompt = """
    You are a stand-up comedian.
    Make all answers funny and entertaining.
    Use jokes and puns whenever possible.
    """


class HistoryTeacher(Agent):
    model = Model.OpenAI("gpt-4o-mini")
    system_prompt = """
    You are a history teacher.
    Explain historical events in an educational way.
    Always include dates and important figures.
    Suggest related topics to learn about.
    """
```

## Understanding the Response

When you call `agent.response()`, you get back information:

```python
response = agent.response("What is AI?")

# The main answer
print(response.content)

# How much did this cost?
print(f"Cost: ${response.cost_usd}")

# How many tokens were used?
print(f"Tokens: {response.tokens}")

# Which model answered?
print(f"Model: {response.model}")

# How long did it take?
print(f"Duration: {response.duration}s")

# What was the raw response from the API?
print(f"Raw: {response.raw}")

# What reason did the AI stop answering?
print(f"Stop reason: {response.stop_reason}")
```

## Advanced: Interactive Q&A

Make it interactive so users can ask questions:

```python
from syrin import Agent
from syrin.model import Model


class InteractiveQAAgent(Agent):
    model = Model.OpenAI("gpt-4o-mini")
    system_prompt = "You are a helpful assistant."


def main():
    agent = InteractiveQAAgent()
    
    print("🤖 Interactive Q&A Agent")
    print("Type your question and press Enter. Type 'quit' to exit.\n")
    
    total_cost = 0.0
    
    while True:
        question = input("You: ")
        
        if question.lower() == "quit":
            print(f"\nGoodbye! Total spent: ${total_cost:.4f}")
            break
        
        if not question.strip():
            continue
        
        response = agent.response(question)
        
        print(f"Agent: {response.content}\n")
        total_cost += response.cost_usd
        
        print(f"[Cost: ${response.cost_usd:.4f}, Total: ${total_cost:.4f}]\n")


if __name__ == "__main__":
    main()
```

## Multiple Different Agents

Create different agents for different purposes:

```python
from syrin import Agent
from syrin.model import Model


class GrammarChecker(Agent):
    model = Model.OpenAI("gpt-4o-mini")
    system_prompt = """
    You are a grammar expert.
    Check the given text for grammar mistakes.
    Provide corrections and explanations.
    """


class CodeExplainer(Agent):
    model = Model.OpenAI("gpt-4o-mini")
    system_prompt = """
    You are a programming expert.
    Explain code in simple terms.
    Break down complex concepts.
    Provide examples when helpful.
    """


class CreativeWriter(Agent):
    model = Model.OpenAI("gpt-4o-mini")
    system_prompt = """
    You are a creative writing expert.
    Help users write stories, poems, and creative content.
    Make suggestions to improve their writing.
    Be encouraging and inspiring.
    """


def main():
    grammar_checker = GrammarChecker()
    code_explainer = CodeExplainer()
    creative_writer = CreativeWriter()
    
    # Test each one
    print("Grammar Check:")
    r1 = grammar_checker.response("She don't like pizza")
    print(r1.content)
    
    print("\nCode Explanation:")
    r2 = code_explainer.response("What does list.append() do?")
    print(r2.content)
    
    print("\nCreative Writing:")
    r3 = creative_writer.response("Help me start a story about a dragon")
    print(r3.content)


if __name__ == "__main__":
    main()
```

## Handling Errors

Sometimes things go wrong. Here's how to handle it:

```python
from syrin import Agent
from syrin.model import Model


class SafeQAAgent(Agent):
    model = Model.OpenAI("gpt-4o-mini")
    system_prompt = "You are a helpful assistant."


def main():
    agent = SafeQAAgent()
    
    questions = ["What is Python?", "Explain AI", "Tell me a joke"]
    
    for question in questions:
        try:
            response = agent.response(question)
            print(f"Q: {question}")
            print(f"A: {response.content}")
            print(f"Cost: ${response.cost_usd:.4f}\n")
            
        except Exception as e:
            print(f"❌ Error for '{question}': {e}\n")
            print("This might happen if:")
            print("  - Your API key is wrong")
            print("  - You ran out of API credits")
            print("  - The API service is down")
            print("  - You have no internet connection\n")


if __name__ == "__main__":
    main()
```

## Tips & Tricks

### 1. Better Prompts = Better Answers

Instead of:
```python
system_prompt = "You are an assistant."
```

Do this:
```python
system_prompt = """
You are an expert customer service representative.
Be friendly, patient, and professional.
If you don't know the answer, offer to transfer to a specialist.
Always be empathetic and try to solve the problem.
"""
```

### 2. Use Different Models for Different Tasks

```python
# Fast and cheap for simple questions
cheap_model = Model.OpenAI("gpt-4o-mini")

# Better quality for complex questions
good_model = Model.OpenAI("gpt-4o")

# Very advanced reasoning
best_model = Model.Anthropic("claude-3-opus")
```

### 3. Provide Context

```python
response = agent.response("""
I'm making a cake. I have:
- Flour
- Eggs
- Sugar
- Butter

What should I do next?
""")
```

### 4. Ask for Specific Formats

```python
response = agent.response("""
List 5 ways to stay healthy.
Format as a numbered list.
Keep each item to one sentence.
""")
```

## Next Steps

- **Add Tools** → Learn [Use Case 2: Research Agent with Tools](research-agent-with-tools.md)
- **Add Memory** → Learn [Use Case 3: Agent with Memory](agent-with-memory.md)
- **Control Costs** → Learn [Use Case 4: Budget Control](budget-control.md)

---

Questions? See the [FAQ in Getting Started](getting-started.md#common-questions)
