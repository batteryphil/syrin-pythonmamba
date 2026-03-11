# Use Case 6: Streaming & Real-Time Updates

## What You'll Learn

How to get **real-time responses** as they're being generated. Perfect for:
- Live chat applications
- Progress updates
- Large responses
- User-friendly interfaces

Instead of waiting for the full response, stream it back piece by piece!

## The Idea

```
Without Streaming:
You: "Write a poem"
Agent: (thinking... 5 seconds)
Agent: "Here's your poem: ..."

With Streaming:
You: "Write a poem"
Agent: "Here's" → "your" → "poem:" → ...
(Words appear in real-time, like ChatGPT!)
```

## Complete Example: Copy & Paste This!

```python
"""
Streaming Response Example
Copy this code and run it!
"""

from syrin import Agent
from syrin.model import Model


class StreamingAgent(Agent):
    """An agent that streams responses."""
    
    model = Model.OpenAI("gpt-4o-mini")
    system_prompt = "You are a helpful assistant."


def main():
    agent = StreamingAgent()
    
    prompt = "Write a haiku about programming"
    
    print("📡 Streaming Response:")
    print("-" * 50)
    print("Prompt: " + prompt)
    print("\nResponse:")
    
    # Stream the response
    total_tokens = 0
    total_cost = 0.0
    
    for chunk in agent.astream(prompt):
        # Print each chunk as it arrives
        print(chunk.text, end="", flush=True)
        
        # Track stats
        if chunk.tokens:
            total_tokens += chunk.tokens
        if chunk.cost_usd:
            total_cost += chunk.cost_usd
    
    print("\n" + "-" * 50)
    print(f"Total tokens: {total_tokens}")
    print(f"Total cost: ${total_cost:.4f}")


if __name__ == "__main__":
    main()
```

## Understanding Streaming

### Normal Response (Blocking)

```python
from syrin import Agent
from syrin.model import Model

class Agent1(Agent):
    model = Model.OpenAI("gpt-4o-mini")

agent = Agent1()

# You wait until entire response is ready
response = agent.response("Write a long story")

# Then you get everything at once
print(response.content)
print("Done!")
```

### Streaming Response (Real-time)

```python
from syrin import Agent
from syrin.model import Model

class Agent2(Agent):
    model = Model.OpenAI("gpt-4o-mini")

agent = Agent2()

# Stream starts immediately
print("Response:")
for chunk in agent.astream("Write a long story"):
    print(chunk.text, end="", flush=True)  # Print immediately
print("\nDone!")
```

## Real-World Example: Chat Application

```python
"""
Simple Chat with Streaming
Like ChatGPT!
"""

from syrin import Agent
from syrin.model import Model


class ChatAgent(Agent):
    """A chat agent with streaming."""
    
    model = Model.OpenAI("gpt-4o-mini")
    
    system_prompt = """
    You are a helpful chat assistant.
    Be friendly and conversational.
    Keep responses concise (2-3 paragraphs max).
    """


def main():
    agent = ChatAgent()
    
    print("💬 Chat with AI (type 'quit' to exit)")
    print("-" * 50)
    
    while True:
        user_input = input("\nYou: ").strip()
        
        if user_input.lower() == "quit":
            print("Goodbye!")
            break
        
        if not user_input:
            continue
        
        print("AI: ", end="", flush=True)
        
        # Stream the response
        try:
            for chunk in agent.astream(user_input):
                print(chunk.text, end="", flush=True)
        except Exception as e:
            print(f"Error: {e}")
        
        print()  # Newline after response


if __name__ == "__main__":
    main()
```

## Getting Information from Chunks

Each streamed chunk has information:

```python
from syrin import Agent
from syrin.model import Model


class DetailedStreamAgent(Agent):
    model = Model.OpenAI("gpt-4o-mini")


def main():
    agent = DetailedStreamAgent()
    
    print("📊 Detailed Streaming Analysis:")
    print("-" * 50)
    
    total_tokens = 0
    total_cost = 0.0
    chunk_count = 0
    
    for chunk in agent.astream("What is AI?"):
        chunk_count += 1
        
        # Print text
        print(chunk.text, end="", flush=True)
        
        # Track statistics
        if hasattr(chunk, 'tokens') and chunk.tokens:
            total_tokens += chunk.tokens
        
        if hasattr(chunk, 'cost_usd') and chunk.cost_usd:
            total_cost += chunk.cost_usd
    
    print("\n" + "-" * 50)
    print(f"Chunks received: {chunk_count}")
    print(f"Total tokens: {total_tokens}")
    print(f"Total cost: ${total_cost:.4f}")


if __name__ == "__main__":
    main()
```

## Streaming with Tools

Stream responses even when using tools:

```python
"""
Streaming with Tool Usage
"""

from syrin import Agent
from syrin.model import Model
from syrin.tool import tool


@tool
def calculate(expression: str) -> dict:
    """Calculate a math expression."""
    return {"result": eval(expression)}


@tool
def search_web(query: str) -> dict:
    """Search the web."""
    return {"results": ["result1", "result2"]}


class ToolStreamAgent(Agent):
    """Agent that uses tools and streams."""
    
    model = Model.OpenAI("gpt-4o-mini")
    system_prompt = "You are helpful."
    tools = [calculate, search_web]


def main():
    agent = ToolStreamAgent()
    
    print("📊 Streaming with Tools:")
    print("-" * 50)
    
    # Query that uses a tool
    for chunk in agent.astream("Calculate 100 * 5 + 20"):
        print(chunk.text, end="", flush=True)
    
    print("\n" + "-" * 50)


if __name__ == "__main__":
    main()
```

## Progress Indicators

Show user a progress indicator:

```python
"""
Streaming with Progress Indicator
"""

from syrin import Agent
from syrin.model import Model
import time


class ProgressStreamAgent(Agent):
    model = Model.OpenAI("gpt-4o-mini")


def main():
    agent = ProgressStreamAgent()
    
    prompt = "Write a 5-paragraph essay about space"
    
    print("📝 Generating essay...")
    print("-" * 50)
    print("Response: ", end="", flush=True)
    
    start_time = time.time()
    word_count = 0
    
    try:
        for chunk in agent.astream(prompt):
            print(chunk.text, end="", flush=True)
            word_count += len(chunk.text.split())
        
        elapsed = time.time() - start_time
        
        print("\n" + "-" * 50)
        print(f"✓ Complete!")
        print(f"  Words: {word_count}")
        print(f"  Time: {elapsed:.1f}s")
        print(f"  Speed: {word_count/elapsed:.0f} words/sec")
        
    except KeyboardInterrupt:
        print("\n✗ Interrupted by user")


if __name__ == "__main__":
    main()
```

## Asynchronous Streaming

For advanced use cases:

```python
"""
Advanced: Async Streaming
For high-performance applications
"""

import asyncio
from syrin import Agent
from syrin.model import Model


class AsyncStreamAgent(Agent):
    model = Model.OpenAI("gpt-4o-mini")


async def stream_response(agent: Agent, prompt: str):
    """Stream a response asynchronously."""
    print("Response: ", end="", flush=True)
    
    async for chunk in agent.astream(prompt):
        print(chunk.text, end="", flush=True)
    
    print()


async def main():
    agent = AsyncStreamAgent()
    
    # Single streaming request
    await stream_response(agent, "Explain quantum computing")
    
    # Multiple streaming requests in parallel
    prompts = [
        "What is Python?",
        "What is JavaScript?",
        "What is Go?"
    ]
    
    print("\n" + "-" * 50)
    print("Multiple streams in parallel:")
    print("-" * 50)
    
    # Run all in parallel
    tasks = [stream_response(agent, p) for p in prompts]
    await asyncio.gather(*tasks)


if __name__ == "__main__":
    asyncio.run(main())
```

## Streaming with Callbacks

Process chunks as they arrive:

```python
"""
Streaming with Custom Callbacks
"""

from syrin import Agent
from syrin.model import Model


class CallbackStreamAgent(Agent):
    model = Model.OpenAI("gpt-4o-mini")


def on_chunk_received(chunk):
    """Called when chunk is received."""
    print(f"[CHUNK: {len(chunk.text)} chars] {chunk.text}", end="", flush=True)


def on_stream_complete(total_tokens, total_cost):
    """Called when streaming completes."""
    print(f"\n[COMPLETE: {total_tokens} tokens, ${total_cost:.4f}]")


def main():
    agent = CallbackStreamAgent()
    
    print("📡 Streaming with Callbacks:")
    print("-" * 50)
    
    total_tokens = 0
    total_cost = 0.0
    
    for chunk in agent.astream("Tell me a story"):
        on_chunk_received(chunk)
        
        if hasattr(chunk, 'tokens') and chunk.tokens:
            total_tokens += chunk.tokens
        
        if hasattr(chunk, 'cost_usd') and chunk.cost_usd:
            total_cost += chunk.cost_usd
    
    on_stream_complete(total_tokens, total_cost)


if __name__ == "__main__":
    main()
```

## Buffering Chunks

Collect chunks before displaying:

```python
"""
Stream with Buffering
Collect chunks and display in batches
"""

from syrin import Agent
from syrin.model import Model


class BufferingStreamAgent(Agent):
    model = Model.OpenAI("gpt-4o-mini")


def main():
    agent = BufferingStreamAgent()
    
    print("📦 Streaming with Buffering:")
    print("-" * 50)
    
    buffer = ""
    buffer_size = 50  # Display every 50 chars
    
    for chunk in agent.astream("Write a poem"):
        buffer += chunk.text
        
        # Display when buffer is full
        if len(buffer) >= buffer_size:
            print(buffer, end="", flush=True)
            buffer = ""
    
    # Display remaining
    if buffer:
        print(buffer, end="", flush=True)
    
    print("\n" + "-" * 50)


if __name__ == "__main__":
    main()
```

## Real-Time Translation

Stream and translate simultaneously:

```python
"""
Real-time Translation Streaming
"""

from syrin import Agent
from syrin.model import Model


class TranslationStreamAgent(Agent):
    model = Model.OpenAI("gpt-4o-mini")
    system_prompt = "Translate all responses to Spanish."


def main():
    agent = TranslationStreamAgent()
    
    print("🌍 Real-time Translation:")
    print("-" * 50)
    print("English prompt -> Spanish response (streaming)")
    print("-" * 50)
    
    print("Response: ", end="", flush=True)
    
    for chunk in agent.astream("Tell me about the weather"):
        print(chunk.text, end="", flush=True)
    
    print("\n" + "-" * 50)


if __name__ == "__main__":
    main()
```

## Best Practices

### 1. Use Streaming for Long Responses

```python
# ✓ Good - stream for long content
for chunk in agent.astream("Write a 1000-word essay"):
    print(chunk.text, end="", flush=True)

# ❌ Not needed - single short response
response = agent.response("What's 2+2?")
```

### 2. Add Progress Feedback

```python
# ✓ Good - user sees progress
print("Generating... ", end="", flush=True)
for chunk in agent.astream(prompt):
    print(".", end="", flush=True)

# ❌ Not good - user doesn't know what's happening
for chunk in agent.astream(prompt):
    pass  # Silent
```

### 3. Handle Errors During Streaming

```python
try:
    for chunk in agent.astream(prompt):
        print(chunk.text, end="", flush=True)
except Exception as e:
    print(f"\n❌ Error: {e}")
```

### 4. Use Async for Performance

```python
# ✓ Good - multiple streams in parallel
async for chunk in agent.astream(prompt):
    # Non-blocking
    pass

# ❌ Not good - blocks
for chunk in agent.astream(prompt):
    # Blocking iteration
    pass
```

## Common Questions

**Q: Does streaming use different tokens?**
A: No, same tokens. Streaming just displays them as they arrive.

**Q: Is streaming faster?**
A: Not overall, but feels faster because user sees progress.

**Q: Can I stream with tools?**
A: No. `stream()` and `astream()` perform a single LLM completion and do not run the tool loop. If your agent has tools, use `response()` or `arun()` instead. For voice or WebSocket integrations that need streaming and tools, call `arun()` and emit the returned text.

**Q: Should I always use streaming?**
A: No. Use streaming for long responses where you want token-by-token output. For short responses or when you need tool execution, use `response()` or `arun()`.

## Next Steps

- **Advanced Features** → See [Feature Reference Guide](reference.md)
- **Back to Start** → Go to [Getting Started](getting-started.md)

---

Questions? See the [FAQ in Getting Started](getting-started.md#common-questions)
