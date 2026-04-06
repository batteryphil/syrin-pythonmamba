---
title: Running Agents
description: How to run your Syrin agent — sync, async, and streaming
weight: 64
---

## Four Ways to Get a Response

Syrin gives you four execution modes: synchronous, asynchronous, synchronous streaming, and asynchronous streaming. The right choice depends on your context — not on the agent itself. The same agent class works with all four.

## Synchronous: `agent.run()`

The simplest and most common mode. Blocks until the agent finishes and returns a `Response` object.

```python
from syrin import Agent, Model

agent = Agent(model=Model.mock(), system_prompt="You are helpful.")

response = agent.run("What is Python?")
print(response.content)
print(f"Cost: ${response.cost:.6f}")
```

Output:

```
Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor incididunt ut labore
Cost: $0.000040
```

Use `run()` when:
- You are writing a script or CLI tool
- You are in a sync web framework (Flask, Django with sync views)
- You want the simplest possible code

## Asynchronous: `agent.arun()`

The async version. Use inside `async def` functions with `await`. Returns the same `Response` object.

```python
import asyncio
from syrin import Agent, Model

agent = Agent(model=Model.mock(), system_prompt="You are helpful.")

async def main():
    response = await agent.arun("Hello async!")
    print(f"Async response: {response.content[:50]}")
    print(f"Cost: ${response.cost:.6f}")

asyncio.run(main())
```

Output:

```
Async response: Lorem ipsum dolor sit amet, consectetur adipiscing
Cost: $0.000041
```

Use `arun()` when:
- You are in an async framework like FastAPI or aiohttp
- You want to run multiple agents concurrently with `asyncio.gather()`
- You are building a high-throughput system

### Running multiple agents concurrently

```python
import asyncio
from syrin import Agent, Model

model = Model.mock()
agent1 = Agent(model=model, system_prompt="You summarize text.")
agent2 = Agent(model=model, system_prompt="You translate text.")
agent3 = Agent(model=model, system_prompt="You classify text.")

async def main():
    text = "Python is a great programming language for AI."
    results = await asyncio.gather(
        agent1.arun(f"Summarize: {text}"),
        agent2.arun(f"Translate to French: {text}"),
        agent3.arun(f"Classify the topic of: {text}"),
    )
    for i, r in enumerate(results, 1):
        print(f"Agent {i}: {r.content[:50]}")

asyncio.run(main())
```

Output:

```
Agent 1: Lorem ipsum dolor sit amet, consectetur adipiscing
Agent 2: Lorem ipsum dolor sit amet, consectetur adipiscing
Agent 3: Lorem ipsum dolor sit amet, consectetur adipiscing
```

All three agents run at the same time. With real models, this dramatically reduces latency compared to running them sequentially.

## Streaming: `agent.stream()`

Returns an iterator of `StreamChunk` objects. Each chunk has a piece of the response as it arrives — token by token or in batches.

```python
from syrin import Agent, Model

agent = Agent(model=Model.mock(), system_prompt="You are helpful.")

print("Response: ", end="")
for chunk in agent.stream("Tell me about Python"):
    print(chunk.text, end="", flush=True)
print()
```

Output:

```
Response: Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor incididunt ut labore
```

The `StreamChunk` object has:
- `chunk.text` — the text in this chunk (may be a single token or several)
- `chunk.accumulated_text` — everything streamed so far
- `chunk.is_final` — `True` on the last chunk
- `chunk.cost_so_far` — running cost estimate
- `chunk.tokens_so_far` — token counts so far
- `chunk.response` — the full `Response` object (only on the final chunk)

Use `stream()` when:
- You are building a chat interface and want text to appear as it is generated
- You are doing long document generation and want to show progress

## Async Streaming: `agent.astream()`

The async version of streaming. Use inside `async def` functions:

```python
import asyncio
from syrin import Agent, Model

agent = Agent(model=Model.mock(), system_prompt="You are helpful.")

async def main():
    print("Response: ", end="")
    async for chunk in agent.astream("Tell me about Python"):
        print(chunk.text, end="", flush=True)
    print()

asyncio.run(main())
```

Use `astream()` when you are in an async framework and want to stream responses to clients (e.g., Server-Sent Events in FastAPI).

## The Response Object

All non-streaming modes return a `Response`. Here is everything it contains:

```python
from syrin import Agent, Budget, Model
from syrin.enums import ExceedPolicy

agent = Agent(
    model=Model.mock(),
    system_prompt="You are helpful.",
    budget=Budget(max_cost=1.00, exceed_policy=ExceedPolicy.WARN),
)
response = agent.run("What is 2 + 2?")

print(f"content:          {response.content[:40]}")
print(f"cost:             ${response.cost:.6f}")
print(f"tokens.input:     {response.tokens.input_tokens}")
print(f"tokens.output:    {response.tokens.output_tokens}")
print(f"tokens.total:     {response.tokens.total_tokens}")
print(f"model:            {response.model}")
print(f"stop_reason:      {response.stop_reason}")
print(f"duration:         {response.duration:.2f}s")
print(f"iterations:       {response.iterations}")
print(f"budget_remaining: ${response.budget_remaining:.6f}")
```

Output:

```
content:          Lorem ipsum dolor sit amet, consectetur
cost:             $0.000041
tokens.input:     7
tokens.output:    25
tokens.total:     32
model:            mock/default
stop_reason:      end_turn
duration:         1.66s
iterations:       1
budget_remaining: $0.999959
```

Key properties:
- `response.content` — the text response
- `response.cost` — USD cost of this run
- `response.tokens` — token counts (input, output, total, cached)
- `response.model` — the model that was used (may differ from configured model if routing or fallback occurred)
- `response.stop_reason` — why the agent stopped (`end_turn` = finished normally, `budget` = hit limit, `max_iterations` = tool loop limit reached)
- `response.duration` — wall-clock seconds
- `response.iterations` — number of LLM calls made (1 for simple responses, more when tools are used)
- `response.budget_remaining` — how much budget is left after this run
- `response.tool_calls` — list of tools called during this run
- `response.trace` — observability spans for the entire run

For structured output responses, `response.output` holds the parsed Pydantic model instance. See [Structured Output](/agent-kit/agent/structured-output).

## Conversation History

Each `run()` call on the same agent instance is part of the same conversation. The agent remembers previous turns:

```python
from syrin import Agent, Model

agent = Agent(model=Model.mock(), system_prompt="You are helpful.")

agent.run("My name is Alex.")
agent.run("I like Python programming.")
response = agent.run("What do you know about me?")
print(response.content[:80])
# With a real model: "Your name is Alex and you like Python programming."
```

To start a fresh conversation, call `agent.reset()`:

```python
agent.reset()
# Now the agent has no memory of the previous conversation
```

## Passing Input Media

For multimodal models, you can pass images, files, and audio alongside text:

```python
from syrin import Agent, Media, Model
from syrin.enums import Media as MediaType

agent = Agent(
    model=Model.mock(),
    system_prompt="You analyze images.",
    input_media={MediaType.IMAGE, MediaType.TEXT},
)

# Pass an image URL with your text
response = agent.run(
    "What is in this image?",
    attachments=["https://example.com/image.png"],
)
```

More on multimodal input in [Input and Output Media](/agent-kit/agent/input-output-media).

## Loop Strategies

The `loop=` parameter controls how the agent reasons through a request. It is a first-class parameter on `Agent`, not an advanced override.

```python
from syrin import Agent, Model
from syrin.loop import ReactLoop, PlanExecuteLoop, SingleShotLoop, CodeActionLoop
from syrin.hitl import HumanInTheLoop
from syrin import ApprovalGate

# SingleShotLoop — one LLM call, no tool iteration (fastest, cheapest)
agent = Agent(model=Model.mock(), system_prompt="You answer questions.", loop=SingleShotLoop())

# ReactLoop — default: reason → act (call tools) → observe → repeat
agent = Agent(model=Model.mock(), system_prompt="You research and act.", loop=ReactLoop())

# PlanExecuteLoop — first plans all steps, then executes them in order
agent = Agent(model=Model.mock(), system_prompt="You decompose and execute.", loop=PlanExecuteLoop())

# CodeActionLoop — writes and runs Python code as its action mechanism
agent = Agent(model=Model.mock(), system_prompt="You write and run code.", loop=CodeActionLoop())

# HumanInTheLoop — pauses before each tool call to request human approval
gate = ApprovalGate(callback=lambda msg, timeout, ctx: input("Approve? [y/n]: ") == "y")
agent = Agent(
    model=Model.mock(),
    system_prompt="You take actions with human oversight.",
    approval_gate=gate,
    loop=HumanInTheLoop(approval_gate=gate),
)
```

All five loop strategies implement the same `Loop` protocol — you can write your own by subclassing it. The loop you choose determines the shape of `response.iterations` and what hooks fire during a run.

See [Loop Strategies](/agent-kit/agent/loops) for the full reference — constructor parameters, comparison table, and how to write a custom loop.

## What's Next

- [Loop Strategies](/agent-kit/agent/loops) — Full reference: SingleShot, ReAct, PlanExecute, CodeAction, HumanInTheLoop
- [Response Object](/agent-kit/agent/response-object) — The full `Response` reference
- [Structured Output](/agent-kit/agent/structured-output) — Get typed, validated responses
- [Streaming](/agent-kit/agent/streaming) — Streaming in depth
- [Tools](/agent-kit/agent/tools) — How tool calls work inside the agent loop
