# Agent Properties & State

Properties and state exposed on the agent after configuration and runs.

## Properties

### budget_state

Current budget state (limit, remaining, spent, percent_used). None when agent has no run budget.

```python
state = agent.budget_state
if state:
    print(f"Spent ${state.spent:.4f}, {state.percent_used:.1f}% used")
```

### memory

Current memory configuration (persistent or conversation).

```python
mem = agent.memory
```

### conversation_memory

Conversation memory instance, if set.

```python
conv = agent.conversation_memory
```

### persistent_memory

Persistent memory config, if set.

```python
cfg = agent.persistent_memory
```

### context

Context object for message preparation.

```python
ctx = agent.context
```

### context_stats

Stats from the last run (tokens, compression, etc.).

```python
stats = agent.context_stats
```

### rate_limit

Rate limit config, if set.

```python
rl = agent.rate_limit
```

### rate_limit_stats

Rate limit usage from the last run.

```python
stats = agent.rate_limit_stats
```

### report

Aggregated report for the last `response()` or `arun()`.

```python
r = agent.report
r.guardrail
r.context
r.memory
r.budget_remaining
r.budget_used
r.tokens
r.output
r.ratelimits
r.checkpoints
```

## Methods for State Change

### switch_model()

Change the model mid-run (e.g. from a budget threshold).

```python
agent.switch_model(Model.OpenAI("gpt-4o-mini"))
```

### complete()

Call the LLM directly. Used by custom loops.

```python
response = await agent.complete(messages, tools)
```

### execute_tool()

Run a tool by name and arguments. Used by custom loops.

```python
result = await agent.execute_tool("search", {"query": "hello"})
```
