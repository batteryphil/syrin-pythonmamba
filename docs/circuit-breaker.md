# Circuit Breaker

## What is a circuit breaker?

Think of the circuit breaker in your home. When a wire overloads or something shorts out, it **trips** — it cuts power to stop the problem from spreading. You don’t keep trying the same faulty outlet; the breaker protects the rest of the house.

A **software circuit breaker** does the same thing for your AI agent. When the LLM provider (e.g. OpenAI, Anthropic) keeps failing — network issues, rate limits, outages — the circuit breaker trips. It stops sending more requests to the failing service and can switch to a backup instead of repeatedly failing.

---

## When should you use it?

Use a circuit breaker when your agent talks to external LLM APIs and you care about reliability.

Examples:

- **Production apps** — Users expect the app to keep working when one provider is down.
- **Costly retries** — Avoid hammering a failing API (e.g. rate limits, 429s).
- **Fallback to local or cheaper models** — When the main API fails, switch to e.g. Ollama on your own machine.

---

## Why use it?

Without a circuit breaker:

1. The primary API fails (timeout, 5xx, 429).
2. Your app retries again and again.
3. Each retry wastes time and may cost money.
4. Users see long waits or repeated errors.
5. The failing provider may stay overloaded.

With a circuit breaker:

1. After a few consecutive failures, the circuit trips.
2. New requests stop hitting the failing provider.
3. If you configured a fallback model, they go there instead.
4. After a short cooldown, one test request checks if the main provider is back.
5. If it succeeds, normal traffic resumes; if not, the circuit stays open and continues using the fallback.

Result: faster failures, less load on a broken API, and a smoother experience when you have a fallback.

---

## Quick start

```python
from syrin import Agent, CircuitBreaker, Model

# Fallback when the main API is down (e.g. local Ollama)
fallback = Model.Ollama("llama3.1")

cb = CircuitBreaker(
    failure_threshold=5,   # Trip after 5 consecutive failures
    recovery_timeout=60,   # Wait 60 seconds before trying the main API again
    fallback=fallback,
)

agent = Agent(
    model=Model.Anthropic("claude-sonnet"),
    system_prompt="You are helpful.",
    circuit_breaker=cb,
)

agent.response("Hello")
```

---

## How it works: states

The circuit breaker moves between three states, like a light switch:

| State      | Meaning                                      | What happens                     |
|-----------|-----------------------------------------------|----------------------------------|
| **CLOSED**   | Everything is fine                            | All requests go to the main API  |
| **OPEN**     | Too many failures — circuit tripped           | Requests use fallback or error   |
| **HALF_OPEN**| Cooldown over; testing if the main API is back| One probe request is allowed     |

Flow:

1. **CLOSED** — Normal operation. Each failure increments a counter; each success resets it.
2. When failures reach `failure_threshold`, the circuit **trips** to **OPEN**.
3. In **OPEN**, no requests go to the main API. After `recovery_timeout` seconds, it moves to **HALF_OPEN**.
4. In **HALF_OPEN**, one probe request is sent to the main API. Success → **CLOSED**. Failure → **OPEN** again.

---

## Configuration

| Option             | Default | Description |
|--------------------|---------|-------------|
| `failure_threshold`| 5       | Consecutive failures before tripping |
| `recovery_timeout` | 60      | Seconds to wait before trying the main API again |
| `half_open_max`    | 1       | Probe requests allowed in HALF_OPEN |
| `fallback`         | None    | Model (or model ID string) to use when circuit is OPEN |
| `on_trip`          | None    | Callback when the circuit trips to OPEN |

---

## Fallback model

When the circuit is **OPEN**, you can:

- **Use a fallback model** — e.g. local Ollama or a cheaper API:

```python
cb = CircuitBreaker(
    failure_threshold=3,
    fallback="ollama/llama3.1",  # or Model.Ollama("llama3.1")
)
```

- **Not configure a fallback** — in that case, the agent raises `CircuitBreakerOpenError` instead of making a request.

---

## Hooks

You can react when the circuit trips or recovers:

| Hook               | When it fires                          |
|--------------------|----------------------------------------|
| `Hook.CIRCUIT_TRIP` | Circuit trips to OPEN                  |
| `Hook.CIRCUIT_RESET`| Circuit recovers (HALF_OPEN → CLOSED)  |

```python
agent.events.on(Hook.CIRCUIT_TRIP, lambda ctx: alert_ops(ctx))
agent.events.on(Hook.CIRCUIT_RESET, lambda ctx: log_recovery())
```

---

## Summary

| Question | Answer |
|----------|--------|
| **What is it?** | A pattern that stops calling a failing service after too many failures. |
| **When to use it?** | Any agent that talks to external LLM APIs, especially in production. |
| **Why use it?** | To fail fast, reduce load on broken APIs, and keep working with a fallback model. |
| **How does it help?** | Trips after N failures, uses fallback or raises, then checks recovery after a timeout. |
