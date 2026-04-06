"""Events and Hooks — Observe everything your agent does.

Hooks fire at every lifecycle moment: LLM calls, tool usage, budget checks, etc.
Use them for logging, metrics, debugging, and custom behavior.

Run:
    python examples/10_observability/events_and_hooks.py
"""

from syrin import Agent, Budget, ExceedPolicy, Hook, Memory, Model
from syrin.enums import MemoryType

model = Model.mock()

# ============================================================
# 1. Basic event handlers — track what happens during a run
# ============================================================
events_log: list[str] = []

agent = Agent(
    model=model,
    budget=Budget(max_cost=1.0, exceed_policy=ExceedPolicy.WARN),
    memory=Memory(),
)
agent.events.on(Hook.AGENT_RUN_START, lambda _: events_log.append("run_start"))
agent.events.on(Hook.AGENT_RUN_END, lambda _: events_log.append("run_end"))
agent.events.on(Hook.LLM_REQUEST_START, lambda _: events_log.append("llm_start"))
agent.events.on(Hook.LLM_REQUEST_END, lambda _: events_log.append("llm_end"))

agent.run("Hello!")
print(f"Events fired: {events_log}")
print()

# ============================================================
# 2. Shortcut methods — common hooks have shortcuts
# ============================================================
agent2 = Agent(model=model)
agent2.events.on_start(lambda ctx: print(f"  [start] input={ctx.get('input', '')[:30]}"))
agent2.events.on_complete(lambda ctx: print(f"  [done]  cost=${ctx.get('cost', 0):.6f}"))
agent2.events.on_tool(lambda ctx: print(f"  [tool]  {ctx.get('tool_name', '')}"))

print("=== Shortcut handlers ===")
agent2.run("Hi there!")
print()

# ============================================================
# 3. Multiple handlers on the same hook
# ============================================================
agent3 = Agent(model=model)
calls: list[str] = []
agent3.events.on(Hook.AGENT_RUN_START, lambda _: calls.append("handler_1"))
agent3.events.on(Hook.AGENT_RUN_START, lambda _: calls.append("handler_2"))
agent3.events.on(Hook.AGENT_RUN_START, lambda _: calls.append("handler_3"))
agent3.run("Hi")
print(f"All 3 handlers fired: {calls}")
print()

# ============================================================
# 4. Cost tracking across multiple calls
# ============================================================
total_cost = 0.0


def track_cost(ctx):
    global total_cost
    total_cost += ctx.get("cost", 0)


agent4 = Agent(model=model)
agent4.events.on(Hook.AGENT_RUN_END, track_cost)

for i in range(3):
    agent4.run(f"Question {i + 1}")

print(f"Total cost across 3 calls: ${total_cost:.6f}")
print()

# ============================================================
# 5. Memory events
# ============================================================
memory_ops: list[str] = []
agent5 = Agent(model=model, memory=Memory())
agent5.events.on(Hook.MEMORY_STORE, lambda _: memory_ops.append("store"))
agent5.events.on(Hook.MEMORY_RECALL, lambda _: memory_ops.append("recall"))

agent5.remember("Python is great", memory_type=MemoryType.FACTS)
agent5.recall("Python")
print(f"Memory operations: {memory_ops}")
print()

# ============================================================
# 6. All hooks at a glance
# ============================================================
print(f"Total available hooks: {len(list(Hook))}")
categories = {}
for h in Hook:
    cat = h.value.split(".")[0]
    categories[cat] = categories.get(cat, 0) + 1
for cat, count in sorted(categories.items()):
    print(f"  {cat}: {count} hooks")
