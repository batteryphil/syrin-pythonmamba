"""Agent Spawning -- a parent agent creates child agents on demand.

Demonstrates:
- parent.spawn(ChildClass, task="...") to spawn and immediately get a response
- parent.spawn(ChildClass) to get a reusable child Agent instance
- Parent-child agent relationships

Run:
    python examples/07_multi_agent/spawn.py
"""

from syrin import Agent, Model

model = Model.Almock()

# ---------------------------------------------------------------------------
# 1. Define parent and child agents
# ---------------------------------------------------------------------------


class Parent(Agent):
    _agent_name = "parent"
    _agent_description = "Coordinator that spawns specialist agents"
    model = model
    system_prompt = "You are a coordinator."


class Child(Agent):
    _agent_name = "child"
    _agent_description = "Specialist agent spawned by parent"
    model = model
    system_prompt = "You are a specialist."


# ---------------------------------------------------------------------------
# 2. Spawn with a task -- returns a Response directly
# ---------------------------------------------------------------------------
print("-- 1. spawn(task=...) returns a Response --")

parent = Parent()
result = parent.spawn(Child, task="What is AI?")
print(f"  Response: {result.content[:80]}...")

# ---------------------------------------------------------------------------
# 3. Spawn without a task -- returns a reusable child Agent
# ---------------------------------------------------------------------------
print("\n-- 2. spawn() returns a child Agent --")

child = parent.spawn(Child)
assert child is not None

r = child.response("Summarize machine learning")
print(f"  Child response: {r.content[:80]}...")

# ---------------------------------------------------------------------------
# Optional: serve with playground UI (requires syrin[serve])
# ---------------------------------------------------------------------------
# parent.serve(port=8000, enable_playground=True, debug=True)
