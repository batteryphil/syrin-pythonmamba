"""Agent Inheritance Example.

Demonstrates:
- Creating agent classes with Python inheritance
- Tool merging from parent classes
- Overriding system prompts and budgets
- Multi-level inheritance

Run: python -m examples.15_advanced.agent_inheritance
Visit: http://localhost:8000/playground
Requires: uv pip install syrin[serve]
"""

from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv

from syrin import Agent, Budget, ExceedPolicy, Model, tool

load_dotenv(Path(__file__).resolve().parent.parent / ".env")


# 1. Basic inheritance
@tool
def repeat(text: str, count: int = 1) -> str:
    """Repeat text count times."""
    return " ".join([text] * count)


class BaseAgent(Agent):
    name = "base-agent"
    description = "Base agent with repeat tool"
    model = Model.mock()
    system_prompt = "You are a helpful assistant."
    tools = [repeat]


class SpecializedAgent(BaseAgent):
    system_prompt = "You are a specialized assistant."


base = BaseAgent()
specialized = SpecializedAgent()
result = specialized.run("Say hello")
print(
    f"Specialized tools: {[t.name for t in specialized._tools]}, response: {result.content[:60]}..."
)


# 2. Adding tools in child
@tool
def greet(name: str) -> str:
    """Greet someone."""
    return f"Hello, {name}!"


class GreetingAgent(BaseAgent):
    tools = [repeat, greet]


agent = GreetingAgent()
print(f"GreetingAgent tools: {[t.name for t in agent._tools]}")


# 3. Budget override
class BudgetBase(Agent):
    model = Model.mock()
    budget = Budget(max_cost=10.0, exceed_policy=ExceedPolicy.WARN)


class TightBudgetAgent(BudgetBase):
    budget = Budget(max_cost=0.10, exceed_policy=ExceedPolicy.WARN)


base = BudgetBase()
tight = TightBudgetAgent()


# 4. Multi-level inheritance
class Level1(Agent):
    model = Model.mock()
    system_prompt = "Level 1 base."


class Level2(Level1):
    system_prompt = "Level 2 specialized."


class Level3(Level2):
    system_prompt = "Level 3 highly specialized."


for cls in [Level1, Level2, Level3]:
    agent = cls()
    print(f"{cls.__name__}: {agent._system_prompt}")

if __name__ == "__main__":
    agent = SpecializedAgent()
    print("Serving at http://localhost:8000/playground")
    agent.serve(port=8000, enable_playground=True, debug=True)
