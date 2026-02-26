# Creating Agents

You can create agents in two ways: **instance-based** (pass arguments to `Agent()`) or **class-based** (declare defaults on a subclass). Both approaches are supported.

## Class-based vs direct instantiation

- **Class-based:** Subclass `Agent`, set `model`, `system_prompt`, `tools`, etc. on the class; instantiate with `MyAgent()`. Use when you have named agent types (e.g. `Researcher`, `Writer`).
- **Direct instantiation:** Call `Agent(model=..., system_prompt=..., tools=[...])` with no subclass. Use for one-off agents or scripts.

## Instance-Based (No Class)

```python
from syrin import Agent
from syrin.model import Model

agent = Agent(
    # model=Model.OpenAI("gpt-4o-mini"),
    model=Model.Almock(),  # No API Key needed
    system_prompt="You are helpful.",
)
response = agent.response("Hello")
```

All configuration is passed to `Agent()` as constructor arguments.

## Class-Based (Subclass)

```python
from syrin import Agent
from syrin.model import Model

class Assistant(Agent):
    # model = Model.OpenAI("gpt-4o-mini")
    model = Model.Almock()  # No API Key needed
    system_prompt = "You are a helpful assistant."

agent = Assistant()
response = agent.response("Hello")
```

Class attributes (`model`, `system_prompt`, `tools`, `budget`, `guardrails`) become defaults. Instance arguments override them.

## Inheritance and MRO

Syrin uses `__init_subclass__` to merge or override class-level attributes along the MRO (Method Resolution Order).

### Merge vs Override

| Attribute        | Behavior | Description                                      |
|------------------|----------|--------------------------------------------------|
| `model`          | Override | First defined in MRO wins                        |
| `system_prompt`  | Override | First defined in MRO wins                        |
| `budget`         | Override | First defined in MRO wins                        |
| `tools`          | Merge    | All tools from the MRO are concatenated          |
| `guardrails`     | Merge    | All guardrails from the MRO are concatenated     |

### Example: Inheritance

```python
from syrin import Agent
from syrin.model import Model
from syrin.tool import tool

@tool
def search(query: str) -> str:
    """Search for information."""
    return f"Results: {query}"

@tool
def calculate(expression: str) -> str:
    """Evaluate a math expression."""
    return str(eval(expression))

class BaseResearcher(Agent):
    # model = Model.OpenAI("gpt-4o")
    model = Model.Almock()  # No API Key needed
    system_prompt = "You are a researcher."
    tools = [search]

class MathResearcher(BaseResearcher):
    model = Model.OpenAI("gpt-4o-mini")  # Overrides parent (use real model when you have a key)
    tools = [calculate]  # Merged with [search] → [search, calculate]

agent = MathResearcher()
# agent has: model=gpt-4o-mini, system_prompt="You are a researcher.",
#            tools=[search, calculate]
```

## Overriding at Instantiation

Constructor arguments override class attributes:

```python
agent = MathResearcher(
    system_prompt="You are a math specialist.",  # Overrides class default
    tools=[calculate],  # Replaces merged tools for this instance
)
```

## Required: Model

`model` is required. It can be set on the class or passed to the constructor. If neither is provided, `TypeError` is raised:

```python
agent = Agent()  # TypeError: Agent requires model
agent = Agent(model=Model.Almock())  # OK (or Model.OpenAI("gpt-4o") when you have a key)
```

## Class-Level Defaults Summary

| Attribute       | Type                       | Default   | Merge? |
|----------------|----------------------------|-----------|--------|
| `model`        | `Model \| ModelConfig`     | None      | No     |
| `system_prompt`| `str`                      | `""`      | No     |
| `tools`        | `list[ToolSpec]`           | `[]`      | Yes    |
| `budget`       | `Budget \| None`           | None      | No     |
| `guardrails`   | `list[Guardrail]`          | `[]`      | Yes    |

## Custom Agent Names (Multi-Agent)

For `DynamicPipeline` and similar patterns, agent names come from the class. You can set a custom name:

```python
class ResearcherAgent(Agent):
    _syrin_name = "research"  # Used in DynamicPipeline
    model = Model.OpenAI("gpt-4o")
    system_prompt = "You are a researcher."
```

If `_syrin_name` is not set, the lowercase class name is used (`ResearcherAgent` → `"researcheragent"`).

## Next Steps

- [Constructor Reference](constructor.md) — All parameters in detail
- [Loop Strategies](loop-strategies.md) — Customizing execution
- [Multi-Agent Patterns](multi-agent-patterns.md) — Pipelines and teams
