"""Dependency Injection Example.

Demonstrates:
- Agent(dependencies=...) for runtime dependencies
- Tools receive ctx: RunContext[MyDeps] with ctx.deps
- Testing with mock dependencies (inject MockDeps)

Run: python -m examples.15_advanced.dependency_injection
Visit: http://localhost:8000/playground
Requires: uv pip install syrin[serve]
"""

from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

from syrin import Agent, Model, RunContext, tool

load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")


# -----------------------------------------------------------------------------
# Define dependencies
# -----------------------------------------------------------------------------


@dataclass
class SearchDeps:
    """Dependencies for search tools."""

    search_client: object
    user_id: str


class RealSearchClient:
    """Production search client."""

    def search(self, query: str) -> str:
        return f"Real results for: {query}"


class MockSearchClient:
    """Mock for tests."""

    def search(self, query: str) -> str:
        return f"[MOCK] Results for: {query}"


# -----------------------------------------------------------------------------
# Agent with DI
# -----------------------------------------------------------------------------


@tool
def search(ctx: RunContext[SearchDeps], query: str) -> str:
    """Search for information. Uses injected search client."""
    return ctx.deps.search_client.search(query)


@tool
def get_user_preference(ctx: RunContext[SearchDeps], key: str) -> str:
    """Get user preference. Uses injected user_id."""
    return f"user={ctx.deps.user_id}, {key}=default"


class SearchAgent(Agent):
    """Agent with dependency injection."""

    name = "search-agent"
    description = "Agent with dependency injection"
    model = Model.mock()
    system_prompt = "Use search and get_user_preference. Be concise."
    tools = [search, get_user_preference]


# -----------------------------------------------------------------------------
# Production: real deps
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    real_deps = SearchDeps(
        search_client=RealSearchClient(),
        user_id="alice",
    )
    agent = SearchAgent(dependencies=real_deps)
    result = agent.run("Search for AI trends")
    print(f"Result: {result.content[:150]}...")
    print(f"Cost: ${result.cost:.6f}")

    # -----------------------------------------------------------------------------
    # Testing: mock deps
    # -----------------------------------------------------------------------------
    mock_deps = SearchDeps(
        search_client=MockSearchClient(),
        user_id="test-user",
    )
    test_agent = SearchAgent(dependencies=mock_deps)
    test_result = test_agent.run("Search for testing")
    print(f"\nWith mock: {test_result.content[:100]}...")
    print("Serving at http://localhost:8000/playground")
    agent.serve(port=8000, enable_playground=True, debug=True)
