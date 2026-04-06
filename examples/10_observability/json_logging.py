"""JSON Structured Logging — Production-grade log output.

Demonstrates:
- SyrinHandler with LogFormat.JSON for machine-readable output (log aggregators)
- SyrinHandler with LogFormat.TEXT for human-readable dev output
- --log-level CLI flag (auto-configured on Agent instantiation)
- Setting log level: ERROR / WARNING / INFO / DEBUG
- Output looks like:
    {"level":"INFO","ts":"2026-03-29T10:00:01Z","logger":"syrin","message":"..."}

Run (no flags — text format, INFO level):
    python examples/10_observability/json_logging.py

Run with JSON to stdout (INFO):
    python examples/10_observability/json_logging.py --log-level INFO

Run errors only (quiet production mode):
    python examples/10_observability/json_logging.py --log-level ERROR
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from syrin import Agent, Model  # noqa: E402
from syrin.logging import LogFormat, SyrinHandler  # noqa: E402


def setup_json_logging(level: int = logging.INFO) -> None:
    """Configure syrin root logger with JSON structured output."""
    logger = logging.getLogger("syrin")
    logger.setLevel(level)
    logger.handlers.clear()

    handler = SyrinHandler(format=LogFormat.JSON)
    handler.setLevel(level)
    logger.addHandler(handler)


def setup_text_logging(level: int = logging.INFO) -> None:
    """Configure syrin root logger with human-readable text output."""
    logger = logging.getLogger("syrin")
    logger.setLevel(level)
    logger.handlers.clear()

    handler = SyrinHandler(format=LogFormat.TEXT)
    handler.setLevel(level)
    logger.addHandler(handler)


class SimpleAgent(Agent):
    name = "simple_agent"
    model = Model.mock(latency_min=1, latency_max=3, lorem_length=800, pricing_tier="high")
    system_prompt = "You are a helpful assistant."


def main() -> None:
    # Determine format from CLI or default to TEXT for this demo
    if "--json" in sys.argv:
        print("# JSON structured logging (production mode)\n")
        setup_json_logging(logging.INFO)
    else:
        print("# Text logging (development mode)\n")
        setup_text_logging(logging.INFO)
        print("Run with --json to see JSON output instead.\n")

    # Note: if --log-level was passed, Agent() auto-configures the handler.
    # The manual setup above demonstrates explicit programmatic control.

    agent = SimpleAgent()
    result = agent.run("What is 2+2?")
    print(f"\nResult: {result.content}")
    print(f"Cost:   ${result.cost:.4f}")

    print("\n--- ERROR-only logging demo ---\n")
    # Switch to errors-only (quiet production mode)
    logger = logging.getLogger("syrin")
    logger.setLevel(logging.ERROR)

    result2 = agent.run("What is the capital of France?")
    print(f"Result: {result2.content}")
    print("(No INFO logs above — only ERRORs would appear in production)")


if __name__ == "__main__":
    main()
