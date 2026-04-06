"""Configuration Example.

Demonstrates:
- configure() for global settings
- get_config() to access current config
- Setting default model, API key, trace
- Environment variable support (SYRIN_TRACE, SYRIN_API_KEY)
- Thread-safe configuration

Run: python -m examples.15_advanced.config
Visit: http://localhost:8000/playground
Requires: uv pip install syrin[serve]
"""

from __future__ import annotations

import os
import threading
from pathlib import Path

from dotenv import load_dotenv

from syrin import configure, get_config

load_dotenv(Path(__file__).resolve().parent.parent / ".env")


# 1. get_config()
config = get_config()
print(f"Trace: {config.trace}, default_model: {config.default_model}")

# 2. configure()
configure(trace=True)
print(f"After configure(trace=True): {get_config().trace}")
configure(trace=False)

# 3. Default model
from syrin.types import ModelConfig

model_config = ModelConfig(
    name="gpt-4o-mini",
    provider="openai",
    model_id="openai/gpt-4o-mini",
)
configure(default_model=model_config)

# 4. API key
configure(default_api_key="sk-test-key123")
config = get_config()
print(f"API key set: {config.default_api_key is not None}")

# 5. Multiple config values
configure(trace=True, default_api_key="sk-test-key")
configure(trace=False)

# 6. Environment variables
print(f"SYRIN_TRACE: {os.environ.get('SYRIN_TRACE', 'not set')}")

# 7. get/set methods
config = get_config()
config.set(trace=True)
value = config.get("nonexistent", "default_value")
configure(trace=False)

# 8. Thread safety
results: list[str] = []


def update_config(value: bool, thread_id: int) -> None:
    configure(trace=value)
    config = get_config()
    results.append(f"Thread {thread_id}: trace={config.trace}")


threads = []
for i in range(3):
    t = threading.Thread(target=update_config, args=(i % 2 == 0, i))
    threads.append(t)
    t.start()
for t in threads:
    t.join()
configure(trace=False)

if __name__ == "__main__":
    from syrin import Agent, Model

    class ConfigDemoAgent(Agent):
        name = "config-demo"
        description = "Agent with global config demo"
        model = Model.mock()
        system_prompt = "You are helpful."

    agent = ConfigDemoAgent()
    print("Serving at http://localhost:8000/playground")
    agent.serve(port=8000, enable_playground=True, debug=True)
