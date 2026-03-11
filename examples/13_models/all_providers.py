"""Model Providers -- every way to configure a model in Syrin.

Demonstrates:
- Model.Almock for testing (no API key needed)
- Model.OpenAI, Model.Anthropic, Model.Google, Model.Ollama, Model.LiteLLM
- Model.Custom for third-party OpenAI-compatible APIs
- Model settings: temperature, max_tokens, context_window
- Structured output with a Pydantic schema
- Fallback chains for resilience
- Using a model with an Agent

Run:
    python examples/13_models/all_providers.py
"""

from pydantic import BaseModel

from syrin import Agent, Model

# ---------------------------------------------------------------------------
# 1. Almock -- mock model for testing (no API key required)
# ---------------------------------------------------------------------------
print("-- 1. Almock (mock model) --")

mock = Model.Almock(latency_seconds=0.01, lorem_length=50)
agent = Agent(model=mock, system_prompt="You are helpful.")
r = agent.response("Hello!")
print(f"  Response: {r.content[:80]}...")

# ---------------------------------------------------------------------------
# 2. Provider namespaces (shows how to configure each provider)
# ---------------------------------------------------------------------------
print("\n-- 2. Provider namespaces --")

# Each provider takes an API key. Here we use placeholder keys for demo.
# In production, pass real keys or set the corresponding env vars.
openai_model = Model.OpenAI("gpt-4o", api_key="sk-test")
print(f"  OpenAI:    provider={openai_model.provider}, model={openai_model.model_id}")

anthropic_model = Model.Anthropic("claude-sonnet-4-5", api_key="sk-test")
print(f"  Anthropic: provider={anthropic_model.provider}, model={anthropic_model.model_id}")

google_model = Model.Google("gemini-2.0-flash", api_key="test")
print(f"  Google:    provider={google_model.provider}, model={google_model.model_id}")

ollama_model = Model.Ollama("llama3")
print(f"  Ollama:    provider={ollama_model.provider}, model={ollama_model.model_id}")

litellm_model = Model.LiteLLM("openai/gpt-4o", api_key="sk-test")
print(f"  LiteLLM:   provider={litellm_model.provider}, model={litellm_model.model_id}")

# ---------------------------------------------------------------------------
# 3. Model configuration (temperature, max_tokens, context_window)
# ---------------------------------------------------------------------------
print("\n-- 3. Model configuration --")

configured = Model.OpenAI(
    "gpt-4o",
    api_key="sk-test",
    temperature=0.7,
    max_tokens=2048,
    context_window=128000,
)
print(f"  temperature={configured.settings.temperature}, max_tokens={configured.settings.max_output_tokens}")

# ---------------------------------------------------------------------------
# 4. Model.Custom for third-party OpenAI-compatible APIs
# ---------------------------------------------------------------------------
print("\n-- 4. Custom provider (OpenAI-compatible API) --")

custom = Model.Custom(
    "deepseek-chat",
    api_base="https://api.deepseek.com/v1",
    api_key="sk-test",
    context_window=128_000,
)
print(f"  Custom: model={custom.model_id}, api_base={custom.api_base}")

# ---------------------------------------------------------------------------
# 5. Structured output with a Pydantic model
# ---------------------------------------------------------------------------
print("\n-- 5. Structured output --")


class SentimentAnalysis(BaseModel):
    sentiment: str
    confidence: float


structured = Model.OpenAI("gpt-4o", output=SentimentAnalysis, api_key="sk-test")
print(f"  Output schema: {structured.output_type}")

# ---------------------------------------------------------------------------
# 6. Fallback chains
# ---------------------------------------------------------------------------
print("\n-- 6. Fallback chain --")

primary = Model.Anthropic("claude-sonnet-4-5", api_key="sk-test").with_fallback(
    Model.OpenAI("gpt-4o", api_key="sk-test"),
    Model.OpenAI("gpt-4o-mini", api_key="sk-test"),
    Model.Ollama("llama3"),
)
print(f"  Primary: {primary.model_id}, fallbacks: {len(primary.fallback)}")

# ---------------------------------------------------------------------------
# Optional: serve with playground UI (requires syrin[serve])
# ---------------------------------------------------------------------------
# agent = Agent(model=Model.Almock(), system_prompt="You are helpful.")
# agent.serve(port=8000, enable_playground=True, debug=True)
