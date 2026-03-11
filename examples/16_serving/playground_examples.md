# Playground Examples

Run any example and visit **http://localhost:8000/playground** to test the Syrin playground UI.

| Example | Run | Description |
|---------|-----|-------------|
| Single agent | `python -m examples.16_serving.playground_single` | One agent with budget |
| Multi-agent | `python -m examples.16_serving.playground_multi_agent` | Researcher + Writer, agent selector |
| Dynamic pipeline | `python -m examples.16_serving.playground_dynamic_pipeline` | 5 agents, pipeline.serve() directly, Almock custom replies so agents spawn, full trace sidebar |
| Guardrails | `python -m examples.16_serving.playground_guardrails` | ContentFilter (blocks spam, scam) |
| Checkpoints | `python -m examples.16_serving.playground_checkpoints` | Step checkpoints, state persistence |
| Original | `python -m examples.16_serving.playground_serve` | Same as playground_single |


