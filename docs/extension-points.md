# Extension Points

Syrin is built around protocols and extension points. Here’s how to implement each one.

## Model

Subclass `Model` and override `complete()` for a custom LLM.

```python
from syrin.model import Model
from syrin.tool import ToolSpec
from syrin.types import Message, ModelConfig, ProviderResponse

class MyModel(Model):
    async def complete(
        self,
        messages: list[Message],
        model: ModelConfig,
        tools: list[ToolSpec] | None = None,
        **kwargs,
    ) -> ProviderResponse:
        # Your implementation
        ...
```

## Provider

Implement `Provider` and register via the registry. See `syrin/providers/` for examples (OpenAI, Anthropic, LiteLLM, etc.).

```python
from syrin.providers.base import Provider
from syrin.tool import ToolSpec
from syrin.types import Message, ModelConfig, ProviderResponse

class MyProvider(Provider):
    async def complete(
        self,
        messages: list[Message],
        model: ModelConfig,
        tools: list[ToolSpec] | None = None,
        **kwargs,
    ) -> ProviderResponse:
        ...
```

## Loop

Implement `Loop` and use `AgentRunContext` (not the raw Agent).

```python
from syrin.loop import Loop
from syrin.agent._run_context import AgentRunContext

class MyLoop(Loop):
    def run(self, ctx: AgentRunContext, user_input: str):
        # Use ctx.build_messages(), ctx.emit_event(), ctx.complete(), etc.
        ...
```

## Guardrail

Implement `Guardrail` and add to the chain. The `evaluate` method must be async.

```python
from syrin.guardrails import Guardrail, GuardrailContext, GuardrailDecision

class MyGuardrail(Guardrail):
    async def evaluate(self, ctx: GuardrailContext) -> GuardrailDecision:
        ...
```

## ContextManager

Implement the protocol for custom context handling.

```python
from syrin.context import ContextManager, ContextPayload
from syrin.context.config import Context, ContextWindowCapacity

class MyContextManager(ContextManager):
    def prepare(
        self,
        messages: list[dict],
        system_prompt: str,
        tools: list[dict],
        memory_context: str,
        capacity: ContextWindowCapacity,
        context: Context | None = None,
    ) -> ContextPayload:
        ...
```

## ContextCompactor

Implement `ContextCompactorProtocol` for custom compaction (truncation, summarization).

```python
from syrin.context.compactors import ContextCompactorProtocol, CompactionResult

class MyCompactor(ContextCompactorProtocol):
    def compact(self, messages: list[dict], budget: int, counter) -> CompactionResult:
        ...
```

## BudgetStore

Implement `BudgetStore` for custom budget persistence.

```python
from syrin.budget_store import BudgetStore
from syrin.budget import BudgetTracker

class MyBudgetStore(BudgetStore):
    def load(self, key: str) -> BudgetTracker | None:
        ...
    def save(self, key: str, tracker: BudgetTracker) -> None:
        ...
```

## CheckpointBackend

Implement `CheckpointBackendProtocol` for custom checkpoint storage.

```python
from syrin.checkpoint import CheckpointBackendProtocol

class MyCheckpointBackend(CheckpointBackendProtocol):
    def save(self, agent_name: str, state: dict) -> str:
        ...
    def load(self, checkpoint_id: str) -> dict | None:
        ...
    def list(self, agent_name: str) -> list[str]:
        ...
    def delete(self, checkpoint_id: str) -> None:
        ...
```

## Memory Backends

Add to `BACKENDS` in `syrin/memory/backends/__init__.py`. Backends need `add`, `get`, `search`, `list`, `update`, `delete`, `clear`.

## RateLimitBackend

Implement `RateLimitBackend` for custom rate limit persistence.

## SpanExporter

Implement `SpanExporter` for observability (e.g. custom tracing backends).
