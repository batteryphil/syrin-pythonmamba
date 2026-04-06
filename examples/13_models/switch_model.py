"""agent.switch_model() — Switch models at runtime.

Demonstrates:
- agent.switch_model(model): switch immediately from any context
- Switch from a budget threshold callback (at 50% usage → cheap model)
- Switch from a hook handler (observe MODEL_SWITCHED event)
- Switch manually from application code
- reason= parameter for tracing why the switch happened

Run:
    python examples/13_models/switch_model.py
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from syrin import Agent, Budget, BudgetThreshold, Model  # noqa: E402
from syrin.enums import Hook  # noqa: E402

# Two models: expensive (powerful) and cheap (fast)
# In a real app: Model.OpenAI("gpt-4o") and Model.OpenAI("gpt-4o-mini")
powerful_model = Model.mock()
cheap_model = Model.mock()  # For demo, both are Model.mock() — in prod use different models


class AdaptiveAgent(Agent):
    name = "adaptive_agent"
    model = powerful_model
    system_prompt = "You are a helpful assistant."
    budget = Budget(
        max_cost=0.50,
        thresholds=[
            # At 50% budget spent: switch to the cheaper model automatically
            BudgetThreshold(
                at=50,
                action=lambda ctx: ctx.agent.switch_model(  # type: ignore[attr-defined]
                    cheap_model,
                    reason="50% budget used — downgrading to cheap model",
                ),
            ),
        ],
    )


def main() -> None:
    print("=" * 60)
    print("agent.switch_model() — Runtime model switching")
    print("=" * 60)

    agent = AdaptiveAgent()

    # --- Observe model switches ---
    def on_model_switched(ctx: object) -> None:
        from_model = getattr(ctx, "from_model", "?")
        to_model = getattr(ctx, "to_model", "?")
        reason = getattr(ctx, "reason", "?")
        print(f"\n  [MODEL_SWITCHED] {from_model} → {to_model}")
        print(f"  reason: {reason}")

    agent.events.on(Hook.MODEL_SWITCHED, on_model_switched)

    # --- 1. Manual switch ---
    print("\n1. Manual switch from application code")
    print(f"   Before: {agent._model_config.model_id if agent._model_config else 'not set'}")  # type: ignore[attr-defined]
    agent.switch_model(cheap_model, reason="manual override for testing")
    print(f"   After:  {agent._model_config.model_id if agent._model_config else 'not set'}")  # type: ignore[attr-defined]

    # Switch back to powerful
    agent.switch_model(powerful_model, reason="restored to powerful model")

    # --- 2. Run with budget threshold trigger ---
    print("\n2. Automatic switch at 50% budget threshold")
    agent2 = AdaptiveAgent()
    agent2.events.on(Hook.MODEL_SWITCHED, on_model_switched)

    for i in range(3):
        r = agent2.run(f"Task {i + 1}: What is {i + 1} * {i + 1}?")
        pct = agent2.budget_state.percent_used if agent2.budget_state else 0
        print(f"   Run {i + 1}: cost=${r.cost:.4f}  budget={pct:.0f}%")

    # --- 3. Switch from hook handler ---
    print("\n3. Switch from hook handler (MODEL_SWITCHED hook)")
    agent3 = Agent(model=powerful_model, system_prompt="Be concise.")
    switch_done = False

    def hook_switch(ctx: object) -> None:
        nonlocal switch_done
        if not switch_done:
            switch_done = True
            agent3.switch_model(cheap_model, reason="hook-triggered switch")

    agent3.events.on(Hook.AGENT_RUN_START, hook_switch)
    agent3.events.on(Hook.MODEL_SWITCHED, on_model_switched)

    r3 = agent3.run("Hello!")
    print(f"   Result: {r3.content[:80]}")

    print("\nDone.")


if __name__ == "__main__":
    main()
