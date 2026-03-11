"""Checkpoints -- save and restore agent state.

Demonstrates:
- save_checkpoint() / load_checkpoint() / list_checkpoints()
- CheckpointConfig with storage backends (memory, sqlite, filesystem)
- Trigger types: MANUAL, STEP, TOOL, ERROR, BUDGET
- Checkpoint report for usage stats

Run:
    python examples/12_checkpoints/basic_checkpoint.py
"""

from syrin import Agent, AgentConfig, CheckpointConfig, CheckpointTrigger, Model

model = Model.Almock()

# ---------------------------------------------------------------------------
# 1. Basic checkpoint: save, list, load
# ---------------------------------------------------------------------------
print("-- 1. Basic checkpointing --")

agent = Agent(model=model, system_prompt="You are a research assistant.")
checkpoint_id = agent.save_checkpoint()
print(f"Saved checkpoint: {checkpoint_id}")

checkpoints = agent.list_checkpoints()
print(f"Available checkpoints: {checkpoints}")

# ---------------------------------------------------------------------------
# 2. All trigger types
# ---------------------------------------------------------------------------
print("\n-- 2. Checkpoint trigger types --")

for trigger in [
    CheckpointTrigger.MANUAL,
    CheckpointTrigger.STEP,
    CheckpointTrigger.TOOL,
    CheckpointTrigger.ERROR,
    CheckpointTrigger.BUDGET,
]:
    Agent(
        model=model,
        config=AgentConfig(
            checkpoint=CheckpointConfig(storage="memory", trigger=trigger),
        ),
    )
    print(f"  Created agent with trigger={trigger.value}")

# ---------------------------------------------------------------------------
# 3. Configured checkpoint (sqlite backend, max 5 saves)
# ---------------------------------------------------------------------------
print("\n-- 3. Configured checkpoint --")

config = CheckpointConfig(
    enabled=True,
    storage="sqlite",
    path="/tmp/research_agent.db",
    trigger=CheckpointTrigger.STEP,
    max_checkpoints=5,
    compress=False,
)
agent = Agent(model=model, config=AgentConfig(checkpoint=config))
print(f"  Storage: sqlite, trigger: STEP, max: 5")

# ---------------------------------------------------------------------------
# 4. Named checkpoint: save and restore by tag
# ---------------------------------------------------------------------------
print("\n-- 4. Named checkpoint (save & restore) --")

agent = Agent(model=model, system_prompt="You are a research assistant.")
cp_id = agent.save_checkpoint("my_research")
print(f"  Saved named checkpoint: {cp_id}")

found = agent.list_checkpoints("my_research")
if found:
    agent.load_checkpoint(found[-1])
    print(f"  Restored from: {found[-1]}")

# ---------------------------------------------------------------------------
# 5. Checkpoint report
# ---------------------------------------------------------------------------
print("\n-- 5. Checkpoint report --")

agent = Agent(model=model)
agent.save_checkpoint()
agent.save_checkpoint()
report = agent.get_checkpoint_report()
print(f"  Saves: {report.checkpoints.saves}, Loads: {report.checkpoints.loads}")

# ---------------------------------------------------------------------------
# 6. Get an actual response from a checkpointed agent
# ---------------------------------------------------------------------------
print("\n-- 6. Agent response --")

agent = Agent(model=model, system_prompt="You are a research assistant.")
r = agent.response("What is machine learning?")
print(f"  Response: {r.content[:80]}...")
agent.save_checkpoint("after_first_question")
print("  Checkpoint saved after first question.")

# ---------------------------------------------------------------------------
# Optional: serve with playground UI (requires syrin[serve])
# ---------------------------------------------------------------------------
# agent.serve(port=8000, enable_playground=True, debug=True)
