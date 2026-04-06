"""Tests for checkpoint integration with Agent."""

from syrin import Agent, CheckpointConfig, Checkpointer, Model


class TestAgentCheckpointIntegration:
    """Tests for checkpoint integration in Agent."""

    def test_agent_with_checkpoint_config(self):
        """Test Agent accepts CheckpointConfig."""
        agent = Agent(
            model=Model(provider="openai", model_id="gpt-4o-mini"),
            checkpoint=CheckpointConfig(storage="memory"),
        )
        assert agent._checkpointer is not None

    def test_agent_without_checkpoint(self):
        """Test Agent works without checkpoint config."""
        agent = Agent(
            model=Model(provider="openai", model_id="gpt-4o-mini"),
        )
        assert agent._checkpointer is None
        assert agent._checkpoint_config is None

    def test_agent_with_checkpointer_instance(self):
        """Test Agent accepts Checkpointer instance."""
        checkpointer = Checkpointer()
        agent = Agent(
            model=Model(provider="openai", model_id="gpt-4o-mini"),
            checkpoint=checkpointer,
        )
        assert agent._checkpointer is checkpointer
        assert agent._checkpoint_config is None

    def test_agent_checkpoint_disabled(self):
        """Test Agent with disabled checkpoint."""
        agent = Agent(
            model=Model(provider="openai", model_id="gpt-4o-mini"),
            checkpoint=CheckpointConfig(enabled=False),
        )
        assert agent._checkpointer is None

    def test_save_checkpoint(self):
        """Test save_checkpoint method."""
        agent = Agent(
            model=Model(provider="openai", model_id="gpt-4o-mini"),
            checkpoint=CheckpointConfig(storage="memory"),
        )
        checkpoint_id = agent.save_checkpoint()
        assert checkpoint_id is not None

    def test_save_checkpoint_with_name(self):
        """Test save_checkpoint with custom name."""
        agent = Agent(
            model=Model(provider="openai", model_id="gpt-4o-mini"),
            checkpoint=CheckpointConfig(storage="memory"),
        )
        checkpoint_id = agent.save_checkpoint("my_agent")
        assert checkpoint_id == "my_agent_1"

    def test_list_checkpoints(self):
        """Test list_checkpoints method."""
        agent = Agent(
            model=Model(provider="openai", model_id="gpt-4o-mini"),
            checkpoint=CheckpointConfig(storage="memory"),
        )
        agent.save_checkpoint("agent1")
        agent.save_checkpoint("agent1")

        checkpoints = agent.list_checkpoints("agent1")
        assert len(checkpoints) == 2

    def test_list_checkpoints_default_agent_name(self):
        """Test list_checkpoints uses agent class name by default."""
        agent = Agent(
            model=Model(provider="openai", model_id="gpt-4o-mini"),
            checkpoint=CheckpointConfig(storage="memory"),
        )
        agent.save_checkpoint()

        checkpoints = agent.list_checkpoints()
        assert len(checkpoints) == 1

    def test_load_checkpoint(self):
        """Test load_checkpoint method."""
        agent = Agent(
            model=Model(provider="openai", model_id="gpt-4o-mini"),
            checkpoint=CheckpointConfig(storage="memory"),
        )
        checkpoint_id = agent.save_checkpoint()
        result = agent.load_checkpoint(checkpoint_id)
        assert result is True

    def test_load_checkpoint_restores_budget_tracker_and_spent(self):
        """Load checkpoint restores budget_tracker state and Budget._spent."""
        from syrin.budget import Budget
        from syrin.types import CostInfo, TokenUsage

        agent = Agent(
            model=Model(provider="openai", model_id="gpt-4o-mini"),
            budget=Budget(max_cost=10.0),
            checkpoint=CheckpointConfig(storage="memory"),
        )
        agent._budget_tracker.record(
            CostInfo(cost_usd=2.5, model_name="gpt-4o-mini", token_usage=TokenUsage())
        )
        agent._budget._set_spent(2.5)
        assert agent._budget_tracker.current_run_cost == 2.5
        assert agent._budget._spent == 2.5

        checkpoint_id = agent.save_checkpoint(reason="budget")
        assert checkpoint_id is not None

        agent._budget_tracker.reset_run()
        agent._budget._set_spent(0)
        assert agent._budget_tracker.current_run_cost == 0.0
        assert agent._budget._spent == 0.0

        result = agent.load_checkpoint(checkpoint_id)
        assert result is True
        assert agent._budget_tracker.current_run_cost == 2.5
        assert agent._budget._spent == 2.5
        assert agent._budget.remaining == 7.5

    def test_load_checkpoint_nonexistent(self):
        """Test loading nonexistent checkpoint returns False."""
        agent = Agent(
            model=Model(provider="openai", model_id="gpt-4o-mini"),
            checkpoint=CheckpointConfig(storage="memory"),
        )
        result = agent.load_checkpoint("nonexistent")
        assert result is False

    def test_load_checkpoint_no_checkpointer(self):
        """Test loading checkpoint when checkpointer is disabled."""
        agent = Agent(
            model=Model(provider="openai", model_id="gpt-4o-mini"),
        )
        result = agent.load_checkpoint("anything")
        assert result is False

    def test_save_checkpoint_no_checkpointer(self):
        """Test saving checkpoint when checkpointer is disabled."""
        agent = Agent(
            model=Model(provider="openai", model_id="gpt-4o-mini"),
        )
        result = agent.save_checkpoint()
        assert result is None

    def test_list_checkpoints_no_checkpointer(self):
        """Test listing checkpoints when checkpointer is disabled."""
        agent = Agent(
            model=Model(provider="openai", model_id="gpt-4o-mini"),
        )
        result = agent.list_checkpoints()
        assert result == []

    def test_get_checkpoint_report(self):
        """Test get_checkpoint_report method."""
        agent = Agent(
            model=Model(provider="openai", model_id="gpt-4o-mini"),
            checkpoint=CheckpointConfig(storage="memory"),
        )
        agent.save_checkpoint()
        agent.save_checkpoint()

        report = agent.get_checkpoint_report()
        assert report.checkpoints.saves >= 2

    def test_checkpoint_with_sqlite_backend(self):
        """Test Agent with SQLite backend."""
        import os
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            agent = Agent(
                model=Model(provider="openai", model_id="gpt-4o-mini"),
                checkpoint=CheckpointConfig(
                    storage="sqlite",
                    path=db_path,
                ),
            )
            assert agent._checkpointer is not None
            checkpoint_id = agent.save_checkpoint("sqlite_agent")
            assert checkpoint_id is not None

            checkpoints = agent.list_checkpoints("sqlite_agent")
            assert len(checkpoints) == 1

    def test_checkpoint_with_filesystem_backend(self):
        """Test Agent with filesystem backend."""
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            agent = Agent(
                model=Model(provider="openai", model_id="gpt-4o-mini"),
                checkpoint=CheckpointConfig(
                    storage="filesystem",
                    path=tmpdir,
                ),
            )
            assert agent._checkpointer is not None
            checkpoint_id = agent.save_checkpoint("fs_agent")
            assert checkpoint_id is not None

            checkpoints = agent.list_checkpoints("fs_agent")
            assert len(checkpoints) == 1


class TestCheckpointTrigger:
    """Tests for checkpoint trigger configuration."""

    def test_trigger_manual(self):
        """Test manual trigger configuration."""
        from syrin import CheckpointTrigger

        config = CheckpointConfig(trigger=CheckpointTrigger.MANUAL)
        assert config.trigger == CheckpointTrigger.MANUAL

    def test_trigger_step(self):
        """Test step trigger configuration."""
        from syrin import CheckpointTrigger

        config = CheckpointConfig(trigger=CheckpointTrigger.STEP)
        assert config.trigger == CheckpointTrigger.STEP

    def test_trigger_tool(self):
        """Test tool trigger configuration."""
        from syrin import CheckpointTrigger

        config = CheckpointConfig(trigger=CheckpointTrigger.TOOL)
        assert config.trigger == CheckpointTrigger.TOOL

    def test_trigger_error(self):
        """Test error trigger configuration."""
        from syrin import CheckpointTrigger

        config = CheckpointConfig(trigger=CheckpointTrigger.ERROR)
        assert config.trigger == CheckpointTrigger.ERROR

    def test_trigger_budget(self):
        """Test budget trigger configuration."""
        from syrin import CheckpointTrigger

        config = CheckpointConfig(trigger=CheckpointTrigger.BUDGET)
        assert config.trigger == CheckpointTrigger.BUDGET
