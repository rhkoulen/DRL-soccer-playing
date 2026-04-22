import numpy as np
from ray.rllib.agents.ppo import PPOTrainer
from soccer_twos import AgentInterface
from common import *

class RLlibLSTMAgent(AgentInterface):
    def __init__(self, env):
        super().__init__()
        self.name = "LSTM Agent"

        config = {
            "num_workers": 0,
            "num_gpus": 0,
            "framework": "torch",
            "env_config": ENV_CONFIG,
            "observation_space": env.observation_space,
            "action_space": env.action_space,
            "model": MODEL_CONFIG,
        }

        self.trainer = PPOTrainer(config=config, env="Soccer")
        self.trainer.restore("checkpoint.pth")
        self.hidden_states = {}

    def act(self, observation):
        actions = {}
        for agent_id, obs in observation.items():
            state = self.hidden_states.get(agent_id, self.trainer.get_policy().get_initial_state())
            action, state, _ = self.trainer.compute_single_action(obs, state=state, explore=False)
            self.hidden_states[agent_id] = state
            actions[agent_id] = action
        return actions

    def reset(self):
        self.hidden_states.clear()