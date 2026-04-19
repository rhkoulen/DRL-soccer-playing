import os
from typing import Dict
import pickle
import gym
import numpy as np
import torch
from ray.rllib.agents.ppo import PPOTrainer
from soccer_twos import AgentInterface, EnvType
import ray

class ShapedRewardAgent(AgentInterface):

    def __init__(self, env):
        super().__init__()
        ray.init(ignore_reinit_error=True, include_dashboard=False, num_gpus=0)

        self.agent = PPOTrainer(config={
            "env": None,
            "observation_space": env.observation_space,
            "action_space": env.action_space,
            "num_workers": 0,
            "num_gpus": 0,
            "framework": "torch",
            "model": {
                "vf_share_layers": True,
                "fcnet_hiddens": [256, 256],
                "fcnet_activation": "relu",
            },
        })

        CHECKPOINT_PATH = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "checkpoint.pth"
        )
        with open(CHECKPOINT_PATH, "rb") as f:
            weights = torch.load(f)
        self.agent.get_policy().model.load_state_dict(weights)
        self.agent.get_policy().model.eval()

    def act(self, observation: Dict[int, np.ndarray]) -> Dict[int, np.ndarray]:
        actions = {}
        for player_id, obs in observation.items():
            action = self.agent.compute_single_action(obs)
            actions[player_id] = action
        return actions