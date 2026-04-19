from typing import Dict
import pickle

import gym
import numpy as np
import torch
from gym_unity.envs import ActionFlattener
from ray.rllib.agents.ppo import PPOTrainer
from soccer_twos import AgentInterface, EnvType
import ray


CHECKPOINT_PATH = "/Users/shourikbanerjee/Desktop/CS 8803/DRL-soccer-playing/ray_results/PPO_SP/PPO_Soccer_1b194_00000_0_2026-04-18_14-55-02/checkpoint_000010/checkpoint-10"


class TestAgent(AgentInterface):

    def __init__(self, env: gym.Env):
        super().__init__()
        ray.init(ignore_reinit_error=True, include_dashboard=False, num_gpus=0)

        self.flattener = ActionFlattener(env.action_space.nvec)

        self.agent = PPOTrainer(config={
            "env": None,
            "observation_space": env.observation_space,
            "action_space": gym.spaces.Discrete(27),
            "num_workers": 0,
            "num_gpus": 0,
            "framework": "torch",
            "model": {
                "vf_share_layers": True,
                "fcnet_hiddens": [512],
            },
        })

        with open(CHECKPOINT_PATH, "rb") as f:
            checkpoint_data = pickle.load(f)
        worker_state = pickle.loads(checkpoint_data["worker"])
        policy_state = worker_state["state"]["default_policy"]
        # load just the torch weights directly, skipping optimizer state
        weights = {k: torch.tensor(v) for k, v in policy_state["weights"].items()}
        self.agent.get_policy().model.load_state_dict(weights)
        self.agent.get_policy().model.eval()


    def act(self, observation: Dict[int, np.ndarray]) -> Dict[int, np.ndarray]:
        actions = {}
        for player_id in observation:
            flat_action = self.agent.compute_single_action(observation[player_id])
            actions[player_id] = self.flattener.lookup_action(flat_action)
        return actions