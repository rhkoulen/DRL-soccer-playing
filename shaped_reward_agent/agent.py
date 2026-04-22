import pickle
import os
from typing import Dict

import gym
import numpy as np
import ray
import torch
from ray import tune
from ray.rllib.env.base_env import BaseEnv
from ray.tune.registry import get_trainable_cls
from soccer_twos import AgentInterface

ALGORITHM = "PPO"
CHECKPOINT_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "../ray_results/PPO_shaped_selfplay/PPO_SoccerShaped_7c1b8_00000_0_2026-04-18_16-52-17/checkpoint_000249/checkpoint-249",
)
POLICY_NAME = "default"


class ShapedRewardAgent(AgentInterface):

    def __init__(self, env: gym.Env):
        super().__init__()
        ray.init(ignore_reinit_error=True)

        config_dir = os.path.dirname(CHECKPOINT_PATH)
        config_path = os.path.join(config_dir, "params.pkl")
        if not os.path.exists(config_path):
            config_path = os.path.join(config_dir, "../params.pkl")

        with open(config_path, "rb") as f:
            config = pickle.load(f)

        config["num_workers"] = 0
        config["num_gpus"] = 0
        config["disable_env_checking"] = True
        config["env"] = None
        config["observation_space"] = env.observation_space
        config["action_space"] = env.action_space

        cls = get_trainable_cls(ALGORITHM)
        agent = cls(config=config)
        with open(CHECKPOINT_PATH, "rb") as f:
            checkpoint_data = pickle.load(f)
        worker_state = pickle.loads(checkpoint_data["worker"])
        weights = {k: torch.tensor(v) for k, v in worker_state["state"]["default"]["weights"].items()}
        agent.get_policy(POLICY_NAME).model.load_state_dict(weights)
        agent.get_policy(POLICY_NAME).model.eval()
        self.policy = agent.get_policy(POLICY_NAME)

    def act(self, observation: Dict[int, np.ndarray]) -> Dict[int, np.ndarray]:
        actions = {}
        for player_id in observation:
            actions[player_id], *_ = self.policy.compute_single_action(
                observation[player_id]
            )
        return actions