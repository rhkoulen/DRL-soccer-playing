import os
from typing import Dict

import numpy as np
import torch
from soccer_twos import AgentInterface

from .model import PPONetwork


class ShapedRewardAgent(AgentInterface):
    """
    Inference wrapper for a PPO policy trained with dense reward shaping.

    Loads a saved PPONetwork checkpoint and implements the act() interface
    required by the soccer_twos evaluation harness. No reward logic is needed
    here — shaped rewards only exist during training.
    """

    def __init__(self, env):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        obs_size = env.observation_space.shape[0]   # 336
        self.model = PPONetwork(obs_size=obs_size).to(self.device)

        weights_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "checkpoint.pth"
        )
        if os.path.isfile(weights_path):
            self.model.load_state_dict(
                torch.load(weights_path, map_location=self.device)
            )
        else:
            print(f"[ShapedRewardAgent] Warning: no checkpoint found at {weights_path}")

        self.model.eval()

    def act(self, observation: Dict[int, np.ndarray]) -> Dict[int, np.ndarray]:
        """
        Args:
            observation: dict mapping player_id → obs array of shape (336,)

        Returns:
            actions: dict mapping player_id → action array of shape (3,)
                     matching the MultiDiscrete([3, 3, 3]) action space
        """
        actions = {}
        with torch.no_grad():
            for player_id, obs in observation.items():
                obs_tensor = torch.from_numpy(obs).float().unsqueeze(0).to(self.device)
                action_list, _, _ = self.model.act(obs_tensor)
                actions[player_id] = np.array(action_list)
        return actions