import pickle
import os
from typing import Dict

import gym
import numpy as np
import torch
import torch.nn as nn
from soccer_twos import AgentInterface

CHECKPOINT_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "PPO_SoccerShaped_7c1b8_00000_0_2026-04-18_16-52-17/checkpoint_000249/checkpoint-249",
)


class FCNet(nn.Module):
    """Mirrors RLlib's FullyConnectedNetwork with vf_share_layers=True."""

    def __init__(self, obs_size: int, action_size: int, hiddens=(256, 256)):
        super().__init__()
        # _hidden_layers
        layers = []
        in_size = obs_size
        for h in hiddens:
            layers.append(nn.Sequential(nn.Linear(in_size, h)))
            in_size = h
        self._hidden_layers = nn.ModuleList(layers)

        # _logits
        self._logits = nn.Sequential(nn.Linear(in_size, action_size))

        # _value_branch (shared — vf_share_layers=True)
        self._value_branch = nn.Sequential(nn.Linear(in_size, 1))

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        x = obs
        for layer in self._hidden_layers:
            x = torch.relu(layer(x))
        return self._logits(x)


class ShapedRewardAgent(AgentInterface):

    def __init__(self, env: gym.Env):
        super().__init__()
        self.name = "ShapedRewardAgent"

        with open(CHECKPOINT_PATH, "rb") as f:
            checkpoint_data = pickle.load(f)
        worker_state = pickle.loads(checkpoint_data["worker"])
        weights = worker_state["state"]["default"]["weights"]

        # Infer sizes from weights
        obs_size = weights["_hidden_layers.0._model.0.weight"].shape[1]
        action_size = weights["_logits._model.0.weight"].shape[0]

        self.model = FCNet(obs_size, action_size)

        # Remap keys: RLlib wraps each layer in a Sequential called `_model`
        # checkpoint: "_hidden_layers.0._model.0.weight"
        # our model:  "_hidden_layers.0.0.weight"
        remapped = {}
        for k, v in weights.items():
            new_k = k.replace("._model.", ".")
            remapped[new_k] = torch.tensor(v, dtype=torch.float32)

        self.model.load_state_dict(remapped)
        self.model.eval()

        # soccer-twos MultiDiscrete [3,3,3] → 9 logits
        if hasattr(env.action_space, "nvec"):
            self.branches = list(env.action_space.nvec)
        else:
            self.branches = None

    @torch.no_grad()
    def act(self, observation: Dict[int, np.ndarray]) -> Dict[int, np.ndarray]:
        actions = {}
        for player_id, obs in observation.items():
            obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            logits = self.model(obs_t)
            if self.branches:
                splits = torch.split(logits, self.branches, dim=-1)
                action = np.array([torch.argmax(s, dim=-1).item() for s in splits])
            else:
                action = torch.argmax(logits, dim=-1).item()
            actions[player_id] = action
        return actions