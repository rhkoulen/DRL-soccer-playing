from typing import Dict

import os
from collections import deque
import gym
import numpy as np
from soccer_twos import AgentInterface
from gym import spaces
import ray
from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.env.multi_agent_env import MultiAgentEnv


ENV = None


class SoccerTwosMultiAgentEnv(MultiAgentEnv):
    """
    Thin wrapper so RLlib can talk to soccer-twos.
    Soccer-twos exposes 4 agents (IDs 0-3); teams are 0+1 vs 2+3.
    Each agent gets its own obs dict key and submits its own action.
    """

    NUM_AGENTS = 4

    def __init__(self, config=None):
        config = config or {}

        self.env = ENV

        self.observation_space = self.env.observation_space
        self.action_space      = self.env.action_space
        self._agent_ids        = set(range(self.NUM_AGENTS))

    # ── gym interface ──────────────────────────────────────────────────────

    def reset(self):
        obs = self.env.reset()  # returns {0: arr, 1: arr, 2: arr, 3: arr}
        return obs

    def step(self, action_dict: dict):
        """action_dict: {agent_id: action, ...} for all live agents."""
        # soccer-twos expects a plain dict {0:.., 1:.., 2:.., 3:..}
        obs, rewards, dones, infos = self.env.step(action_dict)

        # Mark the episode as done for __all__ when any terminal fires
        dones["__all__"] = all(dones.values())
        return obs, rewards, dones, infos

    def close(self):
        self.env.close()


class FrameStackedSoccerEnv(SoccerTwosMultiAgentEnv):
    def __init__(self, config=None):
        config = config or {}
        self.n_stack = config.get("n_stack", 4)
        super().__init__(config)

        single_obs_space = self.env.observation_space
        obs_size = single_obs_space.shape[0]

        low = np.tile(single_obs_space.low, self.n_stack)
        high = np.tile(single_obs_space.high, self.n_stack)
        self.observation_space = spaces.Box(
            low=low, high=high, dtype=single_obs_space.dtype
        )

        self._frames = {
            agent_id: deque(
                [np.zeros(obs_size, dtype=np.float32)] * self.n_stack,
                maxlen=self.n_stack
            )
            for agent_id in range(self.NUM_AGENTS)
        }

    def _update_and_stack(self, obs_dict: dict) -> dict:
        """Push new obs into each agent's deque and return stacked obs."""
        stacked = {}
        for agent_id, agent_obs in obs_dict.items():
            self._frames[agent_id].appendleft(agent_obs)
            stacked[agent_id] = np.concatenate(self._frames[agent_id])
        return stacked
    
    def _reset_frames(self, obs_size: int):
        """Clear all deques back to zeros (call on episode reset)"""
        for agent_id in self._frames:
            self._frames[agent_id] = deque(
                [np.zeros(obs_size, dtype=np.float32)] * self.n_stack,
                maxlen=self.n_stack
            )

    def reset(self):
        obs = super().reset()
        obs_size = next(iter(obs.values())).shape[0]
        self._reset_frames(obs_size)
        return self._update_and_stack(obs)
    
    def step(self, action_dict: dict):
        obs, rewards, dones, infos = super().step(action_dict)
        stacked_obs = self._update_and_stack(obs)
        return stacked_obs, rewards, dones, infos


class ExtendedPPOAgent(AgentInterface):
    def __init__(self, env: gym.Env):
        super().__init__()

        if not ray.is_initialized():
            ray.init(
                ignore_reinit_error=True,
                include_dashboard=False,
                configure_logging=False, # Optional: prevents Ray from hijacking logs
                log_to_driver=False      # Keeps the console clean
            )

        global ENV
        ENV = env

        single_obs_space = env.observation_space
        low = np.tile(single_obs_space.low, 4)
        high = np.tile(single_obs_space.high, 4)
        observation_space = spaces.Box(
            low=low, high=high, dtype=single_obs_space.dtype
        )

        self._frames = {
            i: deque([np.zeros(single_obs_space.shape[0], dtype=np.float32)] * 4, maxlen=4)
            for i in range(4) # Cover all possible agent IDs
        }

        self.name = "Extended PPO Agent"
        config = {
            "env": SoccerTwosMultiAgentEnv,
            "framework": "torch",
            "num_workers": 0,
            "explore": False,
            "model": {
                "fcnet_hiddens": [512, 512],
                "fcnet_activation": "tanh",
                "vf_share_layers": False,
            },
            "multiagent": {
                "policies": {
                    "learner": (None, observation_space, env.action_space, {}),
                    "opponent": (None, observation_space, env.action_space, {
                        "explore": False,
                    }),
                },
                "policy_mapping_fn": lambda agent_id, **kwargs: "learner",
            },
        }
        self.trainer = PPOTrainer(config=config)
        self.trainer.restore(f"{os.path.dirname(os.path.realpath(__file__))}/checkpoint/checkpoint-800")

    def act(self, observation: Dict[int, np.ndarray]) -> Dict[int, np.ndarray]:
        actions = {}
        for agent_id, obs in observation.items():
            self._frames[agent_id].appendleft(obs)
            stacked_obs = np.concatenate(self._frames[agent_id])
            action = self.trainer.compute_action(stacked_obs, policy_id="learner")
            actions[agent_id] = action
        return actions
