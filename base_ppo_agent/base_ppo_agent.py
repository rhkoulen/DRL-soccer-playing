from typing import Dict

import os
import gym
import numpy as np
from soccer_twos import AgentInterface
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


class BasePPOAgent(AgentInterface):
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

        self.name = "Base PPO Agent"
        config = {
            "env": SoccerTwosMultiAgentEnv,
            "framework": "torch",
            "num_workers": 0,
            "num_gpus": 0,
            "explore": False,
            "multiagent": {
                "policies": {
                    "shared_policy": (None, env.observation_space, env.action_space, {}),
                },
                "policy_mapping_fn": lambda agent_id, **kwargs: "shared_policy",
            }
        }
        self.trainer = PPOTrainer(config=config)
        self.trainer.restore(f"{os.path.dirname(os.path.realpath(__file__))}/checkpoint/checkpoint-1254")

    def act(self, observation: Dict[int, np.ndarray]) -> Dict[int, np.ndarray]:
        """The act method is called when the agent is asked to act.
        Args:
            observation: a dictionary where keys are team member ids and
                values are their corresponding observations of the environment,
                as numpy arrays.
        Returns:
            action: a dictionary where keys are team member ids and values
                are their corresponding actions, as np.arrays.
        """
        actions = {}
        for agent_id, obs in observation.items():
            action = self.trainer.compute_action(obs, policy_id="shared_policy")
            actions[agent_id] = action
        return actions