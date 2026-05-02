import os
import itertools
from collections import deque
from typing import Dict

import gym
import numpy as np
import ray
from gym import spaces
from ray.rllib.agents.qmix import QMixTrainer
from ray.tune.registry import register_env
from ray.rllib.env.wrappers.group_agents_wrapper import GroupAgentsWrapper
from soccer_twos import AgentInterface
from ray.rllib.env.multi_agent_env import MultiAgentEnv

# Global reference for the environment if RLlib requires it for initialization
ENV = None

# Action mapping: converts a single integer (0-26) back to [x, y, z] multi-discrete actions
ACTION_MAP = list(itertools.product(range(3), range(3), range(3)))

def discrete_to_multidiscrete(action_int: int) -> np.ndarray:
    return np.array(ACTION_MAP[action_int], dtype=np.int64)

class DummyGroupedEnv(MultiAgentEnv):
    """
    A minimal dummy environment to satisfy RLlib's QMixTrainer initialization.
    Must inherit from MultiAgentEnv to bypass RLlib's strict validation checks.
    """
    def __init__(self, config=None):
        super().__init__()
        single_obs_size = config["single_obs_size"] * 4 # 4 stacked frames
        
        single_obs_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(single_obs_size,), dtype=np.float32
        )
        single_act_space = spaces.Discrete(27)
        
        # Grouped spaces (Tuple of 2 for a single team)
        self.observation_space = spaces.Tuple([single_obs_space, single_obs_space])
        self.action_space = spaces.Tuple([single_act_space, single_act_space])
        
        # RLlib requires MultiAgentEnvs to declare their agent IDs
        self._agent_ids = {"team_0", "team_1"}

    # RLlib expects dictionaries to be returned by multi-agent environments
    def reset(self): 
        return {}
        
    def step(self, action_dict): 
        return {}, {}, {}, {}

class ExtendedQMixAgent(AgentInterface):
    def __init__(self, env: gym.Env):
        super().__init__()

        if not ray.is_initialized():
            ray.init(
                ignore_reinit_error=True,
                include_dashboard=False,
                configure_logging=False,
                log_to_driver=False
            )

        global ENV
        ENV = env

        self.name = "Extended QMIX Agent"
        single_obs_space = env.observation_space

        # 2. Register Dummy Environment for Trainer Initialization
        # RLlib needs to know the Tuple spaces QMIX was trained on
        register_env(
            "dummy_grouped_env", 
            lambda c: DummyGroupedEnv({"single_obs_size": single_obs_space.shape[0]})
        )
        
        # Calculate tuple spaces for the policy config
        obs_size = single_obs_space.shape[0]
        group_obs_space = spaces.Tuple([
            spaces.Box(low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float32),
            spaces.Box(low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float32)
        ])
        group_act_space = spaces.Tuple([
            spaces.Discrete(27), spaces.Discrete(27)
        ])

        # 3. Configure the QMIX Trainer
        config = {
            "env": "dummy_grouped_env",
            "framework": "torch",
            "num_workers": 0,
            "explore": False,
            "mixer": "qmix",
            "mixing_embed_dim": 32,
            "multiagent": {
                "policies": {
                    # Both policies must exist to match the checkpoint structure
                    "default_policy": (None, group_obs_space, group_act_space, {}),
                },
                "policy_mapping_fn": lambda agent_id, **kwargs: "default_policy",
            },
        }

        self._states = {
            "team_0": [np.zeros((2, 64), dtype=np.float32)],
            "team_1": [np.zeros((2, 64), dtype=np.float32)]
        }

        self.trainer = QMixTrainer(config=config)
        
        # Update this to your actual QMIX checkpoint path
        checkpoint_path = f"{os.path.dirname(os.path.realpath(__file__))}/checkpoint/checkpoint-7200"
        self.trainer.restore(checkpoint_path)

    def act(self, observation: Dict[int, np.ndarray]) -> Dict[int, np.ndarray]:
        actions = {}
        agent_ids = sorted(list(observation.keys()))

        for i in range(0, len(agent_ids), 2):
            if i + 1 < len(agent_ids):
                a1, a2 = agent_ids[i], agent_ids[i+1]
                team_key = "team_0" if i == 0 else "team_1"
                
                group_obs = (observation[a1], observation[a2])
                
                # Use compute_single_action
                # It returns: (action, list_of_out_states, info_dict)
                act, state_out, _ = self.trainer.compute_action(
                    observation=group_obs,
                    state=self._states[team_key],
                    policy_id="default_policy",
                    full_fetch=True 
                )
                
                # state_out is already a list, e.g., [ndarray(64,)]
                self._states[team_key] = state_out
                
                actions[a1] = discrete_to_multidiscrete(act[0])
                actions[a2] = discrete_to_multidiscrete(act[1])

        return actions