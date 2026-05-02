"""
Train a QMIX multi-agent policy on the SoccerTwos environment using
observation stacking, an opponent queue, and 512x512 hidden layers.
"""

from collections import deque
import itertools
import os
import ray
import numpy as np
from gym import spaces
from ray import tune
from ray.rllib.agents.qmix import QMixTrainer
from ray.tune.registry import register_env
from ray.rllib.env.wrappers.group_agents_wrapper import GroupAgentsWrapper

from train_ppo import SoccerTwosMultiAgentEnv


ACTION_MAP = list(itertools.product(range(3), range(3), range(3)))


def discrete_to_multidiscrete(action: int) -> np.ndarray:
    return np.array(ACTION_MAP[action], dtype=np.int64)


class DiscretizedSoccerEnv(SoccerTwosMultiAgentEnv):
    """
    Converts each agent's action space from MultiDiscrete([3,3,3])
    to Discrete(27) so QMIX can consume it.
    """

    def __init__(self, config=None):
        super().__init__(config)
        # Override action space to Discrete(27)
        self.action_space = spaces.Discrete(len(ACTION_MAP))

    def step(self, action_dict: dict):
        # Convert each agent's Discrete(27) action back to MultiDiscrete
        converted = {
            agent_id: discrete_to_multidiscrete(action)
            for agent_id, action in action_dict.items()
        }
        return super().step(converted)
                

class FrameStackedSoccerEnv(DiscretizedSoccerEnv):
    """
    Stack the last N observations for each agent into a single flat vector.
    """

    def __init__(self, config=None):
        config = config or {}
        self.n_stack = config.get("n_stack", 4)
        super().__init__(config)

        # Original single-frame space
        single_obs_space = self.env.observation_space
        obs_size = single_obs_space.shape[0]

        # New stacked observation space - just N times wider
        low = np.tile(single_obs_space.low, self.n_stack)
        high = np.tile(single_obs_space.high, self.n_stack)
        self.observation_space = spaces.Box(
            low=low, high=high, dtype=single_obs_space.dtype
        )

        # One deque per agent, pre-filled with zeros
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
    

def make_grouped_env(config=None):
    """
    Wraps DiscretizedSoccerEnv with RLlib's GroupAgentsWrapper.
    After grouping, the env exposes two "super-agents":
      "team_0" → agents [0, 1]
      "team_1" → agents [2, 3]

    Each super-agent receives a Tuple observation and takes a Tuple action.
    """
    config = config or {}
    env = DiscretizedSoccerEnv(config)

    single_obs_space = env.observation_space
    single_act_space = env.action_space  # Discrete(27)

    groups = {
        "team_0": [0, 1],
        "team_1": [2, 3],
    }

    # GroupAgentsWrapper needs explicit Tuple spaces for each group
    group_obs_space = spaces.Tuple([single_obs_space, single_obs_space])
    group_act_space = spaces.Tuple([single_act_space, single_act_space])

    return GroupAgentsWrapper(
        env=env,
        groups=groups,
        obs_space=group_obs_space,
        act_space=group_act_space,
    )

def env_creator(config):
    return make_grouped_env(config)


def start_ray():
    ray.init(
        address=None,
        num_cpus=os.cpu_count(),
        num_gpus=0,
        include_dashboard=False,
        logging_level="WARNING",
        object_store_memory=8 * 1024 * 1024 * 1024,
    )

def get_spaces(n_stack: int = 4):
    """
    Return the grouped observation and action spaces without
    launching a Unity environment.
    """
    obs_size   = 336 * n_stack    # stacked flat obs
    n_actions  = 27               # Discrete(27) = 3*3*3 flattened MultiDiscrete

    single_obs = spaces.Box(
        low=-np.inf,
        high=np.inf,
        shape=(obs_size,),
        dtype=np.float32,
    )
    single_act = spaces.Discrete(n_actions)

    # GroupAgentsWrapper exposes a Tuple space per team (2 agents per team)
    group_obs = spaces.Tuple([single_obs, single_obs])
    group_act = spaces.Tuple([single_act, single_act])

    return group_obs, group_act

def build_config() -> dict:
    # Register the env so RLlib workers can recreate it
    register_env("soccer_qmix", env_creator)

    group_obs_space, group_act_space = get_spaces(n_stack=1)

    # Two policies — one per team, same as self-play PPO setup
    policies = {
        "default_policy":  (None, group_obs_space, group_act_space, {}),
    }

    def policy_mapping_fn(agent_id, **kwargs):
        # agent_id is now "team_0" or "team_1" (group names)
        return "default_policy"

    return {
        # ── environment ────────────────────────────────────────────────
        "env": "soccer_qmix",
        "env_config": {
            "n_stack": 1,
            "time_scale": 20,
            "no_graphics": True,
        },

        # ── multi-agent ────────────────────────────────────────────────
        "multiagent": {
            "policies": policies,
            "policy_mapping_fn": policy_mapping_fn,
            "policies_to_train": ["default_policy"],   # only default_policy updates
        },

        # ── QMIX-specific ──────────────────────────────────────────────
        "mixer": "qmix",          # "qmix" or "vdn" (simpler, no mixing net)
        "mixing_embed_dim": 32,   # size of the mixing network
        "double_q": True,         # helps reduce Q-value overestimation

        # Exploration — QMIX uses epsilon-greedy
        "exploration_config": {
            "type": "EpsilonGreedy",
            "initial_epsilon": 1.0,
            "final_epsilon": 0.05,
            "epsilon_timesteps": 500_000,  # decay over 500k steps
        },

        # Replay buffer — QMIX is off-policy
        "train_batch_size": 64,
        "buffer_size": 300,
        "timesteps_per_iteration": 1000,

        # How often to sync target network
        "target_network_update_freq": 500,

        # Learning rate
        "lr": 5e-4,
        "gamma": 0.99,

        # QMIX uses a single worker to collect experience sequentially
        # (the replay buffer handles the parallelism)
        "num_workers": 1,
        "num_gpus": 0,
        "framework": "torch",
        "log_level": "WARN",

        # Minimum steps in buffer before training starts
        "learning_starts": 100,

        # Rollout length — QMIX needs full episodes for credit assignment
        "horizon": 200,
    }

def main():
    start_ray()

    results = tune.run(
        QMixTrainer,            # <-- QMixTrainer not PPO
        config=build_config(),
        stop={
            "timesteps_total": 10_000_000,
        },
        checkpoint_freq=50,
        checkpoint_at_end=True,
        local_dir="./ray_results",
        name="soccer_twos_qmix",
        verbose=2,
        resume=False,
    )

    best = results.get_best_trial("episode_reward_mean", "max")
    print(f"Best trial: {best.trial_id}")


if __name__ == "__main__":
    main()