"""
Train a PPO multi-agent policy on the SoccerTwos environment using
observation stacking, an opponent queue, tanh activations, 512x512 hidden
layers, and vf_share_layers=False.
"""

from collections import deque
import os
import ray
import soccer_twos
import numpy as np
from gym import spaces
from ray import tune
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.agents.callbacks import DefaultCallbacks

from train_ppo import SoccerTwosMultiAgentEnv
from opponent_pool import OpponentPool


class FrameStackedSoccerEnv(SoccerTwosMultiAgentEnv):
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
    
class SelfPlayCallback(DefaultCallbacks):
    """
    After each training iteration:
      1. Check if the learner's reward beats the pool's worst entry.
         If so, snapshot the learner's weights into the pool.
      2. Every `swap_freq` iterations, sample a random opponent from
         the pool and load its weights into the opponent policy.
    """

    def __init__(self):
        super().__init__()
        self.pool       = OpponentPool(pool_dir="./opponent_pool", max_size=3)
        self.swap_freq  = 20    # swap opponent every N training iterations
        self.iter_count = 0

    def on_train_result(self, *, trainer, result, **kwargs):
        self.iter_count += 1
        reward = result.get("episode_reward_mean", float("-inf"))

        # ── 1. Try to add current learner to the pool ──────────────────
        learner_weights = trainer.get_policy("learner").get_weights()
        inserted = self.pool.try_insert(reward, learner_weights)

        if inserted:
            print(f"[SelfPlayCallback] iter={self.iter_count} "
                  f"reward={reward:.3f} → added to opponent pool")

        # ── 2. Periodically swap the opponent policy ───────────────────
        if self.iter_count % self.swap_freq == 0 and not self.pool.is_empty():
            opponent_weights = self.pool.sample_weights()

            if opponent_weights is not None:
                trainer.set_weights({"opponent": opponent_weights})
                print(f"[SelfPlayCallback] iter={self.iter_count} "
                      f"→ swapped opponent policy")
    
def start_ray():
    ray.init(
        address=None,
        num_cpus=os.cpu_count(),
        num_gpus=0,
        include_dashboard=False,
        logging_level="WARNING",
        _metrics_export_port=0,
    )

def build_config() -> dict:
    N_STACK = 4

    dummy_env = FrameStackedSoccerEnv()
    obs_space = dummy_env.observation_space
    act_space = dummy_env.action_space
    dummy_env.close()

    policies = {
        "learner": (None, obs_space, act_space, {}),
        "opponent": (None, obs_space, act_space, {
            "explore": False,
        }),
    }

    def policy_mapping_fn(agent_id, **kwargs):
        return "learner" if agent_id in (0, 1) else "opponent"

    return {
        "env": FrameStackedSoccerEnv,
        "env_config": {
            "n_stack": N_STACK,
            "time_scale": 20,
            "no_graphics": True,
        },

        # ── multi-agent ────────────────────────────────────────────────
        "multiagent": {
            "policies": policies,
            "policy_mapping_fn": policy_mapping_fn,
            "policies_to_train": ["learner"],
        },

        "callbacks": SelfPlayCallback,

        # Model
        "model": {
            "fcnet_hiddens": [512, 512],
            "fcnet_activation": "tanh",
            "vf_share_layers": False,
        },

        # ── PPO hyperparameters (sensible starting point) ──────────────
        "lr": 3e-4,
        "gamma": 0.99,
        "lambda": 0.95,
        "clip_param": 0.2,
        "entropy_coeff": 0.01,
        "num_sgd_iter": 10,
        "sgd_minibatch_size": 256,
        "train_batch_size": 4000,
        "rollout_fragment_length": 200,

        # ── compute ────────────────────────────────────────────────────
        "num_workers": max(1, os.cpu_count() - 2),  # leave 2 cores free
        "num_envs_per_worker": 1,
        "num_gpus": 0,
        "framework": "torch",

        # ── logging / checkpointing ────────────────────────────────────
        "log_level": "WARN",
        "record_env": False,
    }

def main():
    start_ray()

    results = tune.run(
        "PPO",
        config=build_config(),
        stop={
            "timesteps_total": 10_000_000,
        },
        checkpoint_freq=50,
        checkpoint_at_end=True,
        local_dir="./ray_results",
        name="soccer_twos_ppo_extended",
        verbose=2,
    )

    best = results.get_best_trial("episode_reward_mean", "max")
    print(f"Best trial: {best.trial_id}")
    print(f"  Reward: {best.last_result['episode_reward_mean']:.3f}")
    print(f"  checkpoint: {results.get_best_checkpoint(best, 'episode_reward_mean', 'max')}")


if __name__ == "__main__":
    main()
