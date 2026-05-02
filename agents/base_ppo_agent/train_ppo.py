"""
train_soccer.py
Trains a PPO policy on the SoccerTwos multi-agent environment using
Ray RLlib 1.4.0. The Ray dashboard is bound to 127.0.0.1:8265 so you
can monitor training at http://127.0.0.1:8265 in your browser.
"""

import os
import ray
import soccer_twos
import numpy as np
from gym import spaces
from ray import tune
from ray.rllib.env.multi_agent_env import MultiAgentEnv


# ── 1.  Wrap soccer-twos as an RLlib MultiAgentEnv ──────────────────────────

class SoccerTwosMultiAgentEnv(MultiAgentEnv):
    """
    Thin wrapper so RLlib can talk to soccer-twos.
    Soccer-twos exposes 4 agents (IDs 0-3); teams are 0+1 vs 2+3.
    Each agent gets its own obs dict key and submits its own action.
    """

    NUM_AGENTS = 4

    def __init__(self, config=None):
        config = config or {}

        # worker_index is an attribute on EnvContext in Ray 1.4, NOT a dict key.
        # Offset by 1 so real workers don't collide with the dummy env in build_config().
        worker_id = getattr(config, "worker_index", 0) + 1

        self.env = soccer_twos.make(
            worker_id=worker_id,
            time_scale=config.get("time_scale", 20),
            no_graphics=config.get("no_graphics", True),
        )

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


# ── 2.  Boot Ray with the dashboard bound to localhost ──────────────────────

def start_ray():
    ray.init(
        address=None,                # local cluster (not a remote cluster)
        num_cpus=os.cpu_count(),     # expose all cores to Ray
        num_gpus=0,                  # no GPU needed for this env
        include_dashboard=False,
        logging_level="WARNING",     # quieter console; full logs in dashboard
        _metrics_export_port=0,
    )
    print("Ray dashboard → http://127.0.0.1:8265")


# ── 3.  Build the RLlib / Tune config ───────────────────────────────────────

def build_config() -> dict:
    dummy_env = SoccerTwosMultiAgentEnv()
    obs_space = dummy_env.observation_space
    act_space = dummy_env.action_space
    dummy_env.close()

    # All four agents share one policy ("shared_policy").
    # You can split into two team policies later by mapping agents 0-1
    # and 2-3 to separate policy IDs.
    policies = {
        "shared_policy": (None, obs_space, act_space, {}),
    }

    def policy_mapping_fn(agent_id, **kwargs):
        return "shared_policy"

    return {
        # ── environment ────────────────────────────────────────────────
        "env": SoccerTwosMultiAgentEnv,
        "env_config": {
            "time_scale": 20,        # sim runs 20× real-time
            "no_graphics": True,     # headless; flip to False to watch
        },

        # ── multi-agent ────────────────────────────────────────────────
        "multiagent": {
            "policies": policies,
            "policy_mapping_fn": policy_mapping_fn,
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


# ── 4.  Run training ─────────────────────────────────────────────────────────

def main():
    start_ray()

    results = tune.run(
        "PPO",
        config=build_config(),
        stop={
            "timesteps_total": 10_000_000,   # adjust to taste
            # "episode_reward_mean": 0.9,    # or stop on performance
        },
        checkpoint_freq=50,          # save a checkpoint every 50 iters
        checkpoint_at_end=True,
        local_dir="./ray_results",   # where checkpoints & logs land
        name="soccer_twos_ppo",
        verbose=2,                   # 0=silent 1=status 2=episode stats
    )

    best = results.get_best_trial("episode_reward_mean", "max")
    print(f"Best trial: {best.trial_id}")
    print(f"  reward : {best.last_result['episode_reward_mean']:.3f}")
    print(f"  checkpoint: {results.get_best_checkpoint(best, 'episode_reward_mean', 'max')}")


if __name__ == "__main__":
    main()