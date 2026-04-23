import json
import os
import sys
import shutil
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt
import numpy as np
import ray
from ray import tune
from ray.rllib.agents.callbacks import DefaultCallbacks

from utils import create_rllib_env
from TEAM26_ROBOCUP_AGENT.reward_wrapper import RewardShapingWrapper

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))


NUM_ENVS_PER_WORKER = 3

def policy_mapping_fn(agent_id, *args, **kwargs):
        if agent_id == 0:
            return "default"
        return np.random.choice(
            ["default", "opponent_1", "opponent_2", "opponent_3"],
            p=[0.50, 0.25, 0.125, 0.125],
        )

def create_shaped_env(env_config: dict = {}):
    """Creates a soccer_twos env wrapped with dense reward shaping."""
    env = create_rllib_env(env_config)
    return RewardShapingWrapper(env)


class SelfPlayCallback(DefaultCallbacks):
    def on_train_result(self, **info):
        if info["result"]["episode_reward_mean"] > 0.5:
            print("---- Updating opponents ----")
            trainer = info["trainer"]
            trainer.set_weights({
                "opponent_3": trainer.get_weights(["opponent_2"])["opponent_2"],
                "opponent_2": trainer.get_weights(["opponent_1"])["opponent_1"],
                "opponent_1": trainer.get_weights(["default"])["default"],
            })

if __name__ == "__main__":
    ray.init(include_dashboard=False, num_gpus=0)

    tune.registry.register_env("SoccerShaped", create_shaped_env)

    # Probe a temp env to get spaces — same pattern as train_ray_selfplay.py
    temp_env = create_shaped_env()
    obs_space = temp_env.observation_space
    act_space = temp_env.action_space
    temp_env.close()

    lr = 3e-4
    gamma = 0.99
    lambda_ = 0.95
    clip_param = 0.2
    vf_loss_coeff = 0.5
    entropy_coeff = 0.01
    rollout_fragment_length = 5000
    train_batch_size = 60000
    sgd_minibatch_size = 4096
    num_sgd_iter = 10

    analysis = tune.run(
        "PPO",
        name="PPO_shaped_selfplay",
        config={
            # --- system ---
            "num_gpus": 0,
            "num_workers": 4,
            "num_envs_per_worker": NUM_ENVS_PER_WORKER,
            "log_level": "INFO",
            "framework": "torch",
            "callbacks": SelfPlayCallback,
            # --- env ---
            "env": "SoccerShaped",
            "env_config": {"num_envs_per_worker": NUM_ENVS_PER_WORKER},
            # --- multiagent self-play ---
            "multiagent": {
                "policies": {
                    "default":    (None, obs_space, act_space, {}),
                    "opponent_1": (None, obs_space, act_space, {}),
                    "opponent_2": (None, obs_space, act_space, {}),
                    "opponent_3": (None, obs_space, act_space, {}),
                },
                "policy_mapping_fn": policy_mapping_fn,
                "policies_to_train": ["default"],
            },
            # --- PPO network (matches PPONetwork hidden size) ---
            "model": {
                "vf_share_layers": True,
                "fcnet_hiddens": [256, 256],
                "fcnet_activation": "relu",
            },
            # --- PPO hyperparameters ---
            "lr": lr,
            "gamma": gamma,
            "lambda": lambda_,           # GAE smoothing
            "clip_param": clip_param,        # PPO surrogate clip
            "vf_loss_coeff": vf_loss_coeff,
            "entropy_coeff": entropy_coeff,    # encourages exploration
            "rollout_fragment_length": rollout_fragment_length,
            "batch_mode": "complete_episodes",
            "train_batch_size": train_batch_size,
            "sgd_minibatch_size": sgd_minibatch_size,
            "num_sgd_iter": num_sgd_iter,
        },
        stop={
            "timesteps_total": 15_000_000,
            # "time_total_s": 7200,     # 2 hour hard cap
        },
        checkpoint_freq=50,
        checkpoint_at_end=True,
        local_dir="./ray_results",
    )

    # --- export best weights to shaped_reward_agent/checkpoint.pth ---
    best_trial = analysis.get_best_trial("episode_reward_mean", mode="max")
    best_checkpoint = analysis.get_best_checkpoint(
        trial=best_trial, metric="episode_reward_mean", mode="max"
    )

    src = os.path.join(best_checkpoint, "checkpoint")
    dst = os.path.join(OUTPUT_DIR, "checkpoint.pth")
    shutil.copy(src, dst)
    print(f"Best checkpoint saved to {dst}")

    # --- save hyperparameters log ---
    hparams = {
        "algorithm": "PPO",
        "obs_size": 336,
        "hidden_size": 256,
        "lr": lr,
        "gamma": gamma,
        "lambda": lambda_,
        "clip_param": clip_param,
        "vf_loss_coeff": vf_loss_coeff,
        "entropy_coeff": entropy_coeff,
        "train_batch_size": train_batch_size,
        "sgd_minibatch_size": sgd_minibatch_size,
        "num_sgd_iter": num_sgd_iter,
        "rollout_fragment_length": rollout_fragment_length,
        "num_workers": 8,
        "num_envs_per_worker": NUM_ENVS_PER_WORKER,
        "reward_shaping": {
            "ball_proximity_weight": 0.005,
            "ball_progress_weight": 0.01,
            "possession_weight": 0.002,
            "kick_weight": 0.05,
            "spread_weight": 0.003,
            "kick_threshold": 0.15,
            "spread_threshold": 0.55,
        },
        "self_play_update_threshold": 0.5,
        "timesteps_total": 15_000_000,
    }
    hparams_path = os.path.join(OUTPUT_DIR, "hyperparameters.json")
    with open(hparams_path, "w") as f:
        json.dump(hparams, f, indent=2)
    print(f"Hyperparameters saved to {hparams_path}")

    # --- plot training curves ---
    df = analysis.trial_dataframes[best_trial.logdir]
    steps = df["timesteps_total"]
    mean_reward = df["episode_reward_mean"]
    max_reward = df["episode_reward_max"]
    min_reward = df["episode_reward_min"]

    plt.figure(figsize=(10, 5))
    plt.plot(steps, mean_reward, label="Mean Reward", color="blue")
    plt.fill_between(steps, min_reward, max_reward, alpha=0.2, color="blue", label="Min/Max Range")
    plt.xlabel("Timesteps")
    plt.ylabel("Cumulative Reward")
    plt.title("PPO + Reward Shaping — Training Curve")
    plt.legend()
    plt.tight_layout()

    plot_path = os.path.join(OUTPUT_DIR, "training_curve.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Training curve saved to {plot_path}")