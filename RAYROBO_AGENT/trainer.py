import ray
from ray import tune

from my_utils import create_rllib_env
from common import *



if __name__ == "__main__":
    ray.init()

    tune.registry.register_env("Soccer", create_rllib_env)

    analysis = tune.run(
        "PPO",
        name="PPO_LSTM_GUY",
        config={
            # system settings
            "num_gpus": NUM_GPUS,
            "num_workers": NUM_CPUS,
            "num_envs_per_worker": NUM_ENVS_PER_WORKER,
            "log_level": "INFO",
            "framework": "torch",
            # RL setup
            "env": "Soccer",
            "env_config": ENV_CONFIG,
            "model": MODEL_CONFIG,
            "rollout_fragment_length": 500,
            "train_batch_size": 500 * NUM_CPUS * NUM_ENVS_PER_WORKER,
        },
        stop={
            "timesteps_total": 20000000,
            "time_total_s": 1200,
        },
        checkpoint_freq=1,
        checkpoint_at_end=True,
        local_dir="./ray_results",
        # restore="./ray_results/PPO_selfplay_1/PPO_Soccer_ID/checkpoint_00X/checkpoint-X",
    )

    # Gets best trial based on max accuracy across all training iterations.
    best_trial = analysis.get_best_trial("episode_reward_mean", mode="max")
    print(best_trial)
    # Gets best checkpoint for trial based on accuracy.
    best_checkpoint = analysis.get_best_checkpoint(
        trial=best_trial, metric="episode_reward_mean", mode="max"
    )
    print(best_checkpoint)
    print("Done training")
