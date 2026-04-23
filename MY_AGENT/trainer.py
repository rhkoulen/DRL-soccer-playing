import os
import warnings
warnings.filterwarnings('ignore', category=FutureWarning, module='ray') # yeah man I know I don't have ray[default], take it up with the TAs

import ray
from ray import tune
import matplotlib.pyplot as plt

from my_utils import create_shaped_env, policy_mapping_fn, SelfPlayCallback
from common import *



SAVE_DIR = os.path.dirname(os.path.abspath(__file__))


if __name__ == '__main__':
    ray.init()

    tune.registry.register_env('SoccerShaped', create_shaped_env)

    dummy_env_ = create_shaped_env()
    obs_space = dummy_env_.observation_space
    act_space = dummy_env_.action_space
    dummy_env_.close()

    analysis = tune.run(
        'PPO', # training scheme
        name='PPO_LSTM_attempt',
        config={
            # System Setup
            'num_gpus': NUM_GPUS,
            'num_workers': NUM_CPUS,
            'num_envs_per_worker': NUM_ENVS_PER_WORKER,
            'log_level': 'ERROR',
            'framework': 'torch',
            # Multi-Agent Self-Play RL Setup
            'callbacks': SelfPlayCallback,
            'env': 'SoccerShaped',
            'env_config': ENV_CONFIG,
            'multiagent': {
                'policies': {
                    'default':    (None, obs_space, act_space, {}),
                    'opponent_1': (None, obs_space, act_space, {}),
                    'opponent_2': (None, obs_space, act_space, {}),
                    'opponent_3': (None, obs_space, act_space, {}),
                },
                'policy_mapping_fn': policy_mapping_fn,
                'policies_to_train': ['default'],
            },
            # PPO Setup
            'model': MODEL_CONFIG,
            'rollout_fragment_length': ROLLOUT_LENGTH,
            'train_batch_size': ROLLOUT_LENGTH * NUM_CPUS * NUM_ENVS_PER_WORKER,
            'lr': 5e-5, # can afford to learn slowly
            'gamma': 0.995, # want a high gamma, since we really don't want to discount future goals
            'lambda': 0.9, # since we have a responsive reward wrapper, we shouldn't need a lot of GAE smoothing (could go even lower?)
            'clip_param': 0.2, # [0.8,1.2] is the default PPO range
            'vf_loss_coeff': 0.5, # focus more on the PPO loss than on the value function
            'entropy_coeff': 0.01, # are we even using policy entropy?
            'batch_mode': 'truncate_episodes', # with dense rewards, don't *need* to rollout to a terminal
            'sgd_minibatch_size': 1024, # collate 1024
            'num_sgd_iter': 5, # take several steps
        },
        stop={
            "timesteps_total": 10_000_000,
            # "time_total_s": 1200,
        },
        checkpoint_freq=50,
        checkpoint_at_end=True,
        local_dir='./ray_results',
        # restore='./ray_results/PPO_selfplay_1/PPO_Soccer_ID/checkpoint_00X/checkpoint-X',
    )

    best_trial = analysis.get_best_trial('episode_reward_mean', mode='max')
    print(f'{best_trial=}')
    best_checkpoint = analysis.get_best_checkpoint(trial=best_trial, metric='episode_reward_mean', mode='max')
    print(f'{best_checkpoint=}')
    print('Done training')
