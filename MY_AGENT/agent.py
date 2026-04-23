import ray
from ray import tune

import numpy as np
from ray.rllib.agents.ppo import PPOTrainer
from soccer_twos import AgentInterface

from my_utils import create_rllib_env, create_shaped_env, policy_mapping_fn
from common import *


class CustomAgent(AgentInterface):
    def __init__(self, env):
        super().__init__()
        self.name = 'LSTM Agent'
        ray.init()
        tune.registry.register_env('_', create_shaped_env) # doesn't really matter, since ray won't get any workers, I just need to spin it up to get my LSTM

        self.trainer = PPOTrainer(config={
            'framework': 'torch',
            'num_gpus': 0,
            'num_workers': 0,
            'env': 'SoccerShaped',
            'env_config': ENV_CONFIG,
            'multiagent': {
                'policies': {
                    'default':    (None, env.observation_space, env.action_space, {}),
                    'opponent_1': (None, env.observation_space, env.action_space, {}),
                    'opponent_2': (None, env.observation_space, env.action_space, {}),
                    'opponent_3': (None, env.observation_space, env.action_space, {}),
                },
                'policy_mapping_fn': policy_mapping_fn,
                'policies_to_train': ['default'],
            },
            'model': MODEL_CONFIG,
        })

        self.trainer.restore("checkpoint.pth")
        self.policy = self.trainer.get_policy('default')
        self.hidden_states = dict()


    def act(self, observation):
        actions = dict()
        for agent_id, obs in observation.items():
            if agent_id not in self.hidden_states: self.hidden_states[agent_id] = self.policy.get_initial_state()
            action, self.hidden_states[agent_id], _ = self.trainer.compute_single_action(
                obs,
                state=self.hidden_states[agent_id],
                policy_id='default',
                explore=False,
            )
            actions[agent_id] = action
        return actions


    def reset(self):
        self.hidden_states.clear()