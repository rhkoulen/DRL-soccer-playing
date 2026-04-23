# Max computer resources
NUM_GPUS = 0
NUM_CPUS = 10
NUM_ENVS_PER_WORKER = 1
ROLLOUT_LENGTH = 1000
ENV_CONFIG = {
    'num_envs_per_worker': NUM_ENVS_PER_WORKER,
    'multiagent': True,
}
MODEL_CONFIG = {
    'use_lstm': True,
    'lstm_cell_size': 512,
    'max_seq_len': 50,
    'fcnet_hiddens': [256, 256],
    'vf_share_layers': True,
}