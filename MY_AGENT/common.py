# Max computer resources
from soccer_twos import EnvType
NUM_GPUS = 0
NUM_CPUS = 1
NUM_ENVS_PER_WORKER = 3
ENV_CONFIG = {
    "num_envs_per_worker": NUM_ENVS_PER_WORKER,
    "variation": EnvType.team_vs_policy,
    "multiagent": False,
    "single_player": True,
    "flatten_branched": True,
    "opponent_policy": lambda *_: 0,
}
MODEL_CONFIG = {
    "use_lstm": True,
    "lstm_cell_size": 512,
    "max_seq_len": 50,
    "fcnet_hiddens": [512, 256],
    "vf_share_layers": True,
}