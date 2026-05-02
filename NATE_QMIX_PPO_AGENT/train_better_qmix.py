"""
train_qmix_vs_ppo.py

Train a QMIX policy for team_0 (agents 0, 1) on SoccerTwos against a
frozen PPO opponent for team_1 (agents 2, 3) loaded from a checkpoint.

Team_1 actions are computed inside the environment wrapper, so from
RLlib's perspective this is a single-team problem and QMIX only ever
sees a two-agent grouped env.

Usage:
    python train_qmix_vs_ppo.py --ppo-checkpoint \
        ./ray_results/soccer_twos_ppo/<trial>/checkpoint_050/checkpoint-50
"""

import argparse
import itertools
import os
import pickle

import numpy as np
import ray
from gym import spaces
from ray import tune
from ray.rllib.agents.qmix import QMixTrainer
from ray.rllib.env.wrappers.group_agents_wrapper import GroupAgentsWrapper
from ray.tune.registry import register_env

from train_ppo import SoccerTwosMultiAgentEnv


# ── Ball-position shaping reward ─────────────────────────────────────────────
#
# The field runs along one axis (BALL_AXIS).  One end is team_0's goal
# (negative side) and the other is team_1's goal (positive side).
#
# VERIFY THESE TWO CONSTANTS against your Unity build before training:
#
#   BALL_AXIS
#       Index into the 3-D ball position vector returned by the environment.
#       SoccerTwos typically uses axis 2 (Z).  Set to 0 if your build uses X.
#
#   BALL_TEAM0_GOAL_SIGN
#       +1  →  team_0 defends the positive end of BALL_AXIS
#       -1  →  team_0 defends the negative end  (most common Unity layout)
#
# The reward is linearly interpolated across the full field length:
#
#   ball at team_1's goal  →  +BALL_SHAPING_SCALE   (good for team_0)
#   ball at mid-field      →   0.0
#   ball at team_0's goal  →  -BALL_SHAPING_SCALE   (bad  for team_0)
#
# Keep BALL_SHAPING_SCALE small (≤ 0.05) so it guides without drowning the
# sparse win/loss signal.

BALL_AXIS: int            = 2     # which component of [x, y, z] is the scoring axis
BALL_TEAM0_GOAL_SIGN: int = -1    # sign of team_0's goal end along BALL_AXIS
BALL_FIELD_HALF_LEN: float = 9.0  # half-length of the playable field (Unity units)
BALL_SHAPING_SCALE: float  = 0.02 # max |shaping reward| injected per step


# ── Action discretisation (identical to train_qmix.py) ───────────────────────

ACTION_MAP = list(itertools.product(range(3), range(3), range(3)))  # 27 combos


def discrete_to_multidiscrete(action: int) -> np.ndarray:
    """Map a Discrete(27) index back to a MultiDiscrete([3,3,3]) action."""
    return np.array(ACTION_MAP[action], dtype=np.int64)


# ── PPO policy loader ─────────────────────────────────────────────────────────

# Process-level cache so each Ray worker only loads the checkpoint once,
# even if the env is reset many times within the same worker process.
_PPO_POLICY_CACHE: dict = {}


def load_ppo_policy(checkpoint_path: str):
    """
    Restore 'shared_policy' weights from a PPO checkpoint by reading the
    checkpoint pickle directly — no PPOTrainer or Ray actor is ever spawned.

    Ray 1.x checkpoint files are plain pickles with the structure:
        {
            "worker": <cloudpickle bytes of RolloutWorker.__getstate__()>,
            ...
        }
    The worker state contains:
        {
            "state": { policy_id: policy.get_state(), ... },
            "filters": ...,
        }
    We create a bare PPOTorchPolicy, then call set_state() to load the
    weights — identical to what trainer.restore() does internally, but
    without the overhead of creating workers or environments.
    """
    if checkpoint_path in _PPO_POLICY_CACHE:
        return _PPO_POLICY_CACHE[checkpoint_path]

    from ray.rllib.agents.ppo import DEFAULT_CONFIG as PPO_DEFAULT_CONFIG
    from ray.rllib.agents.ppo.ppo_torch_policy import PPOTorchPolicy

    obs_space = spaces.Box(low=-np.inf, high=np.inf, shape=(336,), dtype=np.float32)
    act_space = spaces.MultiDiscrete([3, 3, 3])

    # Build the minimal config PPOTorchPolicy needs (model defaults, etc.)
    policy_config = dict(PPO_DEFAULT_CONFIG)
    policy_config["framework"] = "torch"

    # Instantiate a fresh policy (random weights at this point)
    policy = PPOTorchPolicy(obs_space, act_space, policy_config)

    # Read the checkpoint file and extract 'shared_policy' state
    with open(checkpoint_path, "rb") as f:
        checkpoint_data = pickle.load(f)

    # The "worker" value is itself a cloudpickle-serialised bytes object
    worker_state = pickle.loads(checkpoint_data["worker"])
    policy_state = worker_state["state"]["shared_policy"]

    # Load weights + optimizer state into the policy
    policy.set_state(policy_state)

    _PPO_POLICY_CACHE[checkpoint_path] = policy
    return policy


# ── Environment ───────────────────────────────────────────────────────────────

class SoccerTwosVsPPOEnv(SoccerTwosMultiAgentEnv):
    """
    SoccerTwos wrapper that hides team_1 from RLlib entirely.

    Exposed to RLlib:
      agents 0, 1  –  Discrete(27) actions  (QMIX trains these)

    Handled internally:
      agents 2, 3  –  MultiDiscrete([3,3,3]) actions computed by the
                      frozen PPO policy at every step

    Required config keys
    --------------------
    ppo_checkpoint_path : str
        Path to the PPO checkpoint to load (e.g. ./ray_results/.../checkpoint-50).
    time_scale : float  (default 20)
    no_graphics : bool  (default True)
    """

    def __init__(self, config=None):
        config = config or {}
        super().__init__(config)

        # Override to Discrete(27) so QMIX can consume the space
        self.action_space = spaces.Discrete(len(ACTION_MAP))

        # Only agents 0 and 1 are visible to RLlib
        self._agent_ids = {0, 1}

        # Load (or reuse) the frozen PPO opponent
        checkpoint = config.get("ppo_checkpoint_path")
        if not checkpoint:
            raise ValueError("config must contain 'ppo_checkpoint_path'")
        self._ppo_policy = load_ppo_policy(checkpoint)

        # Holds the most-recent observations for all 4 agents.
        # Used as a fallback when soccer_twos omits per-agent keys at episode
        # termination (it emits only {"__all__": True} in dones at end-of-ep,
        # leaving individual agent keys absent from obs/rewards/dones/infos).
        self._team1_obs: dict = {}
        self._last_obs: dict = {}
        self._obs_zeros = np.zeros(self.observation_space.shape, dtype=np.float32)

    # ── private helpers ────────────────────────────────────────────────────

    def _get_ball_axis_pos(self, infos: dict):  # -> Optional[float]
        """
        Return the ball's signed position along the scoring axis, or None if
        it cannot be determined.

        Three fallback strategies are tried in order:

        1. info dict  – checked for the keys most commonly set by SoccerTwos
           wrappers: 'ball_position', 'ball_pos', and 'ball_info'.  The value
           is expected to be an array-like [x, y, z]; BALL_AXIS selects the
           relevant component.

        2. Underlying Unity env object  – the base-class attribute `self._env`
           (or `self.env`) is probed for a `ball_position` property, which
           some mlagents wrappers expose directly.

        3. Returns None → shaping reward is silently skipped for that step.
           Add your own extraction logic here if neither strategy works for
           your specific wrapper.
        """
        # Strategy 1: check the per-agent info dict returned by step()
        for aid in (0, 1, 2, 3):
            info = infos.get(aid, {})
            for key in ("ball_position", "ball_pos", "ball_info"):
                pos = info.get(key)
                if pos is not None:
                    try:
                        arr = np.asarray(pos, dtype=np.float32).ravel()
                        if arr.size > BALL_AXIS:
                            return float(arr[BALL_AXIS])
                        # scalar — treat as the scoring axis directly
                        return float(arr[0])
                    except (TypeError, ValueError):
                        pass

        # Strategy 2: probe the underlying environment object
        for attr in ("_env", "env"):
            underlying = getattr(self, attr, None)
            if underlying is None:
                continue
            for prop in ("ball_position", "ball_pos"):
                pos = getattr(underlying, prop, None)
                if pos is not None:
                    try:
                        arr = np.asarray(pos, dtype=np.float32).ravel()
                        if arr.size > BALL_AXIS:
                            return float(arr[BALL_AXIS])
                    except (TypeError, ValueError):
                        pass

        return None  # position unavailable — caller will skip the shaping

    def _ball_shaping_reward(self, infos: dict) -> float:
        """
        Linearly interpolated reward based on ball position along the field.

        Signed so that moving the ball toward the opponent's goal is positive
        for team_0 and moving it toward team_0's own goal is negative.

        The raw axis value is mapped through:

            signed_pos = axis_value * (-BALL_TEAM0_GOAL_SIGN)
                         # positive when ball is on opponent's side

            reward = BALL_SHAPING_SCALE
                     * clamp(signed_pos / BALL_FIELD_HALF_LEN, -1, 1)

        This gives a smooth gradient from -BALL_SHAPING_SCALE (ball at own
        goal) through 0 (mid-field) to +BALL_SHAPING_SCALE (ball at opp goal).
        """
        axis_val = self._get_ball_axis_pos(infos)
        if axis_val is None:
            return 0.0

        # Flip so that "toward opponent goal" is always positive
        signed = axis_val * (-BALL_TEAM0_GOAL_SIGN)
        normalised = np.clip(signed / BALL_FIELD_HALF_LEN, -1.0, 1.0)
        return float(BALL_SHAPING_SCALE * normalised)

    def _compute_team1_actions(self) -> dict:
        """Run the frozen PPO policy for agents 2 and 3."""
        actions = {}
        for agent_id in (2, 3):
            action, _state, _info = self._ppo_policy.compute_single_action(
                self._team1_obs[agent_id]
            )
            actions[agent_id] = action
        return actions

    # ── gym interface ──────────────────────────────────────────────────────

    def reset(self):
        obs = super().reset()            # returns obs for agents 0-3
        self._last_obs = dict(obs)
        self._team1_obs = {2: obs[2], 3: obs[3]}
        return {0: obs[0], 1: obs[1]}   # expose only team_0

    def step(self, action_dict: dict):
        """
        action_dict: {0: Discrete(27), 1: Discrete(27)}  – from QMIX

        Internally merges team_1 actions before forwarding to the base env.
        Returns only team_0 observations, rewards, dones, and infos.
        """
        # Convert team_0 Discrete(27) → MultiDiscrete([3,3,3])
        full_actions = {
            k: discrete_to_multidiscrete(v) for k, v in action_dict.items()
        }
        # Compute and merge frozen PPO actions for team_1
        full_actions.update(self._compute_team1_actions())

        obs, rewards, dones, infos = super().step(full_actions)

        # soccer_twos sometimes omits per-agent keys when "__all__" is True.
        # Fill in safe defaults so we never hit a KeyError downstream.
        all_done = dones.get("__all__", False)
        for aid in range(self.NUM_AGENTS):
            obs.setdefault(aid, self._last_obs.get(aid, self._obs_zeros))
            rewards.setdefault(aid, 0.0)
            dones.setdefault(aid, all_done)
            infos.setdefault(aid, {})

        # Cache latest obs for next step's fallback
        self._last_obs = dict(obs)
        self._team1_obs = {2: obs[2], 3: obs[3]}

        # Compute ball-position shaping reward (same value for both team_0 agents).
        # Positive = ball is closer to team_1's goal; negative = closer to ours.
        shaping = self._ball_shaping_reward(infos)

        return (
            {0: obs[0],                        1: obs[1]},
            {0: rewards[0] + shaping,          1: rewards[1] + shaping},
            {0: dones[0], 1: dones[1], "__all__": all_done},
            {0: infos[0],                      1: infos[1]},
        )


# ── Grouped-env factory (required by QMIX) ───────────────────────────────────

def make_grouped_env(config=None):
    """
    Wrap SoccerTwosVsPPOEnv with RLlib's GroupAgentsWrapper so QMIX
    receives a single 'team_0' super-agent backed by a Tuple obs/act space.
    """
    config = config or {}
    env = SoccerTwosVsPPOEnv(config)

    single_obs = env.observation_space   # Box(336,)
    single_act = env.action_space        # Discrete(27)

    return GroupAgentsWrapper(
        env=env,
        groups={"team_0": [0, 1]},
        obs_space=spaces.Tuple([single_obs, single_obs]),
        act_space=spaces.Tuple([single_act, single_act]),
    )


def env_creator(config):
    return make_grouped_env(config)


# ── Ray init ──────────────────────────────────────────────────────────────────

def start_ray():
    ray.init(
        address=None,
        num_cpus=os.cpu_count(),
        num_gpus=0,
        include_dashboard=False,
        logging_level="WARNING",
        object_store_memory=8 * 1024 * 1024 * 1024,
    )


# ── Static space helpers (no env spawn needed) ────────────────────────────────

def get_spaces():
    """Return the grouped obs/act spaces for the QMIX policy."""
    single_obs = spaces.Box(low=-np.inf, high=np.inf, shape=(336,), dtype=np.float32)
    single_act = spaces.Discrete(27)
    return spaces.Tuple([single_obs, single_obs]), spaces.Tuple([single_act, single_act])


# ── RLlib config ──────────────────────────────────────────────────────────────

def build_config(ppo_checkpoint_path: str) -> dict:
    register_env("soccer_qmix_vs_ppo", env_creator)

    group_obs_space, group_act_space = get_spaces()

    policies = {
        "qmix_policy": (None, group_obs_space, group_act_space, {}),
    }

    def policy_mapping_fn(agent_id, **kwargs):
        # agent_id is "team_0" after GroupAgentsWrapper
        return "qmix_policy"

    return {
        # ── environment ────────────────────────────────────────────────
        "env": "soccer_qmix_vs_ppo",
        "env_config": {
            "ppo_checkpoint_path": ppo_checkpoint_path,
            "time_scale": 20,
            "no_graphics": True,
        },

        # ── multi-agent ────────────────────────────────────────────────
        "multiagent": {
            "policies": policies,
            "policy_mapping_fn": policy_mapping_fn,
            "policies_to_train": ["qmix_policy"],
        },

        # ── QMIX-specific ──────────────────────────────────────────────
        "mixer": "qmix",
        "mixing_embed_dim": 32,
        "double_q": True,

        # Epsilon-greedy exploration
        "exploration_config": {
            "type": "EpsilonGreedy",
            "initial_epsilon": 1.0,
            "final_epsilon": 0.05,
            "epsilon_timesteps": 500_000,
        },

        # Off-policy replay
        "train_batch_size": 64,
        "buffer_size": 1200,
        "timesteps_per_iteration": 1000,
        "target_network_update_freq": 500,
        "learning_starts": 100,

        # Optimisation
        "lr": 5e-4,
        "gamma": 0.99,

        # Compute
        "num_workers": 1,   # QMIX collects experience sequentially
        "num_gpus": 0,
        "framework": "torch",
        "log_level": "WARN",

        # Full-episode rollouts for credit assignment
        "horizon": 200,
    }


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Train QMIX (team_0) against a frozen PPO checkpoint (team_1)."
    )
    parser.add_argument(
        "--ppo-checkpoint",
        required=True,
        metavar="PATH",
        help=(
            "Path to the PPO checkpoint, e.g. "
            "./ray_results/soccer_twos_ppo/<trial>/checkpoint_050/checkpoint-50"
        ),
    )
    args = parser.parse_args()

    start_ray()

    results = tune.run(
        QMixTrainer,
        config=build_config(args.ppo_checkpoint),
        stop={"timesteps_total": 20_000_000},
        checkpoint_freq=50,
        checkpoint_at_end=True,
        local_dir="./ray_results",
        name="soccer_twos_qmix_vs_ppo",
        verbose=2,
        resume=True,
    )

    best = results.get_best_trial("episode_reward_mean", "max")
    print(f"Best trial: {best.trial_id}")


if __name__ == "__main__":
    main()