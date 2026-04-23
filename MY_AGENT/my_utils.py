import numpy as np
from random import uniform as randfloat

import gym
from ray.rllib import MultiAgentEnv
from ray.rllib.agents.callbacks import DefaultCallbacks
import soccer_twos


### Base Environment Wrappers from soccer_twos ###
class RLLibWrapper(gym.core.Wrapper, MultiAgentEnv):
    """
    A RLLib wrapper so our env can inherit from MultiAgentEnv.
    """
    pass


def create_rllib_env(env_config: dict = {}):
    """
    Creates a RLLib environment and prepares it to be instantiated by Ray workers.
    Args:
        env_config: configuration for the environment.
            You may specify the following keys:
            - variation: one of soccer_twos.EnvType. Defaults to EnvType.multiagent_player.
            - opponent_policy: a Callable for your agent to train against. Defaults to a random policy.
    """
    if hasattr(env_config, "worker_index"):
        env_config["worker_id"] = (
            env_config.worker_index * env_config.get("num_envs_per_worker", 1)
            + env_config.vector_index
        )
    env = soccer_twos.make(**env_config)
    # env = TransitionRecorderWrapper(env)
    if "multiagent" in env_config and not env_config["multiagent"]:
        # is multiagent by default, is only disabled if explicitly set to False
        return env
    return RLLibWrapper(env)


def sample_vec(range_dict):
    return [
        randfloat(range_dict["x"][0], range_dict["x"][1]),
        randfloat(range_dict["y"][0], range_dict["y"][1]),
    ]


def sample_val(range_tpl):
    return randfloat(range_tpl[0], range_tpl[1])


def sample_pos_vel(range_dict):
    _s = {}
    if "position" in range_dict:
        _s["position"] = sample_vec(range_dict["position"])
    if "velocity" in range_dict:
        _s["velocity"] = sample_vec(range_dict["velocity"])
    return _s


def sample_player(range_dict):
    _s = sample_pos_vel(range_dict)
    if "rotation_y" in range_dict:
        _s["rotation_y"] = sample_val(range_dict["rotation_y"])
    return _s


### Reward Shaping from Shourik ###
RAY_SIZE = 8
NUM_RAYS = 42
BALL_TAG_IDX = 0
DIST_IDX = 7
KICK_THRESHOLD = 0.15
KICK_ESCAPE_MIN = 0.10
BLUE_AGENT_TAG_IDX = 3
PURPLE_AGENT_TAG_IDX = 4
SPREAD_THRESHOLD = 0.55

class RewardShapingWrapper(RLLibWrapper):
    def __init__(self, env,
        ball_proximity_weight: float = 0.005,
        ball_progress_weight: float = 0.01,
        possession_weight: float = 0.002,
        kick_weight: float = 0.05,
        spread_weight: float = 0.003,
    ):
        super().__init__(env)
        self.ball_proximity_weight = ball_proximity_weight
        self.ball_progress_weight = ball_progress_weight
        self.possession_weight = possession_weight
        self.kick_weight = kick_weight
        self.spread_weight = spread_weight
        self._prev_ball_dist: dict = {}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _min_ball_dist(self, obs: np.ndarray) -> float:
        """Return the normalized distance to the nearest detected ball ray.

        Scans every ray block; if no ray sees the ball returns 1.0 (max).
        """
        min_dist = 1.0
        for i in range(NUM_RAYS):
            base = i * RAY_SIZE
            if obs[base + BALL_TAG_IDX] > 0.5:          # ray detected ball
                min_dist = min(min_dist, float(obs[base + DIST_IDX]))
        return min_dist

    def _teammate_dist(self, obs: np.ndarray, agent_id: int) -> float:
        """Return normalized ray distance to the nearest visible teammate.

        Blue team (ids 0,1) detects teammates via BLUE_AGENT_TAG_IDX.
        Purple team (ids 2,3) detects teammates via PURPLE_AGENT_TAG_IDX.
        Returns 1.0 if teammate is not visible in any ray.
        """
        tag = BLUE_AGENT_TAG_IDX if agent_id < 2 else PURPLE_AGENT_TAG_IDX
        min_dist = 1.0
        for i in range(NUM_RAYS):
            base = i * RAY_SIZE
            if obs[base + tag] > 0.5:
                min_dist = min(min_dist, float(obs[base + DIST_IDX]))
        return min_dist

    def _shape(self, obs: dict, base_rewards: dict) -> dict:
        """Compute shaped rewards for all agents in one step."""
        # --- ball distances per agent ---
        ball_dists = {aid: self._min_ball_dist(o) for aid, o in obs.items()}

        # --- which agent on each team is closest to ball ---
        # team 0 = agent ids 0, 1  |  team 1 = agent ids 2, 3
        team_ids = [[aid for aid in ball_dists if aid < 2],
                    [aid for aid in ball_dists if aid >= 2]]
        closest = {}
        for team in team_ids:
            if team:
                closest[min(team, key=lambda a: ball_dists[a])] = True

        shaped = {}
        for aid, agent_obs in obs.items():
            dist = ball_dists[aid]
            bonus = 0.0

            # 1. proximity: linearly higher the closer the agent is
            bonus += self.ball_proximity_weight * max(0.0, 1.0 - dist)

            # 2. progress: reward shrinking distance vs last step
            prev = self._prev_ball_dist.get(aid, 1.0)
            bonus += self.ball_progress_weight * (prev - dist)
            self._prev_ball_dist[aid] = dist

            # 3. possession: bonus to the closest agent on each team
            if aid in closest:
                bonus += self.possession_weight

            # 4. kick detection: agent was touching the ball and it flew away
            # prev < threshold means agent had contact; dist jumped = kick happened
            if prev < KICK_THRESHOLD and (dist - prev) > KICK_ESCAPE_MIN:
                bonus += self.kick_weight

            # 5. spread: reward when teammate is beyond SPREAD_THRESHOLD distance
            # discourages both agents from crowding the ball
            teammate_dist = self._teammate_dist(agent_obs, aid)
            if teammate_dist > SPREAD_THRESHOLD:
                bonus += self.spread_weight

            shaped[aid] = base_rewards.get(aid, 0.0) + bonus

        return shaped

    def reset(self):
        self._prev_ball_dist = {}
        return self.env.reset()

    def step(self, actions):
        obs, rewards, dones, infos = self.env.step(actions)
        return obs, self._shape(obs, rewards), dones, infos


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


def policy_mapping_fn(agent_id, *_):
    if agent_id == 0:
        return "default"
    return np.random.choice(
        ["default", "opponent_1", "opponent_2", "opponent_3"],
        p=[0.50, 0.25, 0.125, 0.125],
    )