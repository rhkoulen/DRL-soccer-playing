import numpy as np
import gym

# Observation structure: 336 dims = 42 rays × 8 values per ray
# Per ray: [ball, blue_goal, purple_goal, blue_agent, purple_agent, wall, has_hit, distance]
RAY_SIZE = 8
NUM_RAYS = 42
BALL_TAG_IDX = 0   # obs[base + 0] == 1.0  →  this ray sees the ball
DIST_IDX = 7       # obs[base + 7]          →  normalized distance [0, 1]

KICK_THRESHOLD = 0.15   # normalized distance; ball this close = agent is "touching" it
KICK_ESCAPE_MIN = 0.10  # ball must jump at least this much to count as a kick

BLUE_AGENT_TAG_IDX = 3   # tag index within each ray for a blue-team agent
PURPLE_AGENT_TAG_IDX = 4 # tag index within each ray for a purple-team agent
# 0.75 × field width in normalized ray units (field width ≈ 0.73 × max ray length)
SPREAD_THRESHOLD = 0.55


class RewardShapingWrapper(gym.Wrapper):
    """
    Wraps a soccer_twos multiagent env to add dense reward shaping on top of
    the sparse 1 goal reward.

    Extra per-step bonuses (kept small so they don't override the true signal):
      1. Ball proximity   — rewards each agent for being near the ball.
      2. Ball progress    — rewards when an agent's closest ray-to-ball distance
                           decreases relative to the previous step (agent is
                           moving toward / staying near the ball).
      3. Possession       — small bonus to whichever agent on a team is closest
                           to the ball, encouraging one player to "own" it.
      4. Kick detection   — larger bonus when an agent was touching the ball
                           (dist < KICK_THRESHOLD) and the ball then jumps away
                           (dist increases by > KICK_ESCAPE_MIN). Encourages the
                           agent to actually make contact and send the ball
                           somewhere; self-play + the goal reward teach direction.
      5. Spread           — small bonus when the agent's teammate is beyond
                           SPREAD_THRESHOLD (≈0.75× field width), discouraging
                           both players from clustering around the ball.

    Weights are intentionally small so convergence is still driven by goals.
    """

    def __init__(
        self,
        env: gym.Env,
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
