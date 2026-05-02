# RoboCup Farm Team (Team 26)

## Authors
  - Shourik Banerjee (sbanerjee332@gatech.edu)
  - Richard Heinrich Koulen (rkoulen3@gatech.edu)
  - Nathaniel Brian Wert (nwert3@gatech.edu)

## About This Agent
Each of us came together to understand the system and lay out how we wanted to approach the problem, then split up to make our own agents. This agent was primarily coded and trained by Shourik.

This agent's policy is defined by a fully connected MLP, with only two hidden layers of size 256. It has an output head for policy and an output head for value. This was trained using PPO over 15M timesteps. This training was conducted with self-play and a custom dense reward landscape.

The RLLib training environment is wrapped with extra per-step bonuses, with small coefficients so as to not override the goal signal.
  1. Ball proximity — rewards each agent for being near the ball.
  2. Ball progress — rewards when an agent's closest ray-to-ball distance decreases relative to the previous step (agent is moving toward / staying near the ball).
  3. Possession — small bonus to whichever agent on a team is closest to the ball, encouraging one player to "own" it.
  4. Kick detection — larger bonus when an agent was touching the ball (dist < KICK_THRESHOLD) and the ball then jumps away (dist increases by > KICK_ESCAPE_MIN). Encourages the agent to actually make contact and send the ball somewhere rather than push it.
  5. Spread — small bonus when the agent's teammate is beyond SPREAD_THRESHOLD, discouraging both players from clustering around each other.

