import torch
import torch.nn as nn


# Action space is MultiDiscrete([3, 3, 3]) — three independent sub-actions,
# each choosing among 3 options (e.g. forward/back/none, left/right/none, rotate).
ACTION_BRANCHES = [3, 3, 3]


class PPONetwork(nn.Module):
    """
    Actor-Critic network for PPO on the soccer_twos multiagent environment.

    - Actor: one logit vector per action branch (MultiDiscrete policy)
    - Critic: single scalar state value V(s)

    Sharing the trunk between actor and critic lets both heads benefit from the
    same learned representation, which is standard practice for PPO.
    """

    def __init__(self, obs_size: int = 336, hidden_size: int = 256):
        super().__init__()

        # Shared feature extractor (same role as QNetwork's fc1/fc2)
        self.trunk = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )

        # Actor head: one Linear per action branch → independent categorical distributions
        self.actor_heads = nn.ModuleList(
            [nn.Linear(hidden_size, n) for n in ACTION_BRANCHES]
        )

        # Critic head: scalar value estimate V(s)
        self.critic_head = nn.Linear(hidden_size, 1)

    def forward(self, obs: torch.Tensor):
        """
        Args:
            obs: (batch, 336) float tensor

        Returns:
            logits: list of (batch, n_i) tensors, one per action branch
            value:  (batch, 1) tensor
        """
        features = self.trunk(obs)
        logits = [head(features) for head in self.actor_heads]
        value = self.critic_head(features)
        return logits, value

    def act(self, obs: torch.Tensor):
        """
        Sample one action per branch from the current policy (used at inference).

        Args:
            obs: (1, 336) float tensor for a single observation

        Returns:
            actions: list of int, one per branch  e.g. [0, 2, 1]
            log_prob: summed log-probability of the sampled action (for PPO loss)
            value: scalar state value
        """
        logits, value = self.forward(obs)
        actions, log_prob = [], 0.0
        for branch_logits in logits:
            dist = torch.distributions.Categorical(logits=branch_logits)
            a = dist.sample()
            log_prob += dist.log_prob(a)
            actions.append(a.item())
        return actions, log_prob, value.squeeze(-1)