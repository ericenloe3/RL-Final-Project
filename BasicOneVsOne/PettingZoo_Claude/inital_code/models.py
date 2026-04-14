# models.py
"""
ActorCritic policy for continuous action spaces.

No PufferLib dependency — works directly with PettingZoo observations.

Architecture
------------
Both actor and critic share a common MLP encoder (two hidden layers with
Tanh activations, which works well for bounded [-1, 1] actions).

The actor outputs the **mean** of a diagonal Gaussian.  Log-std is a
learned parameter (not state-dependent) — the standard starting point for
PPO in continuous control.

The critic outputs a single scalar state-value V(s).
"""

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal


def _make_mlp(in_dim: int, out_dim: int, hidden: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden),
        nn.Tanh(),
        nn.Linear(hidden, hidden),
        nn.Tanh(),
        nn.Linear(hidden, out_dim),
    )


class ActorCritic(nn.Module):
    """
    Parameters
    ----------
    obs_size    : dimensionality of the (flat) observation vector
    action_size : dimensionality of the continuous action vector
    hidden_size : width of each hidden layer
    """

    def __init__(
        self,
        obs_size: int,
        action_size: int,
        hidden_size: int = 256,
    ):
        super().__init__()

        # Actor — outputs action means; final Tanh bounds outputs to (-1, 1)
        self.actor_mean = nn.Sequential(
            *_make_mlp(obs_size, action_size, hidden_size),
            nn.Tanh(),
        )

        # Learned log-std (state-independent) — initialised to 0 → std = 1
        self.actor_log_std = nn.Parameter(torch.zeros(action_size))

        # Critic — outputs scalar value V(s)
        self.critic = _make_mlp(obs_size, 1, hidden_size)

        # Orthogonal initialisation (common PPO best-practice)
        self._init_weights()

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------
    def _init_weights(self):
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.constant_(module.bias, 0.0)
        # Scale down the final actor layer for small initial actions
        final_actor = self.actor_mean[-2]          # last Linear before Tanh
        nn.init.orthogonal_(final_actor.weight, gain=0.01)

    # ------------------------------------------------------------------
    # Forward helpers
    # ------------------------------------------------------------------
    def _distribution(self, obs: torch.Tensor) -> Normal:
        mean = self.actor_mean(obs)
        std  = self.actor_log_std.exp().expand_as(mean)
        return Normal(mean, std)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        """Return V(s) for the critic loss."""
        return self.critic(obs)

    def get_action_and_value(
        self,
        obs: torch.Tensor,
        action: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample (or evaluate) an action and return all quantities needed for
        the PPO loss.

        Parameters
        ----------
        obs    : (batch, obs_size) observation tensor
        action : if provided, evaluate log-prob of *this* action (used during
                 the PPO update phase); otherwise sample a fresh action.

        Returns
        -------
        action   : (batch, action_size)
        log_prob : (batch,)   — sum over action dimensions
        entropy  : (batch,)   — sum over action dimensions
        value    : (batch, 1)
        """
        dist = self._distribution(obs)

        if action is None:
            action = dist.sample()

        # Clamp to valid action range (safety — Tanh already bounds the mean)
        action = torch.clamp(action, -1.0, 1.0)

        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy  = dist.entropy().sum(dim=-1)
        value    = self.critic(obs)

        return action, log_prob, entropy, value

    @torch.no_grad()
    def get_action(self, obs: torch.Tensor) -> np.ndarray:
        """
        Greedy inference helper for evaluation (no gradients).

        Parameters
        ----------
        obs : (1, obs_size) or (obs_size,) tensor

        Returns
        -------
        numpy array of shape (action_size,)
        """
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        dist   = self._distribution(obs)
        action = dist.sample().squeeze(0)
        action = torch.clamp(action, -1.0, 1.0)
        return action.cpu().numpy()
