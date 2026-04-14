# models.py
"""
ActorCritic policy for continuous action spaces.

Changes from v1
---------------
* Shared trunk — actor and critic share the first MLP block so
  features extracted for value estimation also help the policy
  (improves sample efficiency).

* RunningMeanStd — online Welford estimator used by train.py to
  normalise observations to ≈ N(0, 1) during training.  Its state
  is saved alongside the policy weights and applied at eval time.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal


# ---------------------------------------------------------------------------
# Online observation normaliser
# ---------------------------------------------------------------------------
class RunningMeanStd:
    """
    Welford's online algorithm for tracking running mean and variance.

    Used to normalise observations to ≈ N(0, 1) during training without
    requiring a fixed dataset of experience up front.

    Usage
    -----
    normalizer = RunningMeanStd(shape=(obs_dim,))
    normalizer.update(batch_of_obs)          # call with each rollout batch
    normed = normalizer.normalize(raw_obs)   # apply at inference time
    """

    def __init__(self, shape: tuple):
        self.mean  = np.zeros(shape, dtype=np.float64)
        self.var   = np.ones(shape,  dtype=np.float64)
        self.count = 1e-4          # small init prevents div-by-zero early on

    def update(self, x: np.ndarray) -> None:
        """Update statistics with a batch of observations (shape: [N, *shape])."""
        if x.ndim == 1:
            x = x[np.newaxis]
        batch_mean  = x.mean(axis=0)
        batch_var   = x.var(axis=0)
        batch_count = x.shape[0]

        total = self.count + batch_count
        delta = batch_mean - self.mean

        new_mean = self.mean + delta * batch_count / total
        m_a      = self.var  * self.count
        m_b      = batch_var * batch_count
        m2       = m_a + m_b + delta ** 2 * self.count * batch_count / total

        self.mean  = new_mean
        self.var   = m2 / total
        self.count = total

    def normalize(self, x: np.ndarray) -> np.ndarray:
        """Return (x - mean) / std, clipped to [-10, 10] for stability."""
        normed = (x - self.mean) / (np.sqrt(self.var) + 1e-8)
        return np.clip(normed, -10.0, 10.0).astype(np.float32)

    def state_dict(self) -> dict:
        return {"mean": self.mean.copy(), "var": self.var.copy(), "count": self.count}

    def load_state_dict(self, d: dict) -> None:
        self.mean  = d["mean"].copy()
        self.var   = d["var"].copy()
        self.count = d["count"]


# ---------------------------------------------------------------------------
# Shared-trunk Actor-Critic
# ---------------------------------------------------------------------------
class ActorCritic(nn.Module):
    """
    Architecture
    ------------
    Shared trunk (two hidden layers, Tanh) → split into:
      * Actor head  → action means (bounded by final Tanh to [-1, 1])
      * Critic head → scalar state value V(s)

    The action distribution is a diagonal Gaussian with learned
    (state-independent) log-std parameters.

    Parameters
    ----------
    obs_size    : flat observation dimension
    action_size : continuous action dimension
    hidden_size : width of each hidden layer (default 256)
    """

    def __init__(self, obs_size: int, action_size: int, hidden_size: int = 256):
        super().__init__()

        # Shared feature extractor
        self.trunk = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
        )

        # Actor head — final Tanh bounds mean to (-1, 1)
        self.actor_head = nn.Sequential(
            nn.Linear(hidden_size, action_size),
            nn.Tanh(),
        )

        # Critic head
        self.critic_head = nn.Linear(hidden_size, 1)

        # Learned log-std (not state-dependent) — init to 0 → std = 1
        self.actor_log_std = nn.Parameter(torch.zeros(action_size))

        self._init_weights()

    # ------------------------------------------------------------------
    # Weight initialisation
    # ------------------------------------------------------------------
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0.0)

        # Small gain for action head → small initial actions
        nn.init.orthogonal_(self.actor_head[0].weight, gain=0.01)
        # Standard gain for value head
        nn.init.orthogonal_(self.critic_head.weight, gain=1.0)

    # ------------------------------------------------------------------
    # Forward helpers
    # ------------------------------------------------------------------
    def _features(self, obs: torch.Tensor) -> torch.Tensor:
        return self.trunk(obs)

    def _distribution(self, obs: torch.Tensor) -> Normal:
        mean = self.actor_head(self._features(obs))
        std  = self.actor_log_std.exp().expand_as(mean)
        return Normal(mean, std)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        """Return critic value V(s)."""
        return self.critic_head(self._features(obs))

    def get_action_and_value(
        self,
        obs:    torch.Tensor,
        action: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample (or evaluate) an action.

        Parameters
        ----------
        obs    : (batch, obs_size)
        action : if provided, evaluate log-prob of this action (PPO update phase)

        Returns
        -------
        action, log_prob, entropy, value  — all shaped for PPO loss computation
        """
        feats = self._features(obs)
        mean  = self.actor_head(feats)
        std   = self.actor_log_std.exp().expand_as(mean)
        dist  = Normal(mean, std)

        if action is None:
            action = dist.sample()

        action   = torch.clamp(action, -1.0, 1.0)
        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy  = dist.entropy().sum(dim=-1)
        value    = self.critic_head(feats)

        return action, log_prob, entropy, value

    @torch.no_grad()
    def get_action(self, obs: torch.Tensor) -> np.ndarray:
        """
        Greedy inference helper for evaluation — no gradient tracking.

        Parameters
        ----------
        obs : (1, obs_size) or (obs_size,)

        Returns
        -------
        numpy array of shape (action_size,)
        """
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        feats  = self._features(obs)
        mean   = self.actor_head(feats)
        std    = self.actor_log_std.exp().expand_as(mean)
        action = Normal(mean, std).sample().squeeze(0)
        return torch.clamp(action, -1.0, 1.0).cpu().numpy()
