# models.py
"""
ActorCritic policy and online observation normaliser.

Architecture
------------
Shared MLP trunk (two hidden layers, Tanh activations) feeds into:
  * Actor head  → action means bounded to (-1, 1) via a final Tanh
  * Critic head → scalar state value V(s)

The action distribution is a diagonal Gaussian with learned
(state-independent) log-std parameters.

RunningMeanStd
--------------
Welford online estimator used to normalise observations to ≈ N(0,1)
during training.  Its state is saved inside the checkpoint so that
evaluate.py can apply the same transform at inference time without any
separate calibration step.

Compatible obs_dim: 62  (as produced by my_game_env.py v3)
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

    Parameters
    ----------
    shape : tuple — shape of a single observation vector
    """

    def __init__(self, shape: tuple):
        self.mean  = np.zeros(shape, dtype=np.float64)
        self.var   = np.ones(shape,  dtype=np.float64)
        self.count = 1e-4           # small init to avoid div-by-zero early on

    def update(self, x: np.ndarray) -> None:
        """Update statistics with a batch of RAW observations (N, *shape)."""
        if x.ndim == 1:
            x = x[np.newaxis]
        n           = x.shape[0]
        batch_mean  = x.mean(axis=0)
        batch_var   = x.var(axis=0)
        total       = self.count + n
        delta       = batch_mean - self.mean
        new_mean    = self.mean + delta * n / total
        m2          = (self.var * self.count + batch_var * n
                       + delta ** 2 * self.count * n / total)
        self.mean   = new_mean
        self.var    = m2 / total
        self.count  = total

    def normalize(self, x: np.ndarray) -> np.ndarray:
        """Return (x − mean) / std, clipped to [−10, 10] for stability."""
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
    Parameters
    ----------
    obs_size    : flat observation dimension (62 for v3 env)
    action_size : continuous action dimension (2)
    hidden_size : width of each hidden layer
    """

    def __init__(self, obs_size: int, action_size: int, hidden_size: int = 256):
        super().__init__()

        # Shared feature extractor — Tanh works well with bounded observations
        self.trunk = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
        )

        # Actor head: final Tanh bounds the mean to (−1, 1)
        self.actor_head = nn.Sequential(
            nn.Linear(hidden_size, action_size),
            nn.Tanh(),
        )

        # Critic head: scalar value estimate
        self.critic_head = nn.Linear(hidden_size, 1)

        # Learned log-std (state-independent), initialised to 0 → std = 1
        self.actor_log_std = nn.Parameter(torch.zeros(action_size))

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0.0)
        # Small gain for action head so initial actions are near zero
        nn.init.orthogonal_(self.actor_head[0].weight, gain=0.01)
        nn.init.orthogonal_(self.critic_head.weight,   gain=1.0)

    # ------------------------------------------------------------------
    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        return self.critic_head(self.trunk(obs))

    def get_action_and_value(
        self,
        obs:    torch.Tensor,
        action: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample (or evaluate) an action.

        Returns action, log_prob, entropy, value — all shapes expected by PPO.
        """
        feats  = self.trunk(obs)
        mean   = self.actor_head(feats)
        std    = self.actor_log_std.exp().expand_as(mean)
        dist   = Normal(mean, std)

        if action is None:
            action = dist.sample()
        action   = torch.clamp(action, -1.0, 1.0)

        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy  = dist.entropy().sum(dim=-1)
        value    = self.critic_head(feats)

        return action, log_prob, entropy, value

    @torch.no_grad()
    def get_action(self, obs: torch.Tensor,
                   deterministic: bool = False) -> np.ndarray:
        """Inference-only: returns a numpy action array.

        Parameters
        ----------
        deterministic : bool
            If True, return the distribution mean (no sampling).
            Use this for evaluation/visualisation — sampled actions add
            Gaussian noise on top of the learned policy, making even a
            well-trained agent look erratic.
            If False (default), sample from the distribution as during
            training (useful for measuring exploration at test time).
        """
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        feats  = self.trunk(obs)
        mean   = self.actor_head(feats)
        if deterministic:
            action = mean.squeeze(0)
        else:
            std    = self.actor_log_std.exp().expand_as(mean)
            action = Normal(mean, std).sample().squeeze(0)
        return torch.clamp(action, -1.0, 1.0).cpu().numpy()
