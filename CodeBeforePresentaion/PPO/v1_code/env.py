"""
Hunter-Prey environment — pure numpy, no rendering dependencies.

Design principles:
  - Agents are points in a 2D rectangle.
  - Actions are 2D direction vectors; the env normalises them and moves
    the agent at its rated speed.  Agents ALWAYS move at full speed
    (the policy only controls direction, not magnitude).
  - Rewards depend on each agent's OWN movement, never the joint outcome.
    This prevents the policy-inversion bug (hunter learns to flee, prey
    learns to chase) that occurs when rewards depend on distance change.
  - No wall penalties, no stuck penalties, no proximity bonuses.
    Just heading + terminal + mild time pressure.  Simple = learnable.
"""

import numpy as np


class HunterPreyEnv:
    """Open-field pursuit / evasion between two point agents."""

    def __init__(
        self,
        width:        int   = 800,
        height:       int   = 600,
        hunter_speed: float = 5.0,
        prey_speed:   float = 4.0,
        capture_dist: float = 20.0,
        max_steps:    int   = 500,
    ):
        self.W  = float(width)
        self.H  = float(height)
        self.DIAG = float(np.sqrt(width**2 + height**2))
        self.hunter_speed  = hunter_speed
        self.prey_speed    = prey_speed
        self.capture_dist  = capture_dist
        self.max_steps     = max_steps

        # ---------- reward constants ----------
        # Per-step heading reward: dominant directional signal.
        # Hunter gets +R_HEADING when moving perfectly toward prey.
        # Prey   gets +R_HEADING when moving perfectly away from hunter.
        # These depend on each agent's OWN displacement only.
        self.R_HEADING    =  0.1

        # Terminal rewards
        self.R_CAPTURE_H  =  100.0    # hunter catches prey
        self.R_CAPTURE_P  = -100.0    # prey is caught
        self.R_TIMEOUT_H  =  -10.0    # hunter fails to catch
        self.R_TIMEOUT_P  =   10.0    # prey survives

        # Mild per-step shaping (much smaller than heading)
        self.R_STEP_H     =  -0.01    # tiny time pressure
        self.R_STEP_P     =   0.01    # tiny survival bonus

        # -------- state --------
        self.hunter_pos = np.zeros(2, dtype=np.float64)
        self.prey_pos   = np.zeros(2, dtype=np.float64)
        self._h_last_act = np.zeros(2, dtype=np.float32)
        self._p_last_act = np.zeros(2, dtype=np.float32)
        self.steps = 0
        self.done  = False

    # ---- spaces ----
    @property
    def obs_size(self) -> int:
        # own_pos(2) + rel_other(2) + dist(1) + last_action(2) = 7
        return 7

    @property
    def action_size(self) -> int:
        return 2

    # ---- core API ----
    def reset(self, seed: int | None = None) -> dict:
        if seed is not None:
            np.random.seed(seed)

        pad = 30.0
        self.hunter_pos = np.array([
            np.random.uniform(pad, self.W - pad),
            np.random.uniform(pad, self.H - pad),
        ])
        min_sep = min(self.W, self.H) * 0.3
        for _ in range(200):
            self.prey_pos = np.array([
                np.random.uniform(pad, self.W - pad),
                np.random.uniform(pad, self.H - pad),
            ])
            if np.linalg.norm(self.hunter_pos - self.prey_pos) >= min_sep:
                break

        self._h_last_act = np.zeros(2, dtype=np.float32)
        self._p_last_act = np.zeros(2, dtype=np.float32)
        self.steps = 0
        self.done  = False
        return self._obs()

    def step(self, hunter_action: np.ndarray, prey_action: np.ndarray):
        """
        Parameters
        ----------
        hunter_action, prey_action : (2,) arrays in [-1, 1]².
            Normalised to unit vectors, then scaled by agent speed.

        Returns
        -------
        obs     : dict with "hunter" and "prey" float32 arrays
        rewards : dict with "hunter" and "prey" floats
        done    : bool
        info    : dict with "captured", "distance"
        """
        assert not self.done, "Call reset() before stepping a finished env."

        prev_h = self.hunter_pos.copy()
        prev_p = self.prey_pos.copy()

        # ---- move ----
        self.hunter_pos = self._move(self.hunter_pos, hunter_action, self.hunter_speed)
        self.prey_pos   = self._move(self.prey_pos,   prey_action,   self.prey_speed)

        self._h_last_act = np.clip(hunter_action, -1, 1).astype(np.float32)
        self._p_last_act = np.clip(prey_action,   -1, 1).astype(np.float32)
        self.steps += 1

        # ---- displacements ----
        h_disp = self.hunter_pos - prev_h
        p_disp = self.prey_pos   - prev_p
        dist   = float(np.linalg.norm(self.hunter_pos - self.prey_pos))

        # ---- rewards (per-agent heading) ----
        rewards = {"hunter": 0.0, "prey": 0.0}

        if dist > 1e-6:
            to_prey = (self.prey_pos - self.hunter_pos)
            to_prey = to_prey / np.linalg.norm(to_prey)

            # Hunter: dot(own_displacement, direction_to_prey) / speed
            # +1 when moving directly at prey, -1 when fleeing
            h_approach = float(np.dot(h_disp, to_prey)) / self.hunter_speed
            rewards["hunter"] += self.R_HEADING * h_approach

            # Prey: dot(own_displacement, direction_away_from_hunter) / speed
            # to_prey points away from hunter, so dot > 0 = fleeing ✓
            p_escape = float(np.dot(p_disp, to_prey)) / self.prey_speed
            rewards["prey"] += self.R_HEADING * p_escape

        # Per-step baseline
        rewards["hunter"] += self.R_STEP_H
        rewards["prey"]   += self.R_STEP_P

        # ---- terminal ----
        captured = dist <= self.capture_dist
        timeout  = self.steps >= self.max_steps
        self.done = captured or timeout

        if captured:
            rewards["hunter"] += self.R_CAPTURE_H
            rewards["prey"]   += self.R_CAPTURE_P
        elif timeout:
            rewards["hunter"] += self.R_TIMEOUT_H
            rewards["prey"]   += self.R_TIMEOUT_P

        info = {"captured": captured, "distance": dist, "steps": self.steps}
        return self._obs(), rewards, self.done, info

    # ---- helpers ----
    def _move(self, pos: np.ndarray, action: np.ndarray, speed: float) -> np.ndarray:
        """Normalise action to unit vector, scale by speed, clamp to bounds."""
        a = np.asarray(action, dtype=np.float64)
        mag = np.linalg.norm(a)
        if mag < 1e-8:
            return pos.copy()
        direction = a / mag
        new = pos + direction * speed
        new[0] = np.clip(new[0], 0.0, self.W)
        new[1] = np.clip(new[1], 0.0, self.H)
        return new

    def _obs(self) -> dict:
        dist = float(np.linalg.norm(self.hunter_pos - self.prey_pos))

        hunter_obs = np.array([
            self.hunter_pos[0] / self.W,
            self.hunter_pos[1] / self.H,
            (self.prey_pos[0] - self.hunter_pos[0]) / self.W,
            (self.prey_pos[1] - self.hunter_pos[1]) / self.H,
            dist / self.DIAG,
            self._h_last_act[0],
            self._h_last_act[1],
        ], dtype=np.float32)

        prey_obs = np.array([
            self.prey_pos[0] / self.W,
            self.prey_pos[1] / self.H,
            (self.hunter_pos[0] - self.prey_pos[0]) / self.W,
            (self.hunter_pos[1] - self.prey_pos[1]) / self.H,
            dist / self.DIAG,
            self._p_last_act[0],
            self._p_last_act[1],
        ], dtype=np.float32)

        return {"hunter": hunter_obs, "prey": prey_obs}
