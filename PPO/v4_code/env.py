"""
Hunter-Prey environment — with obstacles (v3: prey intelligence).

Design principles (same as v1, DO NOT CHANGE):
  - Agents are circles in a 2D rectangle.
  - Actions are 2D direction vectors; env normalises to unit, scales by speed.
  - Rewards depend on each agent's OWN movement (per-agent heading reward).

New in v3 (prey tactical intelligence):
  1. LOS-break reward: +0.5 one-time bonus when prey breaks line-of-sight
     (LOS transitions from clear to blocked).  Teaches the prey that hiding
     behind obstacles delays the hunter.  One-time = prey must keep moving.
  2. Direction-to-nearest-cover observation: 2D unit vector toward the nearest
     obstacle center.  Pre-computed directional cue so the policy doesn't
     have to figure out spatial geometry from raw obstacle features.
  3. Cover-seeking reward: +0.03/step for moving toward nearest obstacle
     when the hunter is within THREAT_RADIUS (200px).  Subtle tiebreaker
     that makes "flee toward cover" slightly better than "flee into open."
"""

import numpy as np


class Obstacle:
    """Axis-aligned rectangle obstacle."""

    __slots__ = ("x", "y", "w", "h")

    def __init__(self, x: float, y: float, w: float, h: float):
        self.x, self.y, self.w, self.h = float(x), float(y), float(w), float(h)

    @property
    def cx(self) -> float: return self.x + self.w / 2.0

    @property
    def cy(self) -> float: return self.y + self.h / 2.0

    def contains_circle(self, px: float, py: float, r: float) -> bool:
        """True if a circle of radius r at (px,py) overlaps this rect."""
        nx = np.clip(px, self.x, self.x + self.w)
        ny = np.clip(py, self.y, self.y + self.h)
        return (px - nx) ** 2 + (py - ny) ** 2 <= r * r

    def surface_dist(self, px: float, py: float) -> float:
        """Euclidean distance from (px, py) to the nearest point on this rect."""
        nx = np.clip(px, self.x, self.x + self.w)
        ny = np.clip(py, self.y, self.y + self.h)
        return float(np.sqrt((px - nx) ** 2 + (py - ny) ** 2))


class HunterPreyEnv:
    """Open-field pursuit / evasion with rectangular obstacles."""

    def __init__(
        self,
        width:        int   = 800,
        height:       int   = 600,
        hunter_speed: float = 5.0,
        prey_speed:   float = 4.0,
        capture_dist: float = 20.0,
        max_steps:    int   = 500,
        agent_radius: float = 10.0,
        n_obstacles_range: tuple = (3, 6),
        obstacle_size_range: tuple = (40, 100),
        max_obstacles_obs: int = 5,
    ):
        self.W = float(width); self.H = float(height)
        self.DIAG = float(np.sqrt(width**2 + height**2))
        self.hunter_speed = hunter_speed
        self.prey_speed   = prey_speed
        self.capture_dist = capture_dist
        self.max_steps    = max_steps
        self.agent_radius = float(agent_radius)

        self.n_obstacles_range   = n_obstacles_range
        self.obstacle_size_range = obstacle_size_range
        self.max_obstacles_obs   = max_obstacles_obs

        # ---------- reward constants ----------
        self.R_HEADING    =  0.1       # dominant per-step signal
        self.R_BUMP       = -0.05      # tiny, comparable to step penalty
        self.R_CAPTURE_H  =  100.0
        self.R_CAPTURE_P  = -100.0
        self.R_TIMEOUT_H  =  -10.0
        self.R_TIMEOUT_P  =   10.0
        self.R_STEP_H     =  -0.01
        self.R_STEP_P     =   0.01

        # Prey tactical rewards (v3)
        self.R_LOS_BREAK  =   0.5     # one-time bonus when prey breaks LOS
        self.R_COVER_SEEK =   0.03    # per-step bonus for moving toward cover under threat
        self.THREAT_RADIUS = 200.0    # cover-seek only active within this range

        # ---------- state ----------
        self.hunter_pos = np.zeros(2, dtype=np.float64)
        self.prey_pos   = np.zeros(2, dtype=np.float64)
        self._h_last_act = np.zeros(2, dtype=np.float32)
        self._p_last_act = np.zeros(2, dtype=np.float32)
        self.obstacles = []  # list[Obstacle]
        self._prev_los = 1.0  # track LOS for transition detection
        self.steps = 0
        self.done  = False

    # ---- spaces ----
    @property
    def obs_size(self) -> int:
        # own_pos(2) + rel_other(2) + dist(1) + last_action(2) + los(1)
        #   + wall_distances(4) + cover_dir(2)
        #   + max_obstacles_obs * 5 (rel_cx, rel_cy, w/W, h/H, surf_dist/DIAG)
        return 14 + self.max_obstacles_obs * 5

    @property
    def action_size(self) -> int:
        return 2

    # ---- core API ----
    def reset(self, seed=None) -> dict:
        if seed is not None:
            np.random.seed(seed)

        # 1. Spawn obstacles first
        self.obstacles = self._spawn_obstacles()

        # 2. Spawn agents in free space, far apart
        self.hunter_pos = self._random_free_position()
        min_sep = min(self.W, self.H) * 0.3
        for _ in range(200):
            self.prey_pos = self._random_free_position()
            if np.linalg.norm(self.hunter_pos - self.prey_pos) >= min_sep:
                break

        self._h_last_act = np.zeros(2, dtype=np.float32)
        self._p_last_act = np.zeros(2, dtype=np.float32)
        self._prev_los = 1.0  # assume LOS clear at episode start
        self.steps = 0
        self.done  = False
        return self._obs()

    def step(self, hunter_action, prey_action):
        assert not self.done, "Call reset() before stepping a finished env."

        prev_h = self.hunter_pos.copy()
        prev_p = self.prey_pos.copy()

        # Move with collision handling
        self.hunter_pos, h_bumped = self._move(self.hunter_pos, hunter_action, self.hunter_speed)
        self.prey_pos,   p_bumped = self._move(self.prey_pos,   prey_action,   self.prey_speed)

        self._h_last_act = np.clip(hunter_action, -1, 1).astype(np.float32)
        self._p_last_act = np.clip(prey_action,   -1, 1).astype(np.float32)
        self.steps += 1

        # ---- rewards: heading (per-agent, independent) ----
        h_disp = self.hunter_pos - prev_h
        p_disp = self.prey_pos   - prev_p
        dist   = float(np.linalg.norm(self.hunter_pos - self.prey_pos))

        rewards = {"hunter": 0.0, "prey": 0.0}
        if dist > 1e-6:
            to_prey = (self.prey_pos - self.hunter_pos)
            to_prey = to_prey / np.linalg.norm(to_prey)
            h_approach = float(np.dot(h_disp, to_prey)) / self.hunter_speed
            p_escape   = float(np.dot(p_disp, to_prey)) / self.prey_speed
            rewards["hunter"] += self.R_HEADING * h_approach
            rewards["prey"]   += self.R_HEADING * p_escape

        # ---- LOS-break reward (prey only, transition 1→0) ----
        curr_los = self._line_of_sight()
        los_broken = False
        if self._prev_los > 0.5 and curr_los < 0.5:
            # Prey just broke line-of-sight — one-time bonus
            rewards["prey"] += self.R_LOS_BREAK
            los_broken = True
        self._prev_los = curr_los

        # ---- Cover-seeking reward (prey only, when hunter is close) ----
        if dist < self.THREAT_RADIUS and self.obstacles:
            cover_dir = self._cover_direction(self.prey_pos)
            p_moved = float(np.linalg.norm(p_disp))
            if p_moved > 0.1 and np.linalg.norm(cover_dir) > 0.1:
                # How much did prey move toward the nearest obstacle?
                cover_approach = float(np.dot(p_disp, cover_dir)) / self.prey_speed
                rewards["prey"] += self.R_COVER_SEEK * cover_approach

        # Per-event bump penalty
        if h_bumped: rewards["hunter"] += self.R_BUMP
        if p_bumped: rewards["prey"]   += self.R_BUMP

        # Step baseline
        rewards["hunter"] += self.R_STEP_H
        rewards["prey"]   += self.R_STEP_P

        # Terminal
        captured = dist <= self.capture_dist
        timeout  = self.steps >= self.max_steps
        self.done = captured or timeout
        if captured:
            rewards["hunter"] += self.R_CAPTURE_H
            rewards["prey"]   += self.R_CAPTURE_P
        elif timeout:
            rewards["hunter"] += self.R_TIMEOUT_H
            rewards["prey"]   += self.R_TIMEOUT_P

        info = {"captured": captured, "distance": dist, "steps": self.steps,
                "h_bumped": h_bumped, "p_bumped": p_bumped,
                "los": curr_los, "los_broken": los_broken}
        return self._obs(), rewards, self.done, info

    # ---- curriculum control (set by train.py) ----
    def set_obstacle_range(self, lo: int, hi: int):
        self.n_obstacles_range = (int(lo), int(hi))

    # =================================================================
    # Helpers
    # =================================================================
    def _spawn_obstacles(self):
        lo, hi = self.n_obstacles_range
        if hi <= 0:
            return []
        n = np.random.randint(max(0, lo), max(1, hi) + 1)
        smin, smax = self.obstacle_size_range
        margin = self.agent_radius * 4  # keep obstacles away from walls
        out = []
        attempts = 0
        while len(out) < n and attempts < 200:
            attempts += 1
            w = float(np.random.randint(smin, smax + 1))
            h = float(np.random.randint(smin, smax + 1))
            x = float(np.random.uniform(margin, self.W - w - margin))
            y = float(np.random.uniform(margin, self.H - h - margin))
            cand = Obstacle(x, y, w, h)
            if any(self._rects_overlap(cand, o, pad=self.agent_radius * 2) for o in out):
                continue
            out.append(cand)
        return out

    @staticmethod
    def _rects_overlap(a, b, pad=0.0) -> bool:
        return not (
            a.x + a.w + pad < b.x or b.x + b.w + pad < a.x or
            a.y + a.h + pad < b.y or b.y + b.h + pad < a.y
        )

    def _random_free_position(self) -> np.ndarray:
        pad = self.agent_radius + 5
        for _ in range(500):
            pos = np.array([
                np.random.uniform(pad, self.W - pad),
                np.random.uniform(pad, self.H - pad),
            ])
            if not self._collides_any(pos[0], pos[1]):
                return pos
        return np.array([self.W / 2, self.H / 2])

    def _collides_any(self, x: float, y: float) -> bool:
        return any(o.contains_circle(x, y, self.agent_radius) for o in self.obstacles)

    # Eight evenly-spaced unit escape directions (class-level cache)
    _ESCAPE_ANGLES = np.linspace(0, 2 * np.pi, 9)[:-1]  # 0°,45°,…,315°
    _ESCAPE_DIRS   = np.stack([np.cos(_ESCAPE_ANGLES),
                               np.sin(_ESCAPE_ANGLES)], axis=1)

    def _move(self, pos: np.ndarray, action, speed: float):
        """Normalise action, move, handle wall-clamp + obstacle slide.

        Priority:
          1. Full move        — accepted if clear.
          2. X-only slide     — slide along vertical face.
          3. Y-only slide     — slide along horizontal face.
          4. Escape probe     — 8 directions sorted by similarity to intent,
                                at half speed.  Prevents freeze at wall-obstacle
                                junctions where both axis-slides fail.
          5. Stay put          — all 8 directions blocked (very rare).

        Returns (new_pos, bumped_flag).
        """
        a = np.asarray(action, dtype=np.float64)
        mag = np.linalg.norm(a)
        if mag < 1e-8:
            return pos.copy(), False
        direction = a / mag
        dx, dy = direction * speed

        r = self.agent_radius
        def clamp(p):
            return np.array([
                np.clip(p[0], r, self.W - r),
                np.clip(p[1], r, self.H - r),
            ])

        # 1. Full move
        cand = clamp(pos + np.array([dx, dy]))
        if not self._collides_any(cand[0], cand[1]):
            return cand, False

        # 2. X-only slide
        cand_x = clamp(pos + np.array([dx, 0.0]))
        if not self._collides_any(cand_x[0], cand_x[1]):
            return cand_x, True

        # 3. Y-only slide
        cand_y = clamp(pos + np.array([0.0, dy]))
        if not self._collides_any(cand_y[0], cand_y[1]):
            return cand_y, True

        # 4. Escape probe — 8 directions sorted by closeness to intended direction
        #    Try full speed first, then half speed for tighter spaces
        order = np.argsort(-self._ESCAPE_DIRS @ direction)
        for nudge in [speed, speed * 0.5, speed * 0.25]:
            for i in order:
                cand_e = clamp(pos + self._ESCAPE_DIRS[i] * nudge)
                if not self._collides_any(cand_e[0], cand_e[1]):
                    return cand_e, True

        # 5. Fully blocked
        return pos.copy(), True

    def _line_of_sight(self) -> float:
        """1.0 if the straight line hunter↔prey is clear, else 0.0."""
        if not self.obstacles:
            return 1.0
        for t in np.linspace(0.05, 0.95, 15):
            x = self.hunter_pos[0] + t * (self.prey_pos[0] - self.hunter_pos[0])
            y = self.hunter_pos[1] + t * (self.prey_pos[1] - self.hunter_pos[1])
            for o in self.obstacles:
                if o.x <= x <= o.x + o.w and o.y <= y <= o.y + o.h:
                    return 0.0
        return 1.0

    def _obstacle_features(self, agent_pos: np.ndarray) -> np.ndarray:
        """Nearest-K obstacles as (rel_cx/W, rel_cy/H, w/W, h/H, surf_dist/DIAG).

        Sorted by surface distance, zero-padded to max_obstacles_obs slots.
        Padded slots use surf_dist=1.0 (far) so they don't confuse the policy.
        """
        K = self.max_obstacles_obs
        if not self.obstacles:
            return np.tile([0.0, 0.0, 0.0, 0.0, 1.0], K).astype(np.float32)

        entries = [(o.surface_dist(agent_pos[0], agent_pos[1]), o) for o in self.obstacles]
        entries.sort(key=lambda e: e[0])
        entries = entries[:K]

        feats = []
        for sd, o in entries:
            feats.extend([
                (o.cx - agent_pos[0]) / self.W,
                (o.cy - agent_pos[1]) / self.H,
                o.w / self.W,
                o.h / self.H,
                sd / self.DIAG,
            ])
        for _ in range(K - len(entries)):
            feats.extend([0.0, 0.0, 0.0, 0.0, 1.0])
        return np.array(feats, dtype=np.float32)

    def _cover_direction(self, agent_pos: np.ndarray) -> np.ndarray:
        """Unit vector from agent toward the nearest obstacle center.

        Returns (0,0) if there are no obstacles.  This gives the prey a
        clean directional cue: "dodge THIS way to reach cover."
        """
        if not self.obstacles:
            return np.zeros(2, dtype=np.float32)

        best_dist = float("inf")
        best_dir  = np.zeros(2, dtype=np.float64)
        for o in self.obstacles:
            d = np.array([o.cx - agent_pos[0], o.cy - agent_pos[1]])
            sd = o.surface_dist(agent_pos[0], agent_pos[1])
            if sd < best_dist:
                best_dist = sd
                best_dir  = d

        mag = np.linalg.norm(best_dir)
        if mag < 1e-8:
            return np.zeros(2, dtype=np.float32)
        return (best_dir / mag).astype(np.float32)

    def _wall_distances(self, pos: np.ndarray) -> np.ndarray:
        """Normalised distances to each screen boundary [left, right, top, bottom]."""
        return np.array([
            pos[0] / self.W,
            (self.W - pos[0]) / self.W,
            pos[1] / self.H,
            (self.H - pos[1]) / self.H,
        ], dtype=np.float32)

    def _obs(self) -> dict:
        dist = float(np.linalg.norm(self.hunter_pos - self.prey_pos))
        los  = self._line_of_sight()

        h_cover = self._cover_direction(self.hunter_pos)
        p_cover = self._cover_direction(self.prey_pos)

        h_obs = np.concatenate([
            self.hunter_pos / np.array([self.W, self.H]),
            (self.prey_pos - self.hunter_pos) / np.array([self.W, self.H]),
            [dist / self.DIAG],
            self._h_last_act,
            [los],
            self._wall_distances(self.hunter_pos),
            h_cover,
            self._obstacle_features(self.hunter_pos),
        ]).astype(np.float32)

        p_obs = np.concatenate([
            self.prey_pos / np.array([self.W, self.H]),
            (self.hunter_pos - self.prey_pos) / np.array([self.W, self.H]),
            [dist / self.DIAG],
            self._p_last_act,
            [los],
            self._wall_distances(self.prey_pos),
            p_cover,
            self._obstacle_features(self.prey_pos),
        ]).astype(np.float32)

        return {"hunter": h_obs, "prey": p_obs}
