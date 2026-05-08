"""
Hunter-Prey environment with TWO hunters (multi-hunter v1).

Differences from the single-hunter v5 SAC env:
  - Two hunter agents (h1, h2) chase one prey.  Prey is captured if EITHER
    hunter reaches capture_dist.  Otherwise mechanics are unchanged.
  - prey_speed default raised 4.0 → 4.5 (matches hunter).  In 1v1 the
    hunter's speed advantage is what guaranteed eventual capture; in 2v1
    the geometric advantage of two pursuers replaces that, so equalising
    speeds keeps the contest from becoming trivially hunter-dominated.
  - max_steps 500 → 600.  With 2 hunters captures land faster on average,
    so longer episodes give the prey more opportunity to *learn to* escape
    rather than just timing out before training signal accumulates.
  - R_STEP_P 0.02 → 0.03.  Survival bonus bumped to compensate for the
    larger expected capture penalty under 2v1 (more episodes will end in
    capture; the prey needs more positive signal during survival).

Reward design for two hunters:
  Each hunter gets its own per-step rewards (heading toward prey, bump
  penalty, search-when-LOS-broken).  When EITHER hunter captures, BOTH
  hunters receive R_CAPTURE_H — shared team credit prevents the two from
  racing for capture position at the expense of coordination.

  Prey heading reward averages escape components from both hunters:
    R_HEADING * 0.5 * (escape_from_h1 + escape_from_h2)
  Maximum reward only when moving directly away from both, which
  geometrically requires both hunters to be aligned behind the prey.
  When hunters flank, the average is smaller — the prey can only "half
  escape" — which is exactly the strategic challenge we want to teach.

  LOS-break bonus split: 0.25 per hunter on the transition step where LOS
  to that hunter goes 1 → 0.  Total possible per step is 0.5 (matches v5
  single-hunter).  Splitting per-hunter gives clean credit even when
  breaking sight to one but not both.

Observation structure (50 features, identical layout for all 3 agents):
  [0:2]   own_pos / [W, H]
  [2:4]   own_last_action
  [4:8]   own_wall_distances
  [8:10]  rel_other1 / [W, H]               # primary other
  [10:11] dist_other1 / DIAG
  [11:12] los_to_other1
  [12:14] last_known_other1_dir
  [14:15] staleness_other1
  [15:17] rel_other2 / [W, H]               # secondary other
  [17:18] dist_other2 / DIAG
  [18:19] los_to_other2
  [19:21] last_known_other2_dir
  [21:22] staleness_other2
  [22:23] escape_freedom (relative to other1)
  [23:25] cover_dir (nearest obstacle to self)
  [25:50] obstacle_features (5 nearest * 5)

Mapping per agent (FIXED — does not switch within an episode):
  hunter1: other1 = prey, other2 = hunter2
  hunter2: other1 = prey, other2 = hunter1
  prey:    other1 = hunter1, other2 = hunter2
"""

import numpy as np


class Obstacle:
    """Axis-aligned solid rectangle."""
    __slots__ = ("x", "y", "w", "h")

    def __init__(self, x, y, w, h):
        self.x, self.y, self.w, self.h = float(x), float(y), float(w), float(h)

    @property
    def cx(self): return self.x + self.w / 2.0
    @property
    def cy(self): return self.y + self.h / 2.0

    def contains_circle(self, px, py, r):
        nx = np.clip(px, self.x, self.x + self.w)
        ny = np.clip(py, self.y, self.y + self.h)
        return (px - nx)**2 + (py - ny)**2 <= r * r

    def surface_dist(self, px, py):
        nx = np.clip(px, self.x, self.x + self.w)
        ny = np.clip(py, self.y, self.y + self.h)
        return float(np.sqrt((px - nx)**2 + (py - ny)**2))


class HunterPreyEnv:
    """Two-hunter pursuit / one-prey evasion."""

    def __init__(
        self,
        width:        int   = 800,
        height:       int   = 600,
        hunter_speed: float = 4.5,
        prey_speed:   float = 4.5,    # CHANGED from 4.0 — fairness in 2v1
        capture_dist: float = 20.0,
        max_steps:    int   = 600,    # CHANGED from 500 — longer runway
        agent_radius: float = 10.0,
        n_obstacles_range:   tuple = (3, 6),
        obstacle_size_range: tuple = (40, 100),
        max_obstacles_obs:   int   = 5,
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
        self.R_HEADING    =  0.1
        self.R_BUMP       = -0.05
        self.R_CAPTURE_H  =  100.0
        self.R_CAPTURE_P  = -100.0
        self.R_TIMEOUT_H  =  -20.0
        self.R_TIMEOUT_P  =   20.0
        self.R_STEP_H     =  -0.01
        self.R_STEP_P     =   0.03    # CHANGED 0.02 → 0.03

        # LOS-break: split 0.25 per hunter (total 0.5 per step possible,
        # matches v5 single-hunter R_LOS_BREAK)
        self.R_LOS_BREAK  =   0.25
        self.R_COVER_SEEK =   0.03
        self.THREAT_RADIUS = 200.0

        self.R_WALL_PREY   = -0.08
        self.R_CORNER_PREY = -0.08
        self.WALL_EDGE     =  40.0

        self.R_CLOSING_H  =  0.15
        self.R_CLOSING_P  = -0.15
        self.PROXIMITY_ZONE = 60.0

        # ---------- state ----------
        self.hunter1_pos = np.zeros(2, dtype=np.float64)
        self.hunter2_pos = np.zeros(2, dtype=np.float64)
        self.prey_pos    = np.zeros(2, dtype=np.float64)
        self._h1_last_act = np.zeros(2, dtype=np.float32)
        self._h2_last_act = np.zeros(2, dtype=np.float32)
        self._p_last_act  = np.zeros(2, dtype=np.float32)
        self.obstacles: list = []

        # Per-hunter LOS tracking (for LOS-break detection)
        self._prev_los_h1 = 1.0
        self._prev_los_h2 = 1.0

        # Per-hunter last-known-prey position + staleness
        self._h1_last_known_prey = np.zeros(2, dtype=np.float64)
        self._h2_last_known_prey = np.zeros(2, dtype=np.float64)
        self._h1_steps_since_los = 0
        self._h2_steps_since_los = 0

        # Prey tracks each hunter separately
        self._p_last_known_h1 = np.zeros(2, dtype=np.float64)
        self._p_last_known_h2 = np.zeros(2, dtype=np.float64)
        self._p_steps_since_los_h1 = 0
        self._p_steps_since_los_h2 = 0

        self.steps = 0
        self.done  = False

    # ---- spaces ----
    @property
    def obs_size(self) -> int:
        # 8 (own) + 7 (other1) + 7 (other2) + 1 (escape_freedom) + 2 (cover) + 25 (obstacles) = 50
        return 50

    @property
    def action_size(self) -> int:
        return 2

    # ---- core API ----
    def reset(self, seed=None) -> dict:
        if seed is not None:
            np.random.seed(seed)

        self.obstacles = self._spawn_obstacles()

        # Spawn 3 agents with mutual minimum separation
        min_sep_h_p  = min(self.W, self.H) * 0.30   # hunter↔prey separation
        min_sep_h_h  = min(self.W, self.H) * 0.20   # hunter↔hunter separation

        # Prey: 70% of the time near an obstacle (gives early access to cover)
        if self.obstacles and np.random.random() < 0.7:
            self.prey_pos = self._spawn_near_obstacle()
        else:
            self.prey_pos = self._random_free_position(padding=min(self.W, self.H) * 0.15)

        # Hunters: random free positions with min sep from prey AND each other
        self.hunter1_pos = self._spawn_with_min_sep(
            others=[self.prey_pos], min_seps=[min_sep_h_p],
        )
        self.hunter2_pos = self._spawn_with_min_sep(
            others=[self.prey_pos, self.hunter1_pos],
            min_seps=[min_sep_h_p, min_sep_h_h],
        )

        # Reset all per-agent state
        self._h1_last_act = np.zeros(2, dtype=np.float32)
        self._h2_last_act = np.zeros(2, dtype=np.float32)
        self._p_last_act  = np.zeros(2, dtype=np.float32)

        # All initial LOS clear by default
        self._prev_los_h1 = 1.0
        self._prev_los_h2 = 1.0
        self._h1_last_known_prey = self.prey_pos.copy()
        self._h2_last_known_prey = self.prey_pos.copy()
        self._h1_steps_since_los = 0
        self._h2_steps_since_los = 0
        self._p_last_known_h1 = self.hunter1_pos.copy()
        self._p_last_known_h2 = self.hunter2_pos.copy()
        self._p_steps_since_los_h1 = 0
        self._p_steps_since_los_h2 = 0

        self.steps = 0
        self.done  = False
        return self._obs()

    def step(self, h1_action, h2_action, prey_action):
        """Step env with three actions.

        Returns
        -------
        obs    : dict with "hunter1", "hunter2", "prey"
        rew    : dict with "hunter1", "hunter2", "prey"
        done   : bool
        info   : dict with capture info etc.
        """
        assert not self.done, "Call reset() before stepping a finished env."

        prev_h1 = self.hunter1_pos.copy()
        prev_h2 = self.hunter2_pos.copy()
        prev_p  = self.prey_pos.copy()
        prev_dist_h1 = float(np.linalg.norm(self.hunter1_pos - self.prey_pos))
        prev_dist_h2 = float(np.linalg.norm(self.hunter2_pos - self.prey_pos))

        # Move all three agents
        self.hunter1_pos, h1_bumped = self._move(self.hunter1_pos, h1_action, self.hunter_speed)
        self.hunter2_pos, h2_bumped = self._move(self.hunter2_pos, h2_action, self.hunter_speed)
        self.prey_pos,    p_bumped  = self._move(self.prey_pos,    prey_action, self.prey_speed)

        self._h1_last_act = np.clip(h1_action, -1, 1).astype(np.float32)
        self._h2_last_act = np.clip(h2_action, -1, 1).astype(np.float32)
        self._p_last_act  = np.clip(prey_action, -1, 1).astype(np.float32)
        self.steps += 1

        # Displacements + new distances
        h1_disp = self.hunter1_pos - prev_h1
        h2_disp = self.hunter2_pos - prev_h2
        p_disp  = self.prey_pos    - prev_p
        dist_h1 = float(np.linalg.norm(self.hunter1_pos - self.prey_pos))
        dist_h2 = float(np.linalg.norm(self.hunter2_pos - self.prey_pos))

        rewards = {"hunter1": 0.0, "hunter2": 0.0, "prey": 0.0}

        # ---- Heading rewards ----
        # Each hunter: own approach toward prey
        if dist_h1 > 1e-6:
            to_prey_h1 = (self.prey_pos - self.hunter1_pos) / dist_h1
            h1_approach = float(np.dot(h1_disp, to_prey_h1)) / self.hunter_speed
            rewards["hunter1"] += self.R_HEADING * h1_approach
        if dist_h2 > 1e-6:
            to_prey_h2 = (self.prey_pos - self.hunter2_pos) / dist_h2
            h2_approach = float(np.dot(h2_disp, to_prey_h2)) / self.hunter_speed
            rewards["hunter2"] += self.R_HEADING * h2_approach

        # Prey: average escape from BOTH hunters.  Maximum only when moving
        # directly away from both, which requires hunters to be aligned.
        if dist_h1 > 1e-6 and dist_h2 > 1e-6:
            away_h1 = (self.prey_pos - self.hunter1_pos) / dist_h1
            away_h2 = (self.prey_pos - self.hunter2_pos) / dist_h2
            p_escape_h1 = float(np.dot(p_disp, away_h1)) / self.prey_speed
            p_escape_h2 = float(np.dot(p_disp, away_h2)) / self.prey_speed
            rewards["prey"] += self.R_HEADING * 0.5 * (p_escape_h1 + p_escape_h2)

        # ---- LOS to each hunter (for prey) ----
        los_h1 = self._line_of_sight_between(self.prey_pos, self.hunter1_pos)
        los_h2 = self._line_of_sight_between(self.prey_pos, self.hunter2_pos)

        # LOS-break rewards (split per hunter)
        los_h1_broken = (self._prev_los_h1 > 0.5 and los_h1 < 0.5)
        los_h2_broken = (self._prev_los_h2 > 0.5 and los_h2 < 0.5)
        if los_h1_broken: rewards["prey"] += self.R_LOS_BREAK
        if los_h2_broken: rewards["prey"] += self.R_LOS_BREAK
        self._prev_los_h1 = los_h1
        self._prev_los_h2 = los_h2

        # Track last-known positions and staleness
        if los_h1 > 0.5:
            self._h1_last_known_prey = self.prey_pos.copy()
            self._p_last_known_h1    = self.hunter1_pos.copy()
            self._h1_steps_since_los = 0
            self._p_steps_since_los_h1 = 0
        else:
            self._h1_steps_since_los += 1
            self._p_steps_since_los_h1 += 1
        if los_h2 > 0.5:
            self._h2_last_known_prey = self.prey_pos.copy()
            self._p_last_known_h2    = self.hunter2_pos.copy()
            self._h2_steps_since_los = 0
            self._p_steps_since_los_h2 = 0
        else:
            self._h2_steps_since_los += 1
            self._p_steps_since_los_h2 += 1

        # ---- Cover-seeking (prey, when threatened by EITHER hunter) ----
        closer_dist = min(dist_h1, dist_h2)
        if closer_dist < self.THREAT_RADIUS and self.obstacles:
            cover_dir = self._cover_direction(self.prey_pos)
            p_moved   = float(np.linalg.norm(p_disp))
            if p_moved > 0.1 and np.linalg.norm(cover_dir) > 0.1:
                cover_app = float(np.dot(p_disp, cover_dir)) / self.prey_speed
                rewards["prey"] += self.R_COVER_SEEK * cover_app

        # ---- Hunter search reward (when its LOS to prey is broken) ----
        for hkey, los_val, stale, last_known, h_pos, h_disp_v in [
            ("hunter1", los_h1, self._h1_steps_since_los, self._h1_last_known_prey,
             self.hunter1_pos, h1_disp),
            ("hunter2", los_h2, self._h2_steps_since_los, self._h2_last_known_prey,
             self.hunter2_pos, h2_disp),
        ]:
            if los_val < 0.5 and stale < 100:
                h_moved = float(np.linalg.norm(h_disp_v))
                if h_moved > 0.1:
                    to_lk = last_known - h_pos
                    lk_dist = np.linalg.norm(to_lk)
                    if lk_dist > 5.0:
                        search_dir = to_lk / lk_dist
                        search_app = float(np.dot(h_disp_v, search_dir)) / self.hunter_speed
                        rewards[hkey] += self.R_HEADING * 0.5 * search_app

        # ---- Bump penalties + step baseline ----
        if h1_bumped: rewards["hunter1"] += self.R_BUMP
        if h2_bumped: rewards["hunter2"] += self.R_BUMP
        if p_bumped:  rewards["prey"]    += self.R_BUMP

        rewards["hunter1"] += self.R_STEP_H
        rewards["hunter2"] += self.R_STEP_H
        rewards["prey"]    += self.R_STEP_P

        # ---- Wall/corner penalty (prey only) ----
        px, py = self.prey_pos
        n_walls = (int(px < self.WALL_EDGE) + int(px > self.W - self.WALL_EDGE) +
                   int(py < self.WALL_EDGE) + int(py > self.H - self.WALL_EDGE))
        if n_walls >= 1: rewards["prey"] += self.R_WALL_PREY
        if n_walls >= 2: rewards["prey"] += self.R_CORNER_PREY

        # ---- Closing-velocity proximity reward (per hunter, prey penalty by closest) ----
        for hkey, dist, prev_dist in [
            ("hunter1", dist_h1, prev_dist_h1),
            ("hunter2", dist_h2, prev_dist_h2),
        ]:
            if dist < self.PROXIMITY_ZONE or prev_dist < self.PROXIMITY_ZONE:
                delta_dist = prev_dist - dist
                if delta_dist > 0:
                    cn = delta_dist / self.hunter_speed
                    rewards[hkey] += self.R_CLOSING_H * cn
                    # Prey gets penalty only from the closest hunter to avoid
                    # double-counting in flanking scenarios
                    if dist == min(dist_h1, dist_h2):
                        rewards["prey"] += self.R_CLOSING_P * cn

        # ---- Terminal conditions ----
        captured_by_h1 = dist_h1 <= self.capture_dist
        captured_by_h2 = dist_h2 <= self.capture_dist
        captured = captured_by_h1 or captured_by_h2
        timeout  = self.steps >= self.max_steps
        self.done = captured or timeout

        if captured:
            # BOTH hunters get capture reward (shared team credit)
            rewards["hunter1"] += self.R_CAPTURE_H
            rewards["hunter2"] += self.R_CAPTURE_H
            survival_frac = self.steps / self.max_steps
            rewards["prey"] += self.R_CAPTURE_P * (1.0 - 0.5 * survival_frac)
        elif timeout:
            rewards["hunter1"] += self.R_TIMEOUT_H
            rewards["hunter2"] += self.R_TIMEOUT_H
            rewards["prey"]    += self.R_TIMEOUT_P

        info = {
            "captured": captured,
            "captured_by_h1": captured_by_h1,
            "captured_by_h2": captured_by_h2,
            "dist_h1": dist_h1, "dist_h2": dist_h2,
            "los_h1": los_h1,   "los_h2": los_h2,
            "los_broken_h1": los_h1_broken, "los_broken_h2": los_h2_broken,
            "h1_bumped": h1_bumped, "h2_bumped": h2_bumped, "p_bumped": p_bumped,
            "steps": self.steps,
        }
        return self._obs(), rewards, self.done, info

    # ---- curriculum setters ----
    def set_obstacle_range(self, lo, hi):
        self.n_obstacles_range = (int(lo), int(hi))

    def set_speeds(self, hunter_speed, prey_speed):
        self.hunter_speed = float(hunter_speed)
        self.prey_speed   = float(prey_speed)

    # =================================================================
    # Observation
    # =================================================================
    def _obs(self) -> dict:
        # Pre-compute LOS for both hunter↔prey pairs and h1↔h2
        los_p_h1 = self._line_of_sight_between(self.prey_pos, self.hunter1_pos)
        los_p_h2 = self._line_of_sight_between(self.prey_pos, self.hunter2_pos)
        los_h1_h2 = self._line_of_sight_between(self.hunter1_pos, self.hunter2_pos)

        # ---- Build observations using a fixed mapping ----
        # hunter1: other1 = prey, other2 = hunter2
        # hunter2: other1 = prey, other2 = hunter1
        # prey:    other1 = hunter1, other2 = hunter2

        h1_obs = self._build_obs(
            own_pos=self.hunter1_pos, own_last_act=self._h1_last_act,
            other1_pos=self.prey_pos,    other1_los=los_p_h1,
            other1_last_known=self._h1_last_known_prey,
            other1_staleness=self._h1_steps_since_los,
            other2_pos=self.hunter2_pos, other2_los=los_h1_h2,
            other2_last_known=self.hunter2_pos,  # teammates always "fresh"
            other2_staleness=0,
        )
        h2_obs = self._build_obs(
            own_pos=self.hunter2_pos, own_last_act=self._h2_last_act,
            other1_pos=self.prey_pos,    other1_los=los_p_h2,
            other1_last_known=self._h2_last_known_prey,
            other1_staleness=self._h2_steps_since_los,
            other2_pos=self.hunter1_pos, other2_los=los_h1_h2,
            other2_last_known=self.hunter1_pos,
            other2_staleness=0,
        )
        p_obs = self._build_obs(
            own_pos=self.prey_pos, own_last_act=self._p_last_act,
            other1_pos=self.hunter1_pos, other1_los=los_p_h1,
            other1_last_known=self._p_last_known_h1,
            other1_staleness=self._p_steps_since_los_h1,
            other2_pos=self.hunter2_pos, other2_los=los_p_h2,
            other2_last_known=self._p_last_known_h2,
            other2_staleness=self._p_steps_since_los_h2,
        )

        return {"hunter1": h1_obs, "hunter2": h2_obs, "prey": p_obs}

    def _build_obs(self, own_pos, own_last_act,
                    other1_pos, other1_los, other1_last_known, other1_staleness,
                    other2_pos, other2_los, other2_last_known, other2_staleness):
        """Build the 50-feature observation vector for one agent."""
        rel1 = other1_pos - own_pos
        rel2 = other2_pos - own_pos
        d1 = float(np.linalg.norm(rel1))
        d2 = float(np.linalg.norm(rel2))

        # Last-known direction unit vectors
        lk1_diff = other1_last_known - own_pos
        lk1_mag  = np.linalg.norm(lk1_diff)
        lk1_dir  = (lk1_diff / lk1_mag) if lk1_mag > 1e-4 else np.zeros(2)

        lk2_diff = other2_last_known - own_pos
        lk2_mag  = np.linalg.norm(lk2_diff)
        lk2_dir  = (lk2_diff / lk2_mag) if lk2_mag > 1e-4 else np.zeros(2)

        # Escape freedom relative to other1 (the primary other)
        freedom = self._escape_freedom(own_pos, other1_pos)

        return np.concatenate([
            own_pos / np.array([self.W, self.H]),                 # 2
            own_last_act,                                         # 2
            self._wall_distances(own_pos),                        # 4
            rel1 / np.array([self.W, self.H]),                    # 2
            [d1 / self.DIAG],                                     # 1
            [other1_los],                                         # 1
            lk1_dir,                                              # 2
            [min(1.0, other1_staleness / 100.0)],                # 1
            rel2 / np.array([self.W, self.H]),                    # 2
            [d2 / self.DIAG],                                     # 1
            [other2_los],                                         # 1
            lk2_dir,                                              # 2
            [min(1.0, other2_staleness / 100.0)],                # 1
            [freedom],                                            # 1
            self._cover_direction(own_pos),                       # 2
            self._obstacle_features(own_pos),                     # 25
        ]).astype(np.float32)

    # =================================================================
    # Spawning
    # =================================================================
    def _spawn_obstacles(self):
        lo, hi = self.n_obstacles_range
        if hi <= 0:
            return []
        n = np.random.randint(max(0, lo), max(1, hi) + 1)
        smin, smax = self.obstacle_size_range
        margin = self.agent_radius * 4
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
    def _rects_overlap(a, b, pad=0.0):
        return not (a.x + a.w + pad < b.x or b.x + b.w + pad < a.x or
                     a.y + a.h + pad < b.y or b.y + b.h + pad < a.y)

    def _random_free_position(self, padding=None):
        pad = padding if padding is not None else self.agent_radius + 5
        for _ in range(500):
            pos = np.array([
                np.random.uniform(pad, self.W - pad),
                np.random.uniform(pad, self.H - pad),
            ])
            if not self._collides_any(pos[0], pos[1]):
                return pos
        return np.array([self.W / 2, self.H / 2])

    def _spawn_near_obstacle(self):
        if not self.obstacles:
            return self._random_free_position()
        r = self.agent_radius
        for _ in range(100):
            o = self.obstacles[np.random.randint(len(self.obstacles))]
            side = np.random.randint(4)
            if side == 0:
                pos = np.array([o.x - r - 2, o.cy + np.random.uniform(-o.h/3, o.h/3)])
            elif side == 1:
                pos = np.array([o.x + o.w + r + 2, o.cy + np.random.uniform(-o.h/3, o.h/3)])
            elif side == 2:
                pos = np.array([o.cx + np.random.uniform(-o.w/3, o.w/3), o.y - r - 2])
            else:
                pos = np.array([o.cx + np.random.uniform(-o.w/3, o.w/3), o.y + o.h + r + 2])
            if (r < pos[0] < self.W - r and r < pos[1] < self.H - r
                    and not self._collides_any(pos[0], pos[1])):
                return pos
        return self._random_free_position()

    def _spawn_with_min_sep(self, others, min_seps):
        """Spawn a free position with minimum separation from each `others[i]`."""
        for _ in range(500):
            pos = self._random_free_position()
            if all(np.linalg.norm(pos - other) >= sep
                   for other, sep in zip(others, min_seps)):
                return pos
        # Fallback: best-effort, far from the first `other`
        return self._random_free_position()

    def _collides_any(self, x, y):
        return any(o.contains_circle(x, y, self.agent_radius) for o in self.obstacles)

    # =================================================================
    # Movement (collision sliding)
    # =================================================================
    _ESCAPE_ANGLES = np.linspace(0, 2 * np.pi, 9)[:-1]
    _ESCAPE_DIRS   = np.stack([np.cos(_ESCAPE_ANGLES), np.sin(_ESCAPE_ANGLES)], axis=1)

    def _move(self, pos, action, speed):
        a = np.asarray(action, dtype=np.float64)
        mag = np.linalg.norm(a)
        if mag < 1e-8:
            return pos.copy(), False
        direction = a / mag
        dx, dy = direction * speed

        r = self.agent_radius
        def clamp(p):
            return np.array([np.clip(p[0], r, self.W - r), np.clip(p[1], r, self.H - r)])

        cand = clamp(pos + np.array([dx, dy]))
        if not self._collides_any(cand[0], cand[1]):
            return cand, False

        cand_x = clamp(pos + np.array([dx, 0.0]))
        if not self._collides_any(cand_x[0], cand_x[1]):
            return cand_x, True

        cand_y = clamp(pos + np.array([0.0, dy]))
        if not self._collides_any(cand_y[0], cand_y[1]):
            return cand_y, True

        order = np.argsort(-self._ESCAPE_DIRS @ direction)
        for nudge in [speed, speed * 0.5, speed * 0.25]:
            for i in order:
                cand_e = clamp(pos + self._ESCAPE_DIRS[i] * nudge)
                if not self._collides_any(cand_e[0], cand_e[1]):
                    return cand_e, True
        return pos.copy(), True

    # =================================================================
    # Geometry helpers
    # =================================================================
    def _line_of_sight_between(self, pos_a, pos_b):
        """1.0 if straight line a↔b is clear of obstacles, else 0.0."""
        if not self.obstacles:
            return 1.0
        for t in np.linspace(0.05, 0.95, 15):
            x = pos_a[0] + t * (pos_b[0] - pos_a[0])
            y = pos_a[1] + t * (pos_b[1] - pos_a[1])
            for o in self.obstacles:
                if o.x <= x <= o.x + o.w and o.y <= y <= o.y + o.h:
                    return 0.0
        return 1.0

    def _wall_distances(self, pos):
        return np.array([pos[0] / self.W, (self.W - pos[0]) / self.W,
                          pos[1] / self.H, (self.H - pos[1]) / self.H], dtype=np.float32)

    def _cover_direction(self, agent_pos):
        if not self.obstacles:
            return np.zeros(2, dtype=np.float32)
        best_dist = float("inf"); best_dir = np.zeros(2, dtype=np.float64)
        for o in self.obstacles:
            d = np.array([o.cx - agent_pos[0], o.cy - agent_pos[1]])
            sd = o.surface_dist(agent_pos[0], agent_pos[1])
            if sd < best_dist:
                best_dist = sd; best_dir = d
        mag = np.linalg.norm(best_dir)
        return np.zeros(2, dtype=np.float32) if mag < 1e-8 else (best_dir / mag).astype(np.float32)

    def _escape_freedom(self, agent_pos, other_pos):
        diff = agent_pos - other_pos; dist = np.linalg.norm(diff)
        if dist < 1e-4: return 1.0
        away = diff / dist
        r = self.agent_radius; probe_dist = 50.0
        open_count = hemi_count = 0
        for i in range(8):
            angle = (i / 8) * 2 * np.pi
            pd = np.array([np.cos(angle), np.sin(angle)])
            if np.dot(pd, away) < 0: continue
            hemi_count += 1
            check = agent_pos + pd * probe_dist
            if r <= check[0] <= self.W - r and r <= check[1] <= self.H - r:
                open_count += 1
        return float(open_count / max(hemi_count, 1))

    def _obstacle_features(self, agent_pos):
        K = self.max_obstacles_obs
        if not self.obstacles:
            return np.tile([0.0, 0.0, 0.0, 0.0, 1.0], K).astype(np.float32)
        entries = [(o.surface_dist(agent_pos[0], agent_pos[1]), o) for o in self.obstacles]
        entries.sort(key=lambda e: e[0])
        entries = entries[:K]
        feats = []
        for sd, o in entries:
            feats.extend([(o.cx - agent_pos[0]) / self.W, (o.cy - agent_pos[1]) / self.H,
                          o.w / self.W, o.h / self.H, sd / self.DIAG])
        for _ in range(K - len(entries)):
            feats.extend([0.0, 0.0, 0.0, 0.0, 1.0])
        return np.array(feats, dtype=np.float32)
