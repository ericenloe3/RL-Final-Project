"""
Hunter-Prey environment — obstacles + terrain effects (v6).

v6 fixes and additions (addresses terrain eval findings):

  1. Displacement tracking in observation (+2 per agent):
       own_disp_frac  — last-step actual displacement / rated_speed (0=fully stuck, 1=free)
       own_last_bumped — 1 if last step resulted in a collision bump
     Together these give the policy an explicit "I am stuck" signal.  Without
     this, the hunter policy had no way to detect the wall-obstacle corner trap
     (it output the same action repeatedly and the env silently discarded it).

  2. Terrain exploitation rewards for prey:
       R_BAIT_ICE (1.0) — one-time bonus when the HUNTER enters an ice zone while
         the prey is within THREAT_RADIUS and NOT on ice itself.  Symmetric to
         R_LOS_BREAK: teaches "luring" the hunter into the zone.
       R_MUD_DRAG (0.04/step) — continuous bonus when the hunter is in mud.
         The hunter's speed advantage (4.5 vs 4.0) is neutralised in mud; this
         reward teaches the prey to recognise and maintain that situation.

  3. Ice-aware cover direction for prey:
     _cover_direction now returns the nearest ice zone centre (not just obstacle
     centre) when the hunter is within THREAT_RADIUS and ice zones exist.
     Gives the prey a concrete "run THERE to bait" directional cue just like
     the existing obstacle cover cue.

  4. obs_size: +2 (disp_frac + last_bumped per agent) → 22 + obs*5 + 4 + terrain*6
"""

import numpy as np


# =====================================================================
# World objects
# =====================================================================

class Obstacle:
    """Axis-aligned solid rectangle — blocks movement."""
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


class TerrainZone:
    """Passable rectangular region with a movement status effect.

    kind = "mud"  — slows agent to mud_slow * rated_speed while inside.
    kind = "ice"  — freezes heading and speed from entry until exit.
    """
    __slots__ = ("x", "y", "w", "h", "kind")

    def __init__(self, x, y, w, h, kind):
        self.x, self.y, self.w, self.h = float(x), float(y), float(w), float(h)
        self.kind = kind

    @property
    def cx(self): return self.x + self.w / 2.0
    @property
    def cy(self): return self.y + self.h / 2.0

    def contains_point(self, px, py):
        return (self.x <= px <= self.x + self.w and
                self.y <= py <= self.y + self.h)

    def surface_dist(self, px, py):
        nx = np.clip(px, self.x, self.x + self.w)
        ny = np.clip(py, self.y, self.y + self.h)
        return float(np.sqrt((px - nx)**2 + (py - ny)**2))


# =====================================================================
# Environment
# =====================================================================

class HunterPreyEnv:
    """Open-field pursuit / evasion with obstacles and terrain effects."""

    def __init__(
        self,
        width:        int   = 800,
        height:       int   = 600,
        hunter_speed: float = 4.5,
        prey_speed:   float = 4.0,
        capture_dist: float = 20.0,
        max_steps:    int   = 500,
        agent_radius: float = 10.0,
        n_obstacles_range:   tuple = (3, 6),
        obstacle_size_range: tuple = (40, 100),
        max_obstacles_obs:   int   = 5,
        n_mud_range:    tuple = (0, 0),
        n_ice_range:    tuple = (0, 0),
        mud_size_range: tuple = (80, 150),
        ice_size_range: tuple = (60, 120),
        mud_slow:       float = 0.5,
        max_terrain_obs: int  = 3,
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

        self.n_mud_range     = n_mud_range
        self.n_ice_range     = n_ice_range
        self.mud_size_range  = mud_size_range
        self.ice_size_range  = ice_size_range
        self.mud_slow        = float(mud_slow)
        self.max_terrain_obs = max_terrain_obs

        # ---------- reward constants ----------
        self.R_HEADING    =  0.1
        self.R_BUMP       = -0.05
        self.R_CAPTURE_H  =  100.0
        self.R_CAPTURE_P  = -100.0
        self.R_TIMEOUT_H  =  -20.0
        self.R_TIMEOUT_P  =   20.0
        self.R_STEP_H     =  -0.01
        self.R_STEP_P     =   0.02

        self.R_LOS_BREAK   =   0.5
        self.R_COVER_SEEK  =   0.03
        self.THREAT_RADIUS = 200.0

        self.R_WALL_PREY   = -0.08
        self.R_CORNER_PREY = -0.08
        self.WALL_EDGE     =  40.0

        self.R_CLOSING_H   =  0.15
        self.R_CLOSING_P   = -0.15
        self.PROXIMITY_ZONE = 60.0

        # v6: terrain exploitation rewards (prey only)
        # R_BAIT_ICE: one-time bonus when hunter enters ice while prey is within
        #   THREAT_RADIUS and not on ice itself.  Equal weight to R_LOS_BREAK.
        self.R_BAIT_ICE  = 1.0
        # R_MUD_DRAG: per-step bonus while hunter is in mud.  The speed advantage
        #   (4.5 vs 4.0) is neutralised; prey learns to recognise and hold position.
        self.R_MUD_DRAG  = 0.04

        # ---------- state ----------
        self.hunter_pos  = np.zeros(2, dtype=np.float64)
        self.prey_pos    = np.zeros(2, dtype=np.float64)
        self._h_last_act = np.zeros(2, dtype=np.float32)
        self._p_last_act = np.zeros(2, dtype=np.float32)
        self.obstacles:     list = []
        self.terrain_zones: list = []
        self._prev_los = 1.0
        self.steps = 0
        self.done  = False

        self._h_last_known_prey   = np.zeros(2, dtype=np.float64)
        self._p_last_known_hunter = np.zeros(2, dtype=np.float64)
        self._steps_since_los = 0

        # Ice state
        self._h_on_ice     = False
        self._h_ice_action = np.zeros(2, dtype=np.float32)
        self._p_on_ice     = False
        self._p_ice_action = np.zeros(2, dtype=np.float32)

        # v6: displacement tracking for stuck-detection observation
        self._h_disp_frac    = 0.0   # last-step |displacement| / hunter_speed
        self._p_disp_frac    = 0.0
        self._h_bumped_last  = False
        self._p_bumped_last  = False

    # ---- spaces ----
    @property
    def obs_size(self) -> int:
        # v5 base(18) + obstacles(max_obs*5) + terrain_status(4) + terrain_zones(max_t*6)
        # v6 adds: disp_frac(1) + last_bumped(1) = +2 per agent → base becomes 20
        return 20 + self.max_obstacles_obs * 5 + 4 + self.max_terrain_obs * 6

    @property
    def action_size(self) -> int:
        return 2

    # ---- core API ----
    def reset(self, seed=None) -> dict:
        if seed is not None:
            np.random.seed(seed)

        self.obstacles     = self._spawn_obstacles()
        self.terrain_zones = self._spawn_terrain_zones()

        self.hunter_pos = self._random_free_position()

        min_sep = min(self.W, self.H) * 0.3
        if self.obstacles and np.random.random() < 0.7:
            self.prey_pos = self._spawn_near_obstacle()
        else:
            self.prey_pos = self._random_free_position(padding=min(self.W, self.H) * 0.15)

        for _ in range(200):
            if np.linalg.norm(self.hunter_pos - self.prey_pos) >= min_sep:
                break
            if self.obstacles and np.random.random() < 0.7:
                self.prey_pos = self._spawn_near_obstacle()
            else:
                self.prey_pos = self._random_free_position(padding=min(self.W, self.H) * 0.15)

        self._h_last_act = np.zeros(2, dtype=np.float32)
        self._p_last_act = np.zeros(2, dtype=np.float32)
        self._prev_los = 1.0
        self._h_last_known_prey   = self.prey_pos.copy()
        self._p_last_known_hunter = self.hunter_pos.copy()
        self._steps_since_los = 0

        self._h_on_ice     = False
        self._h_ice_action = np.zeros(2, dtype=np.float32)
        self._p_on_ice     = False
        self._p_ice_action = np.zeros(2, dtype=np.float32)

        # v6: reset displacement tracking
        self._h_disp_frac   = 0.0
        self._p_disp_frac   = 0.0
        self._h_bumped_last = False
        self._p_bumped_last = False

        self.steps = 0
        self.done  = False
        return self._obs()

    def step(self, hunter_action, prey_action):
        assert not self.done, "Call reset() before stepping a finished env."

        h_eff_act, h_eff_spd = self._resolve_terrain(
            self.hunter_pos, hunter_action,
            self.hunter_speed, self._h_on_ice, self._h_ice_action,
        )
        p_eff_act, p_eff_spd = self._resolve_terrain(
            self.prey_pos, prey_action,
            self.prey_speed, self._p_on_ice, self._p_ice_action,
        )

        prev_h    = self.hunter_pos.copy()
        prev_p    = self.prey_pos.copy()
        prev_dist = float(np.linalg.norm(self.hunter_pos - self.prey_pos))

        # Track whether hunter was previously on ice for bait-ice reward detection
        h_was_on_ice = self._h_on_ice

        self.hunter_pos, h_bumped = self._move(self.hunter_pos, h_eff_act, h_eff_spd)
        self.prey_pos,   p_bumped = self._move(self.prey_pos,   p_eff_act, p_eff_spd)

        self._h_last_act = np.clip(h_eff_act, -1, 1).astype(np.float32)
        self._p_last_act = np.clip(p_eff_act, -1, 1).astype(np.float32)
        self.steps += 1

        # v6: update displacement tracking (for next step's observation)
        h_disp_mag = float(np.linalg.norm(self.hunter_pos - prev_h))
        p_disp_mag = float(np.linalg.norm(self.prey_pos - prev_p))
        self._h_disp_frac   = h_disp_mag / max(h_eff_spd, 1e-6)
        self._p_disp_frac   = p_disp_mag / max(p_eff_spd, 1e-6)
        self._h_bumped_last = h_bumped
        self._p_bumped_last = p_bumped

        # Update ice state
        h_zone = self._terrain_at(self.hunter_pos)
        p_zone = self._terrain_at(self.prey_pos)

        h_now_ice = (h_zone is not None and h_zone.kind == "ice")
        p_now_ice = (p_zone is not None and p_zone.kind == "ice")

        if h_now_ice and not self._h_on_ice:
            self._h_on_ice     = True
            self._h_ice_action = h_eff_act.copy()
        elif not h_now_ice:
            self._h_on_ice = False

        if p_now_ice and not self._p_on_ice:
            self._p_on_ice     = True
            self._p_ice_action = p_eff_act.copy()
        elif not p_now_ice:
            self._p_on_ice = False

        # ----------------------------------------------------------------
        # Rewards
        # ----------------------------------------------------------------
        h_disp_vec = self.hunter_pos - prev_h
        p_disp_vec = self.prey_pos   - prev_p
        dist       = float(np.linalg.norm(self.hunter_pos - self.prey_pos))

        rewards = {"hunter": 0.0, "prey": 0.0}

        if dist > 1e-6:
            to_prey    = (self.prey_pos - self.hunter_pos) / dist
            h_approach = float(np.dot(h_disp_vec, to_prey)) / self.hunter_speed
            p_escape   = float(np.dot(p_disp_vec, to_prey)) / self.prey_speed
            rewards["hunter"] += self.R_HEADING * h_approach
            rewards["prey"]   += self.R_HEADING * p_escape

        curr_los  = self._line_of_sight()
        los_broken = False
        if self._prev_los > 0.5 and curr_los < 0.5:
            rewards["prey"] += self.R_LOS_BREAK
            los_broken = True
        self._prev_los = curr_los

        if curr_los > 0.5:
            self._h_last_known_prey   = self.prey_pos.copy()
            self._p_last_known_hunter = self.hunter_pos.copy()
            self._steps_since_los = 0
        else:
            self._steps_since_los += 1

        # Cover-seeking — prey moves toward nearest obstacle when threatened
        if dist < self.THREAT_RADIUS and self.obstacles:
            cover_dir = self._cover_direction(self.prey_pos, dist)
            p_moved   = float(np.linalg.norm(p_disp_vec))
            if p_moved > 0.1 and np.linalg.norm(cover_dir) > 0.1:
                cover_app = float(np.dot(p_disp_vec, cover_dir)) / self.prey_speed
                rewards["prey"] += self.R_COVER_SEEK * cover_app

        # Hunter search (LOS broken)
        if curr_los < 0.5 and self._steps_since_los < 100:
            h_moved = float(np.linalg.norm(h_disp_vec))
            if h_moved > 0.1:
                to_lk   = self._h_last_known_prey - self.hunter_pos
                lk_dist = np.linalg.norm(to_lk)
                if lk_dist > 5.0:
                    search_dir = to_lk / lk_dist
                    search_app = float(np.dot(h_disp_vec, search_dir)) / self.hunter_speed
                    rewards["hunter"] += self.R_HEADING * 0.5 * search_app

        if h_bumped: rewards["hunter"] += self.R_BUMP
        if p_bumped: rewards["prey"]   += self.R_BUMP

        rewards["hunter"] += self.R_STEP_H
        rewards["prey"]   += self.R_STEP_P

        px, py = self.prey_pos
        n_walls = (int(px < self.WALL_EDGE) + int(px > self.W - self.WALL_EDGE) +
                   int(py < self.WALL_EDGE) + int(py > self.H - self.WALL_EDGE))
        if n_walls >= 1: rewards["prey"] += self.R_WALL_PREY
        if n_walls >= 2: rewards["prey"] += self.R_CORNER_PREY

        if dist < self.PROXIMITY_ZONE or prev_dist < self.PROXIMITY_ZONE:
            delta_dist = prev_dist - dist
            if delta_dist > 0:
                cn = delta_dist / self.hunter_speed
                rewards["hunter"] += self.R_CLOSING_H * cn
                rewards["prey"]   += self.R_CLOSING_P * cn

        # ---- v6: terrain exploitation rewards (prey only) ----
        # R_BAIT_ICE: hunter just entered ice, prey is nearby and not frozen itself
        hunter_just_entered_ice = self._h_on_ice and not h_was_on_ice
        if hunter_just_entered_ice and not self._p_on_ice and dist < self.THREAT_RADIUS:
            rewards["prey"] += self.R_BAIT_ICE

        # R_MUD_DRAG: per-step bonus while hunter is slowed by mud
        h_in_mud = (h_zone is not None and h_zone.kind == "mud")
        if h_in_mud:
            rewards["prey"] += self.R_MUD_DRAG

        # Terminal
        captured = dist <= self.capture_dist
        timeout  = self.steps >= self.max_steps
        self.done = captured or timeout
        if captured:
            rewards["hunter"] += self.R_CAPTURE_H
            survival_frac = self.steps / self.max_steps
            rewards["prey"] += self.R_CAPTURE_P * (1.0 - 0.5 * survival_frac)
        elif timeout:
            rewards["hunter"] += self.R_TIMEOUT_H
            rewards["prey"]   += self.R_TIMEOUT_P

        p_in_mud = (p_zone is not None and p_zone.kind == "mud")

        info = {
            "captured": captured,  "distance": dist,   "steps": self.steps,
            "h_bumped": h_bumped,  "p_bumped": p_bumped,
            "los": curr_los,       "los_broken": los_broken,
            "h_on_ice": self._h_on_ice, "p_on_ice": self._p_on_ice,
            "h_in_mud": h_in_mud,        "p_in_mud": p_in_mud,
        }
        return self._obs(), rewards, self.done, info

    # ---- curriculum setters ----
    def set_obstacle_range(self, lo, hi):
        self.n_obstacles_range = (int(lo), int(hi))

    def set_speeds(self, hunter_speed, prey_speed):
        self.hunter_speed = float(hunter_speed)
        self.prey_speed   = float(prey_speed)

    def set_mud_range(self, lo, hi):
        self.n_mud_range = (int(lo), int(hi))

    def set_ice_range(self, lo, hi):
        self.n_ice_range = (int(lo), int(hi))

    # =================================================================
    # Terrain helpers
    # =================================================================

    def _resolve_terrain(self, pos, action, rated_speed, on_ice, ice_action):
        if on_ice:
            return ice_action, rated_speed
        z = self._terrain_at(pos)
        if z is not None and z.kind == "mud":
            return action, rated_speed * self.mud_slow
        return action, rated_speed

    def _terrain_at(self, pos):
        for z in self.terrain_zones:
            if z.contains_point(pos[0], pos[1]):
                return z
        return None

    def _in_mud(self, pos) -> bool:
        z = self._terrain_at(pos)
        return z is not None and z.kind == "mud"

    def _in_ice(self, pos) -> bool:
        z = self._terrain_at(pos)
        return z is not None and z.kind == "ice"

    def _terrain_features(self, agent_pos: np.ndarray) -> np.ndarray:
        """K nearest terrain zones as (rel_cx, rel_cy, w, h, dist, type)."""
        K = self.max_terrain_obs
        if not self.terrain_zones:
            return np.tile([0.0, 0.0, 0.0, 0.0, 1.0, 0.0], K).astype(np.float32)

        entries = [(z.surface_dist(agent_pos[0], agent_pos[1]), z)
                   for z in self.terrain_zones]
        entries.sort(key=lambda e: e[0])
        entries = entries[:K]

        feats = []
        for sd, z in entries:
            feats.extend([
                (z.cx - agent_pos[0]) / self.W,
                (z.cy - agent_pos[1]) / self.H,
                z.w / self.W, z.h / self.H,
                sd / self.DIAG,
                0.0 if z.kind == "mud" else 1.0,
            ])
        for _ in range(K - len(entries)):
            feats.extend([0.0, 0.0, 0.0, 0.0, 1.0, 0.0])
        return np.array(feats, dtype=np.float32)

    # =================================================================
    # Observation
    # =================================================================

    def _obs(self) -> dict:
        dist = float(np.linalg.norm(self.hunter_pos - self.prey_pos))
        los  = self._line_of_sight()

        h_cover   = self._cover_direction(self.hunter_pos, dist)
        p_cover   = self._cover_direction(self.prey_pos,   dist)
        h_freedom = self._escape_freedom(self.hunter_pos, self.prey_pos)
        p_freedom = self._escape_freedom(self.prey_pos,   self.hunter_pos)

        staleness  = min(1.0, self._steps_since_los / 100.0)
        h_last_dir = self._last_known_dir(self.hunter_pos, self._h_last_known_prey)
        p_last_dir = self._last_known_dir(self.prey_pos,   self._p_last_known_hunter)

        h_in_mud = float(self._in_mud(self.hunter_pos))
        h_in_ice = float(self._h_on_ice)
        p_in_mud = float(self._in_mud(self.prey_pos))
        p_in_ice = float(self._p_on_ice)

        h_terrain = self._terrain_features(self.hunter_pos)
        p_terrain = self._terrain_features(self.prey_pos)

        h_obs = np.concatenate([
            self.hunter_pos / np.array([self.W, self.H]),            # 2
            (self.prey_pos - self.hunter_pos) / np.array([self.W, self.H]),  # 2
            [dist / self.DIAG],                                       # 1
            self._h_last_act,                                         # 2
            [los],                                                    # 1
            self._wall_distances(self.hunter_pos),                   # 4
            h_cover,                                                  # 2
            [h_freedom],                                              # 1
            h_last_dir,                                               # 2
            [staleness],                                              # 1
            [self._h_disp_frac, float(self._h_bumped_last)],        # 2  ← v6 stuck signal
            self._obstacle_features(self.hunter_pos),                # max_obs*5
            [h_in_mud, h_in_ice, p_in_mud, p_in_ice],               # 4
            h_terrain,                                                # max_terrain*6
        ]).astype(np.float32)

        p_obs = np.concatenate([
            self.prey_pos / np.array([self.W, self.H]),
            (self.hunter_pos - self.prey_pos) / np.array([self.W, self.H]),
            [dist / self.DIAG],
            self._p_last_act,
            [los],
            self._wall_distances(self.prey_pos),
            p_cover,
            [p_freedom],
            p_last_dir,
            [staleness],
            [self._p_disp_frac, float(self._p_bumped_last)],        # 2  ← v6 stuck signal
            self._obstacle_features(self.prey_pos),
            [p_in_mud, p_in_ice, h_in_mud, h_in_ice],               # own first, then other
            p_terrain,
        ]).astype(np.float32)

        return {"hunter": h_obs, "prey": p_obs}

    # =================================================================
    # Spawning
    # =================================================================

    def _spawn_obstacles(self):
        lo, hi = self.n_obstacles_range
        if hi <= 0:
            return []
        n     = np.random.randint(max(0, lo), max(1, hi) + 1)
        smin, smax = self.obstacle_size_range
        margin = self.agent_radius * 4
        out   = []
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

    def _spawn_terrain_zones(self):
        zones  = []
        margin = self.agent_radius * 3

        for kind, (lo, hi), size_range in [
            ("mud", self.n_mud_range, self.mud_size_range),
            ("ice", self.n_ice_range, self.ice_size_range),
        ]:
            if hi <= 0:
                continue
            n    = np.random.randint(max(0, lo), max(1, hi) + 1)
            smin, smax = size_range
            placed = attempts = 0
            while placed < n and attempts < 300:
                attempts += 1
                w    = float(np.random.randint(smin, smax + 1))
                h    = float(np.random.randint(smin, smax + 1))
                x    = float(np.random.uniform(margin, self.W - w - margin))
                y    = float(np.random.uniform(margin, self.H - h - margin))
                cand = TerrainZone(x, y, w, h, kind)
                if any(self._rects_overlap(cand, o, pad=5) for o in self.obstacles):
                    continue
                if any(self._rects_overlap(cand, z, pad=5) for z in zones):
                    continue
                zones.append(cand)
                placed += 1
        return zones

    @staticmethod
    def _rects_overlap(a, b, pad=0.0) -> bool:
        return not (
            a.x + a.w + pad < b.x or b.x + b.w + pad < a.x or
            a.y + a.h + pad < b.y or b.y + b.h + pad < a.y
        )

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
            o    = self.obstacles[np.random.randint(len(self.obstacles))]
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

    def _collides_any(self, x, y):
        return any(o.contains_circle(x, y, self.agent_radius) for o in self.obstacles)

    _ESCAPE_ANGLES = np.linspace(0, 2 * np.pi, 9)[:-1]
    _ESCAPE_DIRS   = np.stack([np.cos(_ESCAPE_ANGLES), np.sin(_ESCAPE_ANGLES)], axis=1)

    def _move(self, pos, action, speed):
        a   = np.asarray(action, dtype=np.float64)
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

    def _line_of_sight(self):
        if not self.obstacles:
            return 1.0
        for t in np.linspace(0.05, 0.95, 15):
            x = self.hunter_pos[0] + t * (self.prey_pos[0] - self.hunter_pos[0])
            y = self.hunter_pos[1] + t * (self.prey_pos[1] - self.hunter_pos[1])
            for o in self.obstacles:
                if o.x <= x <= o.x + o.w and o.y <= y <= o.y + o.h:
                    return 0.0
        return 1.0

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

    def _cover_direction(self, agent_pos: np.ndarray, dist_to_other: float = float("inf")) -> np.ndarray:
        """Unit vector toward best cover target.

        v6 change: when prey is under threat (dist < THREAT_RADIUS) and ice
        zones exist, the nearest ICE zone is included as a candidate cover target
        alongside solid obstacles.  The policy can then discover that running
        toward ice lures the hunter in, producing the R_BAIT_ICE reward.

        For the hunter, dist_to_other is not used (no ice-bait incentive) so
        the method behaves identically to the original for hunters.
        """
        candidates = []  # (surface_dist, direction_vector)

        for o in self.obstacles:
            d  = np.array([o.cx - agent_pos[0], o.cy - agent_pos[1]])
            sd = o.surface_dist(agent_pos[0], agent_pos[1])
            candidates.append((sd, d))

        # Include nearest ice zone as a cover target when prey is threatened
        if dist_to_other < self.THREAT_RADIUS:
            ice_zones = [z for z in self.terrain_zones if z.kind == "ice"]
            for z in ice_zones:
                d  = np.array([z.cx - agent_pos[0], z.cy - agent_pos[1]])
                sd = z.surface_dist(agent_pos[0], agent_pos[1])
                candidates.append((sd, d))

        if not candidates:
            return np.zeros(2, dtype=np.float32)

        best_sd, best_dir = min(candidates, key=lambda c: c[0])
        mag = np.linalg.norm(best_dir)
        if mag < 1e-8:
            return np.zeros(2, dtype=np.float32)
        return (best_dir / mag).astype(np.float32)

    def _wall_distances(self, pos):
        return np.array([pos[0] / self.W, (self.W - pos[0]) / self.W,
                          pos[1] / self.H, (self.H - pos[1]) / self.H], dtype=np.float32)

    def _escape_freedom(self, agent_pos, other_pos):
        diff = agent_pos - other_pos; dist = np.linalg.norm(diff)
        if dist < 1e-4: return 1.0
        away = diff / dist; r = self.agent_radius; probe_dist = 50.0
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

    def _last_known_dir(self, agent_pos, last_known_other):
        diff = last_known_other - agent_pos; mag = np.linalg.norm(diff)
        return np.zeros(2, dtype=np.float32) if mag < 1e-4 else (diff / mag).astype(np.float32)
