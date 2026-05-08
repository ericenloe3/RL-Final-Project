"""
Hunter-Prey environment — with obstacles (v5: hunter interception fixes).

Core design unchanged:
  - Per-agent heading reward is the dominant signal.
  - No large wall penalties that override heading.

v5 changes (fixes hunter jitter/stalling):
  1. Velocity observations: each agent now observes the other's actual
     displacement from the previous step (normalised by rated speed).
     This gives the hunter enough information to learn *interception*
     (cut-off trajectories) rather than pure tail-chasing, which can never
     close the gap against a prey running perpendicular at 4.0 speed.
  2. Obstacle-bypass navigation direction: new feature added to each agent.
     Hunter gets the direction to the lowest-cost bypass waypoint (obstacle
     corner) when the direct path to prey is blocked.  Prey gets the
     symmetric escape-bypass direction.  Replaces the vague cover_dir signal
     with a concrete "this is where to go next" vector.
  3. Obstacle-aware heading reward: when LOS is blocked the hunter's heading
     reward is computed against the bypass direction, not through the wall.
     This removes the reward gradient that was teaching the hunter to push
     into obstacle faces and oscillate.
  4. Hunter movement bonus (R_MOVE_H): tiny per-step reward for moving
     freely without bumping.  Breaks the stuck-at-obstacle local optimum
     where staying still and bumping have the same heading reward.
  5. Old "hunter search reward" removed — superseded by the nav-dir heading
     reward which already handles the LOS-blocked case correctly.
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
        hunter_speed: float = 4.5,       # default = final curriculum speed
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
        self.R_TIMEOUT_H  =  -20.0     # v6: stronger timeout punishment for hunter
        self.R_TIMEOUT_P  =   20.0     # v6: stronger survival bonus (was 10)
        self.R_STEP_H     =  -0.01
        self.R_STEP_P     =   0.02     # v6: doubled (was 0.01) → 10 total over 500 steps

        # Prey tactical rewards (v3)
        self.R_LOS_BREAK  =   0.5     # one-time bonus when prey breaks LOS
        self.R_COVER_SEEK =   0.03    # per-step bonus for moving toward cover under threat
        self.THREAT_RADIUS = 200.0    # cover-seek only active within this range

        # Anti-wall-camping (v6: stronger than v5)
        self.R_WALL_PREY  =  -0.08    # penalty at edge (80% of heading)
        self.R_CORNER_PREY=  -0.08    # ADDITIONAL in corners → -0.16 total
        self.WALL_EDGE    =  40.0     # pixels — wider zone

        # Anti-jitter: proximity bonus/penalty when agents are very close
        # Without this, deterministic policies oscillate at dist ≈ capture_dist
        # because both get positive heading reward in the standoff.
        self.R_PROXIMITY_H =  0.10    # hunter bonus when very close (strong lunge incentive)
        self.R_PROXIMITY_P = -0.10    # prey penalty when very close (strong break-away incentive)
        self.PROXIMITY_ZONE = 60.0    # activates within this distance

        # v5: Hunter movement bonus — small reward for moving freely (not bumping).
        # Breaks the zero-gradient trap at obstacle faces where standing still
        # and pushing into the wall have the same heading reward (≈ 0 sideways).
        self.R_MOVE_H = 0.015

        # ---------- state ----------
        self.hunter_pos = np.zeros(2, dtype=np.float64)
        self.prey_pos   = np.zeros(2, dtype=np.float64)
        self._h_last_act = np.zeros(2, dtype=np.float32)
        self._p_last_act = np.zeros(2, dtype=np.float32)
        # v5: velocity tracking — stores each agent's actual displacement from
        # the previous step, normalised by rated speed, so the policy can learn
        # interception (cut-off) rather than pure tail-chasing.
        self._h_vel = np.zeros(2, dtype=np.float32)   # hunter velocity (prev step)
        self._p_vel = np.zeros(2, dtype=np.float32)   # prey velocity (prev step)
        self.obstacles = []  # list[Obstacle]
        self._prev_los = 1.0  # track LOS for transition detection
        self.steps = 0
        self.done  = False

        # Last-known position tracking — when LOS is broken, agents
        # remember where the other agent was last seen.
        self._h_last_known_prey = np.zeros(2, dtype=np.float64)
        self._p_last_known_hunter = np.zeros(2, dtype=np.float64)
        self._steps_since_los = 0  # how many steps since LOS was clear

    # ---- spaces ----
    @property
    def obs_size(self) -> int:
        # own_pos(2) + rel_other(2) + dist(1) + last_action(2) + los(1)
        #   + wall_distances(4) + cover_dir(2) + escape_freedom(1)
        #   + last_known_other_dir(2) + staleness(1)
        #   + other_vel(2)       ← v5: other agent's velocity from prev step
        #   + nav_dir(2)         ← v5: obstacle-bypass navigation direction
        #   + max_obstacles_obs * 5
        return 22 + self.max_obstacles_obs * 5

    @property
    def action_size(self) -> int:
        return 2

    # ---- core API ----
    def reset(self, seed=None) -> dict:
        if seed is not None:
            np.random.seed(seed)

        # 1. Spawn obstacles first
        self.obstacles = self._spawn_obstacles()

        # 2. Spawn hunter anywhere in free space
        self.hunter_pos = self._random_free_position()

        # 3. Spawn prey — near an obstacle if available (gives prey
        #    immediate cover access), otherwise away from walls.
        min_sep = min(self.W, self.H) * 0.3
        if self.obstacles and np.random.random() < 0.7:
            # 70% of the time: spawn prey adjacent to a random obstacle
            self.prey_pos = self._spawn_near_obstacle()
        else:
            prey_pad = min(self.W, self.H) * 0.15   # ~90px
            self.prey_pos = self._random_free_position(padding=prey_pad)

        # Ensure minimum separation from hunter
        for _ in range(200):
            if np.linalg.norm(self.hunter_pos - self.prey_pos) >= min_sep:
                break
            if self.obstacles and np.random.random() < 0.7:
                self.prey_pos = self._spawn_near_obstacle()
            else:
                prey_pad = min(self.W, self.H) * 0.15
                self.prey_pos = self._random_free_position(padding=prey_pad)

        self._h_last_act = np.zeros(2, dtype=np.float32)
        self._p_last_act = np.zeros(2, dtype=np.float32)
        self._prev_los = 1.0  # assume LOS clear at episode start
        self._h_last_known_prey = self.prey_pos.copy()
        self._p_last_known_hunter = self.hunter_pos.copy()
        self._steps_since_los = 0
        self._h_vel = np.zeros(2, dtype=np.float32)   # v5: velocities start at zero
        self._p_vel = np.zeros(2, dtype=np.float32)
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
        # v5: LOS is computed HERE (before heading) so the hunter's heading
        # reward can use the obstacle-bypass direction when the path is blocked.
        curr_los = self._line_of_sight()

        h_disp = self.hunter_pos - prev_h
        p_disp = self.prey_pos   - prev_p
        dist   = float(np.linalg.norm(self.hunter_pos - self.prey_pos))

        rewards = {"hunter": 0.0, "prey": 0.0}
        if dist > 1e-6:
            to_prey_dir = (self.prey_pos - self.hunter_pos) / dist

            # Prey: always escape directly away from hunter.
            p_escape = float(np.dot(p_disp, to_prey_dir)) / self.prey_speed
            rewards["prey"] += self.R_HEADING * p_escape

            # Hunter: obstacle-aware heading reward (v5).
            #   • LOS clear  → reward approaching prey directly.
            #   • LOS blocked → reward moving toward the best obstacle-bypass
            #     waypoint.  This removes the gradient pointing through walls
            #     that was causing oscillation at obstacle faces.
            if curr_los > 0.5:
                h_approach = float(np.dot(h_disp, to_prey_dir)) / self.hunter_speed
            else:
                nav_dir = self._compute_nav_dir(self.hunter_pos, self.prey_pos)
                h_approach = float(np.dot(h_disp, nav_dir)) / self.hunter_speed
            rewards["hunter"] += self.R_HEADING * h_approach

        # v5: Hunter movement bonus — tiny reward for actually moving without
        # bumping.  Prevents the zero-gradient trap where the hunter sits at an
        # obstacle face (h_approach ≈ 0, bump_penalty cancels step_penalty).
        h_moved = float(np.linalg.norm(h_disp))
        if h_moved > 0.5 and not h_bumped:
            rewards["hunter"] += self.R_MOVE_H

        # ---- LOS-break reward (prey only, transition 1→0) ----
        los_broken = False
        if self._prev_los > 0.5 and curr_los < 0.5:
            # Prey just broke line-of-sight — one-time bonus
            rewards["prey"] += self.R_LOS_BREAK
            los_broken = True
        self._prev_los = curr_los

        # ---- Update last-known positions ----
        # When LOS is clear, both agents update their memory of
        # where the other agent is.  When LOS is broken, memory
        # goes stale — the staleness counter tells the policy
        # how old the information is.
        if curr_los > 0.5:
            self._h_last_known_prey = self.prey_pos.copy()
            self._p_last_known_hunter = self.hunter_pos.copy()
            self._steps_since_los = 0
        else:
            self._steps_since_los += 1

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

        # v5: Update velocity state for next step's observation.
        # Stored as actual pixel displacement so _obs() can normalise by speed.
        self._h_vel = h_disp.astype(np.float32)
        self._p_vel = p_disp.astype(np.float32)

        # ---- Prey wall/corner penalty (v5) ----
        # Edge penalty: -0.05 when within WALL_EDGE of any boundary
        # Corner penalty: additional -0.05 when near TWO boundaries
        # Combined: -0.10/step in corner vs heading max +0.10 → net zero at best
        px, py = self.prey_pos
        near_left   = px < self.WALL_EDGE
        near_right  = px > self.W - self.WALL_EDGE
        near_top    = py < self.WALL_EDGE
        near_bottom = py > self.H - self.WALL_EDGE
        n_walls = int(near_left) + int(near_right) + int(near_top) + int(near_bottom)
        if n_walls >= 1:
            rewards["prey"] += self.R_WALL_PREY       # -0.05 for any wall
        if n_walls >= 2:
            rewards["prey"] += self.R_CORNER_PREY     # additional -0.05 for corner

        # ---- Anti-jitter: proximity bonus/penalty ----
        # When agents are within PROXIMITY_ZONE, the hunter gets a bonus that
        # scales UP as distance decreases (incentivize closing the final gap).
        # The prey gets a penalty that scales UP as distance decreases
        # (incentivize breaking away rather than maintaining standoff).
        # At dist=60 → factor=0, at dist=20 → factor=1.0
        if dist < self.PROXIMITY_ZONE:
            prox_factor = 1.0 - (dist - self.capture_dist) / (self.PROXIMITY_ZONE - self.capture_dist)
            prox_factor = max(0.0, min(1.0, prox_factor))
            rewards["hunter"] += self.R_PROXIMITY_H * prox_factor
            rewards["prey"]   += self.R_PROXIMITY_P * prox_factor

        # Terminal
        captured = dist <= self.capture_dist
        timeout  = self.steps >= self.max_steps
        self.done = captured or timeout
        if captured:
            rewards["hunter"] += self.R_CAPTURE_H
            # Time-scaled capture penalty: getting caught at step 450 hurts
            # LESS than step 50.  This gives the prey a gradient even when
            # capture is inevitable: "delay as long as possible."
            # Scale: at step 0 → full -100, at step 500 → -50.
            survival_frac = self.steps / self.max_steps
            scaled_penalty = self.R_CAPTURE_P * (1.0 - 0.5 * survival_frac)
            rewards["prey"] += scaled_penalty
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

    def set_speeds(self, hunter_speed: float, prey_speed: float):
        """Set agent speeds — called by train.py for speed curriculum."""
        self.hunter_speed = float(hunter_speed)
        self.prey_speed   = float(prey_speed)

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

    def _random_free_position(self, padding: float = None) -> np.ndarray:
        pad = padding if padding is not None else self.agent_radius + 5
        for _ in range(500):
            pos = np.array([
                np.random.uniform(pad, self.W - pad),
                np.random.uniform(pad, self.H - pad),
            ])
            if not self._collides_any(pos[0], pos[1]):
                return pos
        return np.array([self.W / 2, self.H / 2])

    def _spawn_near_obstacle(self) -> np.ndarray:
        """Spawn a position adjacent to a random obstacle.

        Picks a random obstacle, then a random side, and places the agent
        just outside that face.  Gives the prey immediate cover access
        at the start of an episode — critical for learning to use obstacles.
        """
        if not self.obstacles:
            return self._random_free_position()

        r = self.agent_radius
        for _ in range(100):
            o = self.obstacles[np.random.randint(len(self.obstacles))]
            side = np.random.randint(4)
            if side == 0:    # left
                pos = np.array([o.x - r - 2, o.cy + np.random.uniform(-o.h/3, o.h/3)])
            elif side == 1:  # right
                pos = np.array([o.x + o.w + r + 2, o.cy + np.random.uniform(-o.h/3, o.h/3)])
            elif side == 2:  # top
                pos = np.array([o.cx + np.random.uniform(-o.w/3, o.w/3), o.y - r - 2])
            else:            # bottom
                pos = np.array([o.cx + np.random.uniform(-o.w/3, o.w/3), o.y + o.h + r + 2])

            # Verify in bounds and not colliding with another obstacle
            if (r < pos[0] < self.W - r and r < pos[1] < self.H - r
                    and not self._collides_any(pos[0], pos[1])):
                return pos

        return self._random_free_position()  # fallback

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

    def _line_of_sight_between(self, pos_a: np.ndarray, pos_b: np.ndarray) -> float:
        """1.0 if the straight line between pos_a and pos_b is clear of obstacles."""
        if not self.obstacles:
            return 1.0
        for t in np.linspace(0.05, 0.95, 15):
            x = pos_a[0] + t * (pos_b[0] - pos_a[0])
            y = pos_a[1] + t * (pos_b[1] - pos_a[1])
            for o in self.obstacles:
                if o.x <= x <= o.x + o.w and o.y <= y <= o.y + o.h:
                    return 0.0
        return 1.0

    def _line_of_sight(self) -> float:
        """1.0 if the straight line hunter↔prey is clear, else 0.0."""
        return self._line_of_sight_between(self.hunter_pos, self.prey_pos)

    def _compute_nav_dir(self, from_pos: np.ndarray, to_pos: np.ndarray) -> np.ndarray:
        """Navigation direction from `from_pos` toward `to_pos`, routing around obstacles.

        When the direct path is clear, returns the unit vector toward `to_pos`.
        When blocked, evaluates the four corners of every obstacle as potential
        bypass waypoints and returns the direction toward the corner that minimises
        total travel distance (from_pos → corner → to_pos).  This gives the policy
        a concrete "turn left/right around the obstacle" signal instead of the
        useless "push through the wall" gradient from direct heading.

        Used for:
          • Hunter observation (nav direction toward prey).
          • Hunter heading reward when LOS is blocked.
          • Prey observation (nav direction toward escape target).
        """
        diff = to_pos - from_pos
        dist_to_goal = float(np.linalg.norm(diff))
        if dist_to_goal < 1e-4:
            return np.zeros(2, dtype=np.float32)
        direct_dir = diff / dist_to_goal

        if not self.obstacles:
            return direct_dir.astype(np.float32)

        if self._line_of_sight_between(from_pos, to_pos) > 0.5:
            return direct_dir.astype(np.float32)

        # Path is blocked — find the lowest-cost bypass corner.
        r = self.agent_radius + 4.0  # clearance around obstacle corners
        best_dir  = direct_dir
        best_cost = float("inf")

        for o in self.obstacles:
            corners = [
                np.array([o.x - r,       o.y - r      ]),
                np.array([o.x + o.w + r, o.y - r      ]),
                np.array([o.x - r,       o.y + o.h + r]),
                np.array([o.x + o.w + r, o.y + o.h + r]),
            ]
            for corner in corners:
                # Must be in-bounds and not inside another obstacle
                if not (self.agent_radius < corner[0] < self.W - self.agent_radius and
                        self.agent_radius < corner[1] < self.H - self.agent_radius):
                    continue
                if self._collides_any(corner[0], corner[1]):
                    continue

                to_corner = corner - from_pos
                c_dist = float(np.linalg.norm(to_corner))
                if c_dist < 1e-4:
                    continue

                # Cost = distance from agent to corner + from corner to goal.
                # Prefer corners that are close to agent AND close to goal.
                c_to_goal = float(np.linalg.norm(to_pos - corner))
                cost = c_dist + c_to_goal

                if cost < best_cost:
                    best_cost = cost
                    best_dir  = to_corner / c_dist

        return best_dir.astype(np.float32)

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

    def _escape_freedom(self, agent_pos: np.ndarray, other_pos: np.ndarray) -> float:
        """Fraction of the escape hemisphere that is open (not wall-blocked).

        Samples 8 directions in the half-plane away from the other agent.
        Returns 1.0 when all directions are far from walls (centre of field),
        lower values near edges/corners.  Gives the value function a clean
        signal that "I'm running out of room" before hitting the wall.
        """
        diff = agent_pos - other_pos
        dist = np.linalg.norm(diff)
        if dist < 1e-4:
            return 1.0

        away = diff / dist
        r = self.agent_radius
        probe_dist = 50.0
        open_count = 0
        hemi_count = 0

        for i in range(8):
            angle = (i / 8) * 2 * np.pi
            probe_dir = np.array([np.cos(angle), np.sin(angle)])
            # Only count directions in the escape hemisphere
            if np.dot(probe_dir, away) < 0:
                continue
            hemi_count += 1
            # Check if probe_dist in this direction stays in bounds
            check = agent_pos + probe_dir * probe_dist
            if (check[0] >= r and check[0] <= self.W - r and
                check[1] >= r and check[1] <= self.H - r):
                open_count += 1

        return float(open_count / max(hemi_count, 1))

    def _last_known_dir(self, agent_pos: np.ndarray, last_known_other: np.ndarray) -> np.ndarray:
        """Unit vector from agent toward the other agent's last-known position."""
        diff = last_known_other - agent_pos
        mag = np.linalg.norm(diff)
        if mag < 1e-4:
            return np.zeros(2, dtype=np.float32)
        return (diff / mag).astype(np.float32)

    def _obs(self) -> dict:
        dist = float(np.linalg.norm(self.hunter_pos - self.prey_pos))
        los  = self._line_of_sight()

        h_cover = self._cover_direction(self.hunter_pos)
        p_cover = self._cover_direction(self.prey_pos)

        h_freedom = self._escape_freedom(self.hunter_pos, self.prey_pos)
        p_freedom = self._escape_freedom(self.prey_pos, self.hunter_pos)

        # Last-known direction and staleness (0=fresh, 1=very stale)
        staleness = min(1.0, self._steps_since_los / 100.0)
        h_last_dir = self._last_known_dir(self.hunter_pos, self._h_last_known_prey)
        p_last_dir = self._last_known_dir(self.prey_pos, self._p_last_known_hunter)

        # v5: Velocity observations — normalised actual displacement from last step.
        # Hunter sees prey velocity (enables interception / cut-off learning).
        # Prey sees hunter velocity (enables prediction-based evasion).
        h_prey_vel    = self._p_vel / max(self.prey_speed,   1e-6)  # shape (2,)
        p_hunter_vel  = self._h_vel / max(self.hunter_speed, 1e-6)  # shape (2,)

        # v5: Obstacle-bypass navigation directions.
        # Hunter: toward prey, routed around blocking obstacles.
        # Prey:   away from hunter toward an unblocked escape point.
        h_nav_dir = self._compute_nav_dir(self.hunter_pos, self.prey_pos)

        # Prey escape target: 250 px directly away from hunter (clamped to bounds).
        if dist > 1e-4:
            away = (self.prey_pos - self.hunter_pos) / dist
        else:
            away = np.array([1.0, 0.0])
        escape_target = np.clip(
            self.prey_pos + away * 250.0,
            [self.agent_radius, self.agent_radius],
            [self.W - self.agent_radius, self.H - self.agent_radius],
        )
        p_nav_dir = self._compute_nav_dir(self.prey_pos, escape_target)

        h_obs = np.concatenate([
            self.hunter_pos / np.array([self.W, self.H]),
            (self.prey_pos - self.hunter_pos) / np.array([self.W, self.H]),
            [dist / self.DIAG],
            self._h_last_act,
            [los],
            self._wall_distances(self.hunter_pos),
            h_cover,
            [h_freedom],
            h_last_dir,
            [staleness],
            h_prey_vel,      # v5: prey velocity (2)
            h_nav_dir,       # v5: bypass navigation direction (2)
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
            [p_freedom],
            p_last_dir,
            [staleness],
            p_hunter_vel,    # v5: hunter velocity (2)
            p_nav_dir,       # v5: escape navigation direction (2)
            self._obstacle_features(self.prey_pos),
        ]).astype(np.float32)

        return {"hunter": h_obs, "prey": p_obs}
