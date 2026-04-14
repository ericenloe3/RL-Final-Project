# my_game_env.py
"""
'The Most Dangerous Game' — PettingZoo ParallelEnv (v4).

What changed from v3 and why
------------------------------
CORE FIX – Separate HUNTER_SPEED and PREY_SPEED
    v3 used a single AGENT_SPEED = 5 for both agents.  This is the root cause
    of training failure: a prey running directly away at full speed can never
    be caught by a hunter of equal speed, so the distance stays constant,
    the capture reward (the only strong terminal signal) is never observed,
    and both agents converge to doing nothing purposeful.

    HUNTER_SPEED = 5.5, PREY_SPEED = 4.5 gives the hunter a ~22% advantage.
    In open space the hunter closes distance at 1 px/step, enough to cross
    half the screen in 400 steps — definitely catchable.  With good obstacle
    navigation the prey can still outmanoeuvre the hunter and escape, keeping
    the game competitive.

TUNED – Hunter obstacle hit penalty reduced (-2.0 → -0.5)
    The old penalty was so large that the hunter learned to stay away from
    walls entirely, which prevents it from learning to navigate around
    obstacles to pursue prey.  A smaller penalty still discourages unnecessary
    wall-bumping without making obstacle-adjacent pursuit impossible.

ADDED – Proximity shaping reward
    A dense, shaped bonus kicks in when the hunter is within PROXIMITY_THRESH
    pixels of the prey.  This bridges the large gap between the progress
    reward (which only fires when distance changes) and the terminal capture
    reward (+100), giving the hunter a gradient to follow when it is close.

Observation vector (62 floats, all ≈ [-1, 1] or [0, 1]):
    [0:2]   own position / [W, H]
    [2:4]   (other_pos − own_pos) / [W, H]
    [4]     euclidean distance / diagonal
    [5:7]   last action taken (velocity proxy)
    [7]     distance to left wall   / W
    [8]     distance to right wall  / W
    [9]     distance to top wall    / H
    [10]    distance to bottom wall / H
    [11]    line-of-sight to other agent (0 or 1)
    [12:62] obstacle features (max_obstacles × 5), sorted nearest-first:
              [rel_cx/W, rel_cy/H, width/W, height/H, surface_dist/diagonal]
"""

import pygame
import numpy as np
from gymnasium import spaces
from pettingzoo import ParallelEnv
from pettingzoo.utils import wrappers

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
BLACK         = (0,   0,   0)
WHITE         = (255, 255, 255)
BLUE          = (0,   0,   255)
RED           = (255, 0,   0)

SCREEN_WIDTH  = 800
SCREEN_HEIGHT = 600
AGENT_SIZE    = 15

# --- Speed asymmetry (KEY FIX) ---
# Hunter is ~22% faster, enough to catch fleeing prey in open space over
# 1000 steps.  Prey can still escape by navigating obstacles cleverly.
HUNTER_SPEED  = 5.5
PREY_SPEED    = 4.5

MAX_TIMESTEPS = 1_000

# Reward scales
R_CAPTURE_HUNTER   =  100.0
R_CAPTURE_PREY     = -100.0
R_TIMEOUT_HUNTER   =  -10.0
R_TIMEOUT_PREY     =   10.0
R_STEP_HUNTER      =   -0.05   # time pressure on hunter
R_STEP_PREY        =    0.05   # survival bonus for prey
R_PROGRESS_SCALE   =    2.0    # scales (Δdist / avg_speed) each step
R_OBSTACLE_HIT     =   -0.5    # hunter bumps a wall (was -2.0; reduced so
                                # the hunter still navigates near obstacles)
R_OBSTACLE_HIT_PREY=   -1.0    # prey bumps a wall
R_WALL_PENALTY     =    0.8    # max per-step penalty for hugging screen edge
WALL_MARGIN        =   60      # pixels — inside this zone penalty ramps up

# Proximity shaping: dense bonus when hunter is within this range of prey.
# Bridges the gap between sparse progress reward and the terminal +100.
R_PROXIMITY_SCALE  =    1.5    # max per-step proximity bonus
PROXIMITY_THRESH   =  200.0    # pixels — shaping active inside this radius

# Capture condition
CAPTURE_DIST       = AGENT_SIZE * 2   # 30 px

# LOS sampling
LOS_SAMPLES = 15


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------
def env(**kwargs):
    from pettingzoo.utils import parallel_to_aec
    aec = parallel_to_aec(raw_env(**kwargs))
    aec = wrappers.AssertOutOfBoundsWrapper(aec)
    aec = wrappers.OrderEnforcingWrapper(aec)
    return aec

def parallel_env(**kwargs):
    return MostDangerousGameEnv(**kwargs)

def raw_env(**kwargs):
    return MostDangerousGameEnv(**kwargs)


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
class MostDangerousGameEnv(ParallelEnv):

    metadata = {
        "name": "most_dangerous_game_v4",
        "render_modes": ["human", "rgb_array"],
        "render_fps": 30,
        "is_parallelizable": True,
    }

    _W    = float(SCREEN_WIDTH)
    _H    = float(SCREEN_HEIGHT)
    _DIAG = float(np.sqrt(SCREEN_WIDTH**2 + SCREEN_HEIGHT**2))

    def __init__(self, render_mode=None):
        super().__init__()
        self.render_mode = render_mode
        self.screen = None
        self.clock  = None
        self._font  = None

        self.possible_agents = ["hunter", "prey"]

        # Obstacle bookkeeping — can be overridden via set_obstacle_range()
        self.max_obstacles     = 10
        self.min_obstacle_size = 30
        self.max_obstacle_size = 80
        self._obs_min = 0        # curriculum: current obstacle count range
        self._obs_max = 5
        self.obstacles: list[pygame.Rect] = []

        # Runtime state
        self.hunter_pos:         np.ndarray = np.zeros(2, dtype=np.float32)
        self.prey_pos:           np.ndarray = np.zeros(2, dtype=np.float32)
        self.hunter_last_action: np.ndarray = np.zeros(2, dtype=np.float32)
        self.prey_last_action:   np.ndarray = np.zeros(2, dtype=np.float32)
        self.timestep = 0

        # obs = 2+2+1+2+4+1+max_obstacles*5 = 62
        self._obs_dim = 2 + 2 + 1 + 2 + 4 + 1 + self.max_obstacles * 5

        self._observation_spaces = {
            a: spaces.Box(-np.inf, np.inf, shape=(self._obs_dim,), dtype=np.float32)
            for a in self.possible_agents
        }
        self._action_spaces = {
            a: spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
            for a in self.possible_agents
        }

    # ------------------------------------------------------------------
    # Curriculum control (called from train.py)
    # ------------------------------------------------------------------
    def set_obstacle_range(self, min_n: int, max_n: int) -> None:
        """Set the range of obstacles spawned on each reset."""
        self._obs_min = int(min_n)
        self._obs_max = int(max_n)

    # ------------------------------------------------------------------
    # PettingZoo API
    # ------------------------------------------------------------------
    def observation_space(self, agent):
        return self._observation_spaces[agent]

    def action_space(self, agent):
        return self._action_spaces[agent]

    # ------------------------------------------------------------------
    # Observation helpers
    # ------------------------------------------------------------------
    def _wall_distances(self, pos: np.ndarray) -> np.ndarray:
        """Normalised distances to each screen boundary [left, right, top, bottom]."""
        return np.array([
            pos[0]                  / self._W,
            (self._W - pos[0])      / self._W,
            pos[1]                  / self._H,
            (self._H - pos[1])      / self._H,
        ], dtype=np.float32)

    def _line_of_sight(self, pos_a: np.ndarray, pos_b: np.ndarray) -> float:
        """1.0 if the straight line between pos_a and pos_b is clear, else 0.0."""
        for t in np.linspace(0.05, 0.95, LOS_SAMPLES):
            pt = pos_a + t * (pos_b - pos_a)
            r  = pygame.Rect(int(pt[0]) - 1, int(pt[1]) - 1, 2, 2)
            if any(r.colliderect(o) for o in self.obstacles):
                return 0.0
        return 1.0

    def _obstacle_features(self, agent_pos: np.ndarray) -> np.ndarray:
        """
        (max_obstacles × 5) array sorted nearest-first.
        Per entry: [rel_cx/W, rel_cy/H, w/W, h/H, surf_dist/DIAG]
        Padding uses dist=1.0 (harmlessly far).
        """
        W, H, DIAG = self._W, self._H, self._DIAG
        entries = []
        for r in self.obstacles:
            cx = r.x + r.width  / 2.0
            cy = r.y + r.height / 2.0
            nx = float(np.clip(agent_pos[0], r.x, r.x + r.width))
            ny = float(np.clip(agent_pos[1], r.y, r.y + r.height))
            sd = float(np.linalg.norm(agent_pos - np.array([nx, ny])))
            entries.append((sd,
                            (cx - agent_pos[0]) / W,
                            (cy - agent_pos[1]) / H,
                            r.width  / W,
                            r.height / H,
                            sd / DIAG))

        entries.sort(key=lambda e: e[0])
        feats: list[float] = []
        for e in entries:
            feats.extend(e[1:])

        n_pad = self.max_obstacles - len(entries)
        feats += [0.0, 0.0, 0.0, 0.0, 1.0] * n_pad
        return np.array(feats, dtype=np.float32)

    def _get_obs(self) -> dict[str, np.ndarray]:
        W, H, DIAG = self._W, self._H, self._DIAG
        dist = float(np.linalg.norm(self.hunter_pos - self.prey_pos))
        los  = self._line_of_sight(self.hunter_pos, self.prey_pos)

        hunter_obs = np.concatenate([
            self.hunter_pos / np.array([W, H]),                       # own pos
            (self.prey_pos - self.hunter_pos) / np.array([W, H]),     # rel prey
            [dist / DIAG],                                             # distance
            self.hunter_last_action,                                   # velocity proxy
            self._wall_distances(self.hunter_pos),                    # boundary awareness
            [los],                                                     # line of sight
            self._obstacle_features(self.hunter_pos),
        ]).astype(np.float32)

        prey_obs = np.concatenate([
            self.prey_pos / np.array([W, H]),                         # own pos
            (self.hunter_pos - self.prey_pos) / np.array([W, H]),     # rel hunter
            [dist / DIAG],                                             # distance
            self.prey_last_action,                                     # velocity proxy
            self._wall_distances(self.prey_pos),                      # boundary awareness
            [los],                                                     # line of sight
            self._obstacle_features(self.prey_pos),
        ]).astype(np.float32)

        return {"hunter": hunter_obs, "prey": prey_obs}

    def _get_infos(self) -> dict[str, dict]:
        dist = float(np.linalg.norm(self.hunter_pos - self.prey_pos))
        shared = {
            "distance":   dist,
            "hunter_pos": self.hunter_pos.copy(),
            "prey_pos":   self.prey_pos.copy(),
        }
        return {a: shared.copy() for a in self.possible_agents}

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------
    def _get_random_safe_position(self) -> np.ndarray:
        for _ in range(1000):#number of steps??
            pos = np.array([
                np.random.uniform(AGENT_SIZE * 3, SCREEN_WIDTH  - AGENT_SIZE * 3),
                np.random.uniform(AGENT_SIZE * 3, SCREEN_HEIGHT - AGENT_SIZE * 3),
            ], dtype=np.float32)
            rect = pygame.Rect(
                int(pos[0]) - AGENT_SIZE, int(pos[1]) - AGENT_SIZE,
                AGENT_SIZE * 2, AGENT_SIZE * 2,
            )
            if not any(rect.colliderect(o) for o in self.obstacles):
                return pos
        # Fallback: return centre if map is impossibly crowded
        return np.array([SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2], dtype=np.float32)

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)

        pygame.init()
        self.agents   = self.possible_agents[:]
        self.timestep = 0
        self.hunter_last_action = np.zeros(2, dtype=np.float32)
        self.prey_last_action   = np.zeros(2, dtype=np.float32)

        # Spawn obstacles within the current curriculum range
        self.obstacles.clear()
        if self._obs_max > 0:
            n_obs = np.random.randint(
                max(0, self._obs_min),
                max(1, self._obs_max) + 1,
            )
            for _ in range(n_obs):
                w = np.random.randint(self.min_obstacle_size, self.max_obstacle_size)
                h = np.random.randint(self.min_obstacle_size, self.max_obstacle_size)
                x = np.random.uniform(0, SCREEN_WIDTH  - w)
                y = np.random.uniform(0, SCREEN_HEIGHT - h)
                self.obstacles.append(pygame.Rect(int(x), int(y), w, h))

        # Place agents at least 30% of the short screen dimension apart
        self.hunter_pos = self._get_random_safe_position()
        self.prey_pos   = self._get_random_safe_position()
        min_sep = min(SCREEN_WIDTH, SCREEN_HEIGHT) * 0.3
        for _ in range(300):
            if np.linalg.norm(self.hunter_pos - self.prey_pos) >= min_sep:
                break
            self.prey_pos = self._get_random_safe_position()

        if self.render_mode == "human":
            self._render_frame()

        return self._get_obs(), self._get_infos()

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------
    def _collides(self, pos: np.ndarray) -> bool:
        """True if a circle at pos overlaps any obstacle rect."""
        rect = pygame.Rect(
            int(pos[0]) - AGENT_SIZE, int(pos[1]) - AGENT_SIZE,
            AGENT_SIZE * 2, AGENT_SIZE * 2,
        )
        return any(rect.colliderect(o) for o in self.obstacles)

    def _try_move(self, pos: np.ndarray, action: np.ndarray, speed: float):
        """Move pos by action*speed with sliding collision response.

        Instead of a full stop on any collision, we try each axis separately
        so agents slide along obstacle edges and screen boundaries rather than
        freezing in place.  Returns (new_pos, hit_obstacle).

        Priority:
          1. Full move  -- accepted if clear.
          2. X-only     -- slide along Y face of obstacle.
          3. Y-only     -- slide along X face of obstacle.
          4. No move    -- only if both axes are individually blocked too.
        """
        delta = action * speed

        def clamp(p):
            q = p.copy()
            q[0] = float(min(max(q[0], AGENT_SIZE), SCREEN_WIDTH  - AGENT_SIZE))
            q[1] = float(min(max(q[1], AGENT_SIZE), SCREEN_HEIGHT - AGENT_SIZE))
            return q

        # 1. Full move
        full = clamp(pos + delta)
        if not self._collides(full):
            return full, False

        # 2. X-only (slide parallel to vertical obstacle face)
        x_only = clamp(pos + np.array([delta[0], 0.0], dtype=np.float32))
        if not self._collides(x_only):
            return x_only, True

        # 3. Y-only (slide parallel to horizontal obstacle face)
        y_only = clamp(pos + np.array([0.0, delta[1]], dtype=np.float32))
        if not self._collides(y_only):
            return y_only, True

        # 4. Truly cornered -- stay put
        return pos.copy(), True

    def _wall_penalty(self, pos: np.ndarray) -> float:
        """Smooth penalty that ramps from 0 at WALL_MARGIN to R_WALL_PENALTY at the edge."""
        d = min(
            pos[0],
            self._W - pos[0],
            pos[1],
            self._H - pos[1],
        )
        if d >= WALL_MARGIN:
            return 0.0
        return R_WALL_PENALTY * (1.0 - d / WALL_MARGIN)

    def step(self, actions: dict):
        rewards      = {a: 0.0 for a in self.possible_agents}
        terminations = {a: False for a in self.possible_agents}
        truncations  = {a: False for a in self.possible_agents}

        prev_dist = float(np.linalg.norm(self.hunter_pos - self.prey_pos))

        # ---- Apply actions (with per-agent speeds) -------------------
        h_act = np.clip(np.array(actions.get("hunter", [0, 0]), dtype=np.float32), -1, 1)
        p_act = np.clip(np.array(actions.get("prey",   [0, 0]), dtype=np.float32), -1, 1)

        new_hunter, h_wall = self._try_move(self.hunter_pos, h_act, HUNTER_SPEED)
        new_prey,   p_wall = self._try_move(self.prey_pos,   p_act, PREY_SPEED)

        self.hunter_pos         = new_hunter
        self.prey_pos           = new_prey
        self.hunter_last_action = h_act
        self.prey_last_action   = p_act

        if h_wall:
            rewards["hunter"] += R_OBSTACLE_HIT        # -0.5 (reduced from -2.0)
        if p_wall:
            rewards["prey"]   += R_OBSTACLE_HIT_PREY   # -1.0

        self.timestep += 1
        curr_dist = float(np.linalg.norm(self.hunter_pos - self.prey_pos))

        # ---- Dense progress reward -----------------------------------
        # Normalise by average speed so the scale is consistent regardless
        # of which agent moved more.
        avg_speed  = (HUNTER_SPEED + PREY_SPEED) / 2.0
        dist_delta = prev_dist - curr_dist              # >0 → hunter closed in
        progress   = dist_delta / avg_speed
        rewards["hunter"] += R_PROGRESS_SCALE *  progress
        rewards["prey"]   += R_PROGRESS_SCALE * -progress

        # ---- Proximity shaping reward --------------------------------
        # Dense bonus that scales linearly from 0 at PROXIMITY_THRESH
        # down to R_PROXIMITY_SCALE at distance 0.  This bridges the gap
        # between per-step progress rewards and the large terminal capture
        # reward, giving the hunter a clear gradient to follow when close.
        if curr_dist < PROXIMITY_THRESH:
            prox = R_PROXIMITY_SCALE * (1.0 - curr_dist / PROXIMITY_THRESH)
            rewards["hunter"] += prox
            rewards["prey"]   -= prox

        # ---- Wall-proximity penalty (both agents) --------------------
        rewards["hunter"] -= self._wall_penalty(self.hunter_pos)
        rewards["prey"]   -= self._wall_penalty(self.prey_pos)

        # ---- Per-step baseline --------------------------------------
        rewards["hunter"] += R_STEP_HUNTER
        rewards["prey"]   += R_STEP_PREY

        # ---- Terminal conditions ------------------------------------
        if curr_dist <= CAPTURE_DIST:
            rewards["hunter"] += R_CAPTURE_HUNTER
            rewards["prey"]   += R_CAPTURE_PREY
            terminations = {a: True for a in self.possible_agents}

        elif self.timestep >= MAX_TIMESTEPS:
            rewards["hunter"] += R_TIMEOUT_HUNTER
            rewards["prey"]   += R_TIMEOUT_PREY
            truncations = {a: True for a in self.possible_agents}

        self.agents = [
            a for a in self.agents
            if not terminations[a] and not truncations[a]
        ]

        if self.render_mode == "human":
            self._render_frame()

        return self._get_obs(), rewards, terminations, truncations, self._get_infos()

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------
    def render(self):
        if self.render_mode == "human":
            self._render_frame()
        elif self.render_mode == "rgb_array":
            return self._get_rgb_array()

    def _render_frame(self):
        if self.screen is None:
            pygame.display.init()
            pygame.display.set_caption("The Most Dangerous Game")
            self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        if self.clock is None:
            self.clock = pygame.time.Clock()
        if self._font is None:
            pygame.font.init()
            self._font = pygame.font.SysFont(None, 22)

        self.screen.fill(WHITE)
        for r in self.obstacles:
            pygame.draw.rect(self.screen, BLACK, r)

        pygame.draw.circle(self.screen, BLUE,
                           self.hunter_pos.astype(int).tolist(), AGENT_SIZE)
        pygame.draw.circle(self.screen, RED,
                           self.prey_pos.astype(int).tolist(), AGENT_SIZE)

        # HUD
        dist = np.linalg.norm(self.hunter_pos - self.prey_pos)
        los  = "Y" if self._line_of_sight(self.hunter_pos, self.prey_pos) else "N"
        hud  = self._font.render(
            f"step {self.timestep:>4}  dist {dist:>5.0f} px  LOS {los}"
            f"  H:{HUNTER_SPEED:.1f}  P:{PREY_SPEED:.1f}",
            True, (80, 80, 80),
        )
        self.screen.blit(hud, (8, 8))

        pygame.display.flip()
        self.clock.tick(self.metadata["render_fps"])

    def _get_rgb_array(self) -> np.ndarray:
        if self.screen is None:
            pygame.display.init()
            self.screen = pygame.display.set_mode(
                (SCREEN_WIDTH, SCREEN_HEIGHT), flags=pygame.NOFRAME
            )
        self._render_frame()
        return np.transpose(
            np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
        )

    def close(self):
        if self.screen is not None:
            pygame.display.quit()
            self.screen = None
            self.clock  = None
