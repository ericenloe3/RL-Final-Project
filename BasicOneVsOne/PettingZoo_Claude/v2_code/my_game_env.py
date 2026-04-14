# my_game_env.py
"""
'The Most Dangerous Game' — PettingZoo ParallelEnv (v2, improved).

Key improvements over v1
------------------------
1. Relative, normalised observations
     Each agent sees the other as a (Δx/W, Δy/H) vector plus an explicit
     normalised distance.  All values live in roughly [-1, 1] so gradient
     magnitudes stay well-behaved without a separate normaliser.

2. Obstacles sorted nearest-first per agent
     Feature index 0 always describes the closest obstacle, giving the
     network a stable spatial reference.  Distance is computed to the
     nearest surface point (not centre) for accuracy.

3. Last action (velocity proxy) in state
     Including the agent's previous action lets the policy model momentum
     without a recurrent architecture.

4. Dense progress-based reward shaping
     At every step the hunter earns R_PROGRESS_SCALE * (Δdist / AGENT_SPEED)
     (positive when closing in) and the prey earns the mirror image.
     This creates a smooth gradient at every timestep rather than waiting
     for the sparse terminal signal.

5. Heading bonus
     Each agent additionally earns R_HEADING_SCALE * dot(action, ideal_dir),
     directly rewarding movement in the correct direction.

Install
-------
    pip install pettingzoo pygame numpy gymnasium
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
BLUE          = (0,   0,   255)   # Hunter
RED           = (255, 0,   0)     # Prey
GREEN         = (0,   180, 0)     # Heading-arrow colour

SCREEN_WIDTH  = 800
SCREEN_HEIGHT = 600
AGENT_SIZE    = 15
AGENT_SPEED   = 5
MAX_TIMESTEPS = 1_000

# Reward scales — edit here without touching game logic
R_CAPTURE_HUNTER  =  100.0
R_CAPTURE_PREY    = -100.0
R_TIMEOUT_HUNTER  =  -10.0
R_TIMEOUT_PREY    =   10.0
R_STEP_HUNTER     =   -0.05   # per-step time pressure
R_STEP_PREY       =    0.05   # per-step survival bonus
R_PROGRESS_SCALE  =    2.0    # scales (Δdist / AGENT_SPEED) each step
R_HEADING_SCALE   =    0.5    # scales dot(action, ideal_direction)
R_OBSTACLE_HIT    =   -2.0    # hunter hits a wall
R_OBSTACLE_HIT_PREY = -1.0   # prey hits a wall


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
    """
    Observation vector (57 floats, all approximately in [-1, 1]):

        [0:2]   own position / [W, H]
        [2:4]   (other_pos - own_pos) / [W, H]   ← relative direction to target
        [4]     euclidean distance / diagonal      ← how close is the encounter
        [5:7]   last action taken (velocity proxy)
        [7:57]  obstacle features (max_obstacles=10, 5 per obstacle),
                sorted by surface distance nearest-first:
                  [rel_cx/W, rel_cy/H, width/W, height/H, surface_dist/diagonal]
    """

    metadata = {
        "name": "most_dangerous_game_v1",
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

        self.possible_agents = ["hunter", "prey"]

        self.max_obstacles     = 10
        self.min_obstacle_size = 30
        self.max_obstacle_size = 80
        self.obstacles: list[pygame.Rect] = []

        self.hunter_pos:         np.ndarray = np.zeros(2, dtype=np.float32)
        self.prey_pos:           np.ndarray = np.zeros(2, dtype=np.float32)
        self.hunter_last_action: np.ndarray = np.zeros(2, dtype=np.float32)
        self.prey_last_action:   np.ndarray = np.zeros(2, dtype=np.float32)
        self.timestep = 0

        # obs = 2 + 2 + 1 + 2 + max_obstacles*5 = 57
        self._obs_dim = 2 + 2 + 1 + 2 + self.max_obstacles * 5

        self._observation_spaces = {
            a: spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(self._obs_dim,), dtype=np.float32,
            )
            for a in self.possible_agents
        }
        self._action_spaces = {
            a: spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
            for a in self.possible_agents
        }

    # ------------------------------------------------------------------
    # PettingZoo API
    # ------------------------------------------------------------------
    def observation_space(self, agent):
        return self._observation_spaces[agent]

    def action_space(self, agent):
        return self._action_spaces[agent]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _obstacle_features(self, agent_pos: np.ndarray) -> np.ndarray:
        """
        Returns a (max_obstacles * 5,) array of obstacle features,
        sorted so that the nearest obstacle is always first.
        Padding uses dist=1.0 (far away) for missing obstacles.
        """
        W, H, DIAG = self._W, self._H, self._DIAG
        entries = []
        for r in self.obstacles:
            cx = r.x + r.width  / 2.0
            cy = r.y + r.height / 2.0
            # Surface distance (not centre distance — more actionable)
            nx = float(np.clip(agent_pos[0], r.x, r.x + r.width))
            ny = float(np.clip(agent_pos[1], r.y, r.y + r.height))
            surf_dist = float(np.linalg.norm(agent_pos - np.array([nx, ny])))
            entries.append((
                surf_dist,                          # for sorting only
                (cx - agent_pos[0]) / W,            # rel centre x
                (cy - agent_pos[1]) / H,            # rel centre y
                r.width  / W,                       # normalised width
                r.height / H,                       # normalised height
                surf_dist / DIAG,                   # normalised surface dist
            ))

        entries.sort(key=lambda e: e[0])

        feats: list[float] = []
        for e in entries:
            feats.extend(e[1:])                     # skip raw dist used for sort

        # Pad: rel positions 0, sizes 0, dist=1.0 (far/harmless)
        n_pad = self.max_obstacles - len(entries)
        feats += [0.0, 0.0, 0.0, 0.0, 1.0] * n_pad
        return np.array(feats, dtype=np.float32)

    def _get_obs(self) -> dict[str, np.ndarray]:
        W, H, DIAG = self._W, self._H, self._DIAG
        dist = float(np.linalg.norm(self.hunter_pos - self.prey_pos))

        # Hunter: "other" is prey  → relative vector points toward prey
        hunter_obs = np.concatenate([
            self.hunter_pos / np.array([W, H], dtype=np.float32),
            (self.prey_pos   - self.hunter_pos) / np.array([W, H], dtype=np.float32),
            np.array([dist / DIAG], dtype=np.float32),
            self.hunter_last_action,
            self._obstacle_features(self.hunter_pos),
        ])

        # Prey: "other" is hunter → relative vector points toward hunter
        # (prey should learn to move AWAY from this vector)
        prey_obs = np.concatenate([
            self.prey_pos / np.array([W, H], dtype=np.float32),
            (self.hunter_pos - self.prey_pos) / np.array([W, H], dtype=np.float32),
            np.array([dist / DIAG], dtype=np.float32),
            self.prey_last_action,
            self._obstacle_features(self.prey_pos),
        ])

        return {
            "hunter": hunter_obs.astype(np.float32),
            "prey":   prey_obs.astype(np.float32),
        }

    def _get_infos(self) -> dict[str, dict]:
        dist = float(np.linalg.norm(self.hunter_pos - self.prey_pos))
        shared = {
            "distance":   dist,
            "hunter_pos": self.hunter_pos.copy(),
            "prey_pos":   self.prey_pos.copy(),
        }
        return {a: shared.copy() for a in self.possible_agents}

    def _get_random_safe_position(self) -> np.ndarray:
        """Sample a random position that does not overlap any obstacle."""
        while True:
            pos = np.array([
                np.random.uniform(AGENT_SIZE * 2, SCREEN_WIDTH  - AGENT_SIZE * 2),
                np.random.uniform(AGENT_SIZE * 2, SCREEN_HEIGHT - AGENT_SIZE * 2),
            ], dtype=np.float32)
            rect = pygame.Rect(
                int(pos[0]) - AGENT_SIZE, int(pos[1]) - AGENT_SIZE,
                AGENT_SIZE * 2, AGENT_SIZE * 2,
            )
            if not any(rect.colliderect(o) for o in self.obstacles):
                return pos

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------
    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)

        pygame.init()
        self.agents   = self.possible_agents[:]
        self.timestep = 0
        self.hunter_last_action = np.zeros(2, dtype=np.float32)
        self.prey_last_action   = np.zeros(2, dtype=np.float32)

        # Fresh random obstacles
        self.obstacles.clear()
        n_obs = np.random.randint(3, self.max_obstacles // 2 + 1)
        for _ in range(n_obs):
            w = np.random.randint(self.min_obstacle_size, self.max_obstacle_size)
            h = np.random.randint(self.min_obstacle_size, self.max_obstacle_size)
            x = np.random.uniform(0, SCREEN_WIDTH  - w)
            y = np.random.uniform(0, SCREEN_HEIGHT - h)
            self.obstacles.append(pygame.Rect(int(x), int(y), w, h))

        # Place agents with at least 30% of the shorter screen dimension between them
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
    def _try_move(self, pos: np.ndarray, action: np.ndarray):
        """Apply action with boundary clipping; revert on obstacle collision."""
        new_pos = pos + action * AGENT_SPEED
        new_pos[0] = np.clip(new_pos[0], AGENT_SIZE, SCREEN_WIDTH  - AGENT_SIZE)
        new_pos[1] = np.clip(new_pos[1], AGENT_SIZE, SCREEN_HEIGHT - AGENT_SIZE)

        rect = pygame.Rect(
            int(new_pos[0]) - AGENT_SIZE, int(new_pos[1]) - AGENT_SIZE,
            AGENT_SIZE * 2, AGENT_SIZE * 2,
        )
        if any(rect.colliderect(o) for o in self.obstacles):
            return pos.copy(), True     # reverted; hit_wall=True
        return new_pos, False

    def step(self, actions: dict):
        rewards      = {a: 0.0 for a in self.possible_agents}
        terminations = {a: False for a in self.possible_agents}
        truncations  = {a: False for a in self.possible_agents}

        # Distance BEFORE moving (needed for progress reward)
        prev_dist = float(np.linalg.norm(self.hunter_pos - self.prey_pos))

        # ---- Apply actions --------------------------------------------
        h_act = np.clip(
            np.array(actions.get("hunter", [0, 0]), dtype=np.float32), -1, 1
        )
        p_act = np.clip(
            np.array(actions.get("prey",   [0, 0]), dtype=np.float32), -1, 1
        )

        new_hunter, h_wall = self._try_move(self.hunter_pos, h_act)
        new_prey,   p_wall = self._try_move(self.prey_pos,   p_act)

        self.hunter_pos        = new_hunter
        self.prey_pos          = new_prey
        self.hunter_last_action = h_act
        self.prey_last_action   = p_act

        if h_wall:
            rewards["hunter"] += R_OBSTACLE_HIT
        if p_wall:
            rewards["prey"]   += R_OBSTACLE_HIT_PREY

        self.timestep += 1
        curr_dist = float(np.linalg.norm(self.hunter_pos - self.prey_pos))

        # ---- Dense progress reward -----------------------------------
        # dist_delta > 0 means hunter got closer → good for hunter, bad for prey
        dist_delta = prev_dist - curr_dist
        progress   = dist_delta / AGENT_SPEED       # normalised to approx [-1, 1]
        rewards["hunter"] += R_PROGRESS_SCALE * progress
        rewards["prey"]   -= R_PROGRESS_SCALE * progress

        # ---- Heading bonus -------------------------------------------
        # Hunter should move TOWARD prey; prey should move AWAY from hunter
        to_prey      = self.prey_pos   - self.hunter_pos
        to_prey_n    = to_prey / (np.linalg.norm(to_prey) + 1e-8)
        away_hunter  = self.prey_pos   - self.hunter_pos   # same direction for prey
        away_hunter_n = away_hunter / (np.linalg.norm(away_hunter) + 1e-8)

        rewards["hunter"] += R_HEADING_SCALE * float(np.dot(h_act, to_prey_n))
        rewards["prey"]   += R_HEADING_SCALE * float(np.dot(p_act, away_hunter_n))

        # ---- Per-step baseline ---------------------------------------
        rewards["hunter"] += R_STEP_HUNTER
        rewards["prey"]   += R_STEP_PREY

        # ---- Terminal conditions -------------------------------------
        if curr_dist <= AGENT_SIZE * 2:
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

        self.screen.fill(WHITE)
        for r in self.obstacles:
            pygame.draw.rect(self.screen, BLACK, r)

        # Draw heading arrows (green) so you can see intended directions
        arrow_len = AGENT_SIZE * 2.5
        for pos, act in (
            (self.hunter_pos, self.hunter_last_action),
            (self.prey_pos,   self.prey_last_action),
        ):
            mag = np.linalg.norm(act)
            if mag > 0.05:
                tip = pos + act / mag * arrow_len
                pygame.draw.line(
                    self.screen, GREEN,
                    pos.astype(int).tolist(), tip.astype(int).tolist(), 2
                )

        pygame.draw.circle(self.screen, BLUE,
                           self.hunter_pos.astype(int).tolist(), AGENT_SIZE)
        pygame.draw.circle(self.screen, RED,
                           self.prey_pos.astype(int).tolist(), AGENT_SIZE)

        # Minimal HUD
        if not hasattr(self, "_font"):
            pygame.font.init()
            self._font = pygame.font.SysFont(None, 22)
        dist = np.linalg.norm(self.hunter_pos - self.prey_pos)
        hud  = self._font.render(
            f"step {self.timestep:>4}   dist {dist:>5.0f} px", True, (80, 80, 80)
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
