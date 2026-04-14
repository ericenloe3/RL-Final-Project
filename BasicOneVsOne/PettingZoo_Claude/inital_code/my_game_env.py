# my_game_env.py
"""
'The Most Dangerous Game' — PettingZoo ParallelEnv implementation.

Both the Hunter and Prey choose actions simultaneously each timestep,
which maps naturally onto PettingZoo's ParallelEnv API.

Install dependencies:
    pip install pettingzoo pygame numpy gymnasium
"""

import pygame
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from pettingzoo import ParallelEnv
from pettingzoo.utils import wrappers

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
BLACK         = (0,   0,   0)
WHITE         = (255, 255, 255)
BLUE          = (0,   0,   255)   # Hunter colour
RED           = (255, 0,   0)     # Prey colour
SCREEN_WIDTH  = 800
SCREEN_HEIGHT = 600
AGENT_SIZE    = 15                # Radius used for rendering & collision
AGENT_SPEED   = 5                 # Pixels moved per timestep at full action magnitude
MAX_TIMESTEPS = 1_000


# ---------------------------------------------------------------------------
# Factory helpers expected by PettingZoo tooling / SuperSuit
# ---------------------------------------------------------------------------
def env(**kwargs):
    """Returns an AEC-wrapped version of the environment (for compatibility)."""
    from pettingzoo.utils import parallel_to_aec
    aec = parallel_to_aec(raw_env(**kwargs))
    aec = wrappers.AssertOutOfBoundsWrapper(aec)
    aec = wrappers.OrderEnforcingWrapper(aec)
    return aec


def parallel_env(**kwargs):
    """Returns the raw ParallelEnv — preferred for training."""
    return MostDangerousGameEnv(**kwargs)


def raw_env(**kwargs):
    return MostDangerousGameEnv(**kwargs)


# ---------------------------------------------------------------------------
# Main environment
# ---------------------------------------------------------------------------
class MostDangerousGameEnv(ParallelEnv):
    """
    Observation (per agent):
        [own_x, own_y, other_x, other_y,  <-- 4 floats
         obs0_x, obs0_y, obs0_w, obs0_h,  <-- 4 floats per obstacle
         ... (padded to max_obstacles)]

    Action (per agent):
        [dx, dy]  in [-1, 1]  — scaled by AGENT_SPEED internally.

    Agents: "hunter", "prey"
    """

    metadata = {
        "name": "most_dangerous_game_v0",
        "render_modes": ["human", "rgb_array"],
        "render_fps": 30,
        "is_parallelizable": True,
    }

    def __init__(self, render_mode: str | None = None):
        super().__init__()

        self.render_mode = render_mode
        self.screen = None
        self.clock  = None

        # Agent identifiers (PettingZoo convention)
        self.possible_agents = ["hunter", "prey"]

        # Obstacle bookkeeping
        self.max_obstacles    = 10
        self.min_obstacle_size = 30
        self.max_obstacle_size = 80
        self.obstacles: list[pygame.Rect] = []

        # ---- Observation & action spaces --------------------------------
        # obs = [own(2), other(2), obstacles(max_obstacles * 4)]
        obs_dim = 4 + self.max_obstacles * 4

        self._observation_spaces = {
            agent: spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(obs_dim,), dtype=np.float32,
            )
            for agent in self.possible_agents
        }

        self._action_spaces = {
            agent: spaces.Box(
                low=-1.0, high=1.0,
                shape=(2,), dtype=np.float32,
            )
            for agent in self.possible_agents
        }

        # Internal state (initialised properly in reset())
        self.hunter_pos = np.zeros(2, dtype=np.float32)
        self.prey_pos   = np.zeros(2, dtype=np.float32)
        self.timestep   = 0

    # ------------------------------------------------------------------
    # PettingZoo required property overrides
    # ------------------------------------------------------------------
    def observation_space(self, agent: str) -> spaces.Box:
        return self._observation_spaces[agent]

    def action_space(self, agent: str) -> spaces.Box:
        return self._action_spaces[agent]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _build_obstacle_vector(self) -> np.ndarray:
        """Returns a fixed-length padded obstacle feature vector."""
        feats = []
        for r in self.obstacles:
            feats.extend([float(r.x), float(r.y), float(r.width), float(r.height)])
        # Pad with zeros up to max_obstacles
        feats += [0.0] * ((self.max_obstacles - len(self.obstacles)) * 4)
        return np.array(feats, dtype=np.float32)

    def _get_obs(self) -> dict[str, np.ndarray]:
        """
        Each agent sees its OWN position first, then the OTHER agent's
        position, so gradient flow is always from self-relative features.
        """
        obs_vec = self._build_obstacle_vector()
        return {
            "hunter": np.concatenate([self.hunter_pos, self.prey_pos,   obs_vec]),
            "prey":   np.concatenate([self.prey_pos,   self.hunter_pos, obs_vec]),
        }

    def _get_infos(self) -> dict[str, dict]:
        dist = float(np.linalg.norm(self.hunter_pos - self.prey_pos))
        shared = {
            "distance":   dist,
            "hunter_pos": self.hunter_pos.copy(),
            "prey_pos":   self.prey_pos.copy(),
        }
        return {agent: shared.copy() for agent in self.possible_agents}

    def _get_random_safe_position(self) -> np.ndarray:
        """Sample a position that does not overlap any obstacle."""
        while True:
            pos = np.array([
                np.random.uniform(AGENT_SIZE, SCREEN_WIDTH  - AGENT_SIZE),
                np.random.uniform(AGENT_SIZE, SCREEN_HEIGHT - AGENT_SIZE),
            ], dtype=np.float32)
            rect = pygame.Rect(int(pos[0]) - AGENT_SIZE, int(pos[1]) - AGENT_SIZE,
                               AGENT_SIZE * 2, AGENT_SIZE * 2)
            if not any(rect.colliderect(o) for o in self.obstacles):
                return pos

    # ------------------------------------------------------------------
    # Core PettingZoo API
    # ------------------------------------------------------------------
    def reset(
        self,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[dict, dict]:
        if seed is not None:
            np.random.seed(seed)

        # pygame.Rect works without a display — just needs pygame imported
        pygame.init()

        self.agents   = self.possible_agents[:]
        self.timestep = 0

        # ---- Generate random obstacles for this episode ---------------
        self.obstacles.clear()
        n_obs = np.random.randint(3, self.max_obstacles // 2 + 1)
        for _ in range(n_obs):
            w = np.random.randint(self.min_obstacle_size, self.max_obstacle_size)
            h = np.random.randint(self.min_obstacle_size, self.max_obstacle_size)
            x = np.random.uniform(0, SCREEN_WIDTH  - w)
            y = np.random.uniform(0, SCREEN_HEIGHT - h)
            self.obstacles.append(pygame.Rect(int(x), int(y), w, h))

        # ---- Place agents (no overlap, minimum separation) ------------
        self.hunter_pos = self._get_random_safe_position()
        self.prey_pos   = self._get_random_safe_position()
        while np.linalg.norm(self.hunter_pos - self.prey_pos) < AGENT_SIZE * 6:
            self.prey_pos = self._get_random_safe_position()

        if self.render_mode == "human":
            self._render_frame()

        return self._get_obs(), self._get_infos()

    def step(
        self, actions: dict[str, np.ndarray]
    ) -> tuple[dict, dict, dict, dict, dict]:
        """
        actions: {"hunter": np.ndarray(2,), "prey": np.ndarray(2,)}
        Returns: observations, rewards, terminations, truncations, infos
        """
        rewards      = {a: 0.0 for a in self.possible_agents}
        terminations = {a: False for a in self.possible_agents}
        truncations  = {a: False for a in self.possible_agents}

        # ---- Move hunter ----------------------------------------------
        old_hunter = self.hunter_pos.copy()
        self.hunter_pos = self.hunter_pos + actions["hunter"] * AGENT_SPEED
        h_rect = pygame.Rect(
            int(self.hunter_pos[0]) - AGENT_SIZE,
            int(self.hunter_pos[1]) - AGENT_SIZE,
            AGENT_SIZE * 2, AGENT_SIZE * 2,
        )
        if any(h_rect.colliderect(o) for o in self.obstacles):
            self.hunter_pos = old_hunter        # revert
            rewards["hunter"] -= 1.0            # wall-collision penalty

        # ---- Move prey -----------------------------------------------
        old_prey = self.prey_pos.copy()
        self.prey_pos = self.prey_pos + actions["prey"] * AGENT_SPEED
        p_rect = pygame.Rect(
            int(self.prey_pos[0]) - AGENT_SIZE,
            int(self.prey_pos[1]) - AGENT_SIZE,
            AGENT_SIZE * 2, AGENT_SIZE * 2,
        )
        if any(p_rect.colliderect(o) for o in self.obstacles):
            self.prey_pos = old_prey            # revert (no penalty — walls can be useful)

        # ---- Boundary clipping ---------------------------------------
        for pos in (self.hunter_pos, self.prey_pos):
            pos[0] = np.clip(pos[0], AGENT_SIZE, SCREEN_WIDTH  - AGENT_SIZE)
            pos[1] = np.clip(pos[1], AGENT_SIZE, SCREEN_HEIGHT - AGENT_SIZE)

        self.timestep += 1
        distance = float(np.linalg.norm(self.hunter_pos - self.prey_pos))

        # ---- Terminal conditions -------------------------------------
        if distance <= AGENT_SIZE * 2:          # capture!
            rewards["hunter"] += 100.0
            rewards["prey"]   -= 100.0
            terminations = {a: True for a in self.possible_agents}

        elif self.timestep >= MAX_TIMESTEPS:    # timeout
            rewards["hunter"] -= 10.0
            rewards["prey"]   += 10.0
            truncations = {a: True for a in self.possible_agents}

        else:                                   # per-step shaping
            rewards["hunter"] -= 0.1            # urgency
            rewards["prey"]   += 0.1            # survival bonus

        # Remove finished agents from the active list
        self.agents = [
            a for a in self.agents
            if not terminations[a] and not truncations[a]
        ]

        if self.render_mode == "human":
            self._render_frame()

        return (
            self._get_obs(),
            rewards,
            terminations,
            truncations,
            self._get_infos(),
        )

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

        for obs_rect in self.obstacles:
            pygame.draw.rect(self.screen, BLACK, obs_rect)

        pygame.draw.circle(
            self.screen, BLUE,
            self.hunter_pos.astype(int).tolist(), AGENT_SIZE,
        )
        pygame.draw.circle(
            self.screen, RED,
            self.prey_pos.astype(int).tolist(), AGENT_SIZE,
        )

        pygame.display.flip()
        self.clock.tick(self.metadata["render_fps"])

    def _get_rgb_array(self) -> np.ndarray:
        self._render_frame()
        return np.transpose(
            np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
        )

    def close(self):
        if self.screen is not None:
            pygame.display.quit()
            self.screen = None
            self.clock  = None
