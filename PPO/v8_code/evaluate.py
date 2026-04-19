"""
Evaluate trained hunter-prey policies (v2 — with obstacles).

Phase 1: Live pygame window (NUM_LIVE episodes)
Phase 2: Save GIF recordings (NUM_GIFS episodes → gifs/)

Usage:  python evaluate.py
Expects: models/hunter_final.pt  and  models/prey_final.pt
"""

import time
from pathlib import Path
import numpy as np
import torch
from PIL import Image

from env import HunterPreyEnv
from train import ActorCritic, RunningNorm

# ---------- config ----------
HUNTER_PATH = "models/hunter_final.pt"
PREY_PATH   = "models/prey_final.pt"
HIDDEN      = 128   # must match training

NUM_LIVE    = 10
NUM_GIFS    = 5
GIF_DIR     = Path("gifs")
GIF_FPS     = 30
GIF_SKIP    = 2
RENDER_FPS  = 30

SCREEN_W    = 800
SCREEN_H    = 600
AGENT_R     = 10

# Evaluate at final curriculum difficulty
EVAL_OBSTACLE_RANGE = (3, 6)
EVAL_HUNTER_SPEED   = 4.5
EVAL_PREY_SPEED     = 4.0


# =====================================================================
def load_agent(path, obs_dim, act_dim):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    policy = ActorCritic(obs_dim, act_dim, HIDDEN)
    policy.load_state_dict(ckpt["policy"])
    policy.eval()
    norm = RunningNorm(shape=(obs_dim,))
    if "normalizer" in ckpt:
        norm.load_state_dict(ckpt["normalizer"])
    return policy, norm


# =====================================================================
def init_pygame():
    import pygame
    pygame.init(); pygame.font.init()
    screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
    pygame.display.set_caption("Hunter-Prey Evaluation")
    clock = pygame.time.Clock()
    font  = pygame.font.SysFont(None, 24)
    return screen, clock, font


def draw_frame(screen, font, env, step):
    """Draw obstacles, agents, and HUD."""
    import pygame
    screen.fill((255, 255, 255))

    # Obstacles (black rectangles)
    for o in env.obstacles:
        pygame.draw.rect(screen, (30, 30, 30),
                         pygame.Rect(int(o.x), int(o.y), int(o.w), int(o.h)))

    # Line-of-sight indicator (grey line if LOS, none if blocked)
    los = env._line_of_sight()
    if los > 0.5:
        pygame.draw.line(screen, (200, 200, 200),
                         (int(env.hunter_pos[0]), int(env.hunter_pos[1])),
                         (int(env.prey_pos[0]),   int(env.prey_pos[1])), 1)

    # Hunter = blue circle
    pygame.draw.circle(screen, (0, 0, 255),
                       (int(env.hunter_pos[0]), int(env.hunter_pos[1])), AGENT_R)
    # Prey = red circle
    pygame.draw.circle(screen, (255, 0, 0),
                       (int(env.prey_pos[0]), int(env.prey_pos[1])), AGENT_R)

    # HUD
    dist = float(np.linalg.norm(env.hunter_pos - env.prey_pos))
    los_str = "Y" if los > 0.5 else "N"
    hud = font.render(
        f"step {step:>4}  dist {dist:>5.0f}  LOS {los_str}  "
        f"H:{env.hunter_speed:.1f}  P:{env.prey_speed:.1f}  "
        f"obs {len(env.obstacles)}",
        True, (80, 80, 80),
    )
    screen.blit(hud, (8, 8))


def surface_to_array(screen) -> np.ndarray:
    import pygame
    return np.transpose(np.array(pygame.surfarray.pixels3d(screen)), (1, 0, 2))


# =====================================================================
def rollout(env, policies, normalizers, record=False,
            screen=None, font=None, clock=None):
    obs = env.reset()
    done = False; steps = 0; captured = False; frames = []
    # Per-step trajectory for post-processing
    trajectory = []

    while not done:
        if screen is not None:
            import pygame
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    return _pack_result(env, steps, captured, frames, trajectory)
            draw_frame(screen, font, env, steps)
            pygame.display.flip()
            if clock: clock.tick(RENDER_FPS)

        if record and screen is not None and steps % GIF_SKIP == 0:
            frames.append(surface_to_array(screen).copy())

        with torch.no_grad():
            h_obs = torch.FloatTensor(normalizers["hunter"].normalize(obs["hunter"])).unsqueeze(0)
            p_obs = torch.FloatTensor(normalizers["prey"].normalize(obs["prey"])).unsqueeze(0)
            h_act, _, _ = policies["hunter"].act(h_obs, deterministic=True)
            p_act, _, _ = policies["prey"].act(p_obs, deterministic=True)

        # Record state before step
        dist = float(np.linalg.norm(env.hunter_pos - env.prey_pos))
        los  = env._line_of_sight()
        trajectory.append({
            "step": steps,
            "hunter_pos": env.hunter_pos.tolist(),
            "prey_pos":   env.prey_pos.tolist(),
            "distance":   round(dist, 1),
            "los":        int(los),
        })

        obs, _, done, info = env.step(h_act.squeeze(), p_act.squeeze())
        steps += 1
        if info["captured"]:
            captured = True
        # Track LOS breaks for post-processing
        if info.get("los_broken"):
            trajectory[-1]["los_broken"] = True

    if screen is not None:
        draw_frame(screen, font, env, steps)
        import pygame; pygame.display.flip()
        if record:
            frames.append(surface_to_array(screen).copy())

    return _pack_result(env, steps, captured, frames, trajectory)


def _pack_result(env, steps, captured, frames, trajectory):
    """Build a result dict with episode stats + terrain info."""
    # Terrain: obstacle positions and sizes
    terrain = [{
        "x": round(o.x, 1), "y": round(o.y, 1),
        "w": round(o.w, 1), "h": round(o.h, 1),
    } for o in env.obstacles]

    # Count LOS breaks for summary
    los_breaks = sum(1 for t in trajectory if t.get("los_broken"))

    return {
        "steps":      steps,
        "captured":   captured,
        "frames":     frames,
        "terrain":    terrain,
        "trajectory": trajectory,
        "start_dist": trajectory[0]["distance"] if trajectory else 0,
        "end_dist":   trajectory[-1]["distance"] if trajectory else 0,
        "los_breaks": los_breaks,
    }


def save_gif(frames, path, fps=GIF_FPS):
    if not frames: return
    imgs = [Image.fromarray(f) for f in frames]
    imgs[0].save(path, save_all=True, append_images=imgs[1:],
                 loop=0, duration=int(1000/fps), optimize=False)
    print(f"  Saved {path.name}  ({len(frames)} frames, {path.stat().st_size//1024} KB)")


# =====================================================================
def main():
    import json

    env = HunterPreyEnv(n_obstacles_range=EVAL_OBSTACLE_RANGE,
                        hunter_speed=EVAL_HUNTER_SPEED,
                        prey_speed=EVAL_PREY_SPEED)
    obs_dim = env.obs_size; act_dim = env.action_size

    policies, normalizers = {}, {}
    for agent, path in [("hunter", HUNTER_PATH), ("prey", PREY_PATH)]:
        pol, norm = load_agent(path, obs_dim, act_dim)
        policies[agent] = pol; normalizers[agent] = norm
        print(f"Loaded {agent} from {path}")

    # ---- Phase 1: live viewing ----
    print(f"\n{'='*50}")
    print(f"  Phase 1 — Live viewing ({NUM_LIVE} eps, obstacles {EVAL_OBSTACLE_RANGE})")
    print(f"{'='*50}")

    screen, clock, font = init_pygame()
    caps, all_steps = 0, []
    all_results = []       # collect for JSON export

    for ep in range(1, NUM_LIVE + 1):
        r = rollout(env, policies, normalizers, record=False,
                    screen=screen, font=font, clock=clock)
        all_steps.append(r["steps"])
        tag = "CAPTURED" if r["captured"] else "ESCAPED"
        if r["captured"]: caps += 1
        print(f"  Ep {ep:>3}: {tag} in {r['steps']:>4} steps  "
              f"({len(r['terrain'])} obs, LOS_breaks={r['los_breaks']})")
        # Store without frames (too large for JSON)
        all_results.append({
            "episode":    ep,
            "phase":      "live",
            "captured":   r["captured"],
            "steps":      r["steps"],
            "start_dist": r["start_dist"],
            "end_dist":   r["end_dist"],
            "n_obstacles": len(r["terrain"]),
            "los_breaks": r["los_breaks"],
            "terrain":    r["terrain"],
            "trajectory": r["trajectory"],
        })
        time.sleep(0.3)

    print(f"\n  Capture rate: {caps}/{NUM_LIVE} ({100*caps/NUM_LIVE:.0f}%)")
    print(f"  Avg steps:    {np.mean(all_steps):.0f}")

    # ---- Phase 2: GIF recording ----
    print(f"\n{'='*50}")
    print(f"  Phase 2 — GIF recording ({NUM_GIFS} episodes)")
    print(f"{'='*50}")
    GIF_DIR.mkdir(parents=True, exist_ok=True)

    for ep in range(1, NUM_GIFS + 1):
        r = rollout(env, policies, normalizers, record=True,
                    screen=screen, font=font, clock=None)
        tag = "captured" if r["captured"] else "escaped"
        name = f"episode_{ep:02d}_{tag}_{r['steps']}steps.gif"
        save_gif(r["frames"], GIF_DIR / name)
        print(f"  Ep {ep}: {tag.upper()} in {r['steps']} steps → {name}"
              f"  ({len(r['terrain'])} obs, LOS_breaks={r['los_breaks']})")
        all_results.append({
            "episode":    ep,
            "phase":      "gif",
            "gif_name":   name,
            "captured":   r["captured"],
            "steps":      r["steps"],
            "start_dist": r["start_dist"],
            "end_dist":   r["end_dist"],
            "n_obstacles": len(r["terrain"]),
            "los_breaks": r["los_breaks"],
            "terrain":    r["terrain"],
            "trajectory": r["trajectory"],
        })

    import pygame; pygame.quit()

    # ---- Save evaluation log ----
    log_path = GIF_DIR / "eval_log.json"
    with open(log_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nEval log → {log_path} ({len(all_results)} episodes)")
    print(f"  Contains: per-step trajectories, terrain layouts, agent positions")
    print(f"Done! GIFs in {GIF_DIR}/")


if __name__ == "__main__":
    main()
