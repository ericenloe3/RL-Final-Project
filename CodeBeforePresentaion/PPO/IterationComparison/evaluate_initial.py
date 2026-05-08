"""
Evaluate trained hunter-prey policies.

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

# Import model + normaliser from train.py
from train import ActorCritic, RunningNorm

# ---------- config ----------
HUNTER_PATH = "models/hunter_final.pt"
PREY_PATH   = "models/prey_final.pt"
HIDDEN      = 64

NUM_LIVE    = 10
NUM_GIFS    = 5
GIF_DIR     = Path("gifs")
GIF_FPS     = 30
GIF_SKIP    = 2       # record every N-th frame
RENDER_FPS  = 30

SCREEN_W    = 800
SCREEN_H    = 600
AGENT_R     = 10      # draw radius


# =====================================================================
# Loading
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
# Rendering (pygame surface → numpy array)
# =====================================================================
def init_pygame():
    import pygame
    pygame.init()
    pygame.font.init()
    screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
    pygame.display.set_caption("Hunter-Prey Evaluation")
    clock = pygame.time.Clock()
    font  = pygame.font.SysFont(None, 24)
    return screen, clock, font


def draw_frame(screen, font, env, step):
    """Draw current state onto the pygame surface."""
    import pygame
    screen.fill((255, 255, 255))

    # Hunter = blue circle
    hx, hy = int(env.hunter_pos[0]), int(env.hunter_pos[1])
    pygame.draw.circle(screen, (0, 0, 255), (hx, hy), AGENT_R)

    # Prey = red circle
    px, py = int(env.prey_pos[0]), int(env.prey_pos[1])
    pygame.draw.circle(screen, (255, 0, 0), (px, py), AGENT_R)

    # HUD
    dist = np.linalg.norm(env.hunter_pos - env.prey_pos)
    hud = font.render(
        f"step {step:>4}  dist {dist:>5.0f} px  "
        f"H:{env.hunter_speed:.1f}  P:{env.prey_speed:.1f}",
        True, (80, 80, 80),
    )
    screen.blit(hud, (8, 8))


def surface_to_array(screen) -> np.ndarray:
    import pygame
    return np.transpose(
        np.array(pygame.surfarray.pixels3d(screen)), (1, 0, 2)
    )


# =====================================================================
# Rollout
# =====================================================================
def rollout(env, policies, normalizers, record=False, screen=None, font=None, clock=None):
    obs   = env.reset()
    done  = False
    steps = 0
    captured = False
    frames = []

    while not done:
        # Render / record
        if screen is not None:
            import pygame
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    return {"steps": steps, "captured": captured, "frames": frames}
            draw_frame(screen, font, env, steps)
            pygame.display.flip()
            if clock:
                clock.tick(RENDER_FPS)

        if record and screen is not None and steps % GIF_SKIP == 0:
            frames.append(surface_to_array(screen).copy())

        # Act (deterministic)
        with torch.no_grad():
            h_obs = torch.FloatTensor(normalizers["hunter"].normalize(obs["hunter"])).unsqueeze(0)
            p_obs = torch.FloatTensor(normalizers["prey"].normalize(obs["prey"])).unsqueeze(0)
            h_act, _, _ = policies["hunter"].act(h_obs, deterministic=True)
            p_act, _, _ = policies["prey"].act(p_obs, deterministic=True)

        obs, _, done, info = env.step(h_act.squeeze(), p_act.squeeze())
        steps += 1
        if info["captured"]:
            captured = True

    # Final frame
    if screen is not None:
        draw_frame(screen, font, env, steps)
        import pygame; pygame.display.flip()
        if record:
            frames.append(surface_to_array(screen).copy())

    return {"steps": steps, "captured": captured, "frames": frames}


# =====================================================================
# GIF saving
# =====================================================================
def save_gif(frames, path, fps=GIF_FPS):
    if not frames:
        return
    imgs = [Image.fromarray(f) for f in frames]
    imgs[0].save(path, save_all=True, append_images=imgs[1:],
                 loop=0, duration=int(1000/fps), optimize=False)
    print(f"  Saved {path.name}  ({len(frames)} frames, {path.stat().st_size//1024} KB)")


# =====================================================================
# Main
# =====================================================================
def main():
    env = HunterPreyEnv()
    obs_dim = env.obs_size
    act_dim = env.action_size

    policies, normalizers = {}, {}
    for agent, path in [("hunter", HUNTER_PATH), ("prey", PREY_PATH)]:
        pol, norm = load_agent(path, obs_dim, act_dim)
        policies[agent] = pol
        normalizers[agent] = norm
        print(f"Loaded {agent} from {path}")

    # --- Phase 1: live viewing ---
    print(f"\n{'='*50}")
    print(f"  Phase 1 — Live viewing ({NUM_LIVE} episodes)")
    print(f"{'='*50}")

    screen, clock, font = init_pygame()
    caps, all_steps = 0, []
    for ep in range(1, NUM_LIVE + 1):
        r = rollout(env, policies, normalizers, record=False,
                    screen=screen, font=font, clock=clock)
        all_steps.append(r["steps"])
        tag = "CAPTURED" if r["captured"] else "ESCAPED"
        if r["captured"]: caps += 1
        print(f"  Ep {ep:>3}: {tag} in {r['steps']:>4} steps")
        time.sleep(0.3)

    print(f"\n  Capture rate: {caps}/{NUM_LIVE} ({100*caps/NUM_LIVE:.0f}%)")
    print(f"  Avg steps:    {np.mean(all_steps):.0f}")

    # --- Phase 2: GIF recording ---
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
        print(f"  Ep {ep}: {tag.upper()} in {r['steps']} steps → {name}")

    import pygame
    pygame.quit()
    print(f"\nDone! GIFs in {GIF_DIR}/")


if __name__ == "__main__":
    main()
