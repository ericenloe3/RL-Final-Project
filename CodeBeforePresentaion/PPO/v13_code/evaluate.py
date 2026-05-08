"""
Evaluate trained hunter-prey policies (v5 — with obstacles + terrain).

Renders mud zones (sandy brown, semi-transparent) and ice zones (pale blue,
semi-transparent) behind obstacles.  Both agents show terrain status in HUD.

Usage:  python evaluate.py
Expects: models/hunter_final.pt  and  models/prey_final.pt
"""

import time, json
from pathlib import Path
import numpy as np
import torch
from PIL import Image

from env import HunterPreyEnv
from train import ActorCritic, RunningNorm

# ---------- config ----------
HUNTER_PATH = "models/hunter_final.pt"
PREY_PATH   = "models/prey_final.pt"
HIDDEN      = {"hunter": 128, "prey": 192}

NUM_LIVE    = 10
NUM_GIFS    = 5
GIF_DIR     = Path("gifs")
GIF_FPS     = 30
GIF_SKIP    = 2
RENDER_FPS  = 30

SCREEN_W    = 800
SCREEN_H    = 600
AGENT_R     = 10

EVAL_OBSTACLE_RANGE = (3, 6)
EVAL_HUNTER_SPEED   = 4.5
EVAL_PREY_SPEED     = 4.0
EVAL_MUD_RANGE      = (1, 2)
EVAL_ICE_RANGE      = (1, 2)

# Terrain colours (RGBA — alpha for the overlay surface)
MUD_COLOUR     = (139, 115,  85, 160)   # warm brown, semi-transparent
ICE_COLOUR     = (173, 216, 230, 160)   # pale blue,  semi-transparent
MUD_BORDER     = ( 90,  70,  40)        # darker brown border
ICE_BORDER     = ( 70, 130, 180)        # steel blue border


# =====================================================================
def load_agent(path, obs_dim, act_dim, hidden):
    ckpt   = torch.load(path, map_location="cpu", weights_only=False)
    policy = ActorCritic(obs_dim, act_dim, hidden)
    policy.load_state_dict(ckpt["policy"])
    policy.eval()
    norm = RunningNorm(shape=(obs_dim,))
    if "normalizer" in ckpt:
        norm.load_state_dict(ckpt["normalizer"])
    return policy, norm


def init_pygame():
    import pygame
    pygame.init(); pygame.font.init()
    screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
    pygame.display.set_caption("Hunter-Prey — Terrain Evaluation")
    clock = pygame.time.Clock()
    font  = pygame.font.SysFont(None, 24)
    return screen, clock, font


def draw_frame(screen, font, env, step, info=None):
    """Draw terrain → obstacles → agents → HUD (back to front)."""
    import pygame

    screen.fill((255, 255, 255))

    # ---- terrain zones (drawn first, behind everything) ----
    for z in env.terrain_zones:
        if z.kind == "mud":
            fill, border = MUD_COLOUR[:3], MUD_BORDER
        else:
            fill, border = ICE_COLOUR[:3], ICE_BORDER

        # Semi-transparent fill via a temporary surface
        surf = pygame.Surface((int(z.w), int(z.h)), pygame.SRCALPHA)
        alpha = MUD_COLOUR[3] if z.kind == "mud" else ICE_COLOUR[3]
        surf.fill((*fill, alpha))
        screen.blit(surf, (int(z.x), int(z.y)))

        # Border
        pygame.draw.rect(screen, border,
                         pygame.Rect(int(z.x), int(z.y), int(z.w), int(z.h)), 2)

        # Label ("MUD" / "ICE")
        label_font = pygame.font.SysFont(None, 18)
        label = label_font.render(z.kind.upper(), True, border)
        screen.blit(label, (int(z.cx) - label.get_width()//2,
                             int(z.cy) - label.get_height()//2))

    # ---- obstacles (solid black) ----
    for o in env.obstacles:
        pygame.draw.rect(screen, (30, 30, 30),
                         pygame.Rect(int(o.x), int(o.y), int(o.w), int(o.h)))

    # ---- line-of-sight indicator ----
    los = env._line_of_sight()
    if los > 0.5:
        pygame.draw.line(screen, (200, 200, 200),
                         (int(env.hunter_pos[0]), int(env.hunter_pos[1])),
                         (int(env.prey_pos[0]),   int(env.prey_pos[1])), 1)

    # ---- agents ----
    # Hunter: blue circle; orange ring when on ice; yellow ring when in mud
    h_col = (0, 0, 255)
    pygame.draw.circle(screen, h_col,
                       (int(env.hunter_pos[0]), int(env.hunter_pos[1])), AGENT_R)
    if env._h_on_ice:
        pygame.draw.circle(screen, (255, 140, 0),
                           (int(env.hunter_pos[0]), int(env.hunter_pos[1])), AGENT_R + 3, 3)
    elif env._in_mud(env.hunter_pos):
        pygame.draw.circle(screen, (200, 180, 80),
                           (int(env.hunter_pos[0]), int(env.hunter_pos[1])), AGENT_R + 3, 3)

    # Prey: red circle; orange ring on ice; yellow ring in mud
    pygame.draw.circle(screen, (255, 0, 0),
                       (int(env.prey_pos[0]), int(env.prey_pos[1])), AGENT_R)
    if env._p_on_ice:
        pygame.draw.circle(screen, (255, 140, 0),
                           (int(env.prey_pos[0]), int(env.prey_pos[1])), AGENT_R + 3, 3)
    elif env._in_mud(env.prey_pos):
        pygame.draw.circle(screen, (200, 180, 80),
                           (int(env.prey_pos[0]), int(env.prey_pos[1])), AGENT_R + 3, 3)

    # ---- HUD ----
    dist    = float(np.linalg.norm(env.hunter_pos - env.prey_pos))
    los_str = "Y" if los > 0.5 else "N"
    h_status = ("ICE" if env._h_on_ice else
                 "MUD" if env._in_mud(env.hunter_pos) else "  -")
    p_status = ("ICE" if env._p_on_ice else
                 "MUD" if env._in_mud(env.prey_pos) else "  -")

    hud = font.render(
        f"step {step:>4}  dist {dist:>5.0f}  LOS {los_str}  "
        f"H:{env.hunter_speed:.1f}[{h_status}]  P:{env.prey_speed:.1f}[{p_status}]  "
        f"obs {len(env.obstacles)}  mud {sum(1 for z in env.terrain_zones if z.kind=='mud')}  "
        f"ice {sum(1 for z in env.terrain_zones if z.kind=='ice')}",
        True, (80, 80, 80),
    )
    screen.blit(hud, (8, 8))


def surface_to_array(screen):
    import pygame
    return np.transpose(np.array(pygame.surfarray.pixels3d(screen)), (1, 0, 2))


# =====================================================================
def rollout(env, policies, normalizers, record=False,
            screen=None, font=None, clock=None):
    obs = env.reset()
    done = False; steps = 0; captured = False; frames = []
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
            h_obs = torch.FloatTensor(
                normalizers["hunter"].normalize(obs["hunter"])).unsqueeze(0)
            p_obs = torch.FloatTensor(
                normalizers["prey"].normalize(obs["prey"])).unsqueeze(0)
            h_act, _, _ = policies["hunter"].act(h_obs, deterministic=True)
            p_act, _, _ = policies["prey"].act(p_obs, deterministic=True)

        dist = float(np.linalg.norm(env.hunter_pos - env.prey_pos))
        los  = env._line_of_sight()
        trajectory.append({
            "step":       steps,
            "hunter_pos": env.hunter_pos.tolist(),
            "prey_pos":   env.prey_pos.tolist(),
            "distance":   round(dist, 1),
            "los":        int(los),
            "h_on_ice":   env._h_on_ice,
            "p_on_ice":   env._p_on_ice,
            "h_in_mud":   env._in_mud(env.hunter_pos),
            "p_in_mud":   env._in_mud(env.prey_pos),
        })

        obs, _, done, info = env.step(h_act.squeeze(), p_act.squeeze())
        steps += 1
        if info["captured"]: captured = True
        if info.get("los_broken"): trajectory[-1]["los_broken"] = True

    if screen is not None:
        draw_frame(screen, font, env, steps)
        import pygame; pygame.display.flip()
        if record: frames.append(surface_to_array(screen).copy())

    return _pack_result(env, steps, captured, frames, trajectory)


def _pack_result(env, steps, captured, frames, trajectory):
    obstacles = [{"x": round(o.x,1), "y": round(o.y,1),
                  "w": round(o.w,1), "h": round(o.h,1)}
                 for o in env.obstacles]
    terrain   = [{"x": round(z.x,1), "y": round(z.y,1),
                  "w": round(z.w,1), "h": round(z.h,1), "kind": z.kind}
                 for z in env.terrain_zones]
    los_breaks = sum(1 for t in trajectory if t.get("los_broken"))
    ice_steps  = sum(1 for t in trajectory if t.get("h_on_ice"))
    mud_steps  = sum(1 for t in trajectory if t.get("h_in_mud"))
    return {
        "steps": steps, "captured": captured, "frames": frames,
        "obstacles": obstacles, "terrain": terrain,
        "trajectory": trajectory,
        "start_dist": trajectory[0]["distance"] if trajectory else 0,
        "end_dist":   trajectory[-1]["distance"] if trajectory else 0,
        "los_breaks": los_breaks,
        "hunter_ice_steps": ice_steps,
        "hunter_mud_steps": mud_steps,
    }


def save_gif(frames, path, fps=GIF_FPS):
    if not frames: return
    imgs = [Image.fromarray(f) for f in frames]
    imgs[0].save(path, save_all=True, append_images=imgs[1:],
                 loop=0, duration=int(1000 / fps), optimize=False)
    print(f"  Saved {path.name}  ({len(frames)} frames, {path.stat().st_size//1024} KB)")


# =====================================================================
def main():
    env = HunterPreyEnv(
        n_obstacles_range=EVAL_OBSTACLE_RANGE,
        hunter_speed=EVAL_HUNTER_SPEED,
        prey_speed=EVAL_PREY_SPEED,
        n_mud_range=EVAL_MUD_RANGE,
        n_ice_range=EVAL_ICE_RANGE,
    )
    obs_dim = env.obs_size; act_dim = env.action_size

    policies, normalizers = {}, {}
    for agent, path in [("hunter", HUNTER_PATH), ("prey", PREY_PATH)]:
        pol, norm = load_agent(path, obs_dim, act_dim, HIDDEN[agent])
        policies[agent] = pol; normalizers[agent] = norm
        print(f"Loaded {agent} from {path}  (hidden={HIDDEN[agent]})")

    print(f"\n{'='*55}")
    print(f"  Phase 1 — Live viewing ({NUM_LIVE} eps)  [obstacles+terrain]")
    print(f"{'='*55}")

    screen, clock, font = init_pygame()
    caps, all_steps, all_results = 0, [], []

    for ep in range(1, NUM_LIVE + 1):
        r = rollout(env, policies, normalizers, record=False,
                    screen=screen, font=font, clock=clock)
        all_steps.append(r["steps"])
        tag = "CAPTURED" if r["captured"] else "ESCAPED"
        if r["captured"]: caps += 1
        n_mud = sum(1 for z in env.terrain_zones if z.kind == "mud")
        n_ice = sum(1 for z in env.terrain_zones if z.kind == "ice")
        print(f"  Ep {ep:>3}: {tag} in {r['steps']:>4} steps  "
              f"obs={len(r['obstacles'])} mud={n_mud} ice={n_ice}  "
              f"LOS_breaks={r['los_breaks']}  "
              f"H_ice_steps={r['hunter_ice_steps']}  H_mud_steps={r['hunter_mud_steps']}")
        all_results.append({
            "episode": ep, "phase": "live",
            "captured": r["captured"], "steps": r["steps"],
            "start_dist": r["start_dist"], "end_dist": r["end_dist"],
            "n_obstacles": len(r["obstacles"]),
            "n_mud": n_mud, "n_ice": n_ice,
            "los_breaks": r["los_breaks"],
            "hunter_ice_steps": r["hunter_ice_steps"],
            "hunter_mud_steps": r["hunter_mud_steps"],
            "obstacles": r["obstacles"], "terrain": r["terrain"],
            "trajectory": r["trajectory"],
        })
        time.sleep(0.3)

    print(f"\n  Capture rate: {caps}/{NUM_LIVE} ({100*caps/NUM_LIVE:.0f}%)")
    print(f"  Avg steps:    {np.mean(all_steps):.0f}")

    print(f"\n{'='*55}")
    print(f"  Phase 2 — GIF recording ({NUM_GIFS} episodes)")
    print(f"{'='*55}")
    GIF_DIR.mkdir(parents=True, exist_ok=True)

    for ep in range(1, NUM_GIFS + 1):
        r   = rollout(env, policies, normalizers, record=True,
                      screen=screen, font=font, clock=None)
        tag  = "captured" if r["captured"] else "escaped"
        name = f"episode_{ep:02d}_{tag}_{r['steps']}steps.gif"
        save_gif(r["frames"], GIF_DIR / name)
        n_mud = sum(1 for z in env.terrain_zones if z.kind == "mud")
        n_ice = sum(1 for z in env.terrain_zones if z.kind == "ice")
        print(f"  Ep {ep}: {tag.upper()} in {r['steps']} steps → {name}  "
              f"mud={n_mud} ice={n_ice}  "
              f"H_ice={r['hunter_ice_steps']}  H_mud={r['hunter_mud_steps']}")
        all_results.append({
            "episode": ep, "phase": "gif", "gif_name": name,
            "captured": r["captured"], "steps": r["steps"],
            "start_dist": r["start_dist"], "end_dist": r["end_dist"],
            "n_obstacles": len(r["obstacles"]),
            "n_mud": n_mud, "n_ice": n_ice,
            "los_breaks": r["los_breaks"],
            "hunter_ice_steps": r["hunter_ice_steps"],
            "hunter_mud_steps": r["hunter_mud_steps"],
            "obstacles": r["obstacles"], "terrain": r["terrain"],
            "trajectory": r["trajectory"],
        })

    import pygame; pygame.quit()

    log_path = GIF_DIR / "eval_log.json"
    with open(log_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nEval log → {log_path} ({len(all_results)} episodes)")
    print(f"Done! GIFs in {GIF_DIR}/")


if __name__ == "__main__":
    main()
