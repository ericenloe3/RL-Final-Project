"""
Evaluate trained 2-hunter / 1-prey SAC policies.

The hunter network is shared between hunter1 and hunter2 — both load the
same checkpoint but query with their own observations.
"""

import time, json
from pathlib import Path
import numpy as np
import torch
from PIL import Image

from env import HunterPreyEnv
from train_sac import SACGaussianActor, RunningNorm

# ---------- config ----------
HUNTER_PATH = "models/hunter_sac_final.pt"
PREY_PATH   = "models/prey_sac_final.pt"
HIDDEN      = {"hunter": 192, "prey": 192}

NUM_LIVE    = 10
NUM_GIFS    = 5
GIF_DIR     = Path("gifs_sac")
GIF_FPS     = 30
GIF_SKIP    = 2
RENDER_FPS  = 30

SCREEN_W    = 800
SCREEN_H    = 600
AGENT_R     = 10

EVAL_OBSTACLE_RANGE = (3, 6)
# v2: eval at end-of-curriculum settings
EVAL_HUNTER_SPEED   = 4.7   # CHANGED from 4.5 — matches v2 final phase
EVAL_PREY_SPEED     = 4.5
EVAL_MAX_STEPS      = 600   # SAC keeps long episodes


def load_agent(path, obs_dim, act_dim, hidden):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    policy = SACGaussianActor(obs_dim, act_dim, hidden)
    policy.load_state_dict(ckpt["actor"])
    policy.eval()
    norm = RunningNorm(shape=(obs_dim,))
    if "normalizer" in ckpt:
        norm.load_state_dict(ckpt["normalizer"])
    return policy, norm


def init_pygame():
    import pygame
    pygame.init(); pygame.font.init()
    screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
    pygame.display.set_caption("Hunter-Prey SAC (2 hunters)")
    clock = pygame.time.Clock()
    font  = pygame.font.SysFont(None, 24)
    return screen, clock, font


def draw_frame(screen, font, env, step):
    import pygame
    screen.fill((255, 255, 255))
    for o in env.obstacles:
        pygame.draw.rect(screen, (30, 30, 30),
                         pygame.Rect(int(o.x), int(o.y), int(o.w), int(o.h)))

    # LOS lines from prey to each hunter (light gray when clear)
    los_h1 = env._line_of_sight_between(env.prey_pos, env.hunter1_pos)
    los_h2 = env._line_of_sight_between(env.prey_pos, env.hunter2_pos)
    if los_h1 > 0.5:
        pygame.draw.line(screen, (200, 200, 200),
                         (int(env.hunter1_pos[0]), int(env.hunter1_pos[1])),
                         (int(env.prey_pos[0]),    int(env.prey_pos[1])), 1)
    if los_h2 > 0.5:
        pygame.draw.line(screen, (200, 200, 200),
                         (int(env.hunter2_pos[0]), int(env.hunter2_pos[1])),
                         (int(env.prey_pos[0]),    int(env.prey_pos[1])), 1)

    # Two hunters in slightly different blues to distinguish
    pygame.draw.circle(screen, (0, 0, 255),                    # darker blue = h1
                       (int(env.hunter1_pos[0]), int(env.hunter1_pos[1])), AGENT_R)
    pygame.draw.circle(screen, (60, 120, 220),                 # lighter blue = h2
                       (int(env.hunter2_pos[0]), int(env.hunter2_pos[1])), AGENT_R)
    pygame.draw.circle(screen, (255, 0, 0),
                       (int(env.prey_pos[0]), int(env.prey_pos[1])), AGENT_R)

    d_h1 = float(np.linalg.norm(env.hunter1_pos - env.prey_pos))
    d_h2 = float(np.linalg.norm(env.hunter2_pos - env.prey_pos))
    los_h1_str = "Y" if los_h1 > 0.5 else "N"
    los_h2_str = "Y" if los_h2 > 0.5 else "N"
    hud = font.render(
        f"step {step:>4}  h1: {d_h1:>4.0f}/L{los_h1_str}  h2: {d_h2:>4.0f}/L{los_h2_str}  "
        f"H:{env.hunter_speed:.1f}  P:{env.prey_speed:.1f}  obs {len(env.obstacles)}  [SAC 2v1]",
        True, (80, 80, 80),
    )
    screen.blit(hud, (8, 8))


def surface_to_array(screen):
    import pygame
    return np.transpose(np.array(pygame.surfarray.pixels3d(screen)), (1, 0, 2))


def rollout(env, policies, normalizers, record=False,
            screen=None, font=None, clock=None):
    obs = env.reset()
    done = False; steps = 0; captured = False; frames = []; trajectory = []
    cap_h1 = cap_h2 = False

    while not done:
        if screen is not None:
            import pygame
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    return _pack(env, steps, captured, cap_h1, cap_h2,
                                 frames, trajectory)
            draw_frame(screen, font, env, steps)
            pygame.display.flip()
            if clock: clock.tick(RENDER_FPS)
        if record and screen is not None and steps % GIF_SKIP == 0:
            frames.append(surface_to_array(screen).copy())

        with torch.no_grad():
            h1_obs = torch.FloatTensor(normalizers["hunter"].normalize(obs["hunter1"])).unsqueeze(0)
            h2_obs = torch.FloatTensor(normalizers["hunter"].normalize(obs["hunter2"])).unsqueeze(0)
            p_obs  = torch.FloatTensor(normalizers["prey"].normalize(obs["prey"])).unsqueeze(0)
            h1_act, _, _ = policies["hunter"].act(h1_obs, deterministic=True)
            h2_act, _, _ = policies["hunter"].act(h2_obs, deterministic=True)
            p_act,  _, _ = policies["prey"].act(p_obs,   deterministic=True)

        d_h1 = float(np.linalg.norm(env.hunter1_pos - env.prey_pos))
        d_h2 = float(np.linalg.norm(env.hunter2_pos - env.prey_pos))
        los_h1 = env._line_of_sight_between(env.prey_pos, env.hunter1_pos)
        los_h2 = env._line_of_sight_between(env.prey_pos, env.hunter2_pos)
        trajectory.append({
            "step": steps,
            "h1_pos": env.hunter1_pos.tolist(),
            "h2_pos": env.hunter2_pos.tolist(),
            "p_pos":  env.prey_pos.tolist(),
            "d_h1": round(d_h1, 1), "d_h2": round(d_h2, 1),
            "los_h1": int(los_h1), "los_h2": int(los_h2),
        })

        obs, _, done, info = env.step(h1_act.squeeze(), h2_act.squeeze(), p_act.squeeze())
        steps += 1
        if info["captured"]:
            captured = True
            cap_h1 = info["captured_by_h1"]
            cap_h2 = info["captured_by_h2"]
        if info.get("los_broken_h1"): trajectory[-1]["los_broken_h1"] = True
        if info.get("los_broken_h2"): trajectory[-1]["los_broken_h2"] = True

    if screen is not None:
        draw_frame(screen, font, env, steps)
        import pygame; pygame.display.flip()
        if record: frames.append(surface_to_array(screen).copy())

    return _pack(env, steps, captured, cap_h1, cap_h2, frames, trajectory)


def _pack(env, steps, captured, cap_h1, cap_h2, frames, trajectory):
    terrain = [{"x": round(o.x,1), "y": round(o.y,1),
                "w": round(o.w,1), "h": round(o.h,1)} for o in env.obstacles]
    los_breaks_h1 = sum(1 for t in trajectory if t.get("los_broken_h1"))
    los_breaks_h2 = sum(1 for t in trajectory if t.get("los_broken_h2"))
    return {
        "steps": steps, "captured": captured,
        "captured_by_h1": cap_h1, "captured_by_h2": cap_h2,
        "frames": frames, "terrain": terrain, "trajectory": trajectory,
        "start_d_h1": trajectory[0]["d_h1"] if trajectory else 0,
        "start_d_h2": trajectory[0]["d_h2"] if trajectory else 0,
        "end_d_h1":   trajectory[-1]["d_h1"] if trajectory else 0,
        "end_d_h2":   trajectory[-1]["d_h2"] if trajectory else 0,
        "los_breaks_h1": los_breaks_h1,
        "los_breaks_h2": los_breaks_h2,
    }


def save_gif(frames, path, fps=GIF_FPS):
    if not frames: return
    imgs = [Image.fromarray(f) for f in frames]
    imgs[0].save(path, save_all=True, append_images=imgs[1:],
                 loop=0, duration=int(1000/fps), optimize=False)
    print(f"  Saved {path.name}  ({len(frames)} frames, {path.stat().st_size//1024} KB)")


def main():
    env = HunterPreyEnv(n_obstacles_range=EVAL_OBSTACLE_RANGE,
                        hunter_speed=EVAL_HUNTER_SPEED,
                        prey_speed=EVAL_PREY_SPEED,
                        max_steps=EVAL_MAX_STEPS)
    obs_dim = env.obs_size; act_dim = env.action_size

    policies, normalizers = {}, {}
    for role, path in [("hunter", HUNTER_PATH), ("prey", PREY_PATH)]:
        pol, norm = load_agent(path, obs_dim, act_dim, HIDDEN[role])
        policies[role] = pol; normalizers[role] = norm
        print(f"Loaded {role} from {path}  (hidden={HIDDEN[role]})")

    print(f"\n{'='*55}")
    print(f"  Phase 1 — Live viewing ({NUM_LIVE} eps)  [SAC 2-hunter]")
    print(f"{'='*55}")

    screen, clock, font = init_pygame()
    caps, cap_h1s, cap_h2s, all_steps, all_results = 0, 0, 0, [], []

    for ep in range(1, NUM_LIVE + 1):
        r = rollout(env, policies, normalizers, record=False,
                    screen=screen, font=font, clock=clock)
        all_steps.append(r["steps"])
        tag = "CAPTURED" if r["captured"] else "ESCAPED"
        if r["captured"]: caps += 1
        if r["captured_by_h1"]: cap_h1s += 1
        if r["captured_by_h2"]: cap_h2s += 1
        who = "h1" if r["captured_by_h1"] else ("h2" if r["captured_by_h2"] else "-")
        print(f"  Ep {ep:>3}: {tag} in {r['steps']:>4} steps  by={who}  "
              f"({len(r['terrain'])} obs, LOS_breaks h1={r['los_breaks_h1']} h2={r['los_breaks_h2']})")
        all_results.append({
            "episode": ep, "phase": "live", "algorithm": "SAC_2hunter",
            "captured": r["captured"],
            "captured_by_h1": r["captured_by_h1"], "captured_by_h2": r["captured_by_h2"],
            "steps": r["steps"],
            "start_d_h1": r["start_d_h1"], "start_d_h2": r["start_d_h2"],
            "end_d_h1": r["end_d_h1"],     "end_d_h2": r["end_d_h2"],
            "n_obstacles": len(r["terrain"]),
            "los_breaks_h1": r["los_breaks_h1"], "los_breaks_h2": r["los_breaks_h2"],
            "terrain": r["terrain"], "trajectory": r["trajectory"],
        })
        time.sleep(0.3)

    print(f"\n  Capture rate: {caps}/{NUM_LIVE} ({100*caps/NUM_LIVE:.0f}%)  "
          f"by_h1={cap_h1s} by_h2={cap_h2s}")
    print(f"  Avg steps:    {np.mean(all_steps):.0f}")

    print(f"\n{'='*55}")
    print(f"  Phase 2 — GIF recording ({NUM_GIFS} eps) [SAC 2-hunter]")
    print(f"{'='*55}")
    GIF_DIR.mkdir(parents=True, exist_ok=True)

    for ep in range(1, NUM_GIFS + 1):
        r = rollout(env, policies, normalizers, record=True,
                    screen=screen, font=font, clock=None)
        tag = "captured" if r["captured"] else "escaped"
        name = f"episode_{ep:02d}_{tag}_{r['steps']}steps.gif"
        save_gif(r["frames"], GIF_DIR / name)
        who = "h1" if r["captured_by_h1"] else ("h2" if r["captured_by_h2"] else "-")
        print(f"  Ep {ep}: {tag.upper()} in {r['steps']} steps by={who} → {name}")
        all_results.append({
            "episode": ep, "phase": "gif", "algorithm": "SAC_2hunter",
            "gif_name": name, "captured": r["captured"],
            "captured_by_h1": r["captured_by_h1"], "captured_by_h2": r["captured_by_h2"],
            "steps": r["steps"],
            "start_d_h1": r["start_d_h1"], "start_d_h2": r["start_d_h2"],
            "end_d_h1": r["end_d_h1"],     "end_d_h2": r["end_d_h2"],
            "n_obstacles": len(r["terrain"]),
            "los_breaks_h1": r["los_breaks_h1"], "los_breaks_h2": r["los_breaks_h2"],
            "terrain": r["terrain"], "trajectory": r["trajectory"],
        })

    import pygame; pygame.quit()
    log_path = GIF_DIR / "eval_sac_log.json"
    with open(log_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nEval log → {log_path}")
    print(f"Done! GIFs in {GIF_DIR}/")


if __name__ == "__main__":
    main()
