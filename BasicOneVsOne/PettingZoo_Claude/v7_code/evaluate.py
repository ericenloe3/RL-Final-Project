# evaluate.py
"""
Evaluate trained Hunter and Prey policies — The Most Dangerous Game (v3).

Two phases:
  Phase 1 — Live viewing (NUM_EPISODES episodes, pygame window)
  Phase 2 — GIF recording (GIF_EPISODES episodes → gifs/)

Usage
-----
    python evaluate.py

Expects
-------
    models/hunter_policy.pt
    models/prey_policy.pt

Dependencies
------------
    pip install pettingzoo pygame numpy torch pillow
"""

import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from my_game_env import parallel_env, CAPTURE_DIST
from models import ActorCritic, RunningMeanStd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
HUNTER_MODEL_PATH = "models/hunter_policy.pt"
PREY_MODEL_PATH   = "models/prey_policy.pt"
HIDDEN_SIZE       = 256
# CAPTURE_DIST imported from my_game_env so it always matches the env exactly.

# Use deterministic (mean) actions at evaluation time.
# get_action() normally *samples* from the Gaussian policy distribution.
# With std > 0 (which trained policies always have) this adds noise to every
# action, making the agent look erratic even when the mean policy is sensible.
# Setting deterministic=True returns the mean directly.
DETERMINISTIC_EVAL = True

NUM_EPISODES     = 20
RENDER_SLEEP_SEC = 0.033      # ~30 fps cap

GIF_EPISODES  = 5
GIF_DIR       = Path("gifs")
GIF_FPS       = 30
GIF_FRAME_SKIP = 2
GIF_SCALE     = 0.5


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_checkpoint(path: str, obs_size: int, action_size: int):
    """Load policy + normaliser from a checkpoint file."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)

    policy     = ActorCritic(obs_size, action_size, HIDDEN_SIZE)
    normalizer = RunningMeanStd(shape=(obs_size,))

    if isinstance(ckpt, dict) and "policy" in ckpt:
        policy.load_state_dict(ckpt["policy"])
        if "normalizer" in ckpt:
            normalizer.load_state_dict(ckpt["normalizer"])
    else:
        policy.load_state_dict(ckpt)   # legacy bare state-dict

    policy.eval()
    return policy, normalizer


def scale_frame(frame: np.ndarray, scale: float) -> np.ndarray:
    img = Image.fromarray(frame)
    return np.array(img.resize(
        (max(1, int(img.width * scale)), max(1, int(img.height * scale))),
        Image.LANCZOS,
    ))


def save_gif(frames: list, path: Path, fps: int) -> None:
    if not frames:
        print(f"  [warn] no frames for {path.name}")
        return
    pil = [Image.fromarray(f) for f in frames]
    pil[0].save(path, save_all=True, append_images=pil[1:],
                loop=0, duration=int(1000 / fps), optimize=False)
    print(f"  Saved {path.name}  ({len(frames)} frames, {path.stat().st_size//1024} KB)")


# ---------------------------------------------------------------------------
# Rollout
# ---------------------------------------------------------------------------
def _rollout(env, policies, normalizers, record: bool) -> dict:
    obs, _   = env.reset()
    steps    = 0
    captured = False
    frames   = []
    idx      = 0

    while env.agents:
        if record and idx % GIF_FRAME_SKIP == 0:
            raw = env.render()
            if raw is not None:
                frames.append(scale_frame(raw, GIF_SCALE))

        actions = {}
        with torch.no_grad():
            for agent in env.agents:
                normed = normalizers[agent].normalize(obs[agent])
                obs_t  = torch.FloatTensor(normed).unsqueeze(0)
                actions[agent] = policies[agent].get_action(obs_t,
                                    deterministic=DETERMINISTIC_EVAL)

        obs, _, terms, truns, infos = env.step(actions)
        steps += 1
        idx   += 1

        if not record:
            time.sleep(RENDER_SLEEP_SEC)

        if any(terms.values()):
            dist = next(iter(infos.values())).get("distance", float("inf"))
            if dist <= CAPTURE_DIST:
                captured = True

    if record:
        raw = env.render()
        if raw is not None:
            frames.append(scale_frame(raw, GIF_SCALE))

    return {"steps": steps, "captured": captured, "frames": frames}


# ---------------------------------------------------------------------------
# Phase 1 — live viewing
# ---------------------------------------------------------------------------
def run_live(policies, normalizers) -> dict:
    print(f"\n{'='*55}")
    print(f"  PHASE 1 — Live evaluation  ({NUM_EPISODES} episodes)")
    print(f"{'='*55}")

    env = parallel_env(render_mode="human")
    env.reset(seed=0)
    env.set_obstacle_range(3, 5)    # evaluate at full difficulty

    cap_count, cap_steps, all_steps = 0, [], []
    for ep in range(1, NUM_EPISODES + 1):
        r = _rollout(env, policies, normalizers, record=False)
        all_steps.append(r["steps"])
        if r["captured"]:
            cap_count += 1
            cap_steps.append(r["steps"])
            print(f"  Episode {ep:>3}: Prey CAPTURED  in {r['steps']:>4} steps")
        else:
            print(f"  Episode {ep:>3}: Prey ESCAPED   in {r['steps']:>4} steps (timeout)")

    env.close()
    return {"capture_count": cap_count, "capture_steps": cap_steps, "all_ep_steps": all_steps}


# ---------------------------------------------------------------------------
# Phase 2 — GIF recording
# ---------------------------------------------------------------------------
def run_gifs(policies, normalizers) -> None:
    print(f"\n{'='*55}")
    print(f"  PHASE 2 — GIF recording  ({GIF_EPISODES} episodes)")
    print(f"  Output: {GIF_DIR.resolve()}")
    print(f"{'='*55}")

    GIF_DIR.mkdir(parents=True, exist_ok=True)
    env = parallel_env(render_mode="rgb_array")
    env.reset(seed=200)
    env.set_obstacle_range(3, 5)

    for ep in range(1, GIF_EPISODES + 1):
        print(f"\n  Recording episode {ep}/{GIF_EPISODES}...")
        r        = _rollout(env, policies, normalizers, record=True)
        outcome  = "captured" if r["captured"] else "escaped"
        gif_name = f"episode_{ep:02d}_{outcome}_{r['steps']}steps.gif"
        save_gif(r["frames"], GIF_DIR / gif_name, GIF_FPS)
        print(f"  {'CAPTURED' if r['captured'] else 'ESCAPED'}"
              f"  in {r['steps']} steps  →  {gif_name}")

    env.close()


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
def print_summary(stats: dict, n: int) -> None:
    print(f"\n{'='*55}")
    print("               EVALUATION SUMMARY")
    print(f"{'='*55}")
    rate = stats["capture_count"] / n * 100
    print(f"  Episodes        : {n}")
    print(f"  Captures        : {stats['capture_count']}")
    print(f"  Capture rate    : {rate:.1f}%")
    print(f"  Avg ep length   : {np.mean(stats['all_ep_steps']):.1f} steps")
    if stats["capture_steps"]:
        print(f"  Avg capture time: {np.mean(stats['capture_steps']):.1f} steps")
        print(f"  Min capture time: {min(stats['capture_steps'])} steps")
    else:
        print("  No captures — consider training longer.")
    print(f"{'='*55}\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    probe = parallel_env()
    probe.reset()
    agents      = probe.possible_agents
    obs_size    = probe.observation_space(agents[0]).shape[0]
    action_size = probe.action_space(agents[0]).shape[0]
    probe.close()

    print(f"obs_size={obs_size}  action_size={action_size}")
    policies, normalizers = {}, {}
    for a, path in zip(agents, [HUNTER_MODEL_PATH, PREY_MODEL_PATH]):
        pol, norm = load_checkpoint(path, obs_size, action_size)
        policies[a]    = pol
        normalizers[a] = norm
        print(f"  Loaded {a:>6} from '{path}'")

    stats = run_live(policies, normalizers)
    print_summary(stats, NUM_EPISODES)
    run_gifs(policies, normalizers)
    print(f"\nDone!  GIFs saved to '{GIF_DIR}/'")


if __name__ == "__main__":
    main()