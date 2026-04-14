# evaluate.py
"""
Evaluate trained Hunter and Prey policies from The Most Dangerous Game.

Two phases run back-to-back:

  Phase 1 — Live viewing  (NUM_EPISODES episodes, pygame window)
  Phase 2 — GIF recording (GIF_EPISODES episodes saved to gifs/)

Changes from v1
---------------
* Loads the RunningMeanStd normaliser that was saved alongside each
  policy by train.py and applies it at inference time, exactly matching
  the normalisation used during training.

Usage
-----
    python evaluate.py

Expects
-------
    models/hunter_policy.pt   (contains both weights and normaliser state)
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

from my_game_env import parallel_env
from models import ActorCritic, RunningMeanStd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
HUNTER_MODEL_PATH = "models/hunter_policy.pt"
PREY_MODEL_PATH   = "models/prey_policy.pt"
HIDDEN_SIZE       = 256        # must match training
CAPTURE_DIST      = 30         # px — 2 × AGENT_SIZE

NUM_EPISODES     = 20
RENDER_SLEEP_SEC = 0.033

GIF_EPISODES  = 5
GIF_DIR       = Path("gifs")
GIF_FPS       = 30
GIF_FRAME_SKIP = 2
GIF_SCALE     = 0.5


# ---------------------------------------------------------------------------
# Load helpers
# ---------------------------------------------------------------------------
def load_policy_and_normalizer(
    path: str, obs_size: int, action_size: int
) -> tuple[ActorCritic, RunningMeanStd]:
    """
    Load both the policy weights and the normaliser state from a checkpoint
    saved by train.py.  Falls back gracefully if the file is in the old
    format (weights only).
    """

    ckpt = torch.load(path, map_location="cpu", weights_only=False) #make weights only false

    policy = ActorCritic(obs_size, action_size, HIDDEN_SIZE)
    normalizer = RunningMeanStd(shape=(obs_size,))

    if isinstance(ckpt, dict) and "policy" in ckpt:
        # New format: {"policy": state_dict, "normalizer": state_dict}
        policy.load_state_dict(ckpt["policy"])
        if "normalizer" in ckpt:
            normalizer.load_state_dict(ckpt["normalizer"])
    else:
        # Legacy format: bare state_dict from v1
        policy.load_state_dict(ckpt)

    policy.eval()
    return policy, normalizer


# ---------------------------------------------------------------------------
# GIF helpers
# ---------------------------------------------------------------------------
def scale_frame(frame: np.ndarray, scale: float) -> np.ndarray:
    img = Image.fromarray(frame)
    return np.array(
        img.resize(
            (max(1, int(img.width * scale)), max(1, int(img.height * scale))),
            Image.LANCZOS,
        )
    )


def save_gif(frames: list, path: Path, fps: int) -> None:
    if not frames:
        print(f"  [warn] no frames captured for {path.name}")
        return
    pil_frames = [Image.fromarray(f) for f in frames]
    pil_frames[0].save(
        path,
        save_all=True,
        append_images=pil_frames[1:],
        loop=0,
        duration=int(1000 / fps),
        optimize=False,
    )
    print(f"  Saved {path.name}  ({len(frames)} frames, {path.stat().st_size//1024} KB)")


# ---------------------------------------------------------------------------
# Rollout
# ---------------------------------------------------------------------------
def _rollout(env, policies, normalizers, record_frames: bool) -> dict:
    """Run one complete episode.  Returns steps, captured flag, and frames."""
    obs, _   = env.reset()
    ep_steps = 0
    captured = False
    frames   = []
    step_idx = 0

    while env.agents:
        if record_frames and step_idx % GIF_FRAME_SKIP == 0:
            raw = env.render()
            if raw is not None:
                frames.append(scale_frame(raw, GIF_SCALE))

        actions = {}
        with torch.no_grad():
            for agent in env.agents:
                # Apply the same normalisation used during training
                normed = normalizers[agent].normalize(obs[agent])
                obs_t  = torch.FloatTensor(normed).unsqueeze(0)
                actions[agent] = policies[agent].get_action(obs_t)

        obs, rewards, terminations, truncations, infos = env.step(actions)
        ep_steps += 1
        step_idx += 1

        if not record_frames:
            time.sleep(RENDER_SLEEP_SEC)

        if any(terminations.values()):
            dist = next(iter(infos.values())).get("distance", float("inf"))
            if dist <= CAPTURE_DIST:
                captured = True

    if record_frames:
        raw = env.render()
        if raw is not None:
            frames.append(scale_frame(raw, GIF_SCALE))

    return {"steps": ep_steps, "captured": captured, "frames": frames}


# ---------------------------------------------------------------------------
# Phase 1 — live evaluation
# ---------------------------------------------------------------------------
def run_live_evaluation(policies, normalizers) -> dict:
    print(f"\n{'='*55}")
    print(f"  PHASE 1 — Live evaluation  ({NUM_EPISODES} episodes)")
    print(f"{'='*55}")

    env = parallel_env(render_mode="human")
    env.reset(seed=0)

    capture_count = 0
    capture_steps = []
    all_ep_steps  = []

    for ep in range(1, NUM_EPISODES + 1):
        result = _rollout(env, policies, normalizers, record_frames=False)
        all_ep_steps.append(result["steps"])

        if result["captured"]:
            capture_count += 1
            capture_steps.append(result["steps"])
            print(f"  Episode {ep:>3}: Prey CAPTURED  in {result['steps']:>4} steps")
        else:
            print(f"  Episode {ep:>3}: Prey ESCAPED   in {result['steps']:>4} steps (timeout)")

    env.close()
    return {
        "capture_count": capture_count,
        "capture_steps": capture_steps,
        "all_ep_steps":  all_ep_steps,
    }


# ---------------------------------------------------------------------------
# Phase 2 — GIF recording
# ---------------------------------------------------------------------------
def run_gif_recording(policies, normalizers) -> None:
    print(f"\n{'='*55}")
    print(f"  PHASE 2 — GIF recording  ({GIF_EPISODES} episodes)")
    print(f"  Output : {GIF_DIR.resolve()}")
    print(f"  FPS / skip / scale : {GIF_FPS} / {GIF_FRAME_SKIP} / {GIF_SCALE:.0%}")
    print(f"{'='*55}")

    GIF_DIR.mkdir(parents=True, exist_ok=True)
    env = parallel_env(render_mode="rgb_array")
    env.reset(seed=100)

    for ep in range(1, GIF_EPISODES + 1):
        print(f"\n  Recording episode {ep}/{GIF_EPISODES}...")
        result = _rollout(env, policies, normalizers, record_frames=True)

        outcome  = "captured" if result["captured"] else "escaped"
        gif_name = f"episode_{ep:02d}_{outcome}_{result['steps']}steps.gif"
        save_gif(result["frames"], GIF_DIR / gif_name, fps=GIF_FPS)
        print(f"  {'CAPTURED' if result['captured'] else 'ESCAPED'}"
              f"  in {result['steps']} steps → {gif_name}")

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
        print("  No captures recorded — try training for longer.")
    print(f"{'='*55}\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    # Probe env for space sizes
    probe = parallel_env()
    probe.reset()
    agents      = probe.possible_agents
    obs_size    = probe.observation_space(agents[0]).shape[0]
    action_size = probe.action_space(agents[0]).shape[0]
    probe.close()

    print(f"Loading policies  (obs_size={obs_size}, action_size={action_size})")
    policies    = {}
    normalizers = {}
    for a, path in zip(agents, [HUNTER_MODEL_PATH, PREY_MODEL_PATH]):
        pol, norm = load_policy_and_normalizer(path, obs_size, action_size)
        policies[a]    = pol
        normalizers[a] = norm
        print(f"  Loaded {a:>6} from '{path}'")

    stats = run_live_evaluation(policies, normalizers)
    print_summary(stats, NUM_EPISODES)

    run_gif_recording(policies, normalizers)
    print(f"Done!  GIFs saved to '{GIF_DIR}/'")


if __name__ == "__main__":
    main()
