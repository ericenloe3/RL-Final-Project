# evaluate.py
"""
Evaluate trained Hunter and Prey policies from The Most Dangerous Game.

Two phases are run back-to-back:

  Phase 1 — Live viewing
      NUM_EPISODES episodes rendered in a pygame window with full stats.

  Phase 2 — GIF recording
      GIF_EPISODES episodes captured via rgb_array render mode and saved
      to the 'gifs/' directory as animated GIFs.

Usage
-----
    python evaluate.py

Expects
-------
    models/hunter_policy.pt
    models/prey_policy.pt

Both are produced by train.py.

Dependencies
------------
    pip install pettingzoo pygame numpy torch pillow
"""

import os
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from my_game_env import parallel_env
from models import ActorCritic

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
HUNTER_MODEL_PATH = "models/hunter_policy.pt"
PREY_MODEL_PATH   = "models/prey_policy.pt"
HIDDEN_SIZE       = 256        # must match the value used during training
CAPTURE_DIST      = 30         # pixels — 2 × AGENT_SIZE

# Phase 1 — live viewing
NUM_EPISODES     = 20
RENDER_SLEEP_SEC = 0.033       # ~30 fps pacing on top of pygame's own clock

# Phase 2 — GIF recording
GIF_EPISODES     = 5
GIF_DIR          = Path("gifs")
GIF_FPS          = 30          # target playback speed
GIF_FRAME_SKIP   = 2           # record every Nth frame (reduces file size)
GIF_SCALE        = 0.5         # downscale factor applied to each frame


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_policy(path: str, obs_size: int, action_size: int) -> ActorCritic:
    """Load a saved ActorCritic policy from disk."""
    policy = ActorCritic(obs_size, action_size, HIDDEN_SIZE)
    policy.load_state_dict(torch.load(path, map_location="cpu"))
    policy.eval()
    return policy


def scale_frame(frame: np.ndarray, scale: float) -> np.ndarray:
    """Resize an HxWx3 uint8 numpy frame by 'scale' using Pillow."""
    img = Image.fromarray(frame)
    new_w = max(1, int(img.width  * scale))
    new_h = max(1, int(img.height * scale))
    return np.array(img.resize((new_w, new_h), Image.LANCZOS))


def save_gif(frames: list, path: Path, fps: int) -> None:
    """
    Save a list of HxWx3 uint8 numpy arrays as an animated GIF.

    Parameters
    ----------
    frames : list of numpy arrays (already scaled/processed)
    path   : destination file path (will be created)
    fps    : playback speed in frames-per-second
    """
    if not frames:
        print(f"  [warn] No frames to save for {path.name}")
        return

    duration_ms = int(1000 / fps)          # milliseconds between frames
    pil_frames  = [Image.fromarray(f) for f in frames]

    pil_frames[0].save(
        path,
        save_all=True,
        append_images=pil_frames[1:],
        loop=0,                            # 0 = loop forever
        duration=duration_ms,
        optimize=False,
    )
    size_kb = path.stat().st_size / 1024
    print(f"  Saved {path.name}  ({len(frames)} frames, {size_kb:.0f} KB)")


def _rollout(env, policies: dict, record_frames: bool) -> dict:
    """
    Run a single episode to completion.

    Returns a result dict:
        steps     : int
        captured  : bool
        frames    : list[np.ndarray]  (empty when record_frames=False)
    """
    obs, _ = env.reset()
    ep_steps  = 0
    captured  = False
    frames    = []
    step_idx  = 0

    while env.agents:
        # --- Optionally capture a frame before acting ------------------
        if record_frames and (step_idx % GIF_FRAME_SKIP == 0):
            raw = env.render()              # rgb_array mode returns ndarray
            if raw is not None:
                frames.append(scale_frame(raw, GIF_SCALE))

        # --- Select actions for every active agent ---------------------
        actions = {}
        with torch.no_grad():
            for agent in env.agents:
                obs_t = torch.FloatTensor(obs[agent]).unsqueeze(0)
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

    # Capture the final frame so the GIF ends on the terminal state
    if record_frames:
        raw = env.render()
        if raw is not None:
            frames.append(scale_frame(raw, GIF_SCALE))

    return {"steps": ep_steps, "captured": captured, "frames": frames}


# ---------------------------------------------------------------------------
# Phase 1 — live viewing
# ---------------------------------------------------------------------------
def run_live_evaluation(policies: dict) -> dict:
    """
    Run NUM_EPISODES in a pygame window and return aggregate stats.
    """
    print(f"\n{'='*55}")
    print(f"  PHASE 1 — Live evaluation  ({NUM_EPISODES} episodes)")
    print(f"{'='*55}")

    env = parallel_env(render_mode="human")
    env.reset(seed=0)                       # seed the first reset for reproducibility

    capture_count = 0
    capture_steps = []
    all_ep_steps  = []

    for episode in range(1, NUM_EPISODES + 1):
        result = _rollout(env, policies, record_frames=False)

        all_ep_steps.append(result["steps"])
        if result["captured"]:
            capture_count += 1
            capture_steps.append(result["steps"])
            print(f"  Episode {episode:>3}: Prey CAPTURED  in {result['steps']:>4} steps")
        else:
            print(f"  Episode {episode:>3}: Prey ESCAPED   in {result['steps']:>4} steps (timeout)")

    env.close()

    return {
        "capture_count": capture_count,
        "capture_steps": capture_steps,
        "all_ep_steps":  all_ep_steps,
    }


# ---------------------------------------------------------------------------
# Phase 2 — GIF recording
# ---------------------------------------------------------------------------
def run_gif_recording(policies: dict) -> None:
    """
    Record GIF_EPISODES complete episodes in rgb_array mode and save each
    as an animated GIF.  Episodes are named by outcome and step count so
    the files are self-describing.
    """
    print(f"\n{'='*55}")
    print(f"  PHASE 2 — GIF recording  ({GIF_EPISODES} episodes)")
    print(f"  Output directory : {GIF_DIR.resolve()}")
    print(f"  Frame skip       : every {GIF_FRAME_SKIP} steps")
    print(f"  Playback speed   : {GIF_FPS} fps")
    print(f"  Frame scale      : {GIF_SCALE:.0%}")
    print(f"{'='*55}")

    GIF_DIR.mkdir(parents=True, exist_ok=True)

    # rgb_array mode: render() returns a numpy array instead of drawing
    # to a window, so no display is required during this phase.
    env = parallel_env(render_mode="rgb_array")
    env.reset(seed=100)                     # different seed from Phase 1

    for episode in range(1, GIF_EPISODES + 1):
        print(f"\n  Recording episode {episode}/{GIF_EPISODES}...")

        result = _rollout(env, policies, record_frames=True)

        outcome  = "captured" if result["captured"] else "escaped"
        gif_name = f"episode_{episode:02d}_{outcome}_{result['steps']}steps.gif"
        gif_path = GIF_DIR / gif_name

        save_gif(result["frames"], gif_path, fps=GIF_FPS)
        status = "CAPTURED" if result["captured"] else "ESCAPED"
        print(f"  {status}  in {result['steps']} steps  ->  {gif_name}")

    env.close()


# ---------------------------------------------------------------------------
# Summary printer
# ---------------------------------------------------------------------------
def print_summary(stats: dict, num_episodes: int) -> None:
    print(f"\n{'='*55}")
    print("               EVALUATION SUMMARY")
    print(f"{'='*55}")

    capture_rate = stats["capture_count"] / num_episodes * 100
    print(f"  Episodes        : {num_episodes}")
    print(f"  Captures        : {stats['capture_count']}")
    print(f"  Capture rate    : {capture_rate:.1f}%")
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
    # ---- Probe env for space sizes (headless) --------------------------
    probe_env   = parallel_env()
    probe_env.reset()
    agents      = probe_env.possible_agents
    obs_size    = probe_env.observation_space(agents[0]).shape[0]
    action_size = probe_env.action_space(agents[0]).shape[0]
    probe_env.close()

    # ---- Load policies -------------------------------------------------
    print(f"Loading hunter policy from '{HUNTER_MODEL_PATH}'...")
    print(f"Loading prey   policy from '{PREY_MODEL_PATH}'...")
    policies = {
        "hunter": load_policy(HUNTER_MODEL_PATH, obs_size, action_size),
        "prey":   load_policy(PREY_MODEL_PATH,   obs_size, action_size),
    }
    print("Policies loaded.")

    # ---- Phase 1: live viewing -----------------------------------------
    stats = run_live_evaluation(policies)
    print_summary(stats, NUM_EPISODES)

    # ---- Phase 2: GIF recording ----------------------------------------
    run_gif_recording(policies)
    print(f"Done!  GIFs saved to '{GIF_DIR}/'")


if __name__ == "__main__":
    main()
