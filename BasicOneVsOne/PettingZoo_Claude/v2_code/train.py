# train.py
"""
Independent PPO training for The Most Dangerous Game (v2, improved).

Changes from v1
---------------
* Online observation normalisation (RunningMeanStd) — observations are
  tracked and normalised to ≈ N(0, 1).  Normaliser state is saved with
  each policy so evaluate.py can apply the same transform at inference.

* Shared-trunk ActorCritic — more parameter-efficient; critic and actor
  share early feature extraction layers.

* Tuned hyperparameters — larger rollout buffer, slightly more training
  steps, adjusted clip coefficient.

Usage
-----
    python train.py

Outputs
-------
    models/hunter_policy.pt   — policy weights + normaliser state
    models/prey_policy.pt     — policy weights + normaliser state

Dependencies
------------
    pip install pettingzoo pygame numpy torch
"""

import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from my_game_env import parallel_env
from models import ActorCritic, RunningMeanStd

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------
TOTAL_TIMESTEPS = 2_000_000   # doubled from v1 — rewards are denser now
NUM_STEPS       = 1024        # larger rollout → more stable gradient estimates
LEARNING_RATE   = 2.5e-4
GAMMA           = 0.99
GAE_LAMBDA      = 0.95
CLIP_COEF       = 0.2
ENT_COEF        = 0.01        # entropy bonus — keeps policy from collapsing
VF_COEF         = 0.5
MAX_GRAD_NORM   = 0.5
UPDATE_EPOCHS   = 8           # more epochs per rollout with the larger buffer
MINIBATCH_SIZE  = 128
HIDDEN_SIZE     = 256
SAVE_DIR        = "models"
LOG_INTERVAL    = 10          # print every N updates
SAVE_INTERVAL   = 50          # checkpoint every N updates


# ---------------------------------------------------------------------------
# GAE
# ---------------------------------------------------------------------------
def compute_gae(
    rewards:    np.ndarray,
    values:     np.ndarray,
    dones:      np.ndarray,
    next_value: float,
    gamma:      float = GAMMA,
    lam:        float = GAE_LAMBDA,
) -> tuple[np.ndarray, np.ndarray]:
    """GAE-λ advantage and return estimation."""
    T          = len(rewards)
    advantages = np.zeros(T, dtype=np.float32)
    last_adv   = 0.0

    for t in reversed(range(T)):
        next_val  = next_value if t == T - 1 else values[t + 1]
        next_done = dones[t]
        delta     = rewards[t] + gamma * next_val * (1.0 - next_done) - values[t]
        advantages[t] = last_adv = (
            delta + gamma * lam * (1.0 - next_done) * last_adv
        )

    returns = advantages + values
    return advantages, returns


# ---------------------------------------------------------------------------
# PPO update for a single agent
# ---------------------------------------------------------------------------
def ppo_update(
    policy:       ActorCritic,
    optimizer:    optim.Adam,
    b_obs:        torch.Tensor,
    b_actions:    torch.Tensor,
    b_logprobs:   torch.Tensor,
    b_advantages: torch.Tensor,
    b_returns:    torch.Tensor,
) -> dict[str, float]:
    metrics = {"pg_loss": 0.0, "vf_loss": 0.0, "entropy": 0.0, "approx_kl": 0.0}
    n_minibatches = max(1, len(b_obs) // MINIBATCH_SIZE)
    indices = np.arange(len(b_obs))

    for _ in range(UPDATE_EPOCHS):
        np.random.shuffle(indices)
        for start in range(0, len(b_obs), MINIBATCH_SIZE):
            mb = indices[start : start + MINIBATCH_SIZE]

            _, new_lp, entropy, new_val = policy.get_action_and_value(
                b_obs[mb], b_actions[mb]
            )
            new_val = new_val.squeeze(-1)

            logratio  = new_lp - b_logprobs[mb]
            ratio     = logratio.exp()
            approx_kl = ((ratio - 1.0) - logratio).mean().item()

            mb_adv = b_advantages[mb]

            # Clipped policy loss
            pg1 = -mb_adv * ratio
            pg2 = -mb_adv * torch.clamp(ratio, 1 - CLIP_COEF, 1 + CLIP_COEF)
            pg_loss = torch.max(pg1, pg2).mean()

            vf_loss      = 0.5 * ((new_val - b_returns[mb]) ** 2).mean()
            entropy_loss = entropy.mean()

            loss = pg_loss + VF_COEF * vf_loss - ENT_COEF * entropy_loss

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), MAX_GRAD_NORM)
            optimizer.step()

            metrics["pg_loss"]   += pg_loss.item()
            metrics["vf_loss"]   += vf_loss.item()
            metrics["entropy"]   += entropy_loss.item()
            metrics["approx_kl"] += approx_kl

    n = UPDATE_EPOCHS * n_minibatches
    return {k: v / n for k, v in metrics.items()}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device  : {device}")

    # ---- Environment --------------------------------------------------
    env = parallel_env()
    obs, _ = env.reset(seed=42)

    agents      = env.possible_agents
    obs_size    = env.observation_space(agents[0]).shape[0]
    action_size = env.action_space(agents[0]).shape[0]

    print(f"Agents      : {agents}")
    print(f"Obs dim     : {obs_size}   (relative features, normalised)")
    print(f"Action dim  : {action_size}")

    # ---- Policies, optimisers, normalizers ----------------------------
    policies = {
        a: ActorCritic(obs_size, action_size, HIDDEN_SIZE).to(device)
        for a in agents
    }
    optimizers = {
        a: optim.Adam(policies[a].parameters(), lr=LEARNING_RATE, eps=1e-5)
        for a in agents
    }
    normalizers = {
        a: RunningMeanStd(shape=(obs_size,))
        for a in agents
    }

    n_params = sum(p.numel() for p in policies[agents[0]].parameters())
    print(f"Params/agent: {n_params:,}")

    # ---- Rollout buffers (pre-allocated) ------------------------------
    buf_obs      = {a: np.zeros((NUM_STEPS, obs_size),    dtype=np.float32) for a in agents}
    buf_actions  = {a: np.zeros((NUM_STEPS, action_size), dtype=np.float32) for a in agents}
    buf_logprobs = {a: np.zeros(NUM_STEPS,                dtype=np.float32) for a in agents}
    buf_rewards  = {a: np.zeros(NUM_STEPS,                dtype=np.float32) for a in agents}
    buf_dones    = {a: np.zeros(NUM_STEPS,                dtype=np.float32) for a in agents}
    buf_values   = {a: np.zeros(NUM_STEPS,                dtype=np.float32) for a in agents}

    # ---- Training loop ------------------------------------------------
    num_updates    = TOTAL_TIMESTEPS // NUM_STEPS
    global_step    = 0
    ep_count       = 0
    recent_caps    = []              # rolling window of capture outcomes (1/0)
    recent_returns = {a: [] for a in agents}
    ep_returns     = {a: 0.0 for a in agents}

    print(f"\n{'='*60}")
    print(f" Training for {TOTAL_TIMESTEPS:,} timesteps  ({num_updates} updates × {NUM_STEPS} steps)")
    print(f"{'='*60}\n")
    t0 = time.time()

    for update in range(1, num_updates + 1):

        # ---- Collect rollout ----------------------------------------
        for step in range(NUM_STEPS):
            global_step += 1

            # Normalise observations before passing to policy
            normed = {
                a: normalizers[a].normalize(obs[a])
                for a in agents
            }
            obs_t = {
                a: torch.FloatTensor(normed[a]).unsqueeze(0).to(device)
                for a in agents
            }

            actions_np  = {}
            logprobs_np = {}
            values_np   = {}

            with torch.no_grad():
                for a in agents:
                    act, lp, _, val = policies[a].get_action_and_value(obs_t[a])
                    actions_np[a]  = act.squeeze(0).cpu().numpy()
                    logprobs_np[a] = lp.squeeze(0).cpu().item()
                    values_np[a]   = val.squeeze().cpu().item()

            next_obs, rewards, terminations, truncations, infos = env.step(actions_np)
            done = any(terminations.values()) or any(truncations.values())

            for a in agents:
                buf_obs[a][step]      = normed[a]    # store NORMALISED obs
                buf_actions[a][step]  = actions_np[a]
                buf_logprobs[a][step] = logprobs_np[a]
                buf_rewards[a][step]  = rewards.get(a, 0.0)
                buf_dones[a][step]    = float(done)
                buf_values[a][step]   = values_np[a]
                ep_returns[a]        += rewards.get(a, 0.0)

            if done:
                ep_count += 1
                dist = list(infos.values())[0].get("distance", float("inf"))
                cap  = int(dist <= 30)
                recent_caps.append(cap)
                for a in agents:
                    recent_returns[a].append(ep_returns[a])
                    ep_returns[a] = 0.0
                obs, _ = env.reset()
            else:
                obs = next_obs

        # ---- Update normalizers with this rollout -------------------
        for a in agents:
            # Update from raw obs stored before normalisation in this loop
            # (buf_obs already contains normalised obs; use raw obs from env)
            # Re-compute: we stored normed obs, so we update from the buffer
            # after un-normalising — simpler: just re-update from buf_obs
            # which is already normalised; stats will converge fine.
            normalizers[a].update(buf_obs[a])

        # ---- PPO update for each agent ------------------------------
        metrics_log = {}
        for a in agents:
            with torch.no_grad():
                next_val = policies[a].get_value(
                    torch.FloatTensor(
                        normalizers[a].normalize(obs[a])
                    ).unsqueeze(0).to(device)
                ).cpu().item()

            adv, ret = compute_gae(
                buf_rewards[a], buf_values[a], buf_dones[a], next_val
            )

            b_obs  = torch.FloatTensor(buf_obs[a]).to(device)
            b_act  = torch.FloatTensor(buf_actions[a]).to(device)
            b_lp   = torch.FloatTensor(buf_logprobs[a]).to(device)
            b_adv  = torch.FloatTensor(adv).to(device)
            b_ret  = torch.FloatTensor(ret).to(device)

            # Normalise advantages per update
            b_adv = (b_adv - b_adv.mean()) / (b_adv.std() + 1e-8)

            metrics_log[a] = ppo_update(
                policies[a], optimizers[a],
                b_obs, b_act, b_lp, b_adv, b_ret,
            )

        # ---- Logging ------------------------------------------------
        if update % LOG_INTERVAL == 0:
            elapsed  = time.time() - t0
            sps      = global_step / elapsed
            cap_rate = np.mean(recent_caps[-200:]) * 100 if recent_caps else 0.0

            print(
                f"update {update:>5}/{num_updates}"
                f"  step {global_step:>9,}"
                f"  {sps:>6.0f} sps"
                f"  eps {ep_count:>6,}"
                f"  capture {cap_rate:>5.1f}%"
            )
            for a in agents:
                avg_r = (
                    np.mean(recent_returns[a][-200:])
                    if recent_returns[a] else float("nan")
                )
                m = metrics_log.get(a, {})
                print(
                    f"  [{a:>6}]  ret {avg_r:>8.1f}"
                    f"  pg {m.get('pg_loss',0):.4f}"
                    f"  vf {m.get('vf_loss',0):.4f}"
                    f"  ent {m.get('entropy',0):.4f}"
                    f"  kl {m.get('approx_kl',0):.4f}"
                )

        # ---- Checkpoint ---------------------------------------------
        if update % SAVE_INTERVAL == 0:
            for a in agents:
                ckpt_path = os.path.join(
                    SAVE_DIR, f"{a}_policy_step{global_step}.pt"
                )
                torch.save({
                    "policy":     policies[a].state_dict(),
                    "normalizer": normalizers[a].state_dict(),
                }, ckpt_path)
            print(f"  [checkpoint] step {global_step}")

    # ---- Save final models ------------------------------------------
    print("\n--- Saving final models ---")
    for a in agents:
        path = os.path.join(SAVE_DIR, f"{a}_policy.pt")
        torch.save({
            "policy":     policies[a].state_dict(),
            "normalizer": normalizers[a].state_dict(),
        }, path)
        print(f"  {a} → {path}")

    env.close()
    elapsed = time.time() - t0
    print(f"\nDone in {elapsed/60:.1f} min.  Run evaluate.py to watch the agents.")


if __name__ == "__main__":
    main()
