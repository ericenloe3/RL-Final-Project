# train.py
"""
Independent PPO training — The Most Dangerous Game (v3).

What changed from v2 and why
------------------------------
FIX – Normaliser was updating from already-normalised data
    v2 stored normalised observations in the rollout buffer and then called
    normalizer.update(buf_obs), which fed scaled data back into the Welford
    estimator.  Over time this corrupted the running statistics and caused
    the effective input distribution seen by the network to drift.  v3 keeps
    a separate RAW observation buffer used exclusively for normaliser updates;
    the normalised buffer is used for the PPO loss.

NEW – 8 parallel environments (SyncVecEnv)
    A single environment collects one trajectory at a time, producing highly
    auto-correlated batches.  Running NUM_ENVS=8 independent envs and
    interleaving their transitions multiplies batch diversity by 8× with no
    change to the PPO algorithm.  Each update now sees NUM_ENVS × NUM_STEPS
    = 8 192 transitions instead of 1 024.

NEW – Curriculum learning (3 phases)
    Phase 1 (0 – 30 % of total steps):   0 obstacles — agents learn raw
      pursuit and evasion without navigation complexity.
    Phase 2 (30 – 60 %):                 1–3 obstacles — moderate clutter.
    Phase 3 (60 – 100 %):                3–5 obstacles — full difficulty.
    The obstacle range is set on every env instance at each reset, so the
    transition is smooth and automatic.

TUNED – hyperparameters for the larger effective batch size
    NUM_STEPS reduced from 1024 to 512 (faster curriculum transitions);
    MINIBATCH_SIZE increased to 256; UPDATE_EPOCHS kept at 8.

Usage
-----
    python train.py

Outputs
-------
    models/hunter_policy.pt   {policy: state_dict, normalizer: state_dict}
    models/prey_policy.pt

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
TOTAL_TIMESTEPS = 5_000_000
NUM_ENVS        = 8           # parallel environment instances
NUM_STEPS       = 512         # rollout steps per env per update
# Effective batch per update = NUM_ENVS × NUM_STEPS = 4096
LEARNING_RATE   = 2.5e-4
GAMMA           = 0.99
GAE_LAMBDA      = 0.95
CLIP_COEF       = 0.2
ENT_COEF        = 0.01
VF_COEF         = 0.5
MAX_GRAD_NORM   = 0.5
UPDATE_EPOCHS   = 8
MINIBATCH_SIZE  = 256
HIDDEN_SIZE     = 256
SAVE_DIR        = "models"
LOG_INTERVAL    = 10
SAVE_INTERVAL   = 50

# Curriculum phase boundaries (fraction of TOTAL_TIMESTEPS)
CURRICULUM = [
    (0.00, (0, 0)),    # Phase 1: no obstacles
    (0.30, (1, 3)),    # Phase 2: light clutter
    (0.60, (3, 5)),    # Phase 3: full difficulty
]


# ---------------------------------------------------------------------------
# GAE (per-env)
# ---------------------------------------------------------------------------
def compute_gae(rewards, values, dones, next_value, gamma=GAMMA, lam=GAE_LAMBDA):
    T          = len(rewards)
    advantages = np.zeros(T, dtype=np.float32)
    last_adv   = 0.0
    for t in reversed(range(T)):
        nv       = next_value if t == T - 1 else values[t + 1]
        nd       = dones[t]
        delta    = rewards[t] + gamma * nv * (1.0 - nd) - values[t]
        advantages[t] = last_adv = delta + gamma * lam * (1.0 - nd) * last_adv
    return advantages, advantages + values


# ---------------------------------------------------------------------------
# PPO update (single agent)
# ---------------------------------------------------------------------------
def ppo_update(policy, optimizer, b_obs, b_acts, b_lps, b_advs, b_rets):
    metrics = {"pg": 0.0, "vf": 0.0, "ent": 0.0, "kl": 0.0}
    idx     = np.arange(len(b_obs))
    n_mb    = max(1, len(b_obs) // MINIBATCH_SIZE)

    for _ in range(UPDATE_EPOCHS):
        np.random.shuffle(idx)
        for s in range(0, len(b_obs), MINIBATCH_SIZE):
            mb = idx[s : s + MINIBATCH_SIZE]

            _, new_lp, ent, new_val = policy.get_action_and_value(b_obs[mb], b_acts[mb])
            new_val = new_val.squeeze(-1)

            ratio    = (new_lp - b_lps[mb]).exp()
            apx_kl   = ((ratio - 1) - (new_lp - b_lps[mb])).mean().item()
            adv      = b_advs[mb]

            pg  = torch.max(-adv * ratio,
                            -adv * ratio.clamp(1 - CLIP_COEF, 1 + CLIP_COEF)).mean()
            vf  = 0.5 * ((new_val - b_rets[mb]) ** 2).mean()
            loss = pg + VF_COEF * vf - ENT_COEF * ent.mean()

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), MAX_GRAD_NORM)
            optimizer.step()

            metrics["pg"]  += pg.item()
            metrics["vf"]  += vf.item()
            metrics["ent"] += ent.mean().item()
            metrics["kl"]  += apx_kl

    n = UPDATE_EPOCHS * n_mb
    return {k: v / n for k, v in metrics.items()}


# ---------------------------------------------------------------------------
# Curriculum helper
# ---------------------------------------------------------------------------
def current_obstacle_range(global_step: int) -> tuple[int, int]:
    frac = global_step / TOTAL_TIMESTEPS
    obs_range = CURRICULUM[0][1]
    for threshold, rng in CURRICULUM:
        if frac >= threshold:
            obs_range = rng
    return obs_range


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device      : {device}")
    print(f"Num envs    : {NUM_ENVS}")
    print(f"Batch size  : {NUM_ENVS * NUM_STEPS:,} transitions / update")

    # ---- Create environments ----------------------------------------
    envs = [parallel_env() for _ in range(NUM_ENVS)]
    # Initialise obs by resetting all envs
    env_obs = []
    for i, env in enumerate(envs):
        obs, _ = env.reset(seed=42 + i)
        env_obs.append(obs)

    agents      = envs[0].possible_agents
    obs_size    = envs[0].observation_space(agents[0]).shape[0]
    action_size = envs[0].action_space(agents[0]).shape[0]

    print(f"Agents      : {agents}")
    print(f"Obs dim     : {obs_size}")

    # ---- Policies, optimisers, normalisers --------------------------
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

    print(f"Params/agent: {sum(p.numel() for p in policies[agents[0]].parameters()):,}")

    # ---- Rollout buffers (NUM_ENVS × NUM_STEPS) ---------------------
    # Two obs buffers: raw (for normaliser update) and normed (for PPO loss)
    B, T = NUM_ENVS, NUM_STEPS
    raw_buf  = {a: np.zeros((B, T, obs_size),    dtype=np.float32) for a in agents}
    norm_buf = {a: np.zeros((B, T, obs_size),    dtype=np.float32) for a in agents}
    act_buf  = {a: np.zeros((B, T, action_size), dtype=np.float32) for a in agents}
    lp_buf   = {a: np.zeros((B, T),              dtype=np.float32) for a in agents}
    rew_buf  = {a: np.zeros((B, T),              dtype=np.float32) for a in agents}
    val_buf  = {a: np.zeros((B, T),              dtype=np.float32) for a in agents}
    done_buf =     np.zeros((B, T),              dtype=np.float32)

    # ---- Stats tracking ---------------------------------------------
    num_updates  = TOTAL_TIMESTEPS // (NUM_ENVS * NUM_STEPS)
    global_step  = 0
    ep_count     = 0
    recent_caps  : list[int]   = []
    recent_rets  = {a: [] for a in agents}
    ep_rets      = [{a: 0.0 for a in agents} for _ in range(NUM_ENVS)]

    print(f"\n{'='*65}")
    print(f" Training  {TOTAL_TIMESTEPS:,} steps  ·  {num_updates} updates  ·  {NUM_ENVS} envs")
    print(f" Curriculum: {CURRICULUM}")
    print(f"{'='*65}\n")
    t0 = time.time()
    prev_phase = None

    for update in range(1, num_updates + 1):

        # ---- Curriculum: update all envs if phase changed -----------
        obs_range = current_obstacle_range(global_step)
        if obs_range != prev_phase:
            for env in envs:
                env.set_obstacle_range(*obs_range)
            phase_idx = next(
                i for i, (_, r) in enumerate(CURRICULUM) if r == obs_range
            )
            print(f"  [curriculum] phase {phase_idx+1}  obstacles {obs_range}"
                  f"  (step {global_step:,})")
            prev_phase = obs_range

        # ---- Collect rollout across all envs ------------------------
        for step in range(NUM_STEPS):
            global_step += NUM_ENVS

            # Normalise obs for all envs — batch the network forward pass
            normed_all = [
                {a: normalizers[a].normalize(env_obs[ei][a]) for a in agents}
                for ei in range(NUM_ENVS)
            ]

            with torch.no_grad():
                # Build per-agent batches: shape (NUM_ENVS, obs_size)
                obs_batch = {
                    a: torch.FloatTensor(
                        np.array([normed_all[ei][a] for ei in range(NUM_ENVS)])
                    ).to(device)
                    for a in agents
                }
                acts_batch = {}
                lps_batch  = {}
                vals_batch = {}
                for a in agents:
                    act, lp, _, val = policies[a].get_action_and_value(obs_batch[a])
                    acts_batch[a] = act.cpu().numpy()            # (B, action_size)
                    lps_batch[a]  = lp.cpu().numpy()             # (B,)
                    vals_batch[a] = val.squeeze(-1).cpu().numpy()# (B,)

            # Step each env and store transitions
            for ei in range(NUM_ENVS):
                actions = {a: acts_batch[a][ei] for a in agents}

                for a in agents:
                    raw_buf[a][ei, step]  = env_obs[ei][a]       # RAW for normaliser
                    norm_buf[a][ei, step] = normed_all[ei][a]    # normalised for PPO
                    act_buf[a][ei, step]  = acts_batch[a][ei]
                    lp_buf[a][ei, step]   = lps_batch[a][ei]
                    val_buf[a][ei, step]  = vals_batch[a][ei]

                next_obs, rewards, terms, truns, infos = envs[ei].step(actions)
                done = any(terms.values()) or any(truns.values())
                done_buf[ei, step] = float(done)

                for a in agents:
                    rew_buf[a][ei, step] = rewards.get(a, 0.0)
                    ep_rets[ei][a]      += rewards.get(a, 0.0)

                if done:
                    ep_count += 1
                    dist = list(infos.values())[0].get("distance", float("inf"))
                    cap  = int(dist <= 30)
                    recent_caps.append(cap)
                    for a in agents:
                        recent_rets[a].append(ep_rets[ei][a])
                        ep_rets[ei][a] = 0.0
                    env_obs[ei], _ = envs[ei].reset()
                else:
                    env_obs[ei] = next_obs

        # ---- Update normaliser from RAW observations ----------------
        for a in agents:
            # raw_buf[a] shape: (B, T, obs_size) → flatten to (B*T, obs_size)
            normalizers[a].update(raw_buf[a].reshape(-1, obs_size))

        # ---- PPO update for each agent ------------------------------
        metrics_log = {}
        for a in agents:
            # Bootstrap next values (one per env)
            with torch.no_grad():
                next_obs_batch = torch.FloatTensor(
                    np.array([normalizers[a].normalize(env_obs[ei][a])
                              for ei in range(NUM_ENVS)])
                ).to(device)
                next_vals = policies[a].get_value(next_obs_batch).squeeze(-1).cpu().numpy()

            # GAE per env, then flatten
            adv_buf = np.zeros((B, T), dtype=np.float32)
            ret_buf = np.zeros((B, T), dtype=np.float32)
            for ei in range(NUM_ENVS):
                adv_buf[ei], ret_buf[ei] = compute_gae(
                    rew_buf[a][ei], val_buf[a][ei], done_buf[ei], next_vals[ei]
                )

            # Flatten all envs into one big batch
            b_obs  = torch.FloatTensor(norm_buf[a].reshape(-1, obs_size)).to(device)
            b_acts = torch.FloatTensor(act_buf[a].reshape(-1, action_size)).to(device)
            b_lps  = torch.FloatTensor(lp_buf[a].reshape(-1)).to(device)
            b_advs = torch.FloatTensor(adv_buf.reshape(-1)).to(device)
            b_rets = torch.FloatTensor(ret_buf.reshape(-1)).to(device)

            # Normalise advantages across the full batch
            b_advs = (b_advs - b_advs.mean()) / (b_advs.std() + 1e-8)

            metrics_log[a] = ppo_update(
                policies[a], optimizers[a],
                b_obs, b_acts, b_lps, b_advs, b_rets,
            )

        # ---- Logging ------------------------------------------------
        if update % LOG_INTERVAL == 0:
            elapsed  = time.time() - t0
            sps      = global_step / elapsed
            cap_rate = np.mean(recent_caps[-500:]) * 100 if recent_caps else 0.0

            print(
                f"update {update:>5}/{num_updates}"
                f"  step {global_step:>10,}"
                f"  {sps:>6.0f} sps"
                f"  eps {ep_count:>7,}"
                f"  capture {cap_rate:>5.1f}%"
            )
            for a in agents:
                avg_r = (np.mean(recent_rets[a][-500:])
                         if recent_rets[a] else float("nan"))
                m = metrics_log.get(a, {})
                print(
                    f"  [{a:>6}]  ret {avg_r:>8.1f}"
                    f"  pg {m['pg']:.4f}"
                    f"  vf {m['vf']:.4f}"
                    f"  ent {m['ent']:.4f}"
                    f"  kl {m['kl']:.4f}"
                )

        # ---- Checkpoint ---------------------------------------------
        if update % SAVE_INTERVAL == 0:
            for a in agents:
                ckpt = os.path.join(SAVE_DIR, f"{a}_policy_step{global_step}.pt")
                torch.save({
                    "policy":     policies[a].state_dict(),
                    "normalizer": normalizers[a].state_dict(),
                }, ckpt)
            print(f"  [ckpt] step {global_step:,}")

    # ---- Save final models ------------------------------------------
    print("\n--- Saving final models ---")
    for a in agents:
        path = os.path.join(SAVE_DIR, f"{a}_policy.pt")
        torch.save({
            "policy":     policies[a].state_dict(),
            "normalizer": normalizers[a].state_dict(),
        }, path)
        print(f"  {a:>6} → {path}")

    for env in envs:
        env.close()

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed/60:.1f} min.  Run evaluate.py to watch the agents.")


if __name__ == "__main__":
    main()
