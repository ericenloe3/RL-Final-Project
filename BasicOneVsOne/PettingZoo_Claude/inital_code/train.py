# train.py
"""
Independent PPO training for The Most Dangerous Game.

Each agent (hunter, prey) maintains its own ActorCritic network and its
own Adam optimiser — the 'independent learners' baseline described in the
project proposal.

Usage
-----
    python train.py

Saved artefacts
---------------
    models/hunter_policy.pt   — hunter's state_dict
    models/prey_policy.pt     — prey's  state_dict

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
from models import ActorCritic

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------
TOTAL_TIMESTEPS = 1_000_000   # total env steps across all updates
NUM_STEPS       = 512         # rollout length per update
LEARNING_RATE   = 2.5e-4
GAMMA           = 0.99        # discount factor
GAE_LAMBDA      = 0.95        # GAE smoothing
CLIP_COEF       = 0.2         # PPO clip ε
ENT_COEF        = 0.01        # entropy bonus weight
VF_COEF         = 0.5         # value-loss weight
MAX_GRAD_NORM   = 0.5
UPDATE_EPOCHS   = 4           # PPO epochs per rollout
MINIBATCH_SIZE  = 64
HIDDEN_SIZE     = 256
SAVE_DIR        = "models"
LOG_INTERVAL    = 20          # print progress every N updates
SAVE_INTERVAL   = 100         # save checkpoint every N updates


# ---------------------------------------------------------------------------
# GAE helper
# ---------------------------------------------------------------------------
def compute_gae(
    rewards:    np.ndarray,   # (T,)
    values:     np.ndarray,   # (T,)
    dones:      np.ndarray,   # (T,) — 1.0 when episode ended
    next_value: float,
    gamma:      float = GAMMA,
    lam:        float = GAE_LAMBDA,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns (advantages, returns) arrays of shape (T,).
    Standard GAE-λ.  'dones' masks out bootstrapping across episode
    boundaries so that resets inside the rollout are handled correctly.
    """
    advantages = np.zeros_like(rewards, dtype=np.float32)
    last_adv   = 0.0

    for t in reversed(range(len(rewards))):
        next_val  = next_value if t == len(rewards) - 1 else values[t + 1]
        next_done = dones[t]                           # done *after* step t
        delta     = rewards[t] + gamma * next_val * (1.0 - next_done) - values[t]
        advantages[t] = last_adv = delta + gamma * lam * (1.0 - next_done) * last_adv

    returns = advantages + values
    return advantages, returns


# ---------------------------------------------------------------------------
# PPO update for a single agent
# ---------------------------------------------------------------------------
def ppo_update(
    policy:    ActorCritic,
    optimizer: optim.Adam,
    b_obs:      torch.Tensor,
    b_actions:  torch.Tensor,
    b_logprobs: torch.Tensor,
    b_advantages: torch.Tensor,
    b_returns:  torch.Tensor,
) -> dict[str, float]:
    """Run UPDATE_EPOCHS of minibatch PPO on a single agent's rollout buffer."""
    metrics = {"pg_loss": 0.0, "vf_loss": 0.0, "entropy": 0.0, "approx_kl": 0.0}
    indices = np.arange(len(b_obs))

    for _ in range(UPDATE_EPOCHS):
        np.random.shuffle(indices)
        for start in range(0, len(b_obs), MINIBATCH_SIZE):
            mb_idx = indices[start : start + MINIBATCH_SIZE]

            _, new_logprob, entropy, new_value = policy.get_action_and_value(
                b_obs[mb_idx], b_actions[mb_idx]
            )
            new_value = new_value.squeeze(-1)

            logratio  = new_logprob - b_logprobs[mb_idx]
            ratio     = logratio.exp()

            # Approx KL for monitoring
            with torch.no_grad():
                approx_kl = ((ratio - 1.0) - logratio).mean().item()

            mb_adv = b_advantages[mb_idx]

            # Clipped policy loss
            pg_loss = torch.max(
                -mb_adv * ratio,
                -mb_adv * torch.clamp(ratio, 1.0 - CLIP_COEF, 1.0 + CLIP_COEF),
            ).mean()

            # Value loss (unclipped for simplicity)
            vf_loss = 0.5 * ((new_value - b_returns[mb_idx]) ** 2).mean()

            # Entropy bonus (encourages exploration)
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

    n = UPDATE_EPOCHS * max(1, len(b_obs) // MINIBATCH_SIZE)
    return {k: v / n for k, v in metrics.items()}


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------
def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ---- Environment -------------------------------------------------------
    env = parallel_env()
    obs, _ = env.reset(seed=42)

    agents      = env.possible_agents          # ["hunter", "prey"]
    obs_size    = env.observation_space(agents[0]).shape[0]
    action_size = env.action_space(agents[0]).shape[0]

    print(f"Agents       : {agents}")
    print(f"Obs size     : {obs_size}")
    print(f"Action size  : {action_size}")

    # ---- Policies & optimisers --------------------------------------------
    policies = {
        a: ActorCritic(obs_size, action_size, HIDDEN_SIZE).to(device)
        for a in agents
    }
    optimizers = {
        a: optim.Adam(policies[a].parameters(), lr=LEARNING_RATE, eps=1e-5)
        for a in agents
    }

    n_params = sum(p.numel() for p in policies[agents[0]].parameters())
    print(f"Params/agent : {n_params:,}")

    # ---- Rollout storage (pre-allocated) ----------------------------------
    buf_obs      = {a: np.zeros((NUM_STEPS, obs_size),    dtype=np.float32) for a in agents}
    buf_actions  = {a: np.zeros((NUM_STEPS, action_size), dtype=np.float32) for a in agents}
    buf_logprobs = {a: np.zeros(NUM_STEPS,                dtype=np.float32) for a in agents}
    buf_rewards  = {a: np.zeros(NUM_STEPS,                dtype=np.float32) for a in agents}
    buf_dones    = {a: np.zeros(NUM_STEPS,                dtype=np.float32) for a in agents}
    buf_values   = {a: np.zeros(NUM_STEPS,                dtype=np.float32) for a in agents}

    # ---- Training loop ----------------------------------------------------
    num_updates  = TOTAL_TIMESTEPS // NUM_STEPS
    global_step  = 0
    ep_count     = 0
    ep_returns   = {a: 0.0 for a in agents}
    ep_captures  = 0

    # Episode-level accumulators for logging
    recent_returns  = {a: [] for a in agents}
    recent_captures = []

    print(f"\n{'='*60}")
    print(f" Starting training  ({TOTAL_TIMESTEPS:,} total timesteps)")
    print(f" {num_updates} updates  ×  {NUM_STEPS} steps per update")
    print(f"{'='*60}\n")
    t_start = time.time()

    for update in range(1, num_updates + 1):

        # ---- Collect rollout -------------------------------------------
        for step in range(NUM_STEPS):
            global_step += 1

            # Convert current obs to tensors (only needed agents — all active)
            obs_tensors = {
                a: torch.FloatTensor(obs[a]).unsqueeze(0).to(device)
                for a in agents
            }

            actions_np   = {}
            logprobs_np  = {}
            values_np    = {}

            with torch.no_grad():
                for a in agents:
                    act, lp, _, val = policies[a].get_action_and_value(obs_tensors[a])
                    actions_np[a]  = act.squeeze(0).cpu().numpy()
                    logprobs_np[a] = lp.squeeze(0).cpu().item()
                    values_np[a]   = val.squeeze().cpu().item()

            # Step the environment
            next_obs, rewards, terminations, truncations, infos = env.step(actions_np)

            episode_done = any(terminations.values()) or any(truncations.values())

            # Store transition
            for a in agents:
                buf_obs[a][step]      = obs[a]
                buf_actions[a][step]  = actions_np[a]
                buf_logprobs[a][step] = logprobs_np[a]
                buf_rewards[a][step]  = rewards.get(a, 0.0)
                buf_dones[a][step]    = float(episode_done)
                buf_values[a][step]   = values_np[a]
                ep_returns[a]        += rewards.get(a, 0.0)

            if episode_done:
                # Log episode stats
                ep_count += 1
                dist = list(infos.values())[0].get("distance", float("inf"))
                captured = dist <= 30                  # 2 × AGENT_SIZE
                if captured:
                    ep_captures += 1
                recent_captures.append(int(captured))
                for a in agents:
                    recent_returns[a].append(ep_returns[a])
                    ep_returns[a] = 0.0

                obs, _ = env.reset()
            else:
                obs = next_obs

        # ---- PPO update for each agent ---------------------------------
        metrics_log = {}
        for a in agents:
            # Bootstrap next value
            with torch.no_grad():
                next_val = policies[a].get_value(
                    torch.FloatTensor(obs[a]).unsqueeze(0).to(device)
                ).cpu().item()

            adv, ret = compute_gae(
                buf_rewards[a], buf_values[a], buf_dones[a], next_val
            )

            b_obs  = torch.FloatTensor(buf_obs[a]).to(device)
            b_act  = torch.FloatTensor(buf_actions[a]).to(device)
            b_lp   = torch.FloatTensor(buf_logprobs[a]).to(device)
            b_adv  = torch.FloatTensor(adv).to(device)
            b_ret  = torch.FloatTensor(ret).to(device)

            # Normalise advantages (per-agent, per-update)
            b_adv = (b_adv - b_adv.mean()) / (b_adv.std() + 1e-8)

            metrics_log[a] = ppo_update(
                policies[a], optimizers[a],
                b_obs, b_act, b_lp, b_adv, b_ret,
            )

        # ---- Logging ---------------------------------------------------
        if update % LOG_INTERVAL == 0:
            elapsed = time.time() - t_start
            sps     = global_step / elapsed
            cap_rate = np.mean(recent_captures[-200:]) * 100 if recent_captures else 0.0

            print(
                f"Update {update:>5}/{num_updates}  |"
                f"  step {global_step:>9,}  |"
                f"  {sps:>6.0f} sps  |"
                f"  episodes {ep_count:>6,}  |"
                f"  capture% {cap_rate:>5.1f}"
            )
            for a in agents:
                avg_ret = (
                    np.mean(recent_returns[a][-200:])
                    if recent_returns[a] else float("nan")
                )
                m = metrics_log.get(a, {})
                print(
                    f"  [{a:>6}]  ret {avg_ret:>8.2f}"
                    f"  pg {m.get('pg_loss',0):.4f}"
                    f"  vf {m.get('vf_loss',0):.4f}"
                    f"  ent {m.get('entropy',0):.4f}"
                    f"  kl {m.get('approx_kl',0):.4f}"
                )

        # ---- Checkpoint ------------------------------------------------
        if update % SAVE_INTERVAL == 0:
            for a in agents:
                ckpt = os.path.join(SAVE_DIR, f"{a}_policy_step{global_step}.pt")
                torch.save(policies[a].state_dict(), ckpt)
            print(f"  [checkpoint] saved at step {global_step}")

    # ---- Save final models ---------------------------------------------
    print("\n--- Saving final models ---")
    for a in agents:
        path = os.path.join(SAVE_DIR, f"{a}_policy.pt")
        torch.save(policies[a].state_dict(), path)
        print(f"  {a} → {path}")

    env.close()
    elapsed = time.time() - t_start
    print(f"\nTraining finished in {elapsed/60:.1f} min.")
    print("Run evaluate.py to watch the trained agents!")


if __name__ == "__main__":
    main()
