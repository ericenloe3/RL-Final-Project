"""
PPO training for hunter-prey (v2 — with obstacles).

Changes from v1:
  - Obstacle curriculum: 0 → few → more obstacles over the course of training.
    Phase 1 teaches basic pursuit/evasion (identical to v1 policy).
    Phases 2–3 add obstacles gradually so the policy adapts without
    catastrophic forgetting.
  - Obs size grew from 7 → 33 (LOS + 5 nearest obstacles).  Hidden size
    bumped 64 → 128 to accommodate the larger input.

Saving: final models (co-adapted to each other's current skill).
Periodic checkpoints saved for inspection.

Usage:  python train.py
"""

import os, time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
from env import HunterPreyEnv

# ---------- hyperparameters ----------
TOTAL_TIMESTEPS = 5_000_000
NUM_ENVS        = 16
NUM_STEPS       = 256
LEARNING_RATE   = 3e-4
GAMMA           = 0.99
GAE_LAMBDA      = 0.95
CLIP_EPS        = 0.2
ENT_COEF        = 0.01
VF_COEF         = 0.5
MAX_GRAD_NORM   = 0.5
UPDATE_EPOCHS   = 4
MINIBATCH_SIZE  = 256
HIDDEN          = 128
SAVE_DIR        = "models"
LOG_EVERY       = 10
SAVE_EVERY      = 50

# Obstacle curriculum: (start_fraction_of_training, (min_obs, max_obs))
# Phase 1: no obstacles (master pursuit/evasion)
# Phase 2: 1-3 obstacles (introduce cover)
# Phase 3: 3-6 obstacles (full difficulty)
CURRICULUM = [
    (0.00, (0, 0)),
    (0.30, (1, 3)),
    (0.60, (3, 6)),
]

# =====================================================================
class ActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=128):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden),  nn.Tanh(),
        )
        self.actor  = nn.Sequential(nn.Linear(hidden, act_dim), nn.Tanh())
        self.critic = nn.Linear(hidden, 1)
        self.log_std = nn.Parameter(torch.zeros(act_dim))
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0.0)
        nn.init.orthogonal_(self.actor[0].weight, gain=0.01)
        nn.init.orthogonal_(self.critic.weight, gain=1.0)

    def forward(self, obs):
        h = self.trunk(obs)
        return self.actor(h), self.critic(h)

    @torch.no_grad()
    def act(self, obs, deterministic=False):
        mean, val = self(obs)
        if deterministic:
            return mean.cpu().numpy(), np.zeros(obs.shape[0]), val.squeeze(-1).cpu().numpy()
        std  = self.log_std.exp().expand_as(mean)
        dist = Normal(mean, std)
        a    = dist.sample()
        lp   = dist.log_prob(a).sum(-1)
        return a.cpu().numpy(), lp.cpu().numpy(), val.squeeze(-1).cpu().numpy()

    def evaluate(self, obs, actions):
        mean, val = self(obs)
        std  = self.log_std.exp().expand_as(mean)
        dist = Normal(mean, std)
        lp   = dist.log_prob(actions).sum(-1)
        ent  = dist.entropy().sum(-1)
        return lp, ent, val.squeeze(-1)

# =====================================================================
class RunningNorm:
    def __init__(self, shape):
        self.mean = np.zeros(shape, np.float64)
        self.var  = np.ones(shape, np.float64)
        self.count = 1e-4

    def update(self, batch):
        if batch.ndim == 1: batch = batch[np.newaxis]
        n   = batch.shape[0]
        bm  = batch.mean(0); bv = batch.var(0)
        tot = self.count + n; d = bm - self.mean
        self.mean  = self.mean + d * n / tot
        self.var   = (self.var * self.count + bv * n + d**2 * self.count * n / tot) / tot
        self.count = tot

    def normalize(self, x):
        return np.clip((x - self.mean) / (np.sqrt(self.var) + 1e-8), -10, 10).astype(np.float32)

    def state_dict(self):
        return {"mean": self.mean.copy(), "var": self.var.copy(), "count": self.count}

    def load_state_dict(self, d):
        self.mean = d["mean"].copy(); self.var = d["var"].copy(); self.count = d["count"]

# =====================================================================
def compute_gae(rewards, values, dones, last_val, gamma=GAMMA, lam=GAE_LAMBDA):
    T = len(rewards); adv = np.zeros(T, np.float32); g = 0.0
    for t in reversed(range(T)):
        nv = last_val if t == T-1 else values[t+1]
        delta = rewards[t] + gamma * nv * (1-dones[t]) - values[t]
        g = delta + gamma * lam * (1-dones[t]) * g
        adv[t] = g
    return adv, adv + values

def ppo_update(policy, optimizer, obs, acts, old_lp, advs, rets):
    idx = np.arange(len(obs)); n_mb = 0
    metrics = {"pg": 0.0, "vf": 0.0, "ent": 0.0}
    for _ in range(UPDATE_EPOCHS):
        np.random.shuffle(idx)
        for s in range(0, len(obs), MINIBATCH_SIZE):
            mb = idx[s:s+MINIBATCH_SIZE]
            lp, ent, val = policy.evaluate(obs[mb], acts[mb])
            ratio = (lp - old_lp[mb]).exp()
            a = advs[mb]
            pg = torch.max(-a*ratio, -a*ratio.clamp(1-CLIP_EPS, 1+CLIP_EPS)).mean()
            vf = 0.5 * ((val - rets[mb])**2).mean()
            loss = pg + VF_COEF * vf - ENT_COEF * ent.mean()
            optimizer.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), MAX_GRAD_NORM)
            optimizer.step()
            metrics["pg"] += pg.item(); metrics["vf"] += vf.item()
            metrics["ent"] += ent.mean().item(); n_mb += 1
    return {k: v/max(n_mb,1) for k,v in metrics.items()}

# =====================================================================
def current_obstacle_range(global_step: int):
    """Look up the obstacle range for the current training step."""
    frac = global_step / TOTAL_TIMESTEPS
    rng = CURRICULUM[0][1]
    for threshold, r in CURRICULUM:
        if frac >= threshold:
            rng = r
    return rng

# =====================================================================
def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    envs    = [HunterPreyEnv(n_obstacles_range=(0, 0)) for _ in range(NUM_ENVS)]
    env_obs = [e.reset(seed=42+i) for i, e in enumerate(envs)]
    agents  = ["hunter", "prey"]
    obs_dim = envs[0].obs_size
    act_dim = envs[0].action_size

    policies    = {a: ActorCritic(obs_dim, act_dim, HIDDEN).to(device) for a in agents}
    optimizers  = {a: optim.Adam(policies[a].parameters(), lr=LEARNING_RATE, eps=1e-5) for a in agents}
    normalizers = {a: RunningNorm(shape=(obs_dim,)) for a in agents}

    print(f"Device: {device}  Envs: {NUM_ENVS}  Batch: {NUM_ENVS*NUM_STEPS}")
    print(f"Obs: {obs_dim}  Act: {act_dim}  Hidden: {HIDDEN}  "
          f"Params: {sum(p.numel() for p in policies['hunter'].parameters()):,}")
    print(f"Curriculum: {CURRICULUM}")

    B, T = NUM_ENVS, NUM_STEPS
    buf_obs  = {a: np.zeros((B,T,obs_dim), np.float32) for a in agents}
    buf_act  = {a: np.zeros((B,T,act_dim), np.float32) for a in agents}
    buf_lp   = {a: np.zeros((B,T), np.float32) for a in agents}
    buf_val  = {a: np.zeros((B,T), np.float32) for a in agents}
    buf_rew  = {a: np.zeros((B,T), np.float32) for a in agents}
    buf_done =     np.zeros((B,T), np.float32)

    num_updates = TOTAL_TIMESTEPS // (NUM_ENVS * NUM_STEPS)
    global_step = 0; ep_count = 0
    recent_caps = []
    recent_rets = {a: [] for a in agents}
    ep_rets = [{a: 0.0 for a in agents} for _ in range(NUM_ENVS)]
    prev_phase = None
    t0 = time.time()

    print(f"\nTraining {TOTAL_TIMESTEPS:,} steps · {num_updates} updates\n")

    for update in range(1, num_updates + 1):
        # ---- curriculum: update obstacle range if phase changed ----
        obs_range = current_obstacle_range(global_step)
        if obs_range != prev_phase:
            for e in envs:
                e.set_obstacle_range(*obs_range)
            print(f"  [curriculum] step {global_step:,}  obstacles {obs_range}")
            prev_phase = obs_range

        # ---- collect rollout ----
        for t in range(T):
            global_step += B
            normed = [{a: normalizers[a].normalize(env_obs[ei][a]) for a in agents}
                      for ei in range(B)]

            with torch.no_grad():
                acts_np = {}; lps_np = {}; vals_np = {}
                for a in agents:
                    obs_t = torch.FloatTensor(np.array([normed[ei][a] for ei in range(B)])).to(device)
                    act, lp, val = policies[a].act(obs_t)
                    acts_np[a] = act; lps_np[a] = lp; vals_np[a] = val

            for ei in range(B):
                for a in agents:
                    buf_obs[a][ei,t] = normed[ei][a]
                    buf_act[a][ei,t] = acts_np[a][ei]
                    buf_lp[a][ei,t]  = lps_np[a][ei]
                    buf_val[a][ei,t] = vals_np[a][ei]

                obs_new, rew, done, info = envs[ei].step(acts_np["hunter"][ei], acts_np["prey"][ei])
                buf_done[ei,t] = float(done)
                for a in agents:
                    buf_rew[a][ei,t] = rew[a]
                    ep_rets[ei][a]  += rew[a]

                if done:
                    ep_count += 1
                    recent_caps.append(int(info["captured"]))
                    for a in agents:
                        recent_rets[a].append(ep_rets[ei][a])
                        ep_rets[ei][a] = 0.0
                    env_obs[ei] = envs[ei].reset()
                else:
                    env_obs[ei] = obs_new

        # ---- update normaliser ----
        for a in agents:
            normalizers[a].update(buf_obs[a].reshape(-1, obs_dim))

        # ---- PPO updates ----
        logs = {}
        for a in agents:
            with torch.no_grad():
                lo = torch.FloatTensor(np.array([normalizers[a].normalize(env_obs[ei][a]) for ei in range(B)])).to(device)
                _, lv = policies[a](lo); lv = lv.squeeze(-1).cpu().numpy()

            adv_buf = np.zeros((B,T), np.float32); ret_buf = np.zeros((B,T), np.float32)
            for ei in range(B):
                adv_buf[ei], ret_buf[ei] = compute_gae(buf_rew[a][ei], buf_val[a][ei], buf_done[ei], lv[ei])

            def flat(x):
                return torch.FloatTensor(x.reshape(-1) if x.ndim <= 2 else x.reshape(-1, x.shape[-1])).to(device)
            b_obs = flat(buf_obs[a]); b_act = flat(buf_act[a]); b_lp = flat(buf_lp[a])
            b_adv = flat(adv_buf); b_ret = flat(ret_buf)
            b_adv = (b_adv - b_adv.mean()) / (b_adv.std() + 1e-8)
            logs[a] = ppo_update(policies[a], optimizers[a], b_obs, b_act, b_lp, b_adv, b_ret)

        # ---- logging ----
        if update % LOG_EVERY == 0:
            el = time.time() - t0; sps = global_step / el
            cap = 100 * np.mean(recent_caps[-500:]) if recent_caps else 0.0
            print(f"upd {update:>5}/{num_updates}  step {global_step:>9,}  {sps:>5.0f} sps  "
                  f"eps {ep_count:>6,}  cap {cap:>5.1f}%  obs {obs_range}")
            for a in agents:
                r = np.mean(recent_rets[a][-500:]) if recent_rets[a] else float("nan")
                m = logs.get(a, {})
                print(f"  {a:>6}: ret {r:>7.1f}  pg {m.get('pg',0):.4f}  "
                      f"vf {m.get('vf',0):.4f}  ent {m.get('ent',0):.3f}")

        if update % SAVE_EVERY == 0:
            for a in agents:
                torch.save({"policy": policies[a].state_dict(),
                            "normalizer": normalizers[a].state_dict()},
                           os.path.join(SAVE_DIR, f"{a}_step{global_step}.pt"))
            print(f"  [ckpt] step {global_step:,}")

    # ---- final save ----
    print("\n--- Saving final models ---")
    for a in agents:
        path = os.path.join(SAVE_DIR, f"{a}_final.pt")
        torch.save({"policy": policies[a].state_dict(),
                     "normalizer": normalizers[a].state_dict()}, path)
        print(f"  {a} → {path}")

    cap = 100 * np.mean(recent_caps[-1000:]) if recent_caps else 0.0
    print(f"\nDone in {(time.time()-t0)/60:.1f} min.  Final capture rate: {cap:.1f}%")
    print("Run:  python evaluate.py")

if __name__ == "__main__":
    main()
