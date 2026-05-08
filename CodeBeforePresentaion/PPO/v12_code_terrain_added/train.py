"""
PPO training for hunter-prey — obstacles + terrain (v5).

Terrain curriculum additions:
  Phase 4 (60% through): mud zones introduced.  Mud is learned first because
    its effect (speed reduction) is continuous and produces smooth gradients.
    Agents can feel mud via reduced heading reward before they see it.
  Phase 5 (80% through): ice zones added.  Ice is harder: the policy must
    learn to predict consequences BEFORE entering (the entering action is what
    gets frozen), so it needs a solid value estimate of "being on ice is bad
    when pursuing / good when escaping at the right angle."

Why introduce terrain late (not from step 0):
  - Agents need solid obstacle navigation before terrain.  A hunter that
    can't route around walls will have no mental bandwidth to reason about
    ice avoidance simultaneously.
  - Mud/ice add 22 new obs dimensions.  The running normaliser needs time
    to collect statistics on these before the gradient signal is reliable.
  - Gradual difficulty keeps PPO clip ratio stable.

Training adjustments for terrain:
  - Prey entropy end coefficient raised slightly (0.01 → 0.015): the prey
    needs more exploration to discover ice-baiting strategies.
  - TOTAL_TIMESTEPS raised to 15M to give adequate terrain exposure.
    (Original 10M kept the terrain phases too short for solid convergence.)
  - HUNTER_FREEZE_FRAC unchanged at 20%: prey still needs a head start.

Usage:  python train.py
"""

import os, time, json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
from env import HunterPreyEnv

# ---------- hyperparameters ----------
TOTAL_TIMESTEPS = 15_000_000   # raised from 10M for terrain exposure
NUM_ENVS        = 16
NUM_STEPS       = 256
LEARNING_RATE   = 3e-4
GAMMA           = 0.99
GAE_LAMBDA      = 0.95
CLIP_EPS        = 0.2

ENT_COEF = {
    "hunter": {"start": 0.02, "end": 0.003},
    "prey":   {"start": 0.05, "end": 0.015},  # end raised: prey needs ice strategy
}
VF_COEF       = 0.5
MAX_GRAD_NORM = 0.5
MINIBATCH_SIZE = 256
SAVE_DIR       = "models"
LOG_EVERY      = 10
SAVE_EVERY     = 50

PPO_EPOCHS = {"hunter": 3, "prey": 8}
HIDDEN     = {"hunter": 128, "prey": 192}

HUNTER_FREEZE_FRAC = 0.20
PREY_SPEED         = 4.0

# Combined curriculum:
#   (frac_start, obstacle_range, hunter_speed, mud_range, ice_range)
#
# Terrain is introduced AFTER obstacle mastery to keep credit assignment clean.
# Ice comes after mud because ice requires planning (choosing entry angle),
# whereas mud just needs reactive speed-compensation.
CURRICULUM = [
    (0.00, (0, 0), 4.0, (0, 0), (0, 0)),   # open field
    (0.15, (1, 3), 4.2, (0, 0), (0, 0)),   # sparse obstacles
    (0.40, (3, 6), 4.3, (0, 0), (0, 0)),   # dense obstacles
    (0.55, (3, 6), 4.5, (0, 0), (0, 0)),   # full speed, no terrain
    (0.60, (3, 6), 4.5, (1, 2), (0, 0)),   # + mud zones
    (0.80, (3, 6), 4.5, (1, 2), (1, 2)),   # + ice zones
]


# =====================================================================
class ActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=128):
        super().__init__()
        self.trunk  = nn.Sequential(
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
        nn.init.orthogonal_(self.critic.weight,   gain=1.0)

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
        self.mean  = np.zeros(shape, np.float64)
        self.var   = np.ones(shape,  np.float64)
        self.count = 1e-4

    def update(self, batch):
        if batch.ndim == 1: batch = batch[np.newaxis]
        n   = batch.shape[0]; bm = batch.mean(0); bv = batch.var(0)
        tot = self.count + n;  d  = bm - self.mean
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
        nv    = last_val if t == T - 1 else values[t + 1]
        delta = rewards[t] + gamma * nv * (1 - dones[t]) - values[t]
        g     = delta + gamma * lam * (1 - dones[t]) * g
        adv[t] = g
    return adv, adv + values


def ppo_update(policy, optimizer, obs, acts, old_lp, advs, rets, ent_coef, n_epochs=4):
    idx = np.arange(len(obs)); n_mb = 0
    metrics = {"pg": 0.0, "vf": 0.0, "ent": 0.0}
    for _ in range(n_epochs):
        np.random.shuffle(idx)
        for s in range(0, len(obs), MINIBATCH_SIZE):
            mb = idx[s:s + MINIBATCH_SIZE]
            lp, ent, val = policy.evaluate(obs[mb], acts[mb])
            ratio = (lp - old_lp[mb]).exp()
            a  = advs[mb]
            pg = torch.max(-a * ratio, -a * ratio.clamp(1 - CLIP_EPS, 1 + CLIP_EPS)).mean()
            vf = 0.5 * ((val - rets[mb])**2).mean()
            loss = pg + VF_COEF * vf - ent_coef * ent.mean()
            optimizer.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), MAX_GRAD_NORM)
            optimizer.step()
            metrics["pg"] += pg.item(); metrics["vf"] += vf.item()
            metrics["ent"] += ent.mean().item(); n_mb += 1
    return {k: v / max(n_mb, 1) for k, v in metrics.items()}


def anneal(start, end, frac):
    return start + (end - start) * min(1.0, max(0.0, frac))


def current_phase(global_step):
    """Return (obstacle_range, hunter_speed, mud_range, ice_range) for current step."""
    frac      = global_step / TOTAL_TIMESTEPS
    obs_range = CURRICULUM[0][1]
    h_speed   = CURRICULUM[0][2]
    mud_range = CURRICULUM[0][3]
    ice_range = CURRICULUM[0][4]
    for threshold, r, s, m, i in CURRICULUM:
        if frac >= threshold:
            obs_range = r; h_speed = s; mud_range = m; ice_range = i
    return obs_range, h_speed, mud_range, ice_range


# =====================================================================
def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    envs    = [HunterPreyEnv(hunter_speed=4.0, prey_speed=4.0,
                             n_obstacles_range=(0, 0)) for _ in range(NUM_ENVS)]
    env_obs = [e.reset(seed=42 + i) for i, e in enumerate(envs)]
    agents  = ["hunter", "prey"]
    obs_dim = envs[0].obs_size
    act_dim = envs[0].action_size

    policies    = {a: ActorCritic(obs_dim, act_dim, HIDDEN[a]).to(device) for a in agents}
    optimizers  = {a: optim.Adam(policies[a].parameters(), lr=LEARNING_RATE, eps=1e-5)
                   for a in agents}
    normalizers = {a: RunningNorm(shape=(obs_dim,)) for a in agents}

    hunter_freeze_step = int(TOTAL_TIMESTEPS * HUNTER_FREEZE_FRAC)

    print(f"Device: {device}  Envs: {NUM_ENVS}  Batch: {NUM_ENVS*NUM_STEPS}")
    print(f"Obs: {obs_dim}  Act: {act_dim}")
    for a in agents:
        n_params = sum(p.numel() for p in policies[a].parameters())
        print(f"  {a}: hidden={HIDDEN[a]}  params={n_params:,}  ppo_epochs={PPO_EPOCHS[a]}")
    print(f"Hunter frozen first {HUNTER_FREEZE_FRAC:.0%} ({hunter_freeze_step:,} steps)")
    print(f"Curriculum (obs, h_spd, mud, ice): {CURRICULUM}")

    B, T = NUM_ENVS, NUM_STEPS
    buf_obs  = {a: np.zeros((B, T, obs_dim), np.float32) for a in agents}
    buf_act  = {a: np.zeros((B, T, act_dim), np.float32) for a in agents}
    buf_lp   = {a: np.zeros((B, T), np.float32) for a in agents}
    buf_val  = {a: np.zeros((B, T), np.float32) for a in agents}
    buf_rew  = {a: np.zeros((B, T), np.float32) for a in agents}
    buf_done =     np.zeros((B, T), np.float32)

    num_updates = TOTAL_TIMESTEPS // (NUM_ENVS * NUM_STEPS)
    global_step = 0; ep_count = 0
    recent_caps = []; recent_rets = {a: [] for a in agents}; recent_ep_steps = []
    ep_rets   = [{a: 0.0 for a in agents} for _ in range(NUM_ENVS)]
    prev_phase = None
    t0 = time.time(); train_log = []

    print(f"\nTraining {TOTAL_TIMESTEPS:,} steps · {num_updates} updates\n")

    for update in range(1, num_updates + 1):
        # ---- curriculum ----
        obs_range, h_speed, mud_range, ice_range = current_phase(global_step)
        phase_key = (obs_range, h_speed, mud_range, ice_range)
        if phase_key != prev_phase:
            for e in envs:
                e.set_obstacle_range(*obs_range)
                e.set_speeds(h_speed, PREY_SPEED)
                e.set_mud_range(*mud_range)
                e.set_ice_range(*ice_range)
            print(f"  [curriculum] step {global_step:,}  obs {obs_range}  "
                  f"h_spd {h_speed}  mud {mud_range}  ice {ice_range}")
            prev_phase = phase_key

        # ---- collect rollout ----
        for t in range(T):
            global_step += B
            normed = [{a: normalizers[a].normalize(env_obs[ei][a]) for a in agents}
                      for ei in range(B)]

            with torch.no_grad():
                acts_np = {}; lps_np = {}; vals_np = {}
                for a in agents:
                    obs_t = torch.FloatTensor(
                        np.array([normed[ei][a] for ei in range(B)])).to(device)
                    act, lp, val = policies[a].act(obs_t)
                    acts_np[a] = act; lps_np[a] = lp; vals_np[a] = val

            for ei in range(B):
                for a in agents:
                    buf_obs[a][ei, t] = normed[ei][a]
                    buf_act[a][ei, t] = acts_np[a][ei]
                    buf_lp[a][ei, t]  = lps_np[a][ei]
                    buf_val[a][ei, t] = vals_np[a][ei]

                obs_new, rew, done, info = envs[ei].step(
                    acts_np["hunter"][ei], acts_np["prey"][ei])
                buf_done[ei, t] = float(done)
                for a in agents:
                    buf_rew[a][ei, t] = rew[a]
                    ep_rets[ei][a]   += rew[a]

                if done:
                    ep_count += 1
                    recent_caps.append(int(info["captured"]))
                    recent_ep_steps.append(info["steps"])
                    for a in agents:
                        recent_rets[a].append(ep_rets[ei][a])
                        ep_rets[ei][a] = 0.0
                    env_obs[ei] = envs[ei].reset()
                else:
                    env_obs[ei] = obs_new

        # ---- update normaliser ----
        for a in agents:
            normalizers[a].update(buf_obs[a].reshape(-1, obs_dim))

        # ---- PPO update ----
        frac         = global_step / TOTAL_TIMESTEPS
        ent_coefs    = {a: anneal(ENT_COEF[a]["start"], ENT_COEF[a]["end"], frac)
                        for a in agents}
        hunter_frozen = global_step < hunter_freeze_step

        logs = {}
        for a in agents:
            if a == "hunter" and hunter_frozen:
                logs[a] = {"pg": 0, "vf": 0, "ent": 0}
                continue

            with torch.no_grad():
                lo      = torch.FloatTensor(np.array(
                    [normalizers[a].normalize(env_obs[ei][a]) for ei in range(B)])).to(device)
                _, lv   = policies[a](lo); lv = lv.squeeze(-1).cpu().numpy()

            adv_buf = np.zeros((B, T), np.float32); ret_buf = np.zeros((B, T), np.float32)
            for ei in range(B):
                adv_buf[ei], ret_buf[ei] = compute_gae(
                    buf_rew[a][ei], buf_val[a][ei], buf_done[ei], lv[ei])

            def flat(x):
                return torch.FloatTensor(
                    x.reshape(-1) if x.ndim <= 2 else x.reshape(-1, x.shape[-1])).to(device)

            b_obs = flat(buf_obs[a]); b_act = flat(buf_act[a]); b_lp  = flat(buf_lp[a])
            b_adv = flat(adv_buf);    b_ret = flat(ret_buf)
            b_adv = (b_adv - b_adv.mean()) / (b_adv.std() + 1e-8)

            logs[a] = ppo_update(policies[a], optimizers[a], b_obs, b_act, b_lp,
                                 b_adv, b_ret, ent_coefs[a], n_epochs=PPO_EPOCHS[a])

        # ---- logging ----
        if update % LOG_EVERY == 0:
            el        = time.time() - t0; sps = global_step / el
            cap       = 100 * np.mean(recent_caps[-500:]) if recent_caps else 0.0
            avg_steps = np.mean(recent_ep_steps[-500:]) if recent_ep_steps else 0
            freeze_tag = " [H_FROZEN]" if hunter_frozen else ""
            mud_active = mud_range[1] > 0; ice_active = ice_range[1] > 0
            terrain_tag = (f"  mud{mud_range}" if mud_active else "") + \
                          (f"  ice{ice_range}" if ice_active else "")

            print(f"upd {update:>5}/{num_updates}  step {global_step:>9,}  {sps:>5.0f} sps  "
                  f"eps {ep_count:>6,}  cap {cap:>5.1f}%  obs {obs_range}  "
                  f"h_spd {h_speed:.1f}{terrain_tag}{freeze_tag}")

            entry = {
                "update": update, "global_step": global_step, "episodes": ep_count,
                "capture_rate": round(cap / 100, 4),
                "avg_ep_steps": round(float(avg_steps), 1),
                "obstacle_range": list(obs_range), "hunter_speed": h_speed,
                "mud_range": list(mud_range), "ice_range": list(ice_range),
                "hunter_frozen": hunter_frozen,
                "ent_coef_hunter": round(ent_coefs["hunter"], 5),
                "ent_coef_prey":   round(ent_coefs["prey"],   5),
                "elapsed_sec": round(el, 1), "sps": round(sps, 0),
            }
            for a in agents:
                r = np.mean(recent_rets[a][-500:]) if recent_rets[a] else float("nan")
                m = logs.get(a, {})
                entry[f"{a}_avg_return"] = round(float(r), 2) if not np.isnan(r) else None
                entry[f"{a}_pg_loss"]    = round(m.get("pg",  0), 5)
                entry[f"{a}_vf_loss"]    = round(m.get("vf",  0), 5)
                entry[f"{a}_entropy"]    = round(m.get("ent", 0), 4)
                print(f"  {a:>6}: ret {r:>7.1f}  pg {m.get('pg',0):.4f}  "
                      f"vf {m.get('vf',0):.4f}  ent {m.get('ent',0):.3f}")
            train_log.append(entry)

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

    log_path = os.path.join(SAVE_DIR, "train_log.json")
    with open(log_path, "w") as f:
        json.dump(train_log, f, indent=2)
    print(f"  dynamics → {log_path} ({len(train_log)} entries)")

    cap = 100 * np.mean(recent_caps[-1000:]) if recent_caps else 0.0
    print(f"\nDone in {(time.time()-t0)/60:.1f} min.  Final capture rate: {cap:.1f}%")
    print("Run:  python evaluate.py")


if __name__ == "__main__":
    main()
