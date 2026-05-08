"""
PPO training for hunter-prey (v7 — entropy fix).

The diagnostic from comparison runs showed prey entropy growing linearly to
54 nats (log_std ~26, std ~10^11) by the end of 10M steps, meaning the prey
was sampling near-uniform random actions for ~80% of training.  The
deterministic mean policy still benefited from advantage signals so PPO
overall still won the comparison, but the prey was effectively untrained.

v7 changes (target the entropy explosion only — keep everything else identical):
  1. ActorCritic.log_std is clamped to [-3, 0.5] in a property.  All call
     sites (act, evaluate) use the clamped value, so the upper bound on
     std is exp(0.5) ~ 1.65 — enough exploration without ever reaching
     the noise regime.  The Parameter itself can still take any value,
     but its effect on actions and entropy is bounded.
  2. ENT_COEF["prey"]["end"] reduced 0.01 -> 0.005.  With the clamp in
     place, the prey can be allowed to anneal to a tighter policy without
     losing the exploration that the higher start coefficient provides.
  3. Training log now records log_std (raw + clamped) per agent every
     LOG_EVERY updates so future runs can be monitored for collapse or
     persistent saturation against the upper bound.

The clamp value of 0.5 was chosen because:
  - exp(0.5) ~ 1.65, which is large relative to action range [-1, 1] but
    not absurdly so — enough randomness for genuine continuing exploration.
  - It is well above any std the hunter ever reaches in the working PPO
    runs (hunter log_std stays in [-1, 0]), so the clamp is invisible to
    the hunter and only constrains the prey's pathological growth.

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
TOTAL_TIMESTEPS = 10_000_000
NUM_ENVS        = 16
NUM_STEPS       = 256
LEARNING_RATE   = 3e-4
GAMMA           = 0.99
GAE_LAMBDA      = 0.95
CLIP_EPS        = 0.2
# Per-agent entropy
# v7: prey end coefficient reduced 0.01 -> 0.005.  With the log_std clamp
# in place (see ActorCritic), the prey can anneal to a tighter policy
# without losing useful exploration.  The high start (0.05) still gives
# the prey strong early exploration to discover cover strategies.
ENT_COEF = {
    "hunter": {"start": 0.02, "end": 0.003},
    "prey":   {"start": 0.05, "end": 0.005},
}
VF_COEF         = 0.5
MAX_GRAD_NORM   = 0.5
MINIBATCH_SIZE  = 256
SAVE_DIR        = "models"
LOG_EVERY       = 10
SAVE_EVERY      = 50

# --- Asymmetric training (the key fix) ---
# The prey's evasion task is ~5× harder than the hunter's pursuit task.
# Give the prey more gradient steps so it can develop complex strategies
# before the hunter fully converges.
PPO_EPOCHS = {"hunter": 3, "prey": 8}  # prey gets 2.7× more gradient steps
HIDDEN     = {"hunter": 128, "prey": 192}  # prey gets 50% more capacity

# First HUNTER_FREEZE_FRAC of training: hunter policy is FROZEN.
# Prey trains alone against the random (untrained) hunter.
# This guarantees the prey learns basic evasion before facing a skilled hunter.
HUNTER_FREEZE_FRAC = 0.20  # freeze hunter for first 20% of training

# Combined curriculum
PREY_SPEED = 4.0
CURRICULUM = [
    (0.00, (0, 0), 4.0),
    (0.15, (1, 3), 4.2),
    (0.40, (3, 6), 4.3),
    (0.70, (3, 6), 4.5),
]

# =====================================================================
class ActorCritic(nn.Module):
    """PPO actor-critic with CLAMPED log_std.

    v7 fix: log_std is a learnable Parameter but every read of it for
    action sampling, entropy, or log_prob goes through `_clamped_log_std`,
    which applies torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX).

    The bound LOG_STD_MAX = 0.5 caps action std at exp(0.5) ~ 1.65, well
    above any value a converging policy needs.  Without this bound, prey
    log_std grew to ~26 (std ~10^11) over 10M steps, sampling pure noise.
    """

    LOG_STD_MIN = -3.0   # std lower bound exp(-3) ~ 0.05
    LOG_STD_MAX =  0.5   # std upper bound exp(0.5) ~ 1.65

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

    @property
    def _clamped_log_std(self):
        """Bounded log_std used for all action and entropy computations.

        Note: torch.clamp's gradient is zero in the saturation region, so
        once log_std hits a bound it stops growing in that direction —
        which is exactly what we want.  The Parameter remains learnable
        in the interior of the range.
        """
        return self.log_std.clamp(self.LOG_STD_MIN, self.LOG_STD_MAX)

    def forward(self, obs):
        h = self.trunk(obs)
        return self.actor(h), self.critic(h)

    @torch.no_grad()
    def act(self, obs, deterministic=False):
        mean, val = self(obs)
        if deterministic:
            return mean.cpu().numpy(), np.zeros(obs.shape[0]), val.squeeze(-1).cpu().numpy()
        std  = self._clamped_log_std.exp().expand_as(mean)
        dist = Normal(mean, std)
        a    = dist.sample()
        lp   = dist.log_prob(a).sum(-1)
        return a.cpu().numpy(), lp.cpu().numpy(), val.squeeze(-1).cpu().numpy()

    def evaluate(self, obs, actions):
        mean, val = self(obs)
        std  = self._clamped_log_std.exp().expand_as(mean)
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

def ppo_update(policy, optimizer, obs, acts, old_lp, advs, rets, ent_coef, n_epochs=4):
    idx = np.arange(len(obs)); n_mb = 0
    metrics = {"pg": 0.0, "vf": 0.0, "ent": 0.0}
    for _ in range(n_epochs):
        np.random.shuffle(idx)
        for s in range(0, len(obs), MINIBATCH_SIZE):
            mb = idx[s:s+MINIBATCH_SIZE]
            lp, ent, val = policy.evaluate(obs[mb], acts[mb])
            ratio = (lp - old_lp[mb]).exp()
            a = advs[mb]
            pg = torch.max(-a*ratio, -a*ratio.clamp(1-CLIP_EPS, 1+CLIP_EPS)).mean()
            vf = 0.5 * ((val - rets[mb])**2).mean()
            loss = pg + VF_COEF * vf - ent_coef * ent.mean()
            optimizer.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), MAX_GRAD_NORM)
            optimizer.step()
            metrics["pg"] += pg.item(); metrics["vf"] += vf.item()
            metrics["ent"] += ent.mean().item(); n_mb += 1
    out = {k: v/max(n_mb,1) for k,v in metrics.items()}
    # v7: snapshot raw and clamped log_std so saturation can be diagnosed
    with torch.no_grad():
        out["log_std_raw"]     = policy.log_std.mean().item()
        out["log_std_clamped"] = policy._clamped_log_std.mean().item()
    return out

def anneal(start, end, frac):
    """Linear interpolation from start to end, frac in [0, 1]."""
    return start + (end - start) * min(1.0, max(0.0, frac))

# =====================================================================
def current_phase(global_step: int):
    """Look up the (obstacle_range, hunter_speed) for the current step."""
    frac = global_step / TOTAL_TIMESTEPS
    obs_range = CURRICULUM[0][1]
    h_speed   = CURRICULUM[0][2]
    for threshold, r, s in CURRICULUM:
        if frac >= threshold:
            obs_range = r
            h_speed   = s
    return obs_range, h_speed

# =====================================================================
def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Start with equal speeds (curriculum will ramp hunter up)
    envs    = [HunterPreyEnv(hunter_speed=4.0, prey_speed=4.0,
                             n_obstacles_range=(0, 0)) for _ in range(NUM_ENVS)]
    env_obs = [e.reset(seed=42+i) for i, e in enumerate(envs)]
    agents  = ["hunter", "prey"]
    obs_dim = envs[0].obs_size
    act_dim = envs[0].action_size

    policies    = {a: ActorCritic(obs_dim, act_dim, HIDDEN[a]).to(device) for a in agents}
    optimizers  = {a: optim.Adam(policies[a].parameters(), lr=LEARNING_RATE, eps=1e-5) for a in agents}
    normalizers = {a: RunningNorm(shape=(obs_dim,)) for a in agents}

    hunter_freeze_step = int(TOTAL_TIMESTEPS * HUNTER_FREEZE_FRAC)

    print(f"Device: {device}  Envs: {NUM_ENVS}  Batch: {NUM_ENVS*NUM_STEPS}")
    print(f"Obs: {obs_dim}  Act: {act_dim}")
    for a in agents:
        n_params = sum(p.numel() for p in policies[a].parameters())
        print(f"  {a}: hidden={HIDDEN[a]}  params={n_params:,}  ppo_epochs={PPO_EPOCHS[a]}")
    print(f"Hunter frozen for first {HUNTER_FREEZE_FRAC:.0%} ({hunter_freeze_step:,} steps)")
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
    recent_ep_steps = []
    ep_rets = [{a: 0.0 for a in agents} for _ in range(NUM_ENVS)]
    prev_phase = None
    t0 = time.time()

    train_log = []

    print(f"\nTraining {TOTAL_TIMESTEPS:,} steps · {num_updates} updates\n")

    for update in range(1, num_updates + 1):
        # ---- curriculum: update obstacles AND hunter speed ----
        obs_range, h_speed = current_phase(global_step)
        phase_key = (obs_range, h_speed)
        if phase_key != prev_phase:
            for e in envs:
                e.set_obstacle_range(*obs_range)
                e.set_speeds(h_speed, PREY_SPEED)
            print(f"  [curriculum] step {global_step:,}  obstacles {obs_range}  "
                  f"hunter_speed {h_speed}  prey_speed {PREY_SPEED}")
            prev_phase = phase_key

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

        # ---- PPO updates ----
        frac = global_step / TOTAL_TIMESTEPS
        ent_coefs = {a: anneal(ENT_COEF[a]["start"], ENT_COEF[a]["end"], frac) for a in agents}
        hunter_frozen = global_step < hunter_freeze_step

        logs = {}
        for a in agents:
            # Skip hunter updates during freeze phase
            if a == "hunter" and hunter_frozen:
                # v7: still snapshot log_std for the frozen hunter so the log
                # records its initial state (zero) rather than being missing
                with torch.no_grad():
                    ls_raw     = policies[a].log_std.mean().item()
                    ls_clamped = policies[a]._clamped_log_std.mean().item()
                logs[a] = {"pg": 0, "vf": 0, "ent": 0,
                           "log_std_raw": ls_raw, "log_std_clamped": ls_clamped}
                continue

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
            logs[a] = ppo_update(policies[a], optimizers[a], b_obs, b_act, b_lp, b_adv, b_ret,
                                 ent_coefs[a], n_epochs=PPO_EPOCHS[a])

        # ---- logging + dynamics tracking ----
        if update % LOG_EVERY == 0:
            el = time.time() - t0; sps = global_step / el
            cap = 100 * np.mean(recent_caps[-500:]) if recent_caps else 0.0
            avg_steps = np.mean(recent_ep_steps[-500:]) if recent_ep_steps else 0

            freeze_tag = " [H_FROZEN]" if hunter_frozen else ""
            print(f"upd {update:>5}/{num_updates}  step {global_step:>9,}  {sps:>5.0f} sps  "
                  f"eps {ep_count:>6,}  cap {cap:>5.1f}%  obs {obs_range}  "
                  f"h_spd {h_speed:.1f}{freeze_tag}")

            entry = {
                "update": update,
                "global_step": global_step,
                "episodes": ep_count,
                "capture_rate": round(cap / 100, 4),
                "avg_ep_steps": round(float(avg_steps), 1),
                "obstacle_range": list(obs_range),
                "hunter_speed": h_speed,
                "hunter_frozen": hunter_frozen,
                "ent_coef_hunter": round(ent_coefs["hunter"], 5),
                "ent_coef_prey":   round(ent_coefs["prey"], 5),
                "elapsed_sec": round(el, 1),
                "sps": round(sps, 0),
            }
            for a in agents:
                r = np.mean(recent_rets[a][-500:]) if recent_rets[a] else float("nan")
                m = logs.get(a, {})
                entry[f"{a}_avg_return"]   = round(float(r), 2) if not np.isnan(r) else None
                entry[f"{a}_pg_loss"]      = round(m.get("pg", 0), 5)
                entry[f"{a}_vf_loss"]      = round(m.get("vf", 0), 5)
                entry[f"{a}_entropy"]      = round(m.get("ent", 0), 4)
                # v7: log raw and clamped log_std so saturation is visible
                entry[f"{a}_log_std_raw"]     = round(m.get("log_std_raw",     0), 4)
                entry[f"{a}_log_std_clamped"] = round(m.get("log_std_clamped", 0), 4)
                # Visual saturation marker in console: '*' if pinned to upper bound
                ls_raw   = m.get("log_std_raw", 0.0)
                sat_tag  = " *SAT*" if ls_raw > ActorCritic.LOG_STD_MAX - 0.05 else ""
                print(f"  {a:>6}: ret {r:>7.1f}  pg {m.get('pg',0):.4f}  "
                      f"vf {m.get('vf',0):.4f}  ent {m.get('ent',0):.3f}  "
                      f"log_std {ls_raw:+.2f}{sat_tag}")
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

    # ---- save training dynamics log ----
    log_path = os.path.join(SAVE_DIR, "train_log.json")
    with open(log_path, "w") as f:
        json.dump(train_log, f, indent=2)
    print(f"  dynamics → {log_path} ({len(train_log)} entries)")

    cap = 100 * np.mean(recent_caps[-1000:]) if recent_caps else 0.0
    print(f"\nDone in {(time.time()-t0)/60:.1f} min.  Final capture rate: {cap:.1f}%")
    print("Run:  python evaluate.py")

if __name__ == "__main__":
    main()
