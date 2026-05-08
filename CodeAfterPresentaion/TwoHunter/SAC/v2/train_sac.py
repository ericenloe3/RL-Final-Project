"""
SAC training for 2-hunter / 1-prey hunter-prey (v2 — hunter rebalance).

v1 results:
  - Capture rate peaked at 100% around step 730k, sustained ~3M steps.
  - Then it CRASHED to 19% by step 10M as the prey learned faster than
    the hunters could re-adapt (off-policy SAC: replay buffer kept the
    hunters fitting a stale Q-distribution while the prey actively
    explored its way to a better policy).
  - Both αs pinned at the 0.02 floor by step 7M.  v6's α-floor
    mechanism worked (no catastrophic α=0 collapse), but at α=0.02 the
    hunters had so little exploration headroom they couldn't escape
    their now-suboptimal coordinated-orbit pattern.
  - Per-hunter capture splits stayed near 50/50 throughout — parameter
    sharing functioned correctly.

v2 changes (rebalance to keep hunters competitive late):

  1. SPEED CURRICULUM RE-INTRODUCED (hunter-favoured direction).
     v1 had constant 4.5/4.5 throughout.  v2 starts at equal speeds and
     ramps to a small hunter advantage by training's end:
       0-50%:  hunters 4.5  (equal — same as v1)
       50%+:   hunters 4.7  (hunters 0.2 faster than prey 4.5)
     The late-training speed boost gives the hunters a geometric edge
     that compensates for their alpha-floor exploration deficit during
     the period the prey was previously running away with the contest.
     Note the asymmetry vs PPO 2v1 v2 — there I made hunters SLOWER for
     the prey, here I make them FASTER for the hunter, because the
     methods failed in opposite ways at this scale.

  2. ASYMMETRIC α FLOORS.
     Hunter floor raised 0.02 → 0.05.  Prey floor kept at 0.02.
     The hunters have the harder problem (two-agent pinch geometry is
     more complex than single-agent escape), so they need more
     sustained exploration as the contest evolves.  At α=0.05 the
     entropy bonus -α·log_prob ≈ -0.05 × 0.5 = -0.025 stays
     non-negligible relative to typical Q-values, keeping coordination
     adaptive even after auto-tuning saturates.

  3. PREY R_LOS_BREAK reduced 0.25 → 0.15 per hunter (env.py).
     Total possible per step drops 0.5 → 0.3.  Why: in v1 the prey was
     able to accumulate huge R_LOS_BREAK rewards by cycling rapidly
     around obstacles (eval episode 1 logged 179 LOS breaks in 600
     steps — almost one every 3 steps).  This trained the prey to game
     the LOS-break signal rather than execute genuine escape.  Lower
     coefficient keeps the signal but de-emphasises rapid LOS cycling.

Hyperparameters retained from v1:
  - HIDDEN = 192/192
  - Parameter-shared hunter network
  - α-floor via softplus reparam (the load-bearing v6 fix)
  - α init at 0.2, target_entropy = 0.0
  - Critic warmup 5000 steps, UTD = 0.25
  - max_steps 600 (kept long; the hunters need time to coordinate
    pinch-attacks against a competent prey)

Usage:  python train_sac.py
"""

import os, time, json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from env import HunterPreyEnv

# ---------- hyperparameters ----------
TOTAL_TIMESTEPS   = 10_000_000
NUM_ENVS          = 16
LEARNING_RATE     = 3e-4
GAMMA             = 0.99
TAU               = 0.005
BATCH_SIZE        = 256
BUFFER_SIZE       = 1_000_000
LEARNING_STARTS   = 10_000
UPDATE_EVERY      = 4              # UTD = 0.25
HIDDEN            = {"hunter": 192, "prey": 192}
SAVE_DIR          = "models"
LOG_EVERY         = 5000
SAVE_EVERY        = 500_000

# v6 entropy fixes carried over.
# v2: α floors are now ASYMMETRIC.  Hunters need more sustained
# exploration than the prey because the two-agent pinch geometry is
# harder to coordinate than single-agent escape.  Setting hunter floor
# higher keeps coordination adaptive even after auto-tuning saturates.
TARGET_ENTROPY    = {"hunter": 0.0, "prey": 0.0}
ALPHA_FLOOR       = {"hunter": 0.05, "prey": 0.02}   # CHANGED from {0.02, 0.02}
CRITIC_WARMUP     = 5000

# Hunter freeze: random actions for both hunters during first 5% of training.
# Identical to single-hunter SAC: gives the prey a clean training distribution
# against an unbiased opponent before the pursuit policy starts converging.
HUNTER_FREEZE_FRAC = 0.05

# v2: speed curriculum re-introduced.  Equal speeds for first half of
# training (matching v1 dynamics so the hunters can develop pursuit
# fundamentals); small hunter advantage in late training to compensate
# for the α-floor exploration deficit when the prey starts winning.
# (frac_start, obstacle_range, hunter_speed)
CURRICULUM = [
    (0.00, (0, 0), 4.5),     # open field, equal speeds
    (0.15, (1, 3), 4.5),     # sparse obstacles, equal
    (0.40, (3, 6), 4.5),     # full obstacles, equal (matches v1 endpoint)
    (0.50, (3, 6), 4.7),     # late-training hunter boost
]
PREY_SPEED   = 4.5
MAX_STEPS    = 600   # explicit; SAC keeps long episodes (PPO uses 450)

LOG_STD_MIN = -20.0
LOG_STD_MAX =   2.0


# =====================================================================
# Networks
# =====================================================================

class SACGaussianActor(nn.Module):
    """Squashed-Gaussian SAC actor (state-dependent mean and log_std)."""
    def __init__(self, obs_dim, act_dim, hidden):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),  nn.ReLU(),
        )
        self.mean_head    = nn.Linear(hidden, act_dim)
        self.log_std_head = nn.Linear(hidden, act_dim)

    def forward(self, obs):
        h = self.trunk(obs)
        mean = self.mean_head(h)
        log_std = self.log_std_head(h).clamp(LOG_STD_MIN, LOG_STD_MAX)
        return mean, log_std

    def sample(self, obs):
        mean, log_std = self(obs)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        u = normal.rsample()
        a = torch.tanh(u)
        log_prob = normal.log_prob(u) - torch.log(1 - a.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        return a, log_prob

    @torch.no_grad()
    def act(self, obs, deterministic=False):
        if deterministic:
            mean, _ = self(obs)
            a = torch.tanh(mean)
            return a.cpu().numpy(), None, None
        a, lp = self.sample(obs)
        return a.cpu().numpy(), lp.cpu().numpy(), None


class SACCritic(nn.Module):
    """Twin Q-networks taking (obs, action) and outputting (Q1, Q2)."""
    def __init__(self, obs_dim, act_dim, hidden):
        super().__init__()
        self.q1 = nn.Sequential(
            nn.Linear(obs_dim + act_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),            nn.ReLU(),
            nn.Linear(hidden, 1),
        )
        self.q2 = nn.Sequential(
            nn.Linear(obs_dim + act_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, obs, act):
        x = torch.cat([obs, act], dim=-1)
        return self.q1(x), self.q2(x)


# =====================================================================
# Replay buffer + RunningNorm
# =====================================================================

class ReplayBuffer:
    def __init__(self, capacity, obs_dim, act_dim):
        self.capacity = capacity
        self.obs   = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.act   = np.zeros((capacity, act_dim), dtype=np.float32)
        self.rew   = np.zeros(capacity, dtype=np.float32)
        self.next  = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.done  = np.zeros(capacity, dtype=np.float32)
        self.ptr  = 0
        self.size = 0

    def push(self, o, a, r, no, d):
        i = self.ptr
        self.obs[i] = o; self.act[i] = a; self.rew[i] = r
        self.next[i] = no; self.done[i] = d
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, n, device):
        idx = np.random.randint(0, self.size, n)
        return (
            torch.from_numpy(self.obs[idx]).to(device),
            torch.from_numpy(self.act[idx]).to(device),
            torch.from_numpy(self.rew[idx]).to(device).unsqueeze(-1),
            torch.from_numpy(self.next[idx]).to(device),
            torch.from_numpy(self.done[idx]).to(device).unsqueeze(-1),
        )


class RunningNorm:
    def __init__(self, shape):
        self.mean  = np.zeros(shape, np.float64)
        self.var   = np.ones(shape,  np.float64)
        self.count = 1e-4

    def update(self, batch):
        if batch.ndim == 1: batch = batch[np.newaxis]
        n = batch.shape[0]; bm = batch.mean(0); bv = batch.var(0)
        tot = self.count + n; d = bm - self.mean
        self.mean = self.mean + d * n / tot
        self.var  = (self.var * self.count + bv * n + d**2 * self.count * n / tot) / tot
        self.count = tot

    def normalize(self, x):
        return np.clip((x - self.mean) / (np.sqrt(self.var) + 1e-8), -10, 10).astype(np.float32)

    def state_dict(self):
        return {"mean": self.mean.copy(), "var": self.var.copy(), "count": self.count}

    def load_state_dict(self, d):
        self.mean = d["mean"].copy(); self.var = d["var"].copy(); self.count = d["count"]


# =====================================================================
# Curriculum
# =====================================================================
def current_phase(global_step):
    frac = global_step / TOTAL_TIMESTEPS
    obs_range  = CURRICULUM[0][1]
    hunter_spd = CURRICULUM[0][2]
    for threshold, r, s in CURRICULUM:
        if frac >= threshold:
            obs_range  = r
            hunter_spd = s
    return obs_range, hunter_spd


# =====================================================================
# Main
# =====================================================================
def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    initial_hunter_speed = CURRICULUM[0][2]
    envs    = [HunterPreyEnv(hunter_speed=initial_hunter_speed, prey_speed=PREY_SPEED,
                             n_obstacles_range=(0, 0),
                             max_steps=MAX_STEPS) for _ in range(NUM_ENVS)]
    env_obs = [e.reset(seed=42 + i) for i, e in enumerate(envs)]
    obs_dim = envs[0].obs_size
    act_dim = envs[0].action_size

    # Two roles: hunter (shared between h1/h2) and prey
    roles = ["hunter", "prey"]

    # Networks
    actors  = {r: SACGaussianActor(obs_dim, act_dim, HIDDEN[r]).to(device) for r in roles}
    critics = {r: SACCritic(obs_dim, act_dim, HIDDEN[r]).to(device) for r in roles}
    targets = {r: SACCritic(obs_dim, act_dim, HIDDEN[r]).to(device) for r in roles}
    for r in roles:
        targets[r].load_state_dict(critics[r].state_dict())

    actor_opts  = {r: optim.Adam(actors[r].parameters(),  lr=LEARNING_RATE) for r in roles}
    critic_opts = {r: optim.Adam(critics[r].parameters(), lr=LEARNING_RATE) for r in roles}

    # Auto-entropy temperature with softplus floor (v6 entropy fix).
    # v2: floors are now per-role (hunter higher than prey).  α_init still
    # 0.2 for both → solve softplus(x) = 0.2 - floor for each role's floor.
    init_log_alpha_val = {r: float(np.log(np.exp(0.2 - ALPHA_FLOOR[r]) - 1.0))
                          for r in roles}
    log_alphas = {r: torch.tensor([init_log_alpha_val[r]],
                                   requires_grad=True, device=device)
                  for r in roles}
    alpha_opts = {r: optim.Adam([log_alphas[r]], lr=LEARNING_RATE) for r in roles}

    def get_alpha(role):
        return ALPHA_FLOOR[role] + F.softplus(log_alphas[role])

    # Buffers — one per role.  Hunter buffer collects from BOTH hunters
    # at every env step, so the hunter network gets ~2× the data per step.
    buffers     = {r: ReplayBuffer(BUFFER_SIZE, obs_dim, act_dim) for r in roles}
    normalizers = {r: RunningNorm(shape=(obs_dim,)) for r in roles}

    hunter_freeze_step = int(TOTAL_TIMESTEPS * HUNTER_FREEZE_FRAC)
    critic_update_count = {r: 0 for r in roles}

    print(f"Device: {device}  Envs: {NUM_ENVS}")
    print(f"Obs: {obs_dim}  Act: {act_dim}")
    for r in roles:
        n_actor  = sum(p.numel() for p in actors[r].parameters())
        n_critic = sum(p.numel() for p in critics[r].parameters())
        print(f"  {r:>6}: actor={n_actor:,}  critic={n_critic:,}  hidden={HIDDEN[r]}")
    print(f"Hunters share one network (h1 and h2 query the same actor + critic)")
    print(f"Hunter frozen first {HUNTER_FREEZE_FRAC:.0%} ({hunter_freeze_step:,} steps)")
    print(f"v6 entropy: α_floor={ALPHA_FLOOR}  target_entropy={TARGET_ENTROPY}")
    print(f"v6 entropy: critic warmup={CRITIC_WARMUP}  UTD={1/UPDATE_EVERY}")
    print(f"Curriculum (frac, obs, hunter_speed): {CURRICULUM}  prey_speed={PREY_SPEED}")
    print(f"max_steps={MAX_STEPS}")

    # Tracking
    global_step = 0; ep_count = 0
    recent_caps   = []
    recent_caps_h1 = []   # which hunter caught
    recent_caps_h2 = []
    recent_rets   = {a: [] for a in ["hunter1", "hunter2", "prey"]}
    recent_ep_steps = []
    ep_rets = [{a: 0.0 for a in ["hunter1", "hunter2", "prey"]} for _ in range(NUM_ENVS)]

    prev_phase = None
    train_log = []
    t0 = time.time()

    print(f"\nTraining {TOTAL_TIMESTEPS:,} steps\n")

    while global_step < TOTAL_TIMESTEPS:
        # Curriculum: obstacle density AND hunter speed
        obs_range, hunter_spd = current_phase(global_step)
        phase_key = (obs_range, hunter_spd)
        if phase_key != prev_phase:
            for e in envs:
                e.set_obstacle_range(*obs_range)
                e.set_speeds(hunter_spd, PREY_SPEED)
            print(f"  [curriculum] step {global_step:,}  obs {obs_range}  "
                  f"hunter_spd {hunter_spd}  prey_spd {PREY_SPEED}")
            prev_phase = phase_key

        # Collect one env step from every env
        normed = [{
            "hunter1": normalizers["hunter"].normalize(env_obs[ei]["hunter1"]),
            "hunter2": normalizers["hunter"].normalize(env_obs[ei]["hunter2"]),
            "prey":    normalizers["prey"].normalize(env_obs[ei]["prey"]),
        } for ei in range(NUM_ENVS)]

        hunter_frozen = global_step < hunter_freeze_step

        # Sample actions
        with torch.no_grad():
            if hunter_frozen:
                # Random actions for both hunters during freeze
                h1_acts = np.random.uniform(-1, 1, (NUM_ENVS, act_dim)).astype(np.float32)
                h2_acts = np.random.uniform(-1, 1, (NUM_ENVS, act_dim)).astype(np.float32)
            else:
                h1_obs_t = torch.from_numpy(np.array([normed[ei]["hunter1"] for ei in range(NUM_ENVS)])).to(device)
                h2_obs_t = torch.from_numpy(np.array([normed[ei]["hunter2"] for ei in range(NUM_ENVS)])).to(device)
                # Random until LEARNING_STARTS even after freeze
                if global_step < LEARNING_STARTS:
                    h1_acts = np.random.uniform(-1, 1, (NUM_ENVS, act_dim)).astype(np.float32)
                    h2_acts = np.random.uniform(-1, 1, (NUM_ENVS, act_dim)).astype(np.float32)
                else:
                    h1_acts, _, _ = actors["hunter"].act(h1_obs_t)
                    h2_acts, _, _ = actors["hunter"].act(h2_obs_t)

            p_obs_t = torch.from_numpy(np.array([normed[ei]["prey"] for ei in range(NUM_ENVS)])).to(device)
            if global_step < LEARNING_STARTS:
                p_acts = np.random.uniform(-1, 1, (NUM_ENVS, act_dim)).astype(np.float32)
            else:
                p_acts, _, _ = actors["prey"].act(p_obs_t)

        # Step envs
        for ei in range(NUM_ENVS):
            o_new, rew, done, info = envs[ei].step(h1_acts[ei], h2_acts[ei], p_acts[ei])
            global_step += 1

            for a in ["hunter1", "hunter2", "prey"]:
                ep_rets[ei][a] += rew[a]

            # Push transitions:
            #   hunter buffer gets BOTH hunters' transitions (parameter sharing)
            #   prey buffer gets one prey transition
            buffers["hunter"].push(
                normed[ei]["hunter1"], h1_acts[ei], rew["hunter1"],
                normalizers["hunter"].normalize(o_new["hunter1"]), float(done))
            buffers["hunter"].push(
                normed[ei]["hunter2"], h2_acts[ei], rew["hunter2"],
                normalizers["hunter"].normalize(o_new["hunter2"]), float(done))
            buffers["prey"].push(
                normed[ei]["prey"], p_acts[ei], rew["prey"],
                normalizers["prey"].normalize(o_new["prey"]), float(done))

            if done:
                ep_count += 1
                recent_caps.append(int(info["captured"]))
                recent_caps_h1.append(int(info["captured_by_h1"]))
                recent_caps_h2.append(int(info["captured_by_h2"]))
                recent_ep_steps.append(info["steps"])
                for a in ["hunter1", "hunter2", "prey"]:
                    recent_rets[a].append(ep_rets[ei][a])
                    ep_rets[ei][a] = 0.0
                env_obs[ei] = envs[ei].reset()
            else:
                env_obs[ei] = o_new

            # Update normalizers periodically
            if global_step % 1000 == 0:
                normalizers["hunter"].update(np.array([env_obs[ei]["hunter1"]]))
                normalizers["hunter"].update(np.array([env_obs[ei]["hunter2"]]))
                normalizers["prey"].update(np.array([env_obs[ei]["prey"]]))

            # SAC updates
            if global_step >= LEARNING_STARTS and global_step % UPDATE_EVERY == 0:
                for r in roles:
                    if r == "hunter" and hunter_frozen:
                        continue
                    if buffers[r].size < BATCH_SIZE:
                        continue

                    alpha = get_alpha(r).detach()
                    b_obs, b_act, b_rew, b_next, b_done = buffers[r].sample(BATCH_SIZE, device)

                    # Critic update (always, when we update)
                    with torch.no_grad():
                        next_act, next_lp = actors[r].sample(b_next)
                        nq1, nq2 = targets[r](b_next, next_act)
                        nq = torch.min(nq1, nq2) - alpha * next_lp
                        target_q = b_rew + GAMMA * (1 - b_done) * nq

                    q1, q2 = critics[r](b_obs, b_act)
                    c_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)
                    critic_opts[r].zero_grad(); c_loss.backward(); critic_opts[r].step()
                    critic_update_count[r] += 1

                    # Soft update target
                    with torch.no_grad():
                        for p, tp in zip(critics[r].parameters(), targets[r].parameters()):
                            tp.data.mul_(1 - TAU); tp.data.add_(TAU * p.data)

                    # Actor + α updates skipped during critic warmup
                    in_warmup = critic_update_count[r] <= CRITIC_WARMUP
                    if in_warmup:
                        continue

                    # Actor update
                    new_act, new_lp = actors[r].sample(b_obs)
                    aq1, aq2 = critics[r](b_obs, new_act)
                    aq = torch.min(aq1, aq2)
                    a_loss = (alpha * new_lp - aq).mean()
                    actor_opts[r].zero_grad(); a_loss.backward(); actor_opts[r].step()

                    # α update (auto-tuning)
                    alpha_now = get_alpha(r)
                    al_loss = -(alpha_now * (new_lp.detach() + TARGET_ENTROPY[r])).mean()
                    alpha_opts[r].zero_grad(); al_loss.backward(); alpha_opts[r].step()

            # Logging
            if global_step % LOG_EVERY == 0 and global_step > 0:
                el = time.time() - t0; sps = global_step / el
                cap = 100 * np.mean(recent_caps[-500:]) if recent_caps else 0.0
                cap_h1 = 100 * np.mean(recent_caps_h1[-500:]) if recent_caps_h1 else 0.0
                cap_h2 = 100 * np.mean(recent_caps_h2[-500:]) if recent_caps_h2 else 0.0
                avg_steps = np.mean(recent_ep_steps[-500:]) if recent_ep_steps else 0
                freeze_tag = " [H_FROZEN]" if hunter_frozen else ""

                # Live entropy estimate
                live_ent = {}
                with torch.no_grad():
                    for r in roles:
                        if buffers[r].size >= BATCH_SIZE:
                            sb_obs, _, _, _, _ = buffers[r].sample(min(256, buffers[r].size), device)
                            _, lp = actors[r].sample(sb_obs)
                            live_ent[r] = float((-lp).mean().item())
                        else:
                            live_ent[r] = float("nan")

                print(f"step {global_step:>9,}  {sps:>5.0f} sps  eps {ep_count:>6,}  "
                      f"cap {cap:>5.1f}% (h1:{cap_h1:.0f} h2:{cap_h2:.0f})  "
                      f"obs {obs_range}  avg_ep {avg_steps:.0f}{freeze_tag}")

                entry = {
                    "global_step": global_step, "episodes": ep_count,
                    "capture_rate": round(cap / 100, 4),
                    "capture_h1": round(cap_h1 / 100, 4),
                    "capture_h2": round(cap_h2 / 100, 4),
                    "avg_ep_steps": round(float(avg_steps), 1),
                    "obstacle_range": list(obs_range),
                    "hunter_frozen": hunter_frozen,
                    "elapsed_sec": round(el, 1), "sps": round(sps, 0),
                }
                for a in ["hunter1", "hunter2", "prey"]:
                    rrt = np.mean(recent_rets[a][-500:]) if recent_rets[a] else float("nan")
                    entry[f"{a}_avg_return"] = round(float(rrt), 2) if not np.isnan(rrt) else None
                for r in roles:
                    alpha_val  = float(get_alpha(r).item())
                    at_floor   = " *FLR*" if alpha_val <= ALPHA_FLOOR[r] + 1e-3 else ""
                    in_warmup_now = (critic_update_count[r] <= CRITIC_WARMUP and
                                     not (r == "hunter" and hunter_frozen))
                    warmup_tag = " [warmup]" if in_warmup_now else ""
                    entry[f"{r}_alpha"]            = round(alpha_val, 4)
                    entry[f"{r}_live_entropy"]     = round(live_ent[r], 3) if not np.isnan(live_ent[r]) else None
                    entry[f"{r}_target_entropy"]   = TARGET_ENTROPY[r]
                    entry[f"{r}_critic_updates"]   = critic_update_count[r]
                    print(f"  {r:>6}: α={alpha_val:.3f}{at_floor}  "
                          f"H_live={live_ent[r]:+.3f}  "
                          f"crit_upd={critic_update_count[r]:>6}{warmup_tag}")
                train_log.append(entry)

            if global_step % SAVE_EVERY == 0 and global_step > 0:
                for r in roles:
                    torch.save({
                        "actor": actors[r].state_dict(),
                        "critic": critics[r].state_dict(),
                        "normalizer": normalizers[r].state_dict(),
                    }, os.path.join(SAVE_DIR, f"{r}_sac_step{global_step}.pt"))
                print(f"  [ckpt] step {global_step:,}")

    # Final save
    print("\n--- Saving final models ---")
    for r in roles:
        path = os.path.join(SAVE_DIR, f"{r}_sac_final.pt")
        torch.save({
            "actor": actors[r].state_dict(),
            "critic": critics[r].state_dict(),
            "normalizer": normalizers[r].state_dict(),
        }, path)
        print(f"  {r} → {path}")

    log_path = os.path.join(SAVE_DIR, "train_sac_log.json")
    with open(log_path, "w") as f:
        json.dump(train_log, f, indent=2)
    print(f"  log → {log_path} ({len(train_log)} entries)")

    cap = 100 * np.mean(recent_caps[-1000:]) if recent_caps else 0.0
    print(f"\nDone in {(time.time()-t0)/60:.1f} min.  Final capture: {cap:.1f}%")


if __name__ == "__main__":
    main()
