"""
SAC (Soft Actor-Critic) training for hunter-prey (v6 — entropy floor).

Diagnosis from the v5 final-run logs:
  - Prey α dropped 1.0 → 0.005 in the first 200k steps (during freeze).
  - Hunter α dropped below 0.05 at step 670k (130k after unfreeze).
  - Capture rate peaked at 99.8% at step 1.31M, then decayed to 14% by 5M
    as the prey co-evolved against a now-deterministic hunter that could
    not adapt.
  - Final equilibrium: 14.4% capture, hunter return 3.4, prey return 15.

Root cause:
  At target_entropy = -1, a near-deterministic 2D tanh-Gaussian satisfies
  the constraint (its actual -E[log_prob] sits below -1).  The α loss
  receives a continuous "push down" gradient until α reaches ~0, at
  which point exploration stops entirely.  In a co-evolving setting,
  whichever agent loses exploration first becomes statically exploitable.

v6 changes:

  1. target_entropy raised -1.0 → 0.0 (both agents).  This is a smaller
     change than the v5 logs suggested it should be — the load-bearing fix
     is the α floor (#2 below).  The α gradient pushes α toward whatever
     value makes current_entropy ≈ target_entropy.  A converged 2D tanh-
     Gaussian with std~0.5 has H ≈ +1.0 nats; with std~0.2 (near
     deterministic) H ≈ -1.4 nats.  At target=-1, α gets pushed downward
     for any policy with std > 0.2 — the entire useful operating range.
     At target=0 the downward pressure is gentler but still present, so α
     drifts toward the floor over time.  We do not target the natural
     entropy of a converged policy (~+1) because that quantity depends on
     the reward structure and the convergence point, neither of which we
     know in advance.  Letting α settle near the floor and using the floor
     itself as the exploration guarantee is more robust.

  2. α floor via softplus reparameterisation (THE LOAD-BEARING FIX):
        α = ALPHA_FLOOR + softplus(log_alpha_param)
     Regardless of where the target_entropy gradient pulls log_alpha_param,
     α cannot drop below ALPHA_FLOOR=0.02.  Implemented in the
     parameterisation itself (not post-hoc clipping) so the gradient
     w.r.t. log_alpha_param vanishes smoothly as the floor is approached.
     The floor of 0.02 is small enough not to dominate Q-values once the
     critic learns (typical Q-values are ±10 to ±100 in this env), but
     large enough to keep meaningful entropy pressure through co-evolution.

  3. α initialised at 0.2 instead of 1.0.  At α=1, the entropy bonus
     -α*log_prob in the actor loss has magnitude comparable to the
     critic's Q-values during early training (Q starts near zero, and
     early per-step rewards are small heading-reward terms ~0.1).  The
     actor loss is then dominated by entropy maximisation rather than
     reward, biasing the policy toward exploration in a way that triggers
     the α-collapse race.  α=0.2 lets Q-values lead the actor loss as
     soon as the critic produces meaningful estimates.

  4. UTD ratio reduced 0.5 → 0.25 (UPDATE_EVERY 2 → 4).  Lower update-to-
     data ratio gives the buffer time to refresh with diverse transitions
     before the critic over-commits, slowing the early α decline.

  5. Critic warmup: actor and α stay frozen for the first CRITIC_WARMUP
     gradient steps after LEARNING_STARTS.  Without this, the very first
     actor + α updates use a near-random Q function, producing wildly
     inaccurate gradients that immediately push α down.  The hunter's
     warmup counter resets on unfreeze (when its buffer is cleared) so it
     gets a fresh warmup against its rebuilt buffer.

  6. Diagnostic logging: live policy entropy estimate, α-floor saturation
     flag, warmup status, and target_entropy per-update.

Usage:  python train_sac.py
"""

import os, time, json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
from env import HunterPreyEnv

# ---------- hyperparameters ----------
TOTAL_TIMESTEPS   = 10_000_000
NUM_ENVS          = 16
LEARNING_RATE     = 3e-4
GAMMA             = 0.99
TAU               = 0.005          # soft target update rate
BATCH_SIZE        = 256
BUFFER_SIZE       = 1_000_000
LEARNING_STARTS   = 10_000
UPDATE_EVERY      = 4              # v6: 2 → 4 (UTD 0.5 → 0.25)
HIDDEN            = {"hunter": 192, "prey": 192}
SAVE_DIR          = "models"
LOG_EVERY         = 5000
SAVE_EVERY        = 500_000

# v6: target_entropy raised -1.0 → 0.0 — but the *load-bearing* fix is the
# α floor below, not this constant.
#
# 2D tanh-Gaussian entropy at various stds (per-dim entropy ×2):
#   std = 1.0  → entropy ≈ +0.9 nats (initialisation regime)
#   std = 0.5  → entropy ≈ +1.0 nats (typical converged policy)
#   std = 0.2  → entropy ≈ -1.4 nats (effectively deterministic)
#
# The α gradient pulls α toward whatever value makes current_entropy ≈ target.
# At target=-1, a policy with std > 0.2 (i.e. any useful policy) has H above
# the target, so α gets a strong "push down" gradient — this is what caused
# v5's collapse.
#
# At target=0.0 the downward pressure on α is reduced but still present, so α
# drifts toward the floor over training.  We do NOT target the natural entropy
# of a converged policy (~+1) because that quantity depends on the reward
# structure and convergence point, both unknown a priori.  Letting α settle
# near the floor and using the floor itself as the exploration guarantee is
# more robust than tuning the target precisely.
TARGET_ENTROPY    = {"hunter": 0.0, "prey": 0.0}

# v6: α floor — the load-bearing entropy fix.  Even if target_entropy is
# wrong, this prevents the catastrophic α → 0 collapse seen in v5.
# Implemented via softplus reparameterisation:
#     α = ALPHA_FLOOR + softplus(log_alpha_param)
# so the floor is enforced by the parameterisation itself (gradient vanishes
# smoothly near the floor), not by post-hoc clipping.
ALPHA_FLOOR       = 0.02

# v6: critic warmup.  After LEARNING_STARTS, the first CRITIC_WARMUP gradient
# steps update only the critics; actor and α are frozen.  Without this, the
# first actor + α updates use a near-random Q function, producing huge
# misleading gradients that drive α to its floor immediately.
CRITIC_WARMUP     = 5000

HUNTER_FREEZE_FRAC = 0.05
PREY_SPEED         = 4.0
CURRICULUM = [
    (0.00, (0, 0), 4.0),
    (0.15, (1, 3), 4.2),
    (0.40, (3, 6), 4.3),
    (0.70, (3, 6), 4.5),
]


# =====================================================================
# SAC Networks
# =====================================================================
class SACGaussianActor(nn.Module):
    """Squashed Gaussian policy: outputs mean and log_std, both state-dependent."""

    LOG_STD_MIN = -20
    LOG_STD_MAX = 2

    def __init__(self, obs_dim, act_dim, hidden=128):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),  nn.ReLU(),
        )
        self.mean_head    = nn.Linear(hidden, act_dim)
        self.log_std_head = nn.Linear(hidden, act_dim)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0.0)
        nn.init.orthogonal_(self.mean_head.weight, gain=0.01)

    def forward(self, obs):
        h = self.trunk(obs)
        mean    = self.mean_head(h)
        log_std = self.log_std_head(h).clamp(self.LOG_STD_MIN, self.LOG_STD_MAX)
        return mean, log_std

    def sample(self, obs):
        """Reparameterised sample with log_prob (includes Tanh squashing correction)."""
        mean, log_std = self(obs)
        std  = log_std.exp()
        dist = Normal(mean, std)
        x    = dist.rsample()                          # reparameterised
        action = torch.tanh(x)                         # squash to [-1, 1]
        # Log-prob with Tanh correction: log π(a|s) = log π(u|s) - Σ log(1 - tanh²(u))
        log_prob = dist.log_prob(x) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(-1, keepdim=True)
        return action, log_prob

    @torch.no_grad()
    def act(self, obs, deterministic=False):
        """For inference: returns numpy action."""
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        mean, log_std = self(obs)
        if deterministic:
            action = torch.tanh(mean)
        else:
            std = log_std.exp()
            action = torch.tanh(Normal(mean, std).sample())
        return action.cpu().numpy(), np.zeros(obs.shape[0]), np.zeros(obs.shape[0])


class SACQNetwork(nn.Module):
    """Twin Q-network: takes (obs, action) → two Q-values."""

    def __init__(self, obs_dim, act_dim, hidden=128):
        super().__init__()
        inp = obs_dim + act_dim
        self.q1 = nn.Sequential(
            nn.Linear(inp, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )
        self.q2 = nn.Sequential(
            nn.Linear(inp, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0.0)

    def forward(self, obs, action):
        x = torch.cat([obs, action], dim=-1)
        return self.q1(x), self.q2(x)


# =====================================================================
# Replay Buffer
# =====================================================================
class ReplayBuffer:
    def __init__(self, obs_dim, act_dim, max_size=1_000_000):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0
        self.obs      = np.zeros((max_size, obs_dim), dtype=np.float32)
        self.actions  = np.zeros((max_size, act_dim), dtype=np.float32)
        self.rewards  = np.zeros(max_size, dtype=np.float32)
        self.next_obs = np.zeros((max_size, obs_dim), dtype=np.float32)
        self.dones    = np.zeros(max_size, dtype=np.float32)

    def add(self, obs, action, reward, next_obs, done):
        self.obs[self.ptr]      = obs
        self.actions[self.ptr]  = action
        self.rewards[self.ptr]  = reward
        self.next_obs[self.ptr] = next_obs
        self.dones[self.ptr]    = float(done)
        self.ptr  = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size, device):
        idx = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.FloatTensor(self.obs[idx]).to(device),
            torch.FloatTensor(self.actions[idx]).to(device),
            torch.FloatTensor(self.rewards[idx]).unsqueeze(1).to(device),
            torch.FloatTensor(self.next_obs[idx]).to(device),
            torch.FloatTensor(self.dones[idx]).unsqueeze(1).to(device),
        )


# =====================================================================
# Observation Normaliser (same as PPO version)
# =====================================================================
class RunningNorm:
    def __init__(self, shape):
        self.mean = np.zeros(shape, np.float64)
        self.var  = np.ones(shape, np.float64)
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
    obs_range = CURRICULUM[0][1]; h_speed = CURRICULUM[0][2]
    for threshold, r, s in CURRICULUM:
        if frac >= threshold:
            obs_range = r; h_speed = s
    return obs_range, h_speed


# =====================================================================
# Main
# =====================================================================
def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    envs    = [HunterPreyEnv(hunter_speed=4.0, prey_speed=4.0,
                             n_obstacles_range=(0,0)) for _ in range(NUM_ENVS)]
    env_obs = [e.reset(seed=42+i) for i, e in enumerate(envs)]
    agents  = ["hunter", "prey"]
    obs_dim = envs[0].obs_size
    act_dim = envs[0].action_size

    # ---- Per-agent SAC components ----
    actors      = {a: SACGaussianActor(obs_dim, act_dim, HIDDEN[a]).to(device) for a in agents}
    critics     = {a: SACQNetwork(obs_dim, act_dim, HIDDEN[a]).to(device) for a in agents}
    critic_tgts = {a: SACQNetwork(obs_dim, act_dim, HIDDEN[a]).to(device) for a in agents}
    for a in agents:
        critic_tgts[a].load_state_dict(critics[a].state_dict())

    actor_opts  = {a: optim.Adam(actors[a].parameters(), lr=LEARNING_RATE) for a in agents}
    critic_opts = {a: optim.Adam(critics[a].parameters(), lr=LEARNING_RATE) for a in agents}

    # v6: auto-entropy with softplus floor.
    # α = ALPHA_FLOOR + softplus(log_alpha_param)
    #
    # Initial value: α = 0.2 (not 1.0 as in earlier v5/v6 drafts).  At α=1
    # the entropy bonus -α*log_prob in the actor loss has magnitude ~1-2
    # for a typical squashed-Gaussian policy, which is comparable to or
    # larger than the Q-values in the first few thousand updates (Q starts
    # near zero, early rewards are mostly small per-step heading rewards).
    # This biases the actor toward maximum entropy regardless of reward
    # signal, which contributes to the α-collapse race.  α=0.2 puts the
    # entropy bonus an order of magnitude below typical capture/timeout
    # rewards (±100 / ±20), so the Q-term leads the actor loss as soon
    # as the critic produces meaningful estimates.
    #
    # Solving for log_alpha_param: ALPHA_FLOOR + softplus(x) = 0.2
    #   softplus(x) = 0.18  →  x = log(exp(0.18) - 1)
    init_log_alpha_val = float(np.log(np.exp(0.2 - ALPHA_FLOOR) - 1.0))
    log_alphas = {a: torch.tensor([init_log_alpha_val], requires_grad=True, device=device)
                  for a in agents}
    alpha_opts  = {a: optim.Adam([log_alphas[a]], lr=LEARNING_RATE) for a in agents}

    def get_alpha(agent):
        """α with softplus floor — guarantees α >= ALPHA_FLOOR."""
        return ALPHA_FLOOR + F.softplus(log_alphas[agent])

    normalizers = {a: RunningNorm(shape=(obs_dim,)) for a in agents}
    buffers     = {a: ReplayBuffer(obs_dim, act_dim, BUFFER_SIZE) for a in agents}

    hunter_freeze_step = int(TOTAL_TIMESTEPS * HUNTER_FREEZE_FRAC)

    print(f"Device: {device}  Envs: {NUM_ENVS}  Algorithm: SAC (v6)")
    print(f"Obs: {obs_dim}  Act: {act_dim}")
    for a in agents:
        n = sum(p.numel() for p in actors[a].parameters())
        nq = sum(p.numel() for p in critics[a].parameters())
        print(f"  {a}: actor={n:,}  critic={nq:,}  hidden={HIDDEN[a]}  target_ent={TARGET_ENTROPY[a]}")
    print(f"Hunter frozen first {HUNTER_FREEZE_FRAC:.0%}")
    print(f"v6: α floor={ALPHA_FLOOR}  critic warmup={CRITIC_WARMUP}  "
          f"UTD={1.0/UPDATE_EVERY:.2f}")
    print(f"Curriculum: {CURRICULUM}")

    # ---- Tracking ----
    global_step = 0; ep_count = 0
    recent_caps = []; recent_rets = {a: [] for a in agents}; recent_ep_steps = []
    ep_rets = [{a: 0.0 for a in agents} for _ in range(NUM_ENVS)]
    prev_phase = None
    hunter_buf_cleared = False   # v5: flag for one-shot buffer clear on unfreeze
    critic_update_count = {a: 0 for a in agents}   # v6: critic warmup counter
    train_log = []; t0 = time.time()
    raw_obs_buf = []  # for normaliser updates

    print(f"\nTraining {TOTAL_TIMESTEPS:,} steps (SAC)\n")

    while global_step < TOTAL_TIMESTEPS:
        # ---- Curriculum ----
        obs_range, h_speed = current_phase(global_step)
        phase_key = (obs_range, h_speed)
        if phase_key != prev_phase:
            for e in envs:
                e.set_obstacle_range(*obs_range)
                e.set_speeds(h_speed, PREY_SPEED)
            print(f"  [curriculum] step {global_step:,}  obs {obs_range}  h_spd {h_speed}")
            prev_phase = phase_key

        hunter_frozen = global_step < hunter_freeze_step

        # v5: Clear the hunter's replay buffer the first time it unfreezes.
        # During the freeze period the buffer fills with (random_action, reward)
        # transitions.  These cause the critic to immediately over-estimate Q for
        # random actions, which drives alpha to near-zero and locks the hunter into
        # a deterministic orbit policy with no escape route.
        if not hunter_frozen and hunter_buf_cleared is False:
            buffers["hunter"].ptr  = 0
            buffers["hunter"].size = 0
            hunter_buf_cleared = True
            # v6: also reset critic warmup so the hunter trains its critic
            # against the rebuilt buffer before its actor + α can update.
            critic_update_count["hunter"] = 0
            print(f"  [v5] hunter buffer cleared at step {global_step:,} (freeze lifted)")
            print(f"  [v6] hunter critic warmup reset — actor/α frozen for next "
                  f"{CRITIC_WARMUP} updates")

        # ---- Collect one step per env ----
        for ei in range(NUM_ENVS):
            normed = {a: normalizers[a].normalize(env_obs[ei][a]) for a in agents}
            raw_obs_buf.append(np.stack([normed[a] for a in agents]))

            actions = {}
            for a in agents:
                if global_step < LEARNING_STARTS:
                    # Random actions during warmup
                    actions[a] = np.random.uniform(-1, 1, size=act_dim).astype(np.float32)
                else:
                    obs_t = torch.FloatTensor(normed[a]).unsqueeze(0).to(device)
                    act_np, _, _ = actors[a].act(obs_t, deterministic=False)
                    actions[a] = act_np.squeeze(0)

            obs_new, rew, done, info = envs[ei].step(actions["hunter"], actions["prey"])
            normed_new = {a: normalizers[a].normalize(obs_new[a]) for a in agents}

            for a in agents:
                buffers[a].add(normed[a], actions[a], rew[a], normed_new[a], done)
                ep_rets[ei][a] += rew[a]

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

            global_step += 1

        # ---- Update normaliser periodically ----
        if len(raw_obs_buf) >= 1000:
            arr = np.array(raw_obs_buf)
            for i, a in enumerate(agents):
                normalizers[a].update(arr[:, i])
            raw_obs_buf.clear()

        # ---- SAC gradient steps ----
        if global_step >= LEARNING_STARTS and global_step % UPDATE_EVERY == 0:
            for a in agents:
                if a == "hunter" and hunter_frozen:
                    continue
                if buffers[a].size < BATCH_SIZE:
                    continue

                # v6: read α through the floored parameterisation.
                alpha = get_alpha(a).detach()

                b_obs, b_act, b_rew, b_next, b_done = buffers[a].sample(BATCH_SIZE, device)

                # ---- Update critics (always) ----
                with torch.no_grad():
                    next_act, next_lp = actors[a].sample(b_next)
                    tq1, tq2 = critic_tgts[a](b_next, next_act)
                    tq = torch.min(tq1, tq2) - alpha * next_lp
                    target = b_rew + GAMMA * (1 - b_done) * tq

                q1, q2 = critics[a](b_obs, b_act)
                critic_loss = F.mse_loss(q1, target) + F.mse_loss(q2, target)
                critic_opts[a].zero_grad()
                critic_loss.backward()
                critic_opts[a].step()

                # v6: critic warmup — actor and α stay frozen until the
                # critic has had CRITIC_WARMUP gradient steps on real data.
                critic_update_count[a] += 1
                in_warmup = critic_update_count[a] <= CRITIC_WARMUP

                if not in_warmup:
                    # ---- Update actor ----
                    new_act, new_lp = actors[a].sample(b_obs)
                    q1_new, q2_new = critics[a](b_obs, new_act)
                    q_new = torch.min(q1_new, q2_new)
                    actor_loss = (alpha * new_lp - q_new).mean()
                    actor_opts[a].zero_grad()
                    actor_loss.backward()
                    actor_opts[a].step()

                    # v6: α update through the floored parameterisation.
                    # Standard SAC alpha loss form:
                    #   J(α) = -E[α * (log_prob + target_entropy)]
                    # The floor is enforced because get_alpha applies softplus,
                    # so the gradient w.r.t. log_alpha_param naturally vanishes
                    # as α approaches the floor.
                    alpha_now  = get_alpha(a)
                    alpha_loss = -(alpha_now * (new_lp.detach() + TARGET_ENTROPY[a])).mean()
                    alpha_opts[a].zero_grad()
                    alpha_loss.backward()
                    alpha_opts[a].step()

                # ---- Soft update target networks (always) ----
                with torch.no_grad():
                    for p, tp in zip(critics[a].parameters(), critic_tgts[a].parameters()):
                        tp.data.mul_(1 - TAU).add_(p.data * TAU)

        # ---- Logging ----
        if global_step % LOG_EVERY == 0 and ep_count > 0:
            el = time.time() - t0; sps = global_step / el
            cap = 100 * np.mean(recent_caps[-500:]) if recent_caps else 0.0
            avg_steps = np.mean(recent_ep_steps[-500:]) if recent_ep_steps else 0
            freeze_tag = " [H_FROZEN]" if hunter_frozen else ""

            print(f"step {global_step:>10,}  {sps:>5.0f} sps  eps {ep_count:>6,}  "
                  f"cap {cap:>5.1f}%  obs {obs_range}  h_spd {h_speed:.1f}{freeze_tag}")

            entry = {
                "global_step": global_step, "episodes": ep_count,
                "capture_rate": round(cap/100, 4),
                "avg_ep_steps": round(float(avg_steps), 1),
                "obstacle_range": list(obs_range), "hunter_speed": h_speed,
                "hunter_frozen": hunter_frozen,
                "elapsed_sec": round(el, 1), "sps": round(sps, 0),
            }
            for a in agents:
                r = np.mean(recent_rets[a][-500:]) if recent_rets[a] else float("nan")
                # v6: floored α via get_alpha (matches the value used during training).
                with torch.no_grad():
                    alpha_val = float(get_alpha(a).item())

                # v6: live policy entropy from a sample batch — the quantity
                # SAC's α is trying to push toward TARGET_ENTROPY[a].
                live_ent = float("nan")
                if buffers[a].size >= 256:
                    with torch.no_grad():
                        b_obs_diag, *_ = buffers[a].sample(256, device)
                        _, lp_diag = actors[a].sample(b_obs_diag)
                        live_ent = float(-lp_diag.mean().item())  # H ≈ -E[log_prob]

                at_floor = " *FLR*" if alpha_val <= ALPHA_FLOOR + 1e-3 else ""
                in_warmup_now = (critic_update_count[a] <= CRITIC_WARMUP and
                                 not (a == "hunter" and hunter_frozen))
                warmup_tag = " [warmup]" if in_warmup_now else ""

                entry[f"{a}_avg_return"]      = round(float(r), 2) if not np.isnan(r) else None
                entry[f"{a}_alpha"]           = round(alpha_val, 4)
                entry[f"{a}_live_entropy"]    = (round(live_ent, 3)
                                                 if not np.isnan(live_ent) else None)
                entry[f"{a}_target_entropy"]  = TARGET_ENTROPY[a]
                entry[f"{a}_critic_updates"]  = critic_update_count[a]

                live_str = f"{live_ent:+.2f}" if not np.isnan(live_ent) else "  --  "
                print(f"  {a:>6}: ret {r:>7.1f}  α={alpha_val:.3f}{at_floor}  "
                      f"H_live={live_str} (target {TARGET_ENTROPY[a]:+.1f})  "
                      f"buf={buffers[a].size:,}{warmup_tag}")
            train_log.append(entry)

        # ---- Checkpoint ----
        if global_step > 0 and global_step % SAVE_EVERY == 0:
            for a in agents:
                torch.save({
                    "actor": actors[a].state_dict(),
                    "normalizer": normalizers[a].state_dict(),
                }, os.path.join(SAVE_DIR, f"{a}_sac_step{global_step}.pt"))
            print(f"  [ckpt] step {global_step:,}")

    # ---- Final save ----
    print("\n--- Saving final SAC models ---")
    for a in agents:
        path = os.path.join(SAVE_DIR, f"{a}_sac_final.pt")
        torch.save({
            "actor": actors[a].state_dict(),
            "normalizer": normalizers[a].state_dict(),
        }, path)
        print(f"  {a} → {path}")

    log_path = os.path.join(SAVE_DIR, "train_sac_log.json")
    with open(log_path, "w") as f:
        json.dump(train_log, f, indent=2)
    print(f"  dynamics → {log_path}")

    cap = 100 * np.mean(recent_caps[-1000:]) if recent_caps else 0.0
    print(f"\nDone in {(time.time()-t0)/60:.1f} min.  Final capture rate: {cap:.1f}%")
    print("Run:  python evaluate_sac.py")


if __name__ == "__main__":
    main()
