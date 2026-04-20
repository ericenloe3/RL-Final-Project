"""
SAC (Soft Actor-Critic) training for hunter-prey.

Two independent SAC agents, one per role.  SAC differs from PPO in:
  - Off-policy: uses a replay buffer instead of on-policy rollouts.
  - Twin Q-networks: two critics with clipped double-Q for stability.
  - Entropy-regularised objective: maximises reward + entropy, producing
    more exploratory policies than PPO — especially useful for the prey.
  - Automatic entropy tuning: learns the temperature α online.

The same env, curriculum, and asymmetric training from PPO carry over:
  - Speed curriculum (equal → hunter faster)
  - Hunter freeze (first 5% of training)
  - Equal network size (192 hidden for both agents)
  - Per-agent entropy targets

v5 changes (fixes hunter jitter/stalling/orbiting):
  1. TARGET_ENTROPY["hunter"] raised -2.0 → -1.0: prevents alpha from
     collapsing to ~0.004, which locked the hunter into a deterministic
     orbit that the prey learned to exploit.
  2. HUNTER_FREEZE_FRAC reduced 0.20 → 0.05: the 20% freeze filled the
     replay buffer with random-action transitions, poisoning critic
     estimates when the hunter unfroze and immediately driving alpha to zero.
  3. Hunter buffer cleared on unfreeze: removes the random-action
     transitions accumulated during freeze before training begins.
  4. HIDDEN["hunter"] raised 128 → 192: more capacity for the harder
     pursuit + interception + obstacle-navigation task.

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
LEARNING_STARTS   = 10_000         # random actions before training
UPDATE_EVERY      = 2              # env steps per gradient step
HIDDEN            = {"hunter": 192, "prey": 192}   # v5: hunter raised 128→192 (needs more capacity for interception)
SAVE_DIR          = "models"
LOG_EVERY         = 5000           # log every N env steps
SAVE_EVERY        = 500_000

# Target entropy: -dim(action) = -2 is the common default, but for the hunter
# that caused catastrophic alpha collapse (α → 0.004) in the previous run.
# With α ≈ 0 the hunter converges to a deterministic orbit policy and has zero
# ability to escape it.  Setting hunter target higher keeps α ≈ 0.05-0.15,
# preserving enough stochasticity for continued exploration during co-evolution.
TARGET_ENTROPY    = {"hunter": -1.0, "prey": -1.0}  # v5: hunter raised from -2.0

# Asymmetric training
# v5: Reduced freeze from 0.20 → 0.05.  Freezing for 20% meant 2M steps of
# random-action transitions in the replay buffer when the hunter unfreezes —
# poisoning early critic estimates and causing the alpha to collapse immediately.
# With 5% freeze the hunter sees the prey while it's still learning basic
# evasion, giving a gradual difficulty ramp instead of a cold-start against
# a near-optimal evader.
HUNTER_FREEZE_FRAC = 0.05
PREY_SPEED = 4.0
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
    print(f"Using device: {device}")
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

    # Auto-entropy: learn log_alpha
    log_alphas  = {a: torch.zeros(1, requires_grad=True, device=device) for a in agents}
    alpha_opts  = {a: optim.Adam([log_alphas[a]], lr=LEARNING_RATE) for a in agents}

    normalizers = {a: RunningNorm(shape=(obs_dim,)) for a in agents}
    buffers     = {a: ReplayBuffer(obs_dim, act_dim, BUFFER_SIZE) for a in agents}

    hunter_freeze_step = int(TOTAL_TIMESTEPS * HUNTER_FREEZE_FRAC)

    print(f"Device: {device}  Envs: {NUM_ENVS}  Algorithm: SAC")
    print(f"Obs: {obs_dim}  Act: {act_dim}")
    for a in agents:
        n = sum(p.numel() for p in actors[a].parameters())
        nq = sum(p.numel() for p in critics[a].parameters())
        print(f"  {a}: actor={n:,}  critic={nq:,}  hidden={HIDDEN[a]}  target_ent={TARGET_ENTROPY[a]}")
    print(f"Hunter frozen first {HUNTER_FREEZE_FRAC:.0%}")
    print(f"Curriculum: {CURRICULUM}")

    # ---- Tracking ----
    global_step = 0; ep_count = 0
    recent_caps = []; recent_rets = {a: [] for a in agents}; recent_ep_steps = []
    ep_rets = [{a: 0.0 for a in agents} for _ in range(NUM_ENVS)]
    prev_phase = None
    hunter_buf_cleared = False   # v5: flag for one-shot buffer clear on unfreeze
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
            print(f"  [v5] hunter buffer cleared at step {global_step:,} (freeze lifted)")

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

                b_obs, b_act, b_rew, b_next, b_done = buffers[a].sample(BATCH_SIZE, device)
                alpha = log_alphas[a].exp().detach()

                # ---- Update critics ----
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

                # ---- Update actor ----
                new_act, new_lp = actors[a].sample(b_obs)
                q1_new, q2_new = critics[a](b_obs, new_act)
                q_new = torch.min(q1_new, q2_new)
                actor_loss = (alpha * new_lp - q_new).mean()
                actor_opts[a].zero_grad()
                actor_loss.backward()
                actor_opts[a].step()

                # ---- Update alpha (entropy temperature) ----
                alpha_loss = -(log_alphas[a] * (new_lp.detach() + TARGET_ENTROPY[a])).mean()
                alpha_opts[a].zero_grad()
                alpha_loss.backward()
                alpha_opts[a].step()

                # ---- Soft update target networks ----
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
                alpha_val = log_alphas[a].exp().item()
                entry[f"{a}_avg_return"] = round(float(r), 2) if not np.isnan(r) else None
                entry[f"{a}_alpha"] = round(alpha_val, 4)
                print(f"  {a:>6}: ret {r:>7.1f}  α={alpha_val:.3f}  buf={buffers[a].size:,}")
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
