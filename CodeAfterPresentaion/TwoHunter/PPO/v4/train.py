"""
PPO training for 2-hunter / 1-prey hunter-prey (v4 — make hiding genuinely pay).

v3 results showed the prey learned a single "wall trap" strategy: run
in a straight line away from the average hunter direction, terminate
in a corner, get caught.  Eval LOS-breaks: 0-5 per episode (the prey
essentially never used cover).  Capture rate: 100% (15/15), avg ~132
steps.  The R_HIDDEN per-step reward I added in v3 didn't help because
its magnitude (max +0.04/step) was less than the dominant R_HEADING
(+0.10/step), so straight-line running still paid more per step.

v4 changes (env.py + this file):

  ENV.PY:
    1. R_HIDDEN BOOSTED 0.01 → 0.03 (per blocked hunter).
       R_HIDDEN_BOTH BOOSTED 0.02 → 0.05.
       Total when fully hidden: 2*0.03 + 0.05 = 0.11/step, exceeding
       R_HEADING's max of +0.10/step.  This is the per-step reward
       crossover point: hiding now pays more per-step than running.

    2. WALL PENALTY REDESIGNED: graded over 150px instead of binary at 40px.
       Old: -0.08 only within 40px (zero gradient farther out).
       New: continuous quadratic ramp from 0 (at 150px) to -0.05 (at wall).
       The prey now feels increasing pressure away from walls THROUGHOUT
       its trajectory, not only when it's already near the wall.  Cuts
       the "wall trap" attractor: running toward a wall costs reward
       at every step of the approach.

  TRAIN.PY:
    3. CURRICULUM: extended prey-favoured phase 70% → 80%, final hunter
       speed 4.4 → 4.3.  Gives the prey (a) more training time in the
       prey-favoured regime to develop cover-use, and (b) ~5% speed
       margin permanently (was 2% in v3) which is enough for cover-use
       to actually convert into survival.

Other v3 hyperparameters retained:
  - Prey log_std clamp [-3, -0.5]  (still saturated in v3, but tighter
    helps; magnitude shifts above are the load-bearing fix).
  - Prey end ent_coef 0.0001
  - PPO_EPOCHS = 3 (hunter), 8 (prey)
  - HIDDEN = 192/192, parameter-shared hunter
  - HUNTER_FREEZE_FRAC = 0.20
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

# Per-role entropy schedules
# v3: prey end reduced again (0.001 → 0.0001) so the entropy bonus is
# essentially gone by training's end.  v2 still saturated the prey clamp
# from step 0 — the entropy coefficient was still strong enough relative
# to the (weak, negative) policy gradient.  v3 anneals the bonus to ~zero.
ENT_COEF = {
    "hunter": {"start": 0.02, "end": 0.003},
    "prey":   {"start": 0.05, "end": 0.0001},   # CHANGED end 0.001 → 0.0001
}
VF_COEF        = 0.5
MAX_GRAD_NORM  = 0.5
MINIBATCH_SIZE = 256
SAVE_DIR       = "models"
LOG_EVERY      = 10
SAVE_EVERY     = 50

# Asymmetric epochs and capacity
PPO_EPOCHS = {"hunter": 3, "prey": 8}
HIDDEN     = {"hunter": 192, "prey": 192}

# v3: prey clamp tightened further (0.0 → -0.5).  v2's 0.0 cap (std=1.0)
# was still saturated the whole run.  v3's -0.5 cap (std≈0.6) forces
# tighter policy commitment and reduces the noise floor in sampled actions.
# Hunter clamp unchanged (it never saturates).
LOG_STD_BOUNDS = {
    "hunter": (-3.0, 0.5),
    "prey":   (-3.0, -0.5),   # CHANGED upper bound 0.0 → -0.5
}

HUNTER_FREEZE_FRAC = 0.20

# v4 curriculum: extended prey-favoured phase, slower final hunter speed.
# v3 had final hunter speed 4.4 and ramp end at 70%.  Result: prey couldn't
# escape via cover (only 2% margin) and learned the easier "run straight"
# losing strategy.  v4 keeps prey-favoured longer (ramp end 70% → 80%) and
# tops out hunters at 4.3 (4.5/4.3 = ~5% margin).  This is enough margin
# for cover-use to actually pay off in episode survival, while still being
# challenging enough that the prey must use cover (can't just outrun).
# (frac_start, obstacle_range, hunter_speed)
CURRICULUM = [
    (0.00, (0, 0), 4.0),     # open field, hunters slower
    (0.15, (1, 3), 4.1),     # sparse obstacles, hunters still slower
    (0.40, (3, 6), 4.2),     # full obstacles, small margin
    (0.80, (3, 6), 4.3),     # final: ~5% prey speed margin (was 2% in v3)
]
PREY_SPEED = 4.5


# =====================================================================
class ActorCritic(nn.Module):
    """PPO actor-critic with v7 clamped log_std.

    log_std is a learnable Parameter clamped via a property to per-role bounds.
    Prevents the unbounded growth that destroyed prey training in PPO v6.

    v2: bounds are configurable per instance.  The 2v1 prey saturated the
    +0.5 upper bound for 9.8M steps in v1, indicating that bound was too
    high relative to the prey's gradient signal.  v2 uses [-3, 0.0] for
    the prey (std capped at 1.0) while keeping [-3, +0.5] for the hunter
    (which never saturated).
    """

    def __init__(self, obs_dim, act_dim, hidden=192,
                 log_std_min=-3.0, log_std_max=0.5):
        super().__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
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
        nn.init.orthogonal_(self.critic.weight, gain=1.0)

    @property
    def _clamped_log_std(self):
        return self.log_std.clamp(self.log_std_min, self.log_std_max)

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
    out = {k: v / max(n_mb, 1) for k, v in metrics.items()}
    with torch.no_grad():
        out["log_std_raw"]     = policy.log_std.mean().item()
        out["log_std_clamped"] = policy._clamped_log_std.mean().item()
    return out


def anneal(start, end, frac):
    return start + (end - start) * min(1.0, max(0.0, frac))


def current_phase(global_step):
    frac = global_step / TOTAL_TIMESTEPS
    obs_range   = CURRICULUM[0][1]
    hunter_spd  = CURRICULUM[0][2]
    for threshold, r, s in CURRICULUM:
        if frac >= threshold:
            obs_range = r
            hunter_spd = s
    return obs_range, hunter_spd


# =====================================================================
def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialise envs at the curriculum START speeds (hunters slower than prey).
    # The curriculum apply step below also calls set_speeds at every phase change.
    initial_hunter_speed = CURRICULUM[0][2]
    envs    = [HunterPreyEnv(hunter_speed=initial_hunter_speed, prey_speed=PREY_SPEED,
                              n_obstacles_range=(0, 0)) for _ in range(NUM_ENVS)]
    env_obs = [e.reset(seed=42 + i) for i, e in enumerate(envs)]

    # Two roles; hunter network is parameter-shared between h1 and h2
    roles   = ["hunter", "prey"]
    obs_dim = envs[0].obs_size
    act_dim = envs[0].action_size

    policies    = {r: ActorCritic(obs_dim, act_dim, HIDDEN[r],
                                    log_std_min=LOG_STD_BOUNDS[r][0],
                                    log_std_max=LOG_STD_BOUNDS[r][1]).to(device)
                   for r in roles}
    optimizers  = {r: optim.Adam(policies[r].parameters(), lr=LEARNING_RATE, eps=1e-5)
                   for r in roles}
    normalizers = {r: RunningNorm(shape=(obs_dim,)) for r in roles}

    hunter_freeze_step = int(TOTAL_TIMESTEPS * HUNTER_FREEZE_FRAC)

    print(f"Device: {device}  Envs: {NUM_ENVS}  Batch: {NUM_ENVS*NUM_STEPS} (hunter sees 2×)")
    print(f"Obs: {obs_dim}  Act: {act_dim}")
    for r in roles:
        n = sum(p.numel() for p in policies[r].parameters())
        print(f"  {r:>6}: hidden={HIDDEN[r]}  params={n:,}  ppo_epochs={PPO_EPOCHS[r]}")
    print(f"Hunters share one network (h1 and h2 query the same actor + critic)")
    print(f"log_std clamps: hunter={LOG_STD_BOUNDS['hunter']}  prey={LOG_STD_BOUNDS['prey']}")
    print(f"Hunter frozen first {HUNTER_FREEZE_FRAC:.0%} ({hunter_freeze_step:,} steps)")
    print(f"Curriculum (frac, obs, hunter_speed): {CURRICULUM}  prey_speed={PREY_SPEED}")

    # Buffers — hunter has 2*NUM_ENVS slots (h1 in 0..N-1, h2 in N..2N-1)
    B_h, B_p = 2 * NUM_ENVS, NUM_ENVS
    T = NUM_STEPS

    buf_obs  = {"hunter": np.zeros((B_h, T, obs_dim), np.float32),
                "prey":   np.zeros((B_p, T, obs_dim), np.float32)}
    buf_act  = {"hunter": np.zeros((B_h, T, act_dim), np.float32),
                "prey":   np.zeros((B_p, T, act_dim), np.float32)}
    buf_lp   = {"hunter": np.zeros((B_h, T), np.float32),
                "prey":   np.zeros((B_p, T), np.float32)}
    buf_val  = {"hunter": np.zeros((B_h, T), np.float32),
                "prey":   np.zeros((B_p, T), np.float32)}
    buf_rew  = {"hunter": np.zeros((B_h, T), np.float32),
                "prey":   np.zeros((B_p, T), np.float32)}
    buf_done = {"hunter": np.zeros((B_h, T), np.float32),
                "prey":   np.zeros((B_p, T), np.float32)}

    num_updates = TOTAL_TIMESTEPS // (NUM_ENVS * NUM_STEPS)
    global_step = 0; ep_count = 0
    recent_caps = []; recent_caps_h1 = []; recent_caps_h2 = []
    recent_rets = {a: [] for a in ["hunter1", "hunter2", "prey"]}
    recent_ep_steps = []
    ep_rets = [{a: 0.0 for a in ["hunter1", "hunter2", "prey"]} for _ in range(NUM_ENVS)]
    prev_phase = None
    train_log = []
    t0 = time.time()

    print(f"\nTraining {TOTAL_TIMESTEPS:,} steps · {num_updates} updates\n")

    for update in range(1, num_updates + 1):
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

        hunter_frozen = global_step < hunter_freeze_step

        # ---- Collect rollout ----
        for t in range(T):
            global_step += NUM_ENVS

            # Normalised observations for each role
            normed = [{
                "hunter1": normalizers["hunter"].normalize(env_obs[ei]["hunter1"]),
                "hunter2": normalizers["hunter"].normalize(env_obs[ei]["hunter2"]),
                "prey":    normalizers["prey"].normalize(env_obs[ei]["prey"]),
            } for ei in range(NUM_ENVS)]

            with torch.no_grad():
                # Hunter: stack h1 and h2 obs → 2*NUM_ENVS query → split back
                h_obs_arr = np.array(
                    [normed[ei]["hunter1"] for ei in range(NUM_ENVS)] +
                    [normed[ei]["hunter2"] for ei in range(NUM_ENVS)]
                )
                h_obs_t = torch.from_numpy(h_obs_arr).to(device)
                if hunter_frozen:
                    # Random uniform actions (frozen)
                    h_acts = np.random.uniform(-1, 1, (2*NUM_ENVS, act_dim)).astype(np.float32)
                    h_lps  = np.zeros(2*NUM_ENVS, np.float32)
                    _, h_vals_t = policies["hunter"](h_obs_t)
                    h_vals = h_vals_t.squeeze(-1).cpu().numpy()
                else:
                    h_acts, h_lps, h_vals = policies["hunter"].act(h_obs_t)

                # Prey
                p_obs_t = torch.from_numpy(np.array(
                    [normed[ei]["prey"] for ei in range(NUM_ENVS)])).to(device)
                p_acts, p_lps, p_vals = policies["prey"].act(p_obs_t)

            # Step each env, store transitions
            for ei in range(NUM_ENVS):
                # h1 action is in slot ei, h2 action is in slot NUM_ENVS+ei
                h1_act = h_acts[ei]
                h2_act = h_acts[NUM_ENVS + ei]
                p_act  = p_acts[ei]

                # Store hunter buffer
                buf_obs["hunter"][ei,           t] = normed[ei]["hunter1"]
                buf_obs["hunter"][NUM_ENVS+ei,  t] = normed[ei]["hunter2"]
                buf_act["hunter"][ei,           t] = h1_act
                buf_act["hunter"][NUM_ENVS+ei,  t] = h2_act
                buf_lp ["hunter"][ei,           t] = h_lps[ei]
                buf_lp ["hunter"][NUM_ENVS+ei,  t] = h_lps[NUM_ENVS + ei]
                buf_val["hunter"][ei,           t] = h_vals[ei]
                buf_val["hunter"][NUM_ENVS+ei,  t] = h_vals[NUM_ENVS + ei]

                # Store prey buffer
                buf_obs["prey"][ei, t] = normed[ei]["prey"]
                buf_act["prey"][ei, t] = p_act
                buf_lp ["prey"][ei, t] = p_lps[ei]
                buf_val["prey"][ei, t] = p_vals[ei]

                # Step
                obs_new, rew, done, info = envs[ei].step(h1_act, h2_act, p_act)
                buf_done["hunter"][ei,          t] = float(done)
                buf_done["hunter"][NUM_ENVS+ei, t] = float(done)
                buf_done["prey"][ei, t] = float(done)

                buf_rew["hunter"][ei,          t] = rew["hunter1"]
                buf_rew["hunter"][NUM_ENVS+ei, t] = rew["hunter2"]
                buf_rew["prey"][ei, t] = rew["prey"]

                ep_rets[ei]["hunter1"] += rew["hunter1"]
                ep_rets[ei]["hunter2"] += rew["hunter2"]
                ep_rets[ei]["prey"]    += rew["prey"]

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
                    env_obs[ei] = obs_new

        # ---- Update normaliser from collected rollouts ----
        # Hunter normaliser sees both hunters' observations
        normalizers["hunter"].update(buf_obs["hunter"].reshape(-1, obs_dim))
        normalizers["prey"].update(buf_obs["prey"].reshape(-1, obs_dim))

        # ---- PPO updates ----
        frac = global_step / TOTAL_TIMESTEPS
        ent_coefs = {r: anneal(ENT_COEF[r]["start"], ENT_COEF[r]["end"], frac) for r in roles}

        logs = {}
        for r in roles:
            if r == "hunter" and hunter_frozen:
                # Track log_std even during freeze for diagnostics
                with torch.no_grad():
                    ls_raw     = policies[r].log_std.mean().item()
                    ls_clamped = policies[r]._clamped_log_std.mean().item()
                logs[r] = {"pg": 0, "vf": 0, "ent": 0,
                           "log_std_raw": ls_raw, "log_std_clamped": ls_clamped}
                continue

            # Bootstrap value at end of rollout
            with torch.no_grad():
                if r == "hunter":
                    last_obs = np.array(
                        [normalizers["hunter"].normalize(env_obs[ei]["hunter1"]) for ei in range(NUM_ENVS)] +
                        [normalizers["hunter"].normalize(env_obs[ei]["hunter2"]) for ei in range(NUM_ENVS)]
                    )
                else:
                    last_obs = np.array(
                        [normalizers["prey"].normalize(env_obs[ei]["prey"]) for ei in range(NUM_ENVS)])
                lo = torch.from_numpy(last_obs).to(device)
                _, lv = policies[r](lo); lv = lv.squeeze(-1).cpu().numpy()

            # GAE per slot
            n_slots = buf_obs[r].shape[0]
            adv_buf = np.zeros((n_slots, T), np.float32)
            ret_buf = np.zeros((n_slots, T), np.float32)
            for si in range(n_slots):
                adv_buf[si], ret_buf[si] = compute_gae(
                    buf_rew[r][si], buf_val[r][si], buf_done[r][si], lv[si])

            def flat(x):
                return torch.from_numpy(
                    x.reshape(-1) if x.ndim <= 2 else x.reshape(-1, x.shape[-1])).to(device)

            b_obs = flat(buf_obs[r]); b_act = flat(buf_act[r]); b_lp = flat(buf_lp[r])
            b_adv = flat(adv_buf);    b_ret = flat(ret_buf)
            b_adv = (b_adv - b_adv.mean()) / (b_adv.std() + 1e-8)

            logs[r] = ppo_update(policies[r], optimizers[r], b_obs, b_act, b_lp,
                                 b_adv, b_ret, ent_coefs[r], n_epochs=PPO_EPOCHS[r])

        # ---- Logging ----
        if update % LOG_EVERY == 0:
            el = time.time() - t0; sps = global_step / el
            cap = 100 * np.mean(recent_caps[-500:]) if recent_caps else 0.0
            cap_h1 = 100 * np.mean(recent_caps_h1[-500:]) if recent_caps_h1 else 0.0
            cap_h2 = 100 * np.mean(recent_caps_h2[-500:]) if recent_caps_h2 else 0.0
            avg_steps = np.mean(recent_ep_steps[-500:]) if recent_ep_steps else 0
            freeze_tag = " [H_FROZEN]" if hunter_frozen else ""

            print(f"upd {update:>5}/{num_updates}  step {global_step:>9,}  {sps:>5.0f} sps  "
                  f"eps {ep_count:>6,}  cap {cap:>5.1f}% (h1:{cap_h1:.0f} h2:{cap_h2:.0f})  "
                  f"avg_ep {avg_steps:.0f}  obs {obs_range}{freeze_tag}")

            entry = {
                "update": update, "global_step": global_step, "episodes": ep_count,
                "capture_rate": round(cap / 100, 4),
                "capture_h1":   round(cap_h1 / 100, 4),
                "capture_h2":   round(cap_h2 / 100, 4),
                "avg_ep_steps": round(float(avg_steps), 1),
                "obstacle_range": list(obs_range),
                "hunter_frozen": hunter_frozen,
                "ent_coef_hunter": round(ent_coefs["hunter"], 5),
                "ent_coef_prey":   round(ent_coefs["prey"],   5),
                "elapsed_sec": round(el, 1), "sps": round(sps, 0),
            }
            for a in ["hunter1", "hunter2", "prey"]:
                rrt = np.mean(recent_rets[a][-500:]) if recent_rets[a] else float("nan")
                entry[f"{a}_avg_return"] = round(float(rrt), 2) if not np.isnan(rrt) else None
            for r in roles:
                m = logs.get(r, {})
                entry[f"{r}_pg_loss"]          = round(m.get("pg", 0), 5)
                entry[f"{r}_vf_loss"]          = round(m.get("vf", 0), 5)
                entry[f"{r}_entropy"]          = round(m.get("ent", 0), 4)
                entry[f"{r}_log_std_raw"]      = round(m.get("log_std_raw", 0), 4)
                entry[f"{r}_log_std_clamped"]  = round(m.get("log_std_clamped", 0), 4)
                ls_raw  = m.get("log_std_raw", 0.0)
                sat_tag = " *SAT*" if ls_raw > LOG_STD_BOUNDS[r][1] - 0.05 else ""
                rht = entry.get(f"{r if r=='prey' else 'hunter1'}_avg_return", float("nan"))
                print(f"  {r:>6}: pg {m.get('pg',0):+.4f}  vf {m.get('vf',0):.4f}  "
                      f"ent {m.get('ent',0):.3f}  log_std {ls_raw:+.2f}{sat_tag}")
            train_log.append(entry)

        if update % SAVE_EVERY == 0:
            for r in roles:
                torch.save({
                    "policy": policies[r].state_dict(),
                    "normalizer": normalizers[r].state_dict(),
                }, os.path.join(SAVE_DIR, f"{r}_step{global_step}.pt"))
            print(f"  [ckpt] step {global_step:,}")

    # ---- Final save ----
    print("\n--- Saving final models ---")
    for r in roles:
        path = os.path.join(SAVE_DIR, f"{r}_final.pt")
        torch.save({"policy": policies[r].state_dict(),
                    "normalizer": normalizers[r].state_dict()}, path)
        print(f"  {r} → {path}")

    log_path = os.path.join(SAVE_DIR, "train_log.json")
    with open(log_path, "w") as f:
        json.dump(train_log, f, indent=2)
    print(f"  log → {log_path} ({len(train_log)} entries)")

    cap = 100 * np.mean(recent_caps[-1000:]) if recent_caps else 0.0
    print(f"\nDone in {(time.time()-t0)/60:.1f} min.  Final capture rate: {cap:.1f}%")
    print("Run:  python evaluate.py")


if __name__ == "__main__":
    main()
