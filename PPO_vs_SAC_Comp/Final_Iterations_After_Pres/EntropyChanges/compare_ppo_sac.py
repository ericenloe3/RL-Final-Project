"""
PPO vs SAC Hunter-Prey — Training & Evaluation Comparison
==========================================================
Usage:
    python compare_ppo_sac.py

Expects four JSON log files (adjust paths below if needed):
    train_log.json        — PPO training log
    train_sac_log.json    — SAC training log
    eval_log.json         — PPO evaluation log
    eval_sac_log.json     — SAC evaluation log

Outputs:
    comparison_training.png   — 6-panel training curves
    comparison_eval.png       — 4-panel evaluation summary
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
from pathlib import Path

# ── paths ─────────────────────────────────────────────────────────────────────
TRAIN_PPO = "train_log.json"
TRAIN_SAC = "train_sac_log.json"
EVAL_PPO  = "eval_log.json"
EVAL_SAC  = "eval_sac_log.json"

# ── colours ───────────────────────────────────────────────────────────────────
C_PPO  = "#185FA5"   # blue
C_SAC  = "#D85A30"   # coral
C_PPO2 = "#85B7EB"   # lighter blue (secondary lines)
C_SAC2 = "#F0997B"   # lighter coral
ALPHA  = 0.85


# =============================================================================
# Helpers
# =============================================================================

def load(path):
    with open(path) as f:
        return json.load(f)


def smooth(vals, w=5):
    """Simple moving-average smoothing."""
    if len(vals) < w:
        return vals
    kernel = np.ones(w) / w
    return np.convolve(vals, kernel, mode="same").tolist()


def sample(data, n=40):
    """Down-sample a list to at most n points (keep first and last)."""
    if len(data) <= n:
        return data
    idx = list(range(0, len(data), max(1, len(data) // n)))
    if (len(data) - 1) not in idx:
        idx.append(len(data) - 1)
    return [data[i] for i in idx]


def to_M(steps):
    """Convert step list to millions."""
    return [s / 1_000_000 for s in steps]


def extract_train(log):
    s = sample(log, 40)
    return dict(
        steps   = to_M([e["global_step"] for e in s]),
        cap     = [e["capture_rate"] * 100 for e in s],
        ep_len  = [e["avg_ep_steps"] for e in s],
        h_ret   = [e.get("hunter_avg_return") or 0 for e in s],
        p_ret   = [e.get("prey_avg_return")  or 0 for e in s],
        frozen  = [e.get("hunter_frozen", False) for e in s],
    )


def extract_ppo_extra(log):
    s = sample(log, 40)
    return dict(
        steps  = to_M([e["global_step"] for e in s]),
        h_ent  = [e.get("hunter_entropy", 0) for e in s],
        p_ent  = [e.get("prey_entropy",   0) for e in s],
    )


def extract_sac_extra(log):
    s = sample(log, 40)
    return dict(
        steps  = to_M([e["global_step"] for e in s]),
        h_alp  = [e.get("hunter_alpha", 0) for e in s],
        p_alp  = [e.get("prey_alpha",   0) for e in s],
    )


def parse_eval(log):
    eps = []
    for ep in log:
        eps.append(dict(
            phase    = ep["phase"],
            captured = ep["captured"],
            steps    = ep["steps"],
            start_d  = ep["start_dist"],
            end_d    = ep["end_dist"],
            los      = ep["los_breaks"],
            delta    = round(ep["start_dist"] - ep["end_dist"], 1),
        ))
    return eps


# =============================================================================
# Training figure
# =============================================================================

def plot_training(ppo_log, sac_log, out="comparison_training.png"):
    ppo = extract_train(ppo_log)
    sac = extract_train(sac_log)
    ppo_ex = extract_ppo_extra(ppo_log)
    sac_ex = extract_sac_extra(sac_log)

    fig = plt.figure(figsize=(16, 14))
    fig.suptitle("PPO vs SAC — Training Dynamics", fontsize=16, fontweight="bold",
                 y=0.98)
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.35)

    # Freeze shading helper
    def shade_freeze(ax, ppo_d):
        for i, frozen in enumerate(ppo_d["frozen"]):
            if frozen and i + 1 < len(ppo_d["frozen"]):
                ax.axvspan(ppo_d["steps"][i], ppo_d["steps"][i + 1],
                           alpha=0.08, color="gray", linewidth=0)
        ax.axvline(x=2.0, color="gray", lw=0.8, ls=":", alpha=0.6,
                   label="PPO hunter unfreezes (~2M)")

    # ── 1. Capture rate ──────────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(ppo["steps"], smooth(ppo["cap"]), color=C_PPO,  lw=2, label="PPO")
    ax1.plot(sac["steps"], smooth(sac["cap"]), color=C_SAC,  lw=2, label="SAC")
    shade_freeze(ax1, ppo)
    ax1.set_xlabel("Training steps (M)")
    ax1.set_ylabel("Capture rate (%)")
    ax1.set_title("Capture rate")
    ax1.set_ylim(-5, 105)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.25)

    # ── 2. Episode length ────────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(ppo["steps"], smooth(ppo["ep_len"]), color=C_PPO, lw=2, label="PPO")
    ax2.plot(sac["steps"], smooth(sac["ep_len"]), color=C_SAC, lw=2, label="SAC")
    shade_freeze(ax2, ppo)
    ax2.set_xlabel("Training steps (M)")
    ax2.set_ylabel("Avg episode steps")
    ax2.set_title("Avg episode length")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.25)

    # ── 3. Hunter returns ────────────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(ppo["steps"], smooth(ppo["h_ret"]), color=C_PPO, lw=2, label="PPO hunter")
    ax3.plot(sac["steps"], smooth(sac["h_ret"]), color=C_SAC, lw=2, label="SAC hunter")
    shade_freeze(ax3, ppo)
    ax3.axhline(0, color="gray", lw=0.7, ls="--")
    ax3.set_xlabel("Training steps (M)")
    ax3.set_ylabel("Avg return")
    ax3.set_title("Hunter avg return")
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.25)

    # ── 4. Prey returns ──────────────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(ppo["steps"], smooth(ppo["p_ret"]), color=C_PPO, lw=2, label="PPO prey")
    ax4.plot(sac["steps"], smooth(sac["p_ret"]), color=C_SAC, lw=2, label="SAC prey")
    shade_freeze(ax4, ppo)
    ax4.axhline(0, color="gray", lw=0.7, ls="--")
    ax4.set_xlabel("Training steps (M)")
    ax4.set_ylabel("Avg return")
    ax4.set_title("Prey avg return")
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.25)

    # ── 5. PPO entropy ───────────────────────────────────────────────────────
    ax5 = fig.add_subplot(gs[2, 0])
    ax5.plot(ppo_ex["steps"], ppo_ex["h_ent"], color=C_PPO,  lw=2, label="Hunter entropy")
    ax5.plot(ppo_ex["steps"], ppo_ex["p_ent"], color=C_PPO2, lw=2, ls="--",
             label="Prey entropy")
    ax5.set_xlabel("Training steps (M)")
    ax5.set_ylabel("Entropy (nats)")
    ax5.set_title("PPO — Entropy (hunter & prey)")
    ax5.legend(fontsize=9)
    ax5.grid(True, alpha=0.25)
    #ax5.text(0.02, 0.95,
    #         "Prey entropy grows linearly → log_std explosion\n(fixed in v6 with log_std clamp)",
    #         transform=ax5.transAxes, fontsize=8, va="top", color="#993C1D",
    #         bbox=dict(boxstyle="round", facecolor="#FAECE7", alpha=0.8))

    # ── 6. SAC alpha ─────────────────────────────────────────────────────────
    ax6 = fig.add_subplot(gs[2, 1])
    ax6.plot(sac_ex["steps"], sac_ex["h_alp"], color=C_SAC,  lw=2, label="Hunter α")
    ax6.plot(sac_ex["steps"], sac_ex["p_alp"], color=C_SAC2, lw=2, ls="--",
             label="Prey α")
    ax6.set_xlabel("Training steps (M)")
    ax6.set_ylabel("α (entropy temperature)")
    ax6.set_title("SAC — Entropy temperature (α)")
    ax6.legend(fontsize=9)
    ax6.grid(True, alpha=0.25)
    #ax6.text(0.02, 0.95,
    #         "Hunter α collapses to ~0.004 → deterministic orbit\n(fixed in v5: target_entropy raised -2→-1)",
    #         transform=ax6.transAxes, fontsize=8, va="top", color="#993C1D",
    #         bbox=dict(boxstyle="round", facecolor="#FAECE7", alpha=0.8))

    # Legend patches for freeze zone
    freeze_patch = Patch(facecolor="gray", alpha=0.2, label="PPO hunter frozen")
    fig.legend(handles=[
        Patch(facecolor=C_PPO, label="PPO"),
        Patch(facecolor=C_SAC, label="SAC"),
        freeze_patch,
    ], loc="lower center", ncol=3, fontsize=10, frameon=False, bbox_to_anchor=(0.5, 0.0))

    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved → {out}")


# =============================================================================
# Evaluation figure
# =============================================================================

def plot_eval(ppo_log, sac_log, out="comparison_eval.png"):
    ppo_eps = parse_eval(ppo_log)
    sac_eps = parse_eval(sac_log)

    ppo_live = [e for e in ppo_eps if e["phase"] == "live"]
    ppo_gif  = [e for e in ppo_eps if e["phase"] == "gif"]
    sac_live = [e for e in sac_eps if e["phase"] == "live"]
    sac_gif  = [e for e in sac_eps if e["phase"] == "gif"]

    def cap_pct(eps): return 100 * sum(e["captured"] for e in eps) / len(eps) if eps else 0
    def avg(eps, k):  return np.mean([e[k] for e in eps]) if eps else 0

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("PPO vs SAC — Evaluation Comparison", fontsize=15, fontweight="bold",
                 y=0.98)

    # ── 1. Capture rate by phase ──────────────────────────────────────────────
    ax = axes[0, 0]
    phases   = ["Live (10 eps)", "GIF (5 eps)", "Overall"]
    ppo_caps = [cap_pct(ppo_live), cap_pct(ppo_gif),
                cap_pct(ppo_eps)]
    sac_caps = [cap_pct(sac_live), cap_pct(sac_gif),
                cap_pct(sac_eps)]
    x = np.arange(len(phases)); w = 0.35
    b1 = ax.bar(x - w/2, ppo_caps, w, color=C_PPO, alpha=ALPHA, label="PPO")
    b2 = ax.bar(x + w/2, sac_caps, w, color=C_SAC, alpha=ALPHA, label="SAC")
    #for rect, v in [(b, v) for bars in (b1, b2) for b, v in zip(bars, [*ppo_caps, *sac_caps])]:
    #    ax.text(rect.get_x() + rect.get_width() / 2, rect.get_height() + 1.5,
    #            f"{v:.0f}%", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(phases)
    ax.set_ylabel("Capture rate (%)"); ax.set_ylim(0, 115)
    ax.set_title("Capture rate by phase")
    ax.legend(); ax.grid(True, axis="y", alpha=0.3)

    # ── 2. Steps per episode (box + scatter) ─────────────────────────────────
    ax = axes[0, 1]
    all_ppo = [e["steps"] for e in ppo_eps]
    all_sac = [e["steps"] for e in sac_eps]
    bp = ax.boxplot([all_ppo, all_sac], positions=[1, 2], widths=0.5, patch_artist=True,
                    medianprops=dict(color="white", lw=2))
    bp["boxes"][0].set_facecolor(C_PPO); bp["boxes"][0].set_alpha(ALPHA)
    bp["boxes"][1].set_facecolor(C_SAC); bp["boxes"][1].set_alpha(ALPHA)
    jitter = lambda n: np.random.uniform(-0.12, 0.12, n)
    caps_flag = [e["captured"] for e in ppo_eps]
    ax.scatter(np.ones(len(all_ppo)) + jitter(len(all_ppo)), all_ppo,
               c=[C_PPO if c else "#B5D4F4" for c in caps_flag],
               zorder=5, s=30, edgecolors="white", lw=0.5)
    caps_flag2 = [e["captured"] for e in sac_eps]
    ax.scatter(np.ones(len(all_sac))*2 + jitter(len(all_sac)), all_sac,
               c=[C_SAC if c else "#F5C4B3" for c in caps_flag2],
               zorder=5, s=30, edgecolors="white", lw=0.5)
    ax.set_xticks([1, 2]); ax.set_xticklabels(["PPO", "SAC"])
    ax.set_ylabel("Episode steps"); ax.set_title("Episode length distribution")
    ax.axhline(500, color="gray", ls="--", lw=0.8, label="Timeout (500)")
    cap_patch = Patch(facecolor=C_PPO,  label="PPO captured")
    esc_patch = Patch(facecolor="#B5D4F4", label="PPO escaped")
    ax.legend(handles=[cap_patch, esc_patch,
                        Patch(facecolor=C_SAC,    label="SAC captured"),
                        Patch(facecolor="#F5C4B3", label="SAC escaped")],
              fontsize=8, loc="upper right")
    ax.grid(True, axis="y", alpha=0.3)

    # ── 3. Distance closed  ──────────────────────────────────────────────────
    ax = axes[1, 0]
    ppo_deltas = [e["delta"] for e in ppo_eps]
    sac_deltas = [e["delta"] for e in sac_eps]
    bins = np.linspace(
        min(ppo_deltas + sac_deltas) - 20,
        max(ppo_deltas + sac_deltas) + 20, 15)
    ax.hist(ppo_deltas, bins=bins, alpha=0.7, color=C_PPO, label="PPO",
            edgecolor="white", lw=0.5)
    ax.hist(sac_deltas, bins=bins, alpha=0.7, color=C_SAC, label="SAC",
            edgecolor="white", lw=0.5)
    ax.axvline(np.mean(ppo_deltas), color=C_PPO, ls="--", lw=1.5,
               label=f"PPO mean {np.mean(ppo_deltas):.0f}")
    ax.axvline(np.mean(sac_deltas), color=C_SAC, ls="--", lw=1.5,
               label=f"SAC mean {np.mean(sac_deltas):.0f}")
    ax.set_xlabel("Distance closed (start_dist − end_dist, px)")
    ax.set_ylabel("Episodes")
    ax.set_title("Hunter closing distance per episode")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    # ── 4. LOS breaks per episode ─────────────────────────────────────────────
    ax = axes[1, 1]
    ppo_los = [e["los"] for e in ppo_eps]
    sac_los = [e["los"] for e in sac_eps]
    ax.scatter(range(1, len(ppo_eps)+1), ppo_los, color=C_PPO, s=60,
               label="PPO", zorder=3, alpha=ALPHA)
    ax.scatter(range(1, len(sac_eps)+1), sac_los, color=C_SAC, s=60,
               marker="s", label="SAC", zorder=3, alpha=ALPHA)
    ax.axhline(np.mean(ppo_los), color=C_PPO, ls="--", lw=1.3,
               label=f"PPO mean {np.mean(ppo_los):.1f}")
    ax.axhline(np.mean(sac_los), color=C_SAC, ls="--", lw=1.3,
               label=f"SAC mean {np.mean(sac_los):.1f}")
    ax.set_xlabel("Episode index")
    ax.set_ylabel("LOS breaks")
    ax.set_title("Line-of-sight breaks per episode\n(proxy for prey evasion quality)")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    # Print summary table
    print("\n=== Evaluation summary ===")
    rows = [
        ("Capture rate — live",  cap_pct(ppo_live),  cap_pct(sac_live)),
        ("Capture rate — gif",   cap_pct(ppo_gif),   cap_pct(sac_gif)),
        ("Capture rate — total", cap_pct(ppo_eps),   cap_pct(sac_eps)),
        ("Avg steps (all)",      avg(ppo_eps, "steps"),  avg(sac_eps, "steps")),
        ("Avg start dist",       avg(ppo_eps, "start_d"),avg(sac_eps, "start_d")),
        ("Avg end dist",         avg(ppo_eps, "end_d"),  avg(sac_eps, "end_d")),
        ("Avg LOS breaks",       avg(ppo_eps, "los"),    avg(sac_eps, "los")),
        ("Avg dist closed",      avg(ppo_eps, "delta"),  avg(sac_eps, "delta")),
    ]
    print(f"{'Metric':<30} {'PPO':>10} {'SAC':>10}")
    print("-" * 52)
    for label, pv, sv in rows:
        print(f"{label:<30} {pv:>10.1f} {sv:>10.1f}")

    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nSaved → {out}")


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    missing = [p for p in [TRAIN_PPO, TRAIN_SAC, EVAL_PPO, EVAL_SAC]
               if not Path(p).exists()]
    if missing:
        print(f"Missing files: {missing}")
        raise SystemExit(1)

    ppo_train = load(TRAIN_PPO)
    sac_train = load(TRAIN_SAC)
    ppo_eval  = load(EVAL_PPO)
    sac_eval  = load(EVAL_SAC)

    print(f"Loaded PPO train: {len(ppo_train)} entries  (steps "
          f"{ppo_train[0]['global_step']:,} → {ppo_train[-1]['global_step']:,})")
    print(f"Loaded SAC train: {len(sac_train)} entries  (steps "
          f"{sac_train[0]['global_step']:,} → {sac_train[-1]['global_step']:,})")
    print(f"Loaded PPO eval:  {len(ppo_eval)} episodes")
    print(f"Loaded SAC eval:  {len(sac_eval)} episodes")

    plot_training(ppo_train, sac_train)
    plot_eval(ppo_eval, sac_eval)

    print("\nDone. Open comparison_training.png and comparison_eval.png.")
