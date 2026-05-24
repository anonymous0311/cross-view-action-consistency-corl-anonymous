"""
Matched vs. shuffled pair control figure.

Four vertical bars: K=1 matched, K=1 shuffled, K=2 matched, K=2 shuffled.
K=1 in blue, K=2 in green; shuffled bars use hatch to distinguish from matched.

Data sources (LIBERO-Plus camera track, full 4797 unless noted):
  K=1 matched  : A2 (pi05_v4_pair_cv010_no_spatial_aug), 4073/4797 = 84.9%
  K=1 shuffled : phase0b clean-wrong cv010,              2416/4797 = 50.4%
  K=2 matched  : B6b bilateral, mean over 3 seeds = 87.2% ± 0.4 pp
  K=2 shuffled : B6b clean-wrong, screen-120 subset,       93/360  = 25.8%  (*)

(*) Full eval not run; screen-120 collapse (25.8%) is already decisive.

Output: paper/figures/pair_control.pdf

Run from repository root:
  python code/scripts/figure/figure_pair_control.py
"""

import math
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["font.sans-serif"] = ["Liberation Sans", "Arial", "Helvetica", "DejaVu Sans"]
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["font.size"] = 15

def wilson_ci(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    margin = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (max(0.0, center - margin), min(1.0, center + margin))

COLOR_K1 = "#5E88B0"   # muted steel blue  (K=1, matches ablation color)
COLOR_K2 = "#5B9972"   # muted green       (K=2, matches ours color)

# ── Data ─────────────────────────────────────────────────────────────────────
#                           label                   pct    k      n    err_type  err
BARS = [
    ("K=1\nMatched",   84.9, 4073, 4797, "wilson",    None ),
    ("K=1\nShuffled",  50.4, 2416, 4797, "wilson",    None ),
    ("K=2\nMatched",   87.2, None, None, "seed_std",  0.4  ),
    ("K=2\nShuffled",  25.8,   93,  360, "wilson",    None ),
]
COLORS = [COLOR_K1, COLOR_K1, COLOR_K2, COLOR_K2]
HATCH  = [None,     "///",    None,     "///"]

def make_figure(outpath):
    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    fig.patch.set_facecolor("white")

    x = np.arange(len(BARS))
    w = 0.50

    vals = [b[1] for b in BARS]
    lo_errs, hi_errs = [], []
    for label, pct, k, n, err_type, err in BARS:
        if err_type == "wilson":
            lo, hi = wilson_ci(k, n)
            lo_errs.append((pct / 100 - lo) * 100)
            hi_errs.append((hi - pct / 100) * 100)
        else:
            lo_errs.append(err)
            hi_errs.append(err)

    err_kw = dict(elinewidth=1.2, capthick=1.2, capsize=4, ecolor="#555555")

    for i, (label, pct, k, n, err_type, err) in enumerate(BARS):
        hatch = HATCH[i]
        color = COLORS[i]
        ax.bar(x[i], vals[i], width=w, color=color,
               hatch=hatch, edgecolor="#555555" if hatch else color,
               linewidth=0.8, zorder=3)
        ax.errorbar(x[i], vals[i],
                    yerr=[[lo_errs[i]], [hi_errs[i]]],
                    fmt="none", **err_kw)

    # Value labels above error bars
    for i, (v, hi_e) in enumerate(zip(vals, hi_errs)):
        weight = "bold" if HATCH[i] is None else "normal"
        ax.text(x[i], v + hi_e + 0.8, f"{v:.1f}",
                ha="center", va="bottom",
                fontsize=13, fontweight=weight, color="#222222")

    ax.set_xticks(x)
    ax.set_xticklabels([b[0] for b in BARS], fontsize=15)
    ax.set_ylabel("Success rate (%)", fontsize=15)
    ax.set_ylim(0, 104)
    ax.yaxis.grid(True, linestyle="--", alpha=0.3, zorder=0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", labelsize=14)

    # Legend
    p_k1_matched  = mpatches.Patch(facecolor=COLOR_K1, edgecolor=COLOR_K1, label="K=1")
    p_k2_matched  = mpatches.Patch(facecolor=COLOR_K2, edgecolor=COLOR_K2, label="K=2")
    p_shuffled    = mpatches.Patch(facecolor="white",  edgecolor="#555555",
                                   hatch="///", label="shuffled (control)")
    ax.legend(handles=[p_k1_matched, p_k2_matched, p_shuffled],
              fontsize=13, framealpha=0.85, loc="lower left")

    fig.tight_layout()
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    fig.savefig(outpath, dpi=300, bbox_inches="tight", format="pdf")
    plt.close(fig)
    print(f"Saved: {outpath}")

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(script_dir)))
    fig_dir = os.path.join(repo_root, "paper", "figures")
    make_figure(os.path.join(fig_dir, "pair_control.pdf"))
