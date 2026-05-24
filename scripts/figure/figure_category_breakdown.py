"""
Per-category LIBERO-Plus camera-track breakdown figure.

Grouped bar chart: C1 / C2 / C3 on x-axis, two bars per group
(FM-only and Proposed), with Wilson CI / seed-std error bars.

Output: paper/figures/category_breakdown.pdf

Run from repository root:
  python code/scripts/figure/figure_category_breakdown.py
"""

import math
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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

# Colors from main_sim.pdf palette
COLOR_FM   = "#5E88B0"   # muted steel blue  (ablation)
COLOR_OURS = "#5B9972"   # muted green       (ours)

# Category sample sizes (full LIBERO-Plus camera track)
N = {"C1": 939, "C2": 2976, "C3": 882}

# FM-only (seed 42 single-seed Wilson CI)
FM = {"C1": 72.8, "C2": 79.4, "C3": 86.6}

# Proposed: mean across seeds 42/43/44; seed std dev
OURS_MEAN = {"C1": 81.8, "C2": 87.9, "C3": 90.6}
OURS_STD  = {"C1":  1.0, "C2":  0.5, "C3":  0.9}

CATS = ["C1", "C2", "C3"]
CAT_LABELS = ["C1", "C2", "C3"]

def make_figure(outpath):
    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    fig.patch.set_facecolor("white")

    n_cats = len(CATS)
    x = np.arange(n_cats)
    w = 0.32

    fm_vals   = [FM[c]        for c in CATS]
    ours_vals = [OURS_MEAN[c] for c in CATS]

    # Wilson CI for FM-only (single seed)
    fm_lo = []; fm_hi = []
    for c in CATS:
        k = round(FM[c] / 100 * N[c])
        lo, hi = wilson_ci(k, N[c])
        fm_lo.append((FM[c] / 100 - lo) * 100)
        fm_hi.append((hi - FM[c] / 100) * 100)

    # Seed std for Proposed
    ours_err = [OURS_STD[c] for c in CATS]

    err_kw = dict(elinewidth=1.2, capthick=1.2, capsize=4, ecolor="#555555")

    bars_fm = ax.bar(
        x - w / 2, fm_vals,
        width=w, color=COLOR_FM, zorder=3,
        yerr=[fm_lo, fm_hi], error_kw=err_kw,
        label="FM-only (same-state pairs)"
    )
    bars_ours = ax.bar(
        x + w / 2, ours_vals,
        width=w, color=COLOR_OURS, zorder=3,
        yerr=[ours_err, ours_err], error_kw=err_kw,
        label="Proposed (mean ± seed std)"
    )

    # Value labels above each bar
    for bar, v, hi_e in zip(bars_fm, fm_vals, fm_hi):
        ax.text(bar.get_x() + bar.get_width() / 2,
                v + hi_e + 0.8,
                f"{v:.1f}", ha="center", va="bottom",
                fontsize=13, color="#333333")
    for bar, v, e in zip(bars_ours, ours_vals, ours_err):
        ax.text(bar.get_x() + bar.get_width() / 2,
                v + e + 0.8,
                f"{v:.1f}", ha="center", va="bottom",
                fontsize=13, fontweight="bold", color="#222222")

    ax.set_xticks(x)
    ax.set_xticklabels(CAT_LABELS, fontsize=15)
    ax.set_ylabel("Success rate (%)", fontsize=15)
    ax.set_ylim(60, 102)
    ax.yaxis.grid(True, linestyle="--", alpha=0.3, zorder=0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", labelsize=14)

    ax.legend(fontsize=13, framealpha=0.85, loc="lower left")

    fig.tight_layout()
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    fig.savefig(outpath, dpi=300, bbox_inches="tight", format="pdf")
    plt.close(fig)
    print(f"Saved: {outpath}")

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(script_dir)))
    fig_dir = os.path.join(repo_root, "paper", "figures")
    make_figure(os.path.join(fig_dir, "category_breakdown.pdf"))
