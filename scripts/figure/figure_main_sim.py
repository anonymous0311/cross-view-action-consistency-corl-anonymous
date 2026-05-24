"""
Generate main-sim figure: 4-row horizontal bar plot for LIBERO-Plus camera-track results.

Outputs:
  paper/figures/main_sim.pdf

Run from repository root:
  python code/scripts/figure/figure_main_sim.py
"""

import math
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Font configuration
matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["font.sans-serif"] = ["Liberation Sans", "Arial", "Helvetica", "DejaVu Sans"]
matplotlib.rcParams["pdf.fonttype"] = 42

def wilson_ci(k, n, z=1.96):
    """Compute Wilson 95% confidence interval."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    margin = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (max(0.0, center - margin), min(1.0, center + margin))

# Low-saturation color palette (CoRL aesthetic)
COLORS = {
    "floor":    "#A0A0A0",   # muted gray  — nominal-only (no camera training)
    "baseline": "#C4A84A",   # muted gold  — naive mixed-camera SFT
    "ablation": "#5E88B0",   # muted steel blue
    "ours":     "#5B9972",   # muted green
}

N_CAM = 4797

# Data: (label, camera_pct, group, is_multi_seed)
METHODS = [
    ("Nominal-only baseline",                   16.8, "floor",    False),
    ("Naive mixed-camera SFT",                  74.7, "baseline", False),
    ("FM-only on same-state pairs",             79.5, "ablation",  False),
    ("Proposed: bilateral K=2 + Beta(2,3)",    87.2, "ours",      True),
]

# Proposed row: mean=87.2%, individual seeds=[86.9, 87.6, 87.2], std=0.4pp
PROPOSED_SEEDS = [86.9, 87.6, 87.2]
PROPOSED_SEED_STD = 0.4

def make_figure(outpath):
    fig, ax = plt.subplots(figsize=(7.5, 3.0))
    fig.patch.set_facecolor("white")

    labels = []
    vals = []
    lo_errs = []
    hi_errs = []
    colors = []

    for label, cam_pct, group, is_multi_seed in METHODS:
        labels.append(label)
        vals.append(cam_pct)
        colors.append(COLORS[group])

        if is_multi_seed:
            # For Proposed: use seed std dev as error bar (thin)
            lo_errs.append(PROPOSED_SEED_STD)
            hi_errs.append(PROPOSED_SEED_STD)
        else:
            # For single-seed rows: compute Wilson CI from point estimate
            k = round(cam_pct / 100 * N_CAM)
            lo, hi = wilson_ci(k, N_CAM)
            lo_errs.append((cam_pct / 100 - lo) * 100)
            hi_errs.append((hi - cam_pct / 100) * 100)

    y = np.arange(len(labels))
    ax.barh(
        y, vals,
        xerr=[lo_errs, hi_errs],
        color=colors,
        height=0.50,
        capsize=3,
        error_kw=dict(elinewidth=1.0, capthick=1.0, ecolor="#555555"),
        zorder=3,
    )

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=11)
    ax.set_xlabel("Camera-perturbation track success rate (%)", fontsize=11)
    ax.set_xlim(0, 105)
    ax.xaxis.grid(True, linestyle="--", alpha=0.3, zorder=0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="x", labelsize=10)

    # Value labels beyond error bars
    for i, (v, hi_e, group) in enumerate(zip(vals, hi_errs, [m[2] for m in METHODS])):
        weight = "bold" if group == "ours" else "normal"
        label_x = v + hi_e + 1.0
        ax.text(label_x, i, f"{v:.1f}%", va="center", ha="left",
                fontsize=10, fontweight=weight, color="#222222")

    fig.tight_layout()
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    fig.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {outpath}")

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(script_dir)))
    fig_dir = os.path.join(repo_root, "paper", "figures")
    make_figure(os.path.join(fig_dir, "main_sim.pdf"))
