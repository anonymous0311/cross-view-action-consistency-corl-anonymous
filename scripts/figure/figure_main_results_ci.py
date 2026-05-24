"""
Generate main-results figure with Wilson 95% CI error bars.

Outputs:
  paper/figures/main_results_ci.pdf       — camera-track bar chart
  paper/figures/category_breakdown_ci.pdf — C1/C2/C3 grouped bar chart

Run from the repository root:
  python code/scripts/figure/figure_main_results_ci.py
"""

import math
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Use Liberation Sans (metric-compatible with Arial; available on Linux)
matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["font.sans-serif"] = ["Liberation Sans", "Arial", "Helvetica", "DejaVu Sans"]
matplotlib.rcParams["pdf.fonttype"] = 42   # embed fonts as TrueType in PDF

# ---------------------------------------------------------------------------
# Wilson 95% CI
# ---------------------------------------------------------------------------
def wilson_ci(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    margin = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (max(0.0, center - margin), min(1.0, center + margin))

# ---------------------------------------------------------------------------
# Low-saturation color palette
# ---------------------------------------------------------------------------
COLORS = {
    "baseline": "#C4A84A",   # muted yellow/gold
    "ablation": "#5E88B0",   # muted steel blue
    "ours":     "#5B9972",   # muted green
    "shuffled": "#C47840",   # muted orange
}

# ---------------------------------------------------------------------------
# Data  (camera-track, n=4797; ID n=120; C1 n=939; C2 n=2976; C3 n=882)
# ---------------------------------------------------------------------------
N_CAM = 4797
N_C1, N_C2, N_C3 = 939, 2976, 882

# (display_label, cam_pct, id_pct, c1_pct, c2_pct, c3_pct, group)
METHODS = [
    # shuffled controls
    ("Shuffled (bilateral K=2)",              25.8, 50.8, 19.4, 25.7, 33.3, "shuffled"),
    ("Shuffled (single-sample)",              50.4, 80.8, None, None, None, "shuffled"),
    # data-exposure baselines
    ("Scene-only baseline",                   16.8, None,  1.1, 13.2, 45.7, "baseline"),
    ("Naive mixed-camera SFT",                74.7, None, 68.4, 75.7, 78.2, "baseline"),
    # ablations
    ("Flow-matching only",                    79.5, 92.5, 72.8, 79.4, 86.6, "ablation"),
    ("Single-sample CV (λ=0.10)",        84.9, 92.5, 78.5, 86.3, 87.0, "ablation"),
    ("Stop-gradient K=2",                      84.3, 96.7, 77.5, 85.7, 87.0, "ablation"),
    # proposed
    ("Proposed: bilateral K=2 + Beta(2,3)",   86.9, 95.0, 80.7, 88.1, 89.7, "ours"),
]

# ---------------------------------------------------------------------------
# Figure A: camera-track comparison
# ---------------------------------------------------------------------------
def make_camera_track_figure(outpath):
    fig, ax = plt.subplots(figsize=(7.5, 4.0))
    fig.patch.set_facecolor("white")

    labels, vals, lo_errs, hi_errs, colors = [], [], [], [], []
    for label, cam, _, _, _, _, group in METHODS:
        k = round(cam / 100 * N_CAM)
        lo, hi = wilson_ci(k, N_CAM)
        labels.append(label)
        vals.append(cam)
        lo_errs.append((cam / 100 - lo) * 100)
        hi_errs.append((hi - cam / 100) * 100)
        colors.append(COLORS[group])

    y = np.arange(len(labels))
    ax.barh(
        y, vals,
        xerr=[lo_errs, hi_errs],
        color=colors,
        height=0.55,
        capsize=3,
        error_kw=dict(elinewidth=1.0, capthick=1.0, ecolor="#444444"),
        zorder=3,
    )

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=11)
    ax.set_xlabel("Camera-perturbation track success rate (%)", fontsize=11)
    ax.set_xlim(0, 107)
    ax.xaxis.grid(True, linestyle="--", alpha=0.35, zorder=0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="x", labelsize=10)

    # value labels placed beyond upper error bar
    for i, (v, hi_e, group) in enumerate(zip(vals, hi_errs, [m[6] for m in METHODS])):
        weight = "bold" if group == "ours" else "normal"
        label_x = v + hi_e + 1.2
        ax.text(label_x, i, f"{v:.1f}%", va="center", ha="left",
                fontsize=10, fontweight=weight, color="#222222")

    # separator lines between groups
    ax.axhline(1.5, color="#BDBDBD", lw=0.7, ls="--")
    ax.axhline(3.5, color="#BDBDBD", lw=0.7, ls="--")
    ax.axhline(6.5, color="#BDBDBD", lw=0.7, ls="--")

    legend_items = [
        mpatches.Patch(facecolor=COLORS["ours"],     label="Proposed"),
        mpatches.Patch(facecolor=COLORS["ablation"], label="Ablation"),
        mpatches.Patch(facecolor=COLORS["baseline"], label="Data-exposure baseline"),
        mpatches.Patch(facecolor=COLORS["shuffled"], label="Shuffled-pair control"),
    ]
    ax.legend(handles=legend_items, fontsize=8, loc="lower right",
              framealpha=0.9, edgecolor="#CCCCCC")

    fig.tight_layout()
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    fig.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {outpath}")


# ---------------------------------------------------------------------------
# Figure B: per-category C1/C2/C3 breakdown
# ---------------------------------------------------------------------------
def make_category_breakdown_figure(outpath):
    SUBSET = [
        ("Naive mixed-camera SFT",   68.4, 75.7, 78.2, "baseline"),
        ("Flow-matching only",        72.8, 79.4, 86.6, "ablation"),
        ("Single-sample CV",          78.5, 86.3, 87.0, "ablation"),
        ("Stop-gradient K=2",         77.5, 85.7, 87.0, "ablation"),
        ("Proposed",                  80.7, 88.1, 89.7, "ours"),
    ]
    labels_m = [m[0] for m in SUBSET]
    c1_vals  = [m[1] for m in SUBSET]
    c2_vals  = [m[2] for m in SUBSET]
    c3_vals  = [m[3] for m in SUBSET]
    groups   = [m[4] for m in SUBSET]

    N_cats    = [N_C1, N_C2, N_C3]
    cat_vals  = [c1_vals, c2_vals, c3_vals]
    cat_xlabels = [
        "C1: distance/scale\n(n=939)",
        "C2: spherical position\n(n=2,976)",
        "C3: orientation\n(n=882)",
    ]

    fig, axes = plt.subplots(1, 3, figsize=(9.5, 3.4), sharey=True)
    fig.patch.set_facecolor("white")

    x = np.arange(len(labels_m))
    width = 0.55

    for ax, vals, N, xlabel in zip(axes, cat_vals, N_cats, cat_xlabels):
        lo_errs, hi_errs = [], []
        for v, g in zip(vals, groups):
            k = round(v / 100 * N)
            lo, hi = wilson_ci(k, N)
            lo_errs.append((v / 100 - lo) * 100)
            hi_errs.append((hi - v / 100) * 100)

        bar_colors = [COLORS[g] for g in groups]
        ax.bar(
            x, vals, width,
            color=bar_colors,
            yerr=[lo_errs, hi_errs],
            capsize=3,
            error_kw=dict(elinewidth=0.9, capthick=0.9, ecolor="#444444"),
            zorder=3,
        )

        ax.set_xticks(x)
        ax.set_xticklabels(labels_m, rotation=30, ha="right", fontsize=9.5)
        ax.set_xlabel(xlabel, fontsize=10.5)
        ax.yaxis.grid(True, linestyle="--", alpha=0.35, zorder=0)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_ylim(55, 103)
        ax.tick_params(axis="y", labelsize=10)

        # value labels above upper CI cap
        for xi, (v, hi_e) in enumerate(zip(vals, hi_errs)):
            ax.text(xi, v + hi_e + 1.2, f"{v:.1f}", ha="center",
                    fontsize=9, color="#222222")

    axes[0].set_ylabel("Success rate (%)", fontsize=11)
    fig.tight_layout()
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    fig.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {outpath}")


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root  = os.path.dirname(os.path.dirname(os.path.dirname(script_dir)))
    fig_dir    = os.path.join(repo_root, "paper", "figures")

    make_camera_track_figure(os.path.join(fig_dir, "main_results_ci.pdf"))
    make_category_breakdown_figure(os.path.join(fig_dir, "category_breakdown_ci.pdf"))
