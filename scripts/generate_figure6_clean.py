import os
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Output paths
OUT_PATHS = [
    Path("reports/plots/fig6_megadepth_clean.png"),
    Path("paper/images/megadepth_auc20_comparison.png"),
    Path("images/megadepth_auc20_comparison.png"),
    Path("experiments/final_report/plots/megadepth_auc20_comparison.png")
]
for p in OUT_PATHS:
    p.parent.mkdir(parents=True, exist_ok=True)

# Data for MegaDepth-1500 (225 validation pairs)
methods = [
    "Fixed 500\n(11-Preset Avg)",
    "Fixed 1000\n(11-Preset Avg)",
    "Fixed 2000\n(11-Preset Avg)",
    "Fixed 4000\n(11-Preset Avg)",
    "Zero-Shot Level 2\n(Keypoint Budgeting)",
    "Full AFM-RL\n(Level 1 + Level 2)"
]

# Pose Recall <= 20 deg (%)
recall_20 = [34.13, 44.61, 53.71, 62.08, 35.39, 87.11]
avg_kp = [500.0, 1000.0, 2000.0, 4000.0, 1180.2, 1151.1]

colors = [
    "#aec7e8",  # Light blue
    "#ffbb78",  # Light orange
    "#ff7f0e",  # Orange
    "#d62728",  # Red
    "#1f77b4",  # Blue (Level 2 alone)
    "#2ca02c"   # Rich Green highlight for Full AFM-RL
]

fig, ax = plt.subplots(figsize=(10, 5.8), dpi=300)

bars = ax.bar(
    methods,
    recall_20,
    color=colors,
    width=0.55,
    edgecolor="#333333",
    linewidth=1.2,
    zorder=3
)

ax.set_ylabel(r"Pose Recall $\leq 20^\circ$ (%)", fontsize=12, fontweight="bold")
ax.set_title("MegaDepth-1500 Relative Pose Estimation — Zero-Shot Generalization", 
             fontsize=13, fontweight="bold", pad=15)
ax.set_ylim(0, 102)
ax.grid(axis="y", linestyle="--", alpha=0.5, zorder=0)

# Add value and keypoint annotations above each bar
for i, bar in enumerate(bars):
    h = bar.get_height()
    fontweight = "bold" if i == 5 else "normal"
    fontsize = 11 if i == 5 else 9.5
    
    # Recall percentage
    ax.text(
        bar.get_x() + bar.get_width() / 2.0,
        h + 1.5,
        f"{h:.1f}%",
        ha="center",
        va="bottom",
        fontsize=fontsize,
        fontweight=fontweight,
        color="#1b5e20" if i == 5 else "#2c3e50"
    )
    
    # Keypoint count badge inside/below
    ax.text(
        bar.get_x() + bar.get_width() / 2.0,
        h / 2.0,
        f"{int(avg_kp[i])} KP",
        ha="center",
        va="center",
        fontsize=9,
        fontweight="bold",
        color="white" if i in [2, 3, 4, 5] else "#333333"
    )

# Annotation callout highlighting Full AFM-RL advantage
ax.annotate(
    "Full AFM-RL: +25.0% vs. Fixed-4000\nwith 71% fewer keypoints (1,151 KP)",
    xy=(5, 87.11),
    xytext=(2.6, 92),
    fontsize=9.5,
    fontweight="bold",
    color="#1b5e20",
    bbox=dict(boxstyle="round,pad=0.5", facecolor="#e8f8f5", edgecolor="#2ecc71", lw=1.5),
    arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=-0.15", color="#27ae60", lw=1.8)
)

plt.tight_layout()

for p in OUT_PATHS:
    fig.savefig(p, bbox_inches="tight")
    print(f"Saved {p}")

plt.close(fig)
