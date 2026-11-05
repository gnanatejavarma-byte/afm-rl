"""
plot_hierarchical_comparison.py

Generates publication-quality comparison figures for:
- Fixed Baselines (500, 1000, 2000, 4000)
- Level 2 Alone (Adaptive Keypoint Budgeting)
- Full Hierarchical AFM-RL (Level 1 Preset Selection + Level 2 Budgeting)

Outputs saved to:
  - plots/
  - experiments/final_report/plots/
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Directories
OUT_DIRS = [
    Path("plots"),
    Path("experiments/final_report/plots")
]
for d in OUT_DIRS:
    d.mkdir(parents=True, exist_ok=True)

# Styling
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]
plt.rcParams["axes.edgecolor"] = "#cccccc"
plt.rcParams["axes.linewidth"] = 1.0

# Colors
C_F500   = "#aec7e8"
C_F1000  = "#ffbb78"
C_F2000  = "#ff7f0e"
C_F4000  = "#d62728"
C_L2     = "#1f77b4"
C_AFM_RL = "#2ca02c"   # Rich green highlight for full AFM-RL

# Load actual tested AFM-RL Level 1+2 results
df_l1_test = pd.read_csv("logs/level1_results_test.csv")
afm_succ1 = (df_l1_test.best_corner_err < 1.0).mean() * 100
afm_succ3 = (df_l1_test.best_corner_err < 3.0).mean() * 100
afm_succ5 = (df_l1_test.best_corner_err < 5.0).mean() * 100
afm_kp    = df_l1_test.best_n_kp.mean()

# Benchmark data on held-out 85 test pairs
methods = [
    "Fixed 500",
    "Fixed 1000",
    "Fixed 2000",
    "Fixed 4000",
    "Level-2 Alone",
    "Full AFM-RL (L1+L2)"
]

colors = [C_F500, C_F1000, C_F2000, C_F4000, C_L2, C_AFM_RL]

avg_kp = [495.6, 979.5, 1897.0, 3424.4, 1191.7, afm_kp]
succ_1 = [22.57, 26.74, 29.73, 31.55, 31.55, afm_succ1]
succ_3 = [42.67, 48.88, 56.26, 56.79, 64.39, afm_succ3]
succ_5 = [54.01, 60.11, 66.20, 68.66, 78.18, afm_succ5]

# -----------------------------------------------------------------------------
# PLOT 1: Grouped Bar Chart of Success Rates Across Error Thresholds
# -----------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(11, 6), dpi=300)
x = np.arange(len(methods))
width = 0.24

b1 = ax.bar(x - width, succ_1, width, label="Success@1px", color="#9ecae1", edgecolor="#3182bd")
b2 = ax.bar(x,         succ_3, width, label="Success@3px (Standard)", color="#41b6c4", edgecolor="#225ea8")
b3 = ax.bar(x + width, succ_5, width, label="Success@5px", color="#253494", edgecolor="#081d58")

# Highlight full AFM-RL on the x-axis
ax.set_xticks(x)
labels = [
    "Fixed 500", "Fixed 1000", "Fixed 2000", "Fixed 4000",
    "Level 2 Alone\n(Keypoint Budgeting)",
    "Full AFM-RL\n(Level 1 + Level 2)"
]
ax.set_xticklabels(labels, fontsize=10.5, fontweight="semibold")

ax.set_ylabel("Homography Success Rate (%)", fontsize=12, fontweight="bold")
ax.set_title("HPatches Held-Out Test Split (85 Image Pairs)\nFull Hierarchical AFM-RL vs. Baselines", 
             fontsize=14, fontweight="bold", pad=15)
ax.set_ylim(0, 105)
ax.legend(frameon=True, fontsize=11, loc="upper left")

# Value annotations on top of the primary metric (Success@3px)
for i, rect in enumerate(b2):
    height = rect.get_height()
    fontweight = "bold" if i == 5 else "normal"
    fontsize = 10 if i == 5 else 9
    ax.annotate(f"{height:.1f}%",
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 4), textcoords="offset points",
                ha="center", va="bottom", fontsize=fontsize, fontweight=fontweight)

plt.tight_layout()
for d in OUT_DIRS:
    plt.savefig(d / "hierarchical_success_comparison.png")
plt.close()

# -----------------------------------------------------------------------------
# PLOT 2: Pareto Efficiency (Accuracy vs Keypoint Workload)
# -----------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 6.5), dpi=300)

for m, kp, acc, c in zip(methods, avg_kp, succ_3, colors):
    size = 280 if m.startswith("Full") else (200 if "Level-2" in m else 150)
    marker = "*" if m.startswith("Full") else ("D" if "Level-2" in m else "o")
    ax.scatter(kp, acc, s=size, color=c, marker=marker, edgecolors="black", linewidths=1.5, zorder=5)
    
    # Label placement
    offset_x, offset_y = 40, -2
    if m.startswith("Full"):
        offset_x, offset_y = -70, 4
    elif m == "Fixed 4000":
        offset_x, offset_y = -100, -6
    elif m == "Fixed 2000":
        offset_x, offset_y = 30, -5
    elif m == "Fixed 1000":
        offset_x, offset_y = 30, -6

    fontweight = "bold" if m.startswith("Full") else "normal"
    ax.annotate(f"{m}\n({acc:.1f}%, {kp:.0f} pts)",
                (kp, acc), xytext=(offset_x, offset_y), textcoords="offset points",
                fontsize=9.5, fontweight=fontweight,
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=c, alpha=0.85, lw=1.2))

# Optimal Direction Arrow
ax.annotate("Ideal Target:\nHigh Accuracy, Low Cost",
            xy=(600, 88), xytext=(850, 75),
            arrowprops=dict(facecolor="#2ca02c", shrink=0.08, width=2, headwidth=8),
            fontsize=10, fontweight="bold", color="#1b7837")

ax.set_xlabel("Average Keypoint Budget Allocated", fontsize=12, fontweight="bold")
ax.set_ylabel("Success Rate @ 3px (%)", fontsize=12, fontweight="bold")
ax.set_title("Pareto Efficiency: Homography Accuracy vs. Keypoint Cost\n(Lower Keypoints + Higher Accuracy = Better)",
             fontsize=13, fontweight="bold", pad=15)
ax.set_xlim(300, 3900)
ax.set_ylim(35, 95)

plt.tight_layout()
for d in OUT_DIRS:
    plt.savefig(d / "hierarchical_accuracy_vs_cost.png")
plt.close()

# -----------------------------------------------------------------------------
# PLOT 3: Performance Breakdown by Scene Challenge (Illum vs Viewpoint)
# -----------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9, 5.5), dpi=300)

illum_succ = (df_l1_test[df_l1_test.kind == "illumination"].best_corner_err < 3.0).mean() * 100
view_succ  = (df_l1_test[df_l1_test.kind == "viewpoint"].best_corner_err < 3.0).mean() * 100

categories = ["Illumination Variations\n(40 Test Pairs)", "Viewpoint Changes\n(45 Test Pairs)"]
l2_scene = [73.2, 56.5]   # Verified Level 2 alone breakdown from report
afm_scene = [illum_succ, view_succ]

x_cat = np.arange(len(categories))
w_cat = 0.32

r1 = ax.bar(x_cat - w_cat/2, l2_scene, w_cat, label="Level 2 Alone (Arbitrary Presets)", color=C_L2, edgecolor="black")
r2 = ax.bar(x_cat + w_cat/2, afm_scene, w_cat, label="Full AFM-RL (Level 1 + Level 2)", color=C_AFM_RL, edgecolor="black")

ax.set_xticks(x_cat)
ax.set_xticklabels(categories, fontsize=11, fontweight="semibold")
ax.set_ylabel("Success Rate @ 3px (%)", fontsize=12, fontweight="bold")
ax.set_title("Robustness Across Challenging Conditions (Test Split)", fontsize=13, fontweight="bold", pad=15)
ax.set_ylim(0, 110)
ax.legend(frameon=True, fontsize=10.5, loc="upper right")

for rect in list(r1) + list(r2):
    h = rect.get_height()
    ax.annotate(f"{h:.1f}%",
                xy=(rect.get_x() + rect.get_width() / 2, h),
                xytext=(0, 4), textcoords="offset points",
                ha="center", va="bottom", fontsize=10, fontweight="bold")

plt.tight_layout()
for d in OUT_DIRS:
    plt.savefig(d / "hierarchical_scene_breakdown.png")
plt.close()

print("=" * 60)
print("Plots generated successfully in:")
print("  - plots/")
print("  - experiments/final_report/plots/")
print("Generated:")
print("  1. hierarchical_success_comparison.png")
print("  2. hierarchical_accuracy_vs_cost.png")
print("  3. hierarchical_scene_breakdown.png")
print("=" * 60)

