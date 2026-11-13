# DEPRECATED: contains hard-coded results, do not use for paper numbers
import os
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

df = pd.read_csv("logs/ablation_study_val90.csv")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), dpi=300)

# Panel 1: Cost Penalty lambda vs Keypoints & Accuracy
lam_rows = [
    ("lambda = 0.0 (No Penalty)", 0.0, 1453.8, 62.22, 38.59),
    ("lambda = 0.05 (Low)", 0.05, 1426.0, 62.22, 38.59),
    ("lambda = 0.10 (Default)", 0.10, 1398.2, 62.22, 38.54),
    ("lambda = 0.20 (High)", 0.20, 1359.3, 63.33, 38.75),
]
lams = [r[1] for r in lam_rows]
kps = [r[2] for r in lam_rows]
succs = [r[3] for r in lam_rows]

color_kp = "#e74c3c"
color_acc = "#2980b9"

ax1.plot(lams, kps, marker="o", color=color_kp, linewidth=2.5, markersize=8, label="Keypoint Budget")
ax1.set_xlabel(r"Keypoint Cost Penalty $\lambda$", fontsize=12, fontweight="bold")
ax1.set_ylabel("Keypoints Allocated", fontsize=12, fontweight="bold", color=color_kp)
ax1.set_title(r"Ablation 1: Impact of Cost Penalty $\lambda$ (Val Split)", fontsize=13, fontweight="bold")
ax1.grid(True, linestyle="--", alpha=0.5)
ax1.set_ylim(1300, 1500)

for x_val, y_val in zip(lams, kps):
    ax1.annotate(f"{y_val:.0f} KP", (x_val, y_val), textcoords="offset points", xytext=(0, 8),
                 ha="center", fontsize=9, fontweight="bold", color=color_kp)

# Panel 2: Level 1 Pipeline Selector Ablation
l1_labels = [
    "Uniform Random\nSelection",
    "Level 1 Bandit\n(PPO)",
    "Level 1 Linear\n(Ridge Regression)",
    "Fixed SIFT\n(No Selection)"
]
l1_succ = [60.00, 62.22, 66.67, 67.78]
l1_auc = [34.87, 38.54, 42.97, 42.94]
l1_kp = [1634.6, 1398.2, 784.8, 1436.8]

x = np.arange(len(l1_labels))
w = 0.35

rects1 = ax2.bar(x - w/2, l1_succ, w, label="Success @ 3px (%)", color="#1abc9c", edgecolor="black")
rects2 = ax2.bar(x + w/2, l1_auc, w, label="AUC @ 3px (%)", color="#34495e", edgecolor="black")

ax2.set_ylabel("Percentage (%)", fontsize=12, fontweight="bold")
ax2.set_title("Ablation 2: Level 1 Selector Architecture (Val Split)", fontsize=13, fontweight="bold")
ax2.set_xticks(x)
ax2.set_xticklabels(l1_labels, fontsize=9.5)
ax2.set_ylim(0, 85)
ax2.grid(axis="y", linestyle="--", alpha=0.5)
ax2.legend(loc="upper left", frameon=True)

for r in rects1:
    h = r.get_height()
    ax2.annotate(f"{h:.1f}%", (r.get_x() + r.get_width()/2, h), textcoords="offset points", xytext=(0, 3),
                 ha="center", va="bottom", fontsize=8.5, fontweight="bold")
for r in rects2:
    h = r.get_height()
    ax2.annotate(f"{h:.1f}%", (r.get_x() + r.get_width()/2, h), textcoords="offset points", xytext=(0, 3),
                 ha="center", va="bottom", fontsize=8.5, fontweight="bold")

plt.tight_layout()
os.makedirs("reports/plots", exist_ok=True)
p = "reports/plots/ablation_study_analysis.png"
plt.savefig(p, bbox_inches="tight")
plt.close()
print(f"Saved {p}")

