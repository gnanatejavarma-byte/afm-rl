import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Load honest summary results
df = pd.read_csv("logs/honest_summary_table.csv")

# Filter key comparison methods
focus_methods = [
    "Fixed-4000 (11 Classical Avg)",
    "SIFT + Fixed-4000",
    "ALIKED + Fixed-500 (Learned)",
    "Full AFM-RL (L1 + L2, Strict STOP)",
    "Oracle Preset Selection (Upper Bound)"
]
df_f = df[df["Method"].isin(focus_methods)].copy()
order_dict = {m: i for i, m in enumerate(focus_methods)}
df_f["order"] = df_f["Method"].map(order_dict)
df_f = df_f.sort_values("order").reset_index(drop=True)

short_names = [
    "Fixed-4000\n(11-Preset Avg)",
    "SIFT\n(Fixed-4000)",
    "ALIKED\n(Fixed-500)",
    "Full AFM-RL\n(Label-Free)",
    "Oracle Bound\n(Upper Bound)"
]

colors = ["#7f8c8d", "#3498db", "#9b59b6", "#2ecc71", "#e67e22"]

# Figure 1: Accuracy & Workload Comparison (2-panel)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), dpi=300)

x = np.arange(len(short_names))
w = 0.35

# Panel 1: Geometric Alignment Performance
rects1 = ax1.bar(x - w/2, df_f["Success@3px"], w, label="Success @ 3px (%)", color="#1abc9c", edgecolor="black", alpha=0.9)
rects2 = ax1.bar(x + w/2, df_f["AUC@3px"], w, label="AUC @ 3px (%)", color="#2980b9", edgecolor="black", alpha=0.9)

ax1.set_ylabel("Percentage (%)", fontsize=12, fontweight="bold")
ax1.set_title("Geometric Alignment Accuracy (HPatches)", fontsize=13, fontweight="bold")
ax1.set_xticks(x)
ax1.set_xticklabels(short_names, fontsize=10)
ax1.set_ylim(0, 100)
ax1.legend(loc="upper left", frameon=True)
ax1.grid(axis="y", linestyle="--", alpha=0.4)

# Add values above bars
for r in rects1:
    h = r.get_height()
    ax1.annotate(f"{h:.1f}%", xy=(r.get_x() + r.get_width()/2, h), xytext=(0, 3),
                 textcoords="offset points", ha="center", va="bottom", fontsize=8, fontweight="bold")
for r in rects2:
    h = r.get_height()
    ax1.annotate(f"{h:.1f}%", xy=(r.get_x() + r.get_width()/2, h), xytext=(0, 3),
                 textcoords="offset points", ha="center", va="bottom", fontsize=8, fontweight="bold")

# Panel 2: Workload & Runtime Comparison
rects3 = ax2.bar(x - w/2, df_f["Keypoints"], w, label="Final Keypoints", color="#e74c3c", edgecolor="black", alpha=0.85)
ax2_twin = ax2.twinx()
rects4 = ax2_twin.bar(x + w/2, df_f["Time_ms"], w, label="Latency (ms)", color="#f39c12", edgecolor="black", alpha=0.85)

ax2.set_ylabel("Keypoint Count", fontsize=12, fontweight="bold", color="#c0392b")
ax2_twin.set_ylabel("Latency (ms / pair)", fontsize=12, fontweight="bold", color="#d35400")
ax2.set_title("Computational Cost & Runtime (Lower is Better)", fontsize=13, fontweight="bold")
ax2.set_xticks(x)
ax2.set_xticklabels(short_names, fontsize=10)
ax2.set_ylim(0, 4200)
ax2_twin.set_ylim(0, 50)
ax2.grid(axis="y", linestyle="--", alpha=0.4)

# Combine legends for twin axis
lines1, labels1 = ax2.get_legend_handles_labels()
lines2, labels2 = ax2_twin.get_legend_handles_labels()
ax2.legend(lines1 + lines2, labels1 + labels2, loc="upper right", frameon=True)

# Annotations for workload
for r in rects3:
    h = r.get_height()
    ax2.annotate(f"{int(h)}", xy=(r.get_x() + r.get_width()/2, h), xytext=(0, 3),
                 textcoords="offset points", ha="center", va="bottom", fontsize=8, fontweight="bold")
for r in rects4:
    h = r.get_height()
    ax2_twin.annotate(f"{h:.1f}ms", xy=(r.get_x() + r.get_width()/2, h), xytext=(0, 3),
                      textcoords="offset points", ha="center", va="bottom", fontsize=8, fontweight="bold")

plt.tight_layout()
os.makedirs("reports/plots", exist_ok=True)
p1 = "reports/plots/honest_accuracy_and_workload.png"
plt.savefig(p1, bbox_inches="tight")
plt.close()
print(f"Saved {p1}")

# Figure 2: Pareto Efficiency Frontier (Accuracy vs Keypoints & Latency)
plt.figure(figsize=(9, 6), dpi=300)
for idx, row in df.iterrows():
    m = row["Method"]
    if "Oracle" in m:
        c, marker, s = "#e67e22", "*", 250
    elif "AFM-RL" in m:
        c, marker, s = "#2ecc71", "o", 200
    elif "SIFT" in m:
        c, marker, s = "#3498db", "s", 160
    elif "ALIKED" in m or "XFeat" in m:
        c, marker, s = "#9b59b6", "^", 160
    else:
        c, marker, s = "#95a5a6", "D", 120

    plt.scatter(row["Keypoints"], row["Success@3px"], color=c, marker=marker, s=s, edgecolor="black", alpha=0.9, zorder=4)
    # Label placement
    offset_x = 40
    offset_y = 0.5
    if "AFM-RL (L1 + L2, Strict" in m:
        name_str = "Full AFM-RL (Label-Free)\n[1,165 kp, 21 ms]"
        offset_y = -3.2
    elif "SIFT + Fixed-4000" in m:
        name_str = "SIFT-4000 [3,499 kp, 37 ms]"
        offset_y = 1.0
    elif "SIFT + Fixed-2000" in m:
        name_str = "SIFT-2000 [1,971 kp, 132 ms]"
        offset_y = -2.0
    elif "11 Classical Avg" in m:
        name_str = "11-Preset Avg [3,424 kp, 33 ms]"
        offset_y = -2.0
    elif "ALIKED" in m:
        name_str = "ALIKED-500 [495 kp, 42 ms]"
        offset_y = 1.0
    elif "Oracle" in m:
        name_str = "Oracle Upper Bound"
        offset_y = 1.0
    else:
        continue

    plt.annotate(name_str, (row["Keypoints"], row["Success@3px"]),
                 xytext=(row["Keypoints"] + offset_x, row["Success@3px"] + offset_y),
                 fontsize=9, fontweight="bold",
                 arrowprops=dict(arrowstyle="->", color="#34495e", lw=1))

plt.xlabel("Keypoint Budget per Image", fontsize=12, fontweight="bold")
plt.ylabel("Homography Success @ 3px (%)", fontsize=12, fontweight="bold")
plt.title("Pareto Efficiency Frontier on HPatches (Label-Free Inference)", fontsize=13, fontweight="bold")
plt.grid(True, linestyle="--", alpha=0.5)
plt.xlim(200, 4200)
plt.ylim(50, 90)

p2 = "reports/plots/pareto_efficiency_frontier.png"
plt.savefig(p2, bbox_inches="tight")
plt.close()
print(f"Saved {p2}")

