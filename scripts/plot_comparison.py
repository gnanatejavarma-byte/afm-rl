import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("logs/baseline_comparison_val.csv")
os.makedirs("plots", exist_ok=True)

PRESETS = sorted(df.preset.unique())
COLOR_FIXED = "#9aa5b1"   # grey -- "without RL"
COLOR_RL = "#2f80ed"      # blue -- "with RL"

def grouped_bar(values_fixed, values_rl, title, ylabel, filename, fmt="{:.2f}"):
    x = np.arange(len(PRESETS))
    w = 0.38
    fig, ax = plt.subplots(figsize=(11, 5))
    b1 = ax.bar(x - w/2, values_fixed, w, label="Without RL (fixed N)", color=COLOR_FIXED)
    b2 = ax.bar(x + w/2, values_rl, w, label="With RL (adaptive)", color=COLOR_RL)
    ax.set_xticks(x)
    ax.set_xticklabels(PRESETS, rotation=35, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    for bars in (b1, b2):
        for bar in bars:
            h = bar.get_height()
            ax.annotate(fmt.format(h), (bar.get_x() + bar.get_width()/2, h),
                        textcoords="offset points", xytext=(0, 3), ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(f"plots/{filename}", dpi=150)
    plt.close(fig)
    print(f"Saved plots/{filename}")

# ---------- Metric 1: Success rate (h_ok), all pairs ----------
succ = df.groupby(["preset", "method"]).h_ok.mean().unstack().reindex(PRESETS)
grouped_bar(succ["fixed"]*100, succ["rl"]*100,
            "Match Success Rate by Preset", "Success rate (%)",
            "1_success_rate.png", fmt="{:.0f}%")

# ---------- Metric 2: Accuracy among successful pairs (median corner error) ----------
ok = df[df.h_ok == 1]
err = ok.groupby(["preset", "method"]).corner_err.median().unstack().reindex(PRESETS)
grouped_bar(err["fixed"], err["rl"],
            "Accuracy on Successful Matches (lower = better)", "Median corner error (px)",
            "2_accuracy_on_success.png", fmt="{:.2f}")

# ---------- Metric 3: Mean keypoints used (computation cost) ----------
cost = df.groupby(["preset", "method"]).n_kp.mean().unstack().reindex(PRESETS)
grouped_bar(cost["fixed"], cost["rl"],
            "Average Keypoints Used by Preset", "Mean keypoints used",
            "3_keypoints_used.png", fmt="{:.0f}")

# ---------- Metric 4: Success rate by pair type (illumination vs viewpoint) ----------
fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
for ax, kind, title in zip(axes, ["illumination", "viewpoint"], ["Illumination pairs", "Viewpoint pairs"]):
    sub = df[df.kind == kind].groupby(["preset", "method"]).h_ok.mean().unstack().reindex(PRESETS)
    x = np.arange(len(PRESETS))
    w = 0.38
    ax.bar(x - w/2, sub["fixed"]*100, w, label="Without RL", color=COLOR_FIXED)
    ax.bar(x + w/2, sub["rl"]*100, w, label="With RL", color=COLOR_RL)
    ax.set_xticks(x)
    ax.set_xticklabels(PRESETS, rotation=35, ha="right")
    ax.set_title(title)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
axes[0].set_ylabel("Success rate (%)")
axes[0].legend()
fig.suptitle("Success Rate by Pair Difficulty Type")
fig.tight_layout()
fig.savefig("plots/4_success_by_pair_type.png", dpi=150)
plt.close(fig)
print("Saved plots/4_success_by_pair_type.png")

# ---------- Metric 5 (the adaptivity proof): RL's OWN keypoint choice, easy vs hard ----------
rl = df[df.method == "rl"]
adapt = rl.groupby(["preset", "kind"]).n_kp.mean().unstack().reindex(PRESETS)
fig, ax = plt.subplots(figsize=(11, 5))
x = np.arange(len(PRESETS))
w = 0.38
b1 = ax.bar(x - w/2, adapt["illumination"], w, label="Illumination pairs (usually easier)", color="#27ae60")
b2 = ax.bar(x + w/2, adapt["viewpoint"], w, label="Viewpoint pairs (usually harder)", color="#eb5757")
ax.set_xticks(x)
ax.set_xticklabels(PRESETS, rotation=35, ha="right")
ax.set_ylabel("Mean keypoints chosen by the RL agent")
ax.set_title("Proof of Adaptivity: RL Spends More Keypoints Only Where Needed")
ax.legend()
ax.grid(axis="y", linestyle="--", alpha=0.4)
for bars in (b1, b2):
    for bar in bars:
        h = bar.get_height()
        ax.annotate(f"{h:.0f}", (bar.get_x() + bar.get_width()/2, h),
                    textcoords="offset points", xytext=(0, 3), ha="center", fontsize=8)
fig.tight_layout()
fig.savefig("plots/5_adaptivity_proof.png", dpi=150)
plt.close(fig)
print("Saved plots/5_adaptivity_proof.png")

# ---------- Bonus: accuracy vs cost scatter, the single "does RL sit on a better frontier" plot ----------
fig, ax = plt.subplots(figsize=(8, 6))
for method, color, label, marker in [("fixed", COLOR_FIXED, "Without RL", "s"),
                                      ("rl", COLOR_RL, "With RL", "o")]:
    sub = df[df.method == method].groupby("preset").agg(
        n_kp=("n_kp", "mean"), success=("h_ok", "mean")).reset_index()
    ax.scatter(sub.n_kp, sub.success * 100, c=color, label=label, s=90,
               marker=marker, edgecolors="black", linewidths=0.5)
    for _, row in sub.iterrows():
        ax.annotate(row.preset, (row.n_kp, row.success * 100), fontsize=7,
                    xytext=(4, 4), textcoords="offset points")
ax.set_xlabel("Mean keypoints used (computation cost)")
ax.set_ylabel("Success rate (%)")
ax.set_title("Accuracy vs. Computation Cost, per Preset")
ax.legend()
ax.grid(linestyle="--", alpha=0.4)
fig.tight_layout()
fig.savefig("plots/6_accuracy_vs_cost_scatter.png", dpi=150)
plt.close(fig)
print("Saved plots/6_accuracy_vs_cost_scatter.png")

print("\nAll plots saved in the 'plots' folder.")