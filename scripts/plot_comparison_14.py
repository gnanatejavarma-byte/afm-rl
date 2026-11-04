import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("logs/baseline_comparison_val.csv")
OUT = "plots_14preset"
os.makedirs(OUT, exist_ok=True)

PRESETS = sorted(df.preset.unique())
COLOR_FIXED = "#9aa5b1"   # grey -- without RL
COLOR_RL = "#2f80ed"      # blue -- with RL
trapz = getattr(np, "trapezoid", None) or np.trapz


def legend_top(ax, top):
    ax.set_ylim(0, top * 1.22)          # headroom so the legend never covers bars
    ax.legend(loc="upper left", ncol=2, fontsize=9)


def grouped_bar(vf, vr, title, ylabel, filename, fmt="{:.2f}"):
    x = np.arange(len(PRESETS)); w = 0.38
    fig, ax = plt.subplots(figsize=(14, 5.5))
    b1 = ax.bar(x - w/2, vf, w, label="Without RL (fixed N)", color=COLOR_FIXED)
    b2 = ax.bar(x + w/2, vr, w, label="With RL (adaptive)", color=COLOR_RL)
    ax.set_xticks(x); ax.set_xticklabels(PRESETS, rotation=35, ha="right")
    ax.set_ylabel(ylabel); ax.set_title(title)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    for bars in (b1, b2):
        for bar in bars:
            h = bar.get_height()
            ax.annotate(fmt.format(h), (bar.get_x() + bar.get_width()/2, h),
                        textcoords="offset points", xytext=(0, 3), ha="center", fontsize=7.5)
    legend_top(ax, max(np.max(vf), np.max(vr)))
    fig.tight_layout(); fig.savefig(f"{OUT}/{filename}", dpi=150); plt.close(fig)
    print("Saved", f"{OUT}/{filename}")


# 1. Success rate
succ = df.groupby(["preset", "method"]).h_ok.mean().unstack().reindex(PRESETS)
grouped_bar(succ["fixed"]*100, succ["rl"]*100, "Match Success Rate by Preset",
            "Success rate (%)", "1_success_rate.png", fmt="{:.0f}%")

# 2. Precision among each method's own successes
ok = df[df.h_ok == 1]
err = ok.groupby(["preset", "method"]).corner_err.median().unstack().reindex(PRESETS)
grouped_bar(err["fixed"], err["rl"],
            "Precision Among Each Method's Own Successes (lower = better)",
            "Median corner error (px), successful pairs only", "2_precision_on_success.png")

# 3. Keypoints used
cost = df.groupby(["preset", "method"]).n_kp.mean().unstack().reindex(PRESETS)
grouped_bar(cost["fixed"], cost["rl"], "Average Keypoints Used by Preset",
            "Mean keypoints used", "3_keypoints_used.png", fmt="{:.0f}")

# 4. Success by pair type
fig, axes = plt.subplots(1, 2, figsize=(16, 5.5), sharey=True)
for ax, kind, title in zip(axes, ["illumination", "viewpoint"],
                           ["Illumination pairs", "Viewpoint pairs"]):
    sub = df[df.kind == kind].groupby(["preset", "method"]).h_ok.mean().unstack().reindex(PRESETS)
    x = np.arange(len(PRESETS)); w = 0.38
    ax.bar(x - w/2, sub["fixed"]*100, w, label="Without RL", color=COLOR_FIXED)
    ax.bar(x + w/2, sub["rl"]*100, w, label="With RL", color=COLOR_RL)
    ax.set_xticks(x); ax.set_xticklabels(PRESETS, rotation=40, ha="right")
    ax.set_title(title); ax.set_ylim(0, 118)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
axes[0].set_ylabel("Success rate (%)")
axes[0].legend(loc="upper left", ncol=2, fontsize=9)
fig.suptitle("Success Rate by Pair Difficulty Type")
fig.tight_layout(); fig.savefig(f"{OUT}/4_success_by_pair_type.png", dpi=150); plt.close(fig)
print("Saved", f"{OUT}/4_success_by_pair_type.png")

# 5. Adaptivity proof
rl = df[df.method == "rl"]
adapt = rl.groupby(["preset", "kind"]).n_kp.mean().unstack().reindex(PRESETS)
fig, ax = plt.subplots(figsize=(14, 5.5))
x = np.arange(len(PRESETS)); w = 0.38
b1 = ax.bar(x - w/2, adapt["illumination"], w, color="#27ae60",
            label="Illumination pairs (usually easier)")
b2 = ax.bar(x + w/2, adapt["viewpoint"], w, color="#eb5757",
            label="Viewpoint pairs (usually harder)")
ax.set_xticks(x); ax.set_xticklabels(PRESETS, rotation=35, ha="right")
ax.set_ylabel("Mean keypoints chosen by the RL agent")
ax.set_title("Proof of Adaptivity: RL Spends More Keypoints Only Where Needed")
ax.grid(axis="y", linestyle="--", alpha=0.4)
for bars in (b1, b2):
    for bar in bars:
        h = bar.get_height()
        ax.annotate(f"{h:.0f}", (bar.get_x() + bar.get_width()/2, h),
                    textcoords="offset points", xytext=(0, 3), ha="center", fontsize=7.5)
legend_top(ax, adapt.values.max())
fig.tight_layout(); fig.savefig(f"{OUT}/5_adaptivity_proof.png", dpi=150); plt.close(fig)
print("Saved", f"{OUT}/5_adaptivity_proof.png")

# 6. Accuracy vs cost scatter
fig, ax = plt.subplots(figsize=(10, 7))
for method, color, label, marker in [("fixed", COLOR_FIXED, "Without RL", "s"),
                                     ("rl", COLOR_RL, "With RL", "o")]:
    sub = df[df.method == method].groupby("preset").agg(
        n_kp=("n_kp", "mean"), success=("h_ok", "mean")).reset_index()
    ax.scatter(sub.n_kp, sub.success*100, c=color, label=label, s=90,
               marker=marker, edgecolors="black", linewidths=0.5)
    for _, r in sub.iterrows():
        ax.annotate(r.preset, (r.n_kp, r.success*100), fontsize=7,
                    xytext=(4, 4), textcoords="offset points")
ax.set_xlabel("Mean keypoints used (computation cost)")
ax.set_ylabel("Success rate (%)")
ax.set_title("Success Rate vs. Computation Cost, per Preset")
ax.legend(); ax.grid(linestyle="--", alpha=0.4)
fig.tight_layout(); fig.savefig(f"{OUT}/6_accuracy_vs_cost_scatter.png", dpi=150); plt.close(fig)
print("Saved", f"{OUT}/6_accuracy_vs_cost_scatter.png")


# 7. AUC@1, @3, @5 (RIPE-style metric)
def auc(errors, tau, n=400):
    e = np.clip(np.nan_to_num(np.asarray(errors, dtype=float), nan=1e6, posinf=1e6), 0, tau)
    ts = np.linspace(0, tau, n)
    recall = np.array([(e < t).mean() for t in ts])
    return trapz(recall, ts) / tau

rows = []
for (preset, method), g in df.groupby(["preset", "method"]):
    rows.append({"preset": preset, "method": method,
                 **{f"AUC@{t}": auc(g.corner_err, t) for t in (1, 3, 5)}})
res = pd.DataFrame(rows)
res.to_csv("logs/auc_comparison_val.csv", index=False)

fig, axes = plt.subplots(1, 3, figsize=(20, 5.5))
for ax, t in zip(axes, (1, 3, 5)):
    p = res.pivot(index="preset", columns="method", values=f"AUC@{t}").reindex(PRESETS)
    x = np.arange(len(PRESETS)); w = 0.38
    ax.bar(x - w/2, p["fixed"], w, label="Without RL", color=COLOR_FIXED)
    ax.bar(x + w/2, p["rl"], w, label="With RL", color=COLOR_RL)
    ax.set_xticks(x); ax.set_xticklabels(PRESETS, rotation=45, ha="right", fontsize=8)
    ax.set_title(f"AUC@{t}px"); ax.set_ylim(0, 1.0)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
axes[0].set_ylabel("AUC (higher = better)")
axes[0].legend(loc="upper left", ncol=2, fontsize=9)
fig.suptitle("Homography Estimation AUC at 1, 3 and 5 pixel thresholds")
fig.tight_layout(); fig.savefig(f"{OUT}/7_auc.png", dpi=150); plt.close(fig)
print("Saved", f"{OUT}/7_auc.png")

print("\nMean AUC over all presets:")
print(res.groupby("method")[["AUC@1", "AUC@3", "AUC@5"]].mean().round(3).to_string())