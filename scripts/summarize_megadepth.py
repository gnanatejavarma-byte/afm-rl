import sys, os, glob
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import megadepth as md

D = sys.argv[1] if len(sys.argv) > 1 else "logs/megadepth"
df = pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(D, "eval_*.csv"))], ignore_index=True)
for c in ("fixed_err", "best_err", "final_err"):
    df[c] = df[c].astype(float)
print("pairs per preset:\n", df.groupby("preset").size().to_string(), "\n")

M = {"fixed": ("fixed_err", "fixed_kp"), "rl_best": ("best_err", "best_kp"), "rl_final": ("final_err", "final_kp")}
rows = []
for preset, g in df.groupby("preset"):
    for name, (ec, kc) in M.items():
        a = md.pose_auc(g[ec].values)
        rows.append(dict(preset=preset, method=name, auc5=a[0]*100, auc10=a[1]*100,
                         auc20=a[2]*100, mean_kp=g[kc].mean()))
res = pd.DataFrame(rows)
res.to_csv(os.path.join(D, "auc_summary.csv"), index=False)

fixed_n = df.groupby("preset").fixed_n.first()
kp = res.pivot(index="preset", columns="method", values="mean_kp")
for t in (5, 10, 20):
    p = res.pivot(index="preset", columns="method", values=f"auc{t}")
    out = pd.DataFrame({"FixedN": fixed_n, "Fixed": p.fixed, "RL_best": p.rl_best, "RL_final": p.rl_final,
                        "KP_fixed": kp.fixed, "KP_best": kp.rl_best, "KP_final": kp.rl_final,
                        "d_best": p.rl_best - p.fixed, "d_final": p.rl_final - p.fixed})
    out = out.sort_values("RL_best", ascending=False)
    out.loc["MEAN"] = out.mean(numeric_only=True)
    print(f"=== AUC@{t} deg (%) ===")
    print(out.round(1).to_string(), "\n")
    out.round(2).to_csv(os.path.join(D, f"table_auc{t}.csv"))

print("=== by overlap bucket (all presets pooled) ===")
for b, g in df.groupby("bucket"):
    a = {n: md.pose_auc(g[c[0]].values)[0] * 100 for n, c in M.items()}
    print(b, "pairs:", len(g), "| AUC@5 fixed/best/final:", *(f"{a[n]:.1f}" for n in M),
          "| mean kp fixed/best/final:", *(f"{g[c[1]].mean():.0f}" for c in M.values()))

os.makedirs("plots_megadepth", exist_ok=True)
fig, ax = plt.subplots(figsize=(8, 4.6))
mean = res.groupby("method")[["auc5", "auc10", "auc20"]].mean()
for k, (name, col) in enumerate([("fixed", "#9aa5b1"), ("rl_best", "#2f80ed"), ("rl_final", "#6fa8f5")]):
    v = mean.loc[name].values
    ax.bar(np.arange(3) + (k - 1) * 0.27, v, 0.27, label=name, color=col)
    for x, y in zip(np.arange(3) + (k - 1) * 0.27, v):
        ax.text(x, y + 0.5, f"{y:.1f}", ha="center", fontsize=9)
ax.set_xticks(range(3)); ax.set_xticklabels(["AUC@5°", "AUC@10°", "AUC@20°"])
ax.set_ylabel("Mean AUC over presets (%)"); ax.legend(); ax.grid(axis="y", ls="--", alpha=.4)
fig.tight_layout(); fig.savefig("plots_megadepth/auc_summary.png", dpi=150)

fig, ax = plt.subplots(figsize=(8, 4.6))
bk = df.groupby("bucket")[["fixed_kp", "best_kp", "final_kp"]].mean()
bk.plot.bar(ax=ax, color=["#9aa5b1", "#2f80ed", "#6fa8f5"], rot=0)
ax.set_xlabel("Pair overlap range (low = harder)"); ax.set_ylabel("Mean keypoints")
ax.grid(axis="y", ls="--", alpha=.4); fig.tight_layout()
fig.savefig("plots_megadepth/adaptivity_by_overlap.png", dpi=150)
print("saved plots_megadepth/")