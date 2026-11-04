import numpy as np, pandas as pd

def auc(errors, tau, n=400):
    e = np.clip(np.asarray(errors, dtype=float), 0, tau)  # failures capped, so they earn no credit
    ts = np.linspace(0, tau, n)
    recall = np.array([(e < t).mean() for t in ts])
    return np.trapezoid(recall, ts) / tau

df = pd.read_csv("logs/baseline_comparison_val.csv")
df["corner_err"] = df.corner_err.replace([np.inf, -np.inf], 1e6).fillna(1e6)

rows = []
for (preset, method), g in df.groupby(["preset", "method"]):
    rows.append({"preset": preset, "method": method,
                 **{f"AUC@{t}": auc(g.corner_err, t) for t in (1, 3, 5)}})
res = pd.DataFrame(rows)
table = res.pivot(index="preset", columns="method")
print(table.round(3).to_string())
print("\nMean over all 14 presets:")
print(res.groupby("method")[["AUC@1", "AUC@3", "AUC@5"]].mean().round(3).to_string())
res.to_csv("logs/auc_comparison_val.csv", index=False)