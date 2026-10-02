import numpy as np, pandas as pd

df = pd.read_csv("logs/accuracy_test.csv")
order = list(dict.fromkeys(df.preset))
K = 150
err = df.corner_err.replace([np.inf, -np.inf], np.nan).fillna(1e6)

cands = {
    "A precision":                df.precision,
    f"B min(correct,{K})/{K}":    np.minimum(df.n_correct, K) / K,
    "C homography success":       df.h_ok,
    "D exp(-corner_err/5)":       np.exp(-err / 5),
}
cost = df.N / 4000
lams = [0.05, 0.1, 0.2, 0.4]

for name, acc in cands.items():
    print(f"\n=== {name}: best N per preset (reward = acc - lambda*N/4000) ===")
    res = {}
    for lam in lams:
        t = (acc - lam * cost).groupby([df.preset, df.N]).mean().unstack()
        res[f"lam={lam}"] = t.idxmax(axis=1)
    out = pd.DataFrame(res).loc[order]
    print(out.to_string())
    print("share of cells at N=100:", round((out == 100).values.mean(), 2),
          "| at N=4000:", round((out == 4000).values.mean(), 2))

print("\nMean corner error (px, failures excluded) by N:")
ce = df.replace([np.inf, -np.inf], np.nan).groupby("N").corner_err.median()
print(ce.round(1).to_string())
print("Share of runs with no valid homography, by N:")
print((df.corner_err.isin([np.inf, -np.inf])).groupby(df.N).mean().round(2).to_string())