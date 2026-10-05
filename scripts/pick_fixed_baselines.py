import numpy as np, pandas as pd

df = pd.read_csv("logs/accuracy_test.csv")
LAMBDA = 0.1
cost = df.N / 4000
err = df.corner_err.replace([np.inf, -np.inf], np.nan).fillna(1e6)
utility = np.exp(-err / 5) - LAMBDA * cost
df["utility"] = utility

table = df.groupby(["preset", "N"]).utility.mean().unstack()
best_n = table.idxmax(axis=1)

print("Best fixed N per preset (by mean utility across our Step 4 test pairs):\n")
print(best_n.to_string())

print("\nReport-mandated baselines override the above for these two:")
print("  SIFT -> 2000 (per report Section 8)")
print("  ORB  -> 1000 (per report Section 8)")