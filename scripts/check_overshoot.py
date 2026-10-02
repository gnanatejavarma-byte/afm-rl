import numpy as np, pandas as pd

df = pd.read_csv("logs/accuracy_test.csv")
LAMBDA = 0.1
cost = df.N / 4000
utility = df.reward_acc if "reward_acc" in df.columns else np.exp(-df.corner_err.replace([np.inf,-np.inf], np.nan).fillna(1e6)/5)
df["utility"] = utility - LAMBDA * cost

order = list(dict.fromkeys(df.preset))
table = df.groupby(["preset", "N"]).utility.mean().unstack()[sorted(df.N.unique())]
print(table.round(3).loc[order].to_string())

print("\nDoes utility ever DROP from one N to a higher N? (preset: drop size, from->to)")
for preset in order:
    row = table.loc[preset]
    drops = [(row.index[i], row.index[i+1], row.iloc[i]-row.iloc[i+1])
             for i in range(len(row)-1) if row.iloc[i+1] < row.iloc[i] - 0.01]
    if drops:
        print(f"  {preset}: {drops}")
    else:
        print(f"  {preset}: no meaningful drop (monotonic or flat)")