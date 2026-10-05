import pandas as pd
import numpy as np

df = pd.read_csv("logs/baseline_comparison_val.csv")
pd.set_option("display.width", 200)

# Only compare pixel accuracy where matching actually SUCCEEDED -- a "how
# wrong was the failure" number is noise, not signal, since a degenerate
# RANSAC fit on bad matches can land anywhere.
ok = df[df.h_ok == 1]

print("=== Accuracy among SUCCESSFUL pairs only (median corner_err, px) ===")
acc_summary = ok.groupby(["preset", "method"]).corner_err.median().unstack().round(3)
print(acc_summary.to_string())

print("\n=== Success rate, ALL pairs (the headline number) ===")
succ_summary = df.groupby(["preset", "method"]).h_ok.mean().unstack().round(3)
print(succ_summary.to_string())

print("\n=== Mean keypoints used, ALL pairs ===")
n_summary = df.groupby(["preset", "method"]).n_kp.mean().unstack().round(1)
print(n_summary.to_string())

print("\n=== Combined headline table ===")
headline = pd.DataFrame({
    "success_fixed": succ_summary["fixed"], "success_rl": succ_summary["rl"],
    "success_gain": (succ_summary["rl"] - succ_summary["fixed"]).round(3),
    "median_err_fixed_ok": acc_summary["fixed"], "median_err_rl_ok": acc_summary["rl"],
    "n_fixed": n_summary["fixed"].round(0), "n_rl": n_summary["rl"].round(0),
})
print(headline.round(3).to_string())