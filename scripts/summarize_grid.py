import sys, os, glob
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np, pandas as pd
import megadepth as md

D = sys.argv[1] if len(sys.argv) > 1 else "logs/megadepth"
g = pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(D, "grid_*.csv")) if "_vs_rl" not in f])
e = pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(D, "eval_*.csv"))])
GRID = (500, 1000, 2000, 4000)
for ti, t in enumerate((5, 10, 20)):
    rows = []
    for p, gg in g.groupby("preset"):
        a = {n: md.pose_auc(gg[f"err_{n}"].values)[ti] * 100 for n in GRID}
        bn = max(a, key=a.get)
        ee = e[e.preset == p]
        rl = md.pose_auc(ee.final_err.values)[ti] * 100
        rows.append(dict(preset=p, **{f"fix{n}": a[n] for n in GRID}, bestN=bn, best_fixed=a[bn],
                         kp_fixed=gg[f"kp_{bn}"].mean(), rl_best=md.pose_auc(ee.best_err.values)[ti] * 100,
                         rl_final=rl, kp_rl=ee.final_kp.mean(), delta=rl - a[bn],
                         oracle=md.pose_auc(gg[[f"err_{n}" for n in GRID]].min(axis=1).values)[ti] * 100))
    r = pd.DataFrame(rows).set_index("preset").astype(float).sort_values("rl_final", ascending=False)
    r.loc["MEAN"] = r.mean(numeric_only=True)
    print(f"=== AUC@{t} deg (%): best fixed N per preset vs agent ===")
    print(r.round(1).to_string(), "\n")
    r.round(2).to_csv(os.path.join(D, f"grid_vs_rl_auc{t}.csv"))