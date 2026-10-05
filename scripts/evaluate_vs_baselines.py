import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np, pandas as pd
from tqdm import tqdm
from stable_baselines3 import PPO

import pipeline_utils as pu
from metrics import evaluate_matches
from hpatches import load_split, load_pair
from level2_env import Level2Env

MODEL_PATH = "checkpoints/level2_deltaobs_best/best_model.zip"
SPLIT = "val"          # switch to "test" for the one-time final run

FIXED_N = {
    "ORB": 1000, "SIFT": 2000, "AKAZE": 1000, "BRISK": 2000,
    "FAST+BRIEF": 1000, "KAZE": 4000, "GFTT+BRIEF": 1000,
    "GFTT+SIFT": 1000, "STAR+BRIEF": 100, "ORB+FREAK": 2000,
    "FAST+FREAK": 1000,
}
MATCHER, RATIO = "BF", 0.75

model = PPO.load(MODEL_PATH)
seqs = load_split()[SPLIT]
pairs = [(s, k) for s in seqs for k in range(2, 7)]   # every pair, not a hand-picked sample

env = Level2Env(split=SPLIT, seed=123)

rows = []
for seq, k in tqdm(pairs, desc="pairs"):
    img1, img2, H = load_pair(seq, k)
    kind = "illumination" if seq.startswith("i_") else "viewpoint"

    for preset in pu.PRESET_NAMES:
        # ---- Without RL: one fixed-N run ----
        n_fixed = FIXED_N[preset]
        r = pu.run_pipeline(img1, img2, preset, n_fixed, matcher=MATCHER, ratio=RATIO,
                             key1=(seq, 1), key2=(seq, k))
        m = evaluate_matches(r["pts1"], r["pts2"], H, img1.shape)
        rows.append(dict(seq=seq, k=k, kind=kind, preset=preset, method="fixed",
                          n_kp=r["n_kp"], corner_err=m["corner_err"], h_ok=m["h_ok"]))

        # ---- With RL: full episode, take the scoreboard's best ----
        obs, info = env.reset_to(seq, k, preset)
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(int(action))
            if terminated or truncated:
                break
        rows.append(dict(seq=seq, k=k, kind=kind, preset=preset, method="rl",
                          n_kp=info["best_n_kp"], corner_err=info["best_corner_err"],
                          h_ok=info["best_h_ok"]))

df = pd.DataFrame(rows)
df["corner_err_capped"] = df.corner_err.replace([np.inf, -np.inf], 50.0)  # cap failures for averaging
os.makedirs("logs", exist_ok=True)
out_path = f"logs/baseline_comparison_{SPLIT}.csv"
df.to_csv(out_path, index=False)
print(f"\nSaved {out_path} ({len(df)} rows)\n")

pd.set_option("display.width", 200)

print("=== Overall, per preset ===")
summary = df.groupby(["preset", "method"]).agg(
    mean_corner_err=("corner_err_capped", "mean"),
    success_rate=("h_ok", "mean"),
    mean_n_kp=("n_kp", "mean"),
).round(3)
print(summary.to_string())

print("\n=== By pair type (illumination vs viewpoint) ===")
summary2 = df.groupby(["preset", "kind", "method"]).agg(
    mean_corner_err=("corner_err_capped", "mean"),
    success_rate=("h_ok", "mean"),
    mean_n_kp=("n_kp", "mean"),
).round(3)
print(summary2.to_string())

print("\n=== Headline: RL wins vs fixed, per preset (success rate delta, keypoint delta) ===")
piv_err = df.pivot_table(index="preset", columns="method", values="h_ok", aggfunc="mean")
piv_n = df.pivot_table(index="preset", columns="method", values="n_kp", aggfunc="mean")
headline = pd.DataFrame({
    "success_fixed": piv_err["fixed"], "success_rl": piv_err["rl"],
    "success_delta": (piv_err["rl"] - piv_err["fixed"]).round(3),
    "n_fixed": piv_n["fixed"], "n_rl": piv_n["rl"].round(0),
})
print(headline.round(3).to_string())