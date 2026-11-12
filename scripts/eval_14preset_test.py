import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np, pandas as pd
from tqdm import tqdm
from stable_baselines3 import PPO

import pipeline_utils as pu
from metrics import evaluate_matches
from hpatches import load_split, load_pair
from level2_env import Level2Env

MODEL_PATH = "checkpoints/level2_14preset_best/best_model.zip"
SPLIT = "test"

FIXED_N = {
    "ORB": 1000, "SIFT": 2000, "AKAZE": 1000, "BRISK": 2000,
    "FAST+BRIEF": 1000, "KAZE": 4000, "GFTT+BRIEF": 1000,
    "GFTT+SIFT": 1000, "STAR+BRIEF": 100, "ORB+FREAK": 2000,
    "FAST+FREAK": 1000, "XFeat": 1000, "ALIKED": 500, "ORB+BEBLID": 4000,
}
MATCHER, RATIO = "BF", 0.75

model = PPO.load(MODEL_PATH)
seqs = load_split()[SPLIT]
pairs = [(s, k) for s in seqs for k in range(2, 7)]

env = Level2Env(split=SPLIT, seed=123)

rows = []
for seq, k in tqdm(pairs, desc="Test split pairs"):
    img1, img2, H = load_pair(seq, k)
    kind = "illumination" if seq.startswith("i_") else "viewpoint"

    for preset in pu.PRESET_NAMES:
        # Fixed
        n_fixed = FIXED_N[preset]
        r = pu.run_pipeline(img1, img2, preset, n_fixed, matcher=MATCHER, ratio=RATIO,
                             key1=(seq, 1), key2=(seq, k))
        m = evaluate_matches(r["pts1"], r["pts2"], H, img1.shape)
        rows.append(dict(seq=seq, k=k, kind=kind, preset=preset, method="fixed",
                          n_kp=r["n_kp"], corner_err=m["corner_err"], h_ok=m["h_ok"]))

        # RL
        obs, info = env.reset_to(seq, k, preset)
        steps = 0
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(int(action))
            steps += 1
            if terminated or truncated:
                break
        rows.append(dict(seq=seq, k=k, kind=kind, preset=preset, method="rl",
                          n_kp=info["best_n_kp"], corner_err=info["best_corner_err"],
                          h_ok=info["best_h_ok"], steps=steps))

df = pd.DataFrame(rows)
df["corner_err_capped"] = df.corner_err.replace([np.inf, -np.inf], 50.0)
os.makedirs("logs", exist_ok=True)
out_path = f"logs/baseline_comparison_{SPLIT}_14preset.csv"
df.to_csv(out_path, index=False)
print(f"Saved {out_path} ({len(df)} rows)")

def auc(errors, tau, n=400):
    e = np.clip(np.nan_to_num(np.asarray(errors, dtype=float), nan=1e6, posinf=1e6), 0, tau)
    ts = np.linspace(0, tau, n)
    recall = np.array([(e < t).mean() for t in ts])
    return (np.trapezoid(recall, ts) if hasattr(np, 'trapezoid') else np.trapz(recall, ts)) / tau

auc_rows = []
for (preset, method), g in df.groupby(["preset", "method"]):
    auc_rows.append({
        "preset": preset, "method": method,
        "n_kp": g["n_kp"].mean(),
        "AUC@1": auc(g.corner_err, 1),
        "AUC@3": auc(g.corner_err, 3),
        "AUC@5": auc(g.corner_err, 5),
    })
df_auc = pd.DataFrame(auc_rows)
df_auc.to_csv(f"logs/auc_comparison_{SPLIT}_14preset.csv", index=False)
print("Saved AUC results!")
print(df_auc.to_string())
