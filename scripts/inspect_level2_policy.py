import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from stable_baselines3 import PPO
from level2_env import Level2Env
from hpatches import load_split

MODEL_PATH = "checkpoints/level2_deltaobs_best/best_model.zip"

model = PPO.load(MODEL_PATH)
val = load_split()["val"]

seq_i = [s for s in val if s.startswith("i_")][:2]
seq_v = [s for s in val if s.startswith("v_")][:2]
test_seqs = seq_i + seq_v
presets_to_try = ["SIFT", "ORB", "AKAZE", "STAR+BRIEF"]

env = Level2Env(split="val", seed=0)

print(f"{'seq':12s} {'k':>2s} {'preset':11s} {'steps':>5s} {'raw_final_N':>11s} {'best_N':>7s} "
      f"{'best_err':>9s} {'best_prec':>9s} {'best_h_ok':>5s}")
for seq in test_seqs:
    for k in (3, 5):
        for preset in presets_to_try:
            obs, info = env.reset_to(seq, k, preset)
            steps = 0
            while True:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(int(action))
                steps += 1
                if terminated or truncated:
                    break
            ce = info["best_corner_err"]
            ce_str = f"{ce:9.2f}" if ce != float("inf") else f"{'inf':>9s}"
            print(f"{seq:12s} {k:>2d} {preset:11s} {steps:5d} {info['n_kp']:11d} "
                  f"{info['best_n_kp']:7d} {ce_str} {info['best_precision']:9.2f} "
                  f"{info['best_h_ok']:5.0f}")