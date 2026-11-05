import os
import sys
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from stable_baselines3 import PPO

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from level1_env import Level1Env
from hpatches import load_split, list_pairs
import pipeline_utils as pu

def evaluate_level1(model_path="checkpoints/level1_best/best_model.zip", split="val"):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}. Train Level 1 first!")

    print(f"Loading Level 1 policy from {model_path}...")
    l1_model = PPO.load(model_path)

    env = Level1Env(split=split, seed=999)
    pairs = list_pairs(load_split()[split])

    print(f"Evaluating Level 1 on {len(pairs)} pairs ({split} split)...")
    records = []
    preset_counts = {}

    for seq, k in tqdm(pairs, desc=f"Level 1 ({split})"):
        kind = "illumination" if seq.startswith("i_") else "viewpoint"
        obs, _ = env.reset_to(seq, k)

        # 1. Level 1 Agent decision
        action, _ = l1_model.predict(obs, deterministic=True)
        chosen_preset = pu.PRESET_NAMES[int(action)]
        preset_counts[chosen_preset] = preset_counts.get(chosen_preset, 0) + 1

        _, reward, _, _, info = env.step(int(action))

        records.append({
            "seq": seq,
            "k": k,
            "kind": kind,
            "level1_preset": chosen_preset,
            "best_utility": info["best_utility"],
            "best_n_kp": info["best_n_kp"],
            "best_h_ok": info["best_h_ok"],
            "best_corner_err": info["best_corner_err"],
        })

    df = pd.DataFrame(records)
    df["corner_err_capped"] = df["best_corner_err"].replace([np.inf, -np.inf], 50.0)

    os.makedirs("logs", exist_ok=True)
    out_csv = f"logs/level1_results_{split}.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nSaved detailed evaluation to {out_csv}\n")

    print("=" * 60)
    print("Level 1 Pipeline Selection Distribution:")
    for p, count in sorted(preset_counts.items(), key=lambda x: -x[1]):
        pct = (count / len(pairs)) * 100
        print(f"  - {p:15s}: {count:3d} pairs ({pct:5.1f}%)")
    print("=" * 60)

    print("\nLevel 1 Performance Summary:")
    print(f"  Average Utility:      {df['best_utility'].mean():.4f}")
    print(f"  Success Rate (h_ok):  {df['best_h_ok'].mean() * 100:.2f}%")
    print(f"  Average Keypoints:    {df['best_n_kp'].mean():.1f}")
    print(f"  Mean Corner Error:    {df['corner_err_capped'].mean():.2f} px")

    print("\nPerformance by scene type (illumination vs viewpoint):")
    summary = df.groupby("kind").agg(
        success_rate=("best_h_ok", lambda x: f"{x.mean()*100:.1f}%"),
        avg_keypoints=("best_n_kp", "mean"),
        avg_corner_err=("corner_err_capped", "mean"),
        count=("seq", "count")
    )
    print(summary.to_string())
    print("=" * 60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="checkpoints/level1_best/best_model.zip")
    parser.add_argument("--split", type=str, default="val", choices=["val", "test", "train"])
    args = parser.parse_args()
    evaluate_level1(args.model, args.split)

