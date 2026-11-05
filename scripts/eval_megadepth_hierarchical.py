"""
eval_megadepth_hierarchical.py

Independent, isolated evaluation of the full two-level AFM-RL pipeline 
(Level 1 Preset Selection + Level 2 Adaptive Keypoints) on MegaDepth.

- Evaluates on the exact 225-pair validation split (from split_manifest.csv).
- Does not modify any existing code or overwrite any existing logs.
- Outputs results to: logs/megadepth_hierarchical_val225.csv
"""

import sys
import os
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
from stable_baselines3 import PPO

# Paths
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pipeline_utils as pu
import megadepth as md
from megadepth_env import MegaDepthEnv, pair_stats

L1_MODEL_PATH = "checkpoints/level1_best/best_model.zip"
L2_MODEL_PATH = "checkpoints/level2_14preset_best/best_model.zip"
MANIFEST_PATH = "experiments/megadepth_training/split_manifest.csv"
OUT_CSV       = "logs/megadepth_hierarchical_val225.csv"

def run_evaluation(max_pairs=None):
    print("=" * 65)
    print("MegaDepth Hierarchical Zero-Shot Evaluation (225 Validation Pairs)")
    print("=" * 65)

    if not os.path.exists(L1_MODEL_PATH):
        raise FileNotFoundError(f"Level 1 model not found: {L1_MODEL_PATH}")
    if not os.path.exists(L2_MODEL_PATH):
        raise FileNotFoundError(f"Level 2 model not found: {L2_MODEL_PATH}")

    print(f"Loading Level 1 policy: {L1_MODEL_PATH}")
    l1_model = PPO.load(L1_MODEL_PATH, device="cpu")
    print(f"Loading Level 2 policy: {L2_MODEL_PATH}")
    l2_model = PPO.load(L2_MODEL_PATH, device="cpu")

    # Load 225 validation pairs from the official split manifest
    print(f"Loading split manifest: {MANIFEST_PATH}")
    df_manifest = pd.read_csv(MANIFEST_PATH)
    val_indices = df_manifest[df_manifest.split == "val"].pair_idx.tolist()
    all_pairs = md.list_pairs()
    val_pairs = [all_pairs[idx] for idx in val_indices]

    if max_pairs:
        val_pairs = val_pairs[:max_pairs]
    print(f"Total pairs to evaluate: {len(val_pairs)}")

    env = MegaDepthEnv()
    records = []
    preset_counts = {}

    for idx, pair in enumerate(tqdm(val_pairs, desc="MegaDepth Val (225 pairs)")):
        fname, i, j = pair
        bucket = md.bucket(fname)

        try:
            # 1. Level 1: Extract 9 visual stats & choose pipeline preset
            stats = pair_stats(fname, i, j)
            action, _ = l1_model.predict(stats, deterministic=True)
            chosen_preset = pu.PRESET_NAMES[int(action)]
            preset_counts[chosen_preset] = preset_counts.get(chosen_preset, 0) + 1

            # 2. Level 2: Adaptive keypoint rollout on MegaDepth
            obs, _ = env.reset_to(pair, chosen_preset)
            steps = 0
            while True:
                a, _ = l2_model.predict(obs, deterministic=True)
                obs, _, term, trunc, info = env.step(int(a))
                steps += 1
                if term or trunc:
                    break

            pose_err = float(info["best_corner_err"])   # holds pose error in degrees
            best_kp  = int(info["best_n_kp"])
            prec     = float(info.get("best_precision", 0.0))

            records.append({
                "pair_idx": val_indices[idx],
                "file": fname,
                "i": i,
                "j": j,
                "bucket": bucket,
                "chosen_preset": chosen_preset,
                "best_n_kp": best_kp,
                "pose_err_deg": pose_err,
                "precision": prec,
                "steps": steps,
                "pose_ok_5deg": float(pose_err < 5.0),
                "pose_ok_10deg": float(pose_err < 10.0),
                "pose_ok_20deg": float(pose_err < 20.0),
            })
        except Exception as e:
            print(f"\nWarning: Error evaluating pair {pair}: {e}")
            records.append({
                "pair_idx": val_indices[idx],
                "file": fname,
                "i": i,
                "j": j,
                "bucket": bucket,
                "chosen_preset": "ERROR",
                "best_n_kp": 0,
                "pose_err_deg": np.inf,
                "precision": 0.0,
                "steps": 0,
                "pose_ok_5deg": 0.0,
                "pose_ok_10deg": 0.0,
                "pose_ok_20deg": 0.0,
            })

    # Save to dedicated output file
    os.makedirs("logs", exist_ok=True)
    df = pd.DataFrame(records)
    df.to_csv(OUT_CSV, index=False)
    print(f"\nSaved results to: {OUT_CSV}")

    # Compute AUC metrics
    errors = df["pose_err_deg"].values
    auc5, auc10, auc20 = md.pose_auc(errors, thresholds=(5, 10, 20))

    print("\n" + "=" * 65)
    print("Level 1 Preset Selection Distribution on MegaDepth:")
    for p, c in sorted(preset_counts.items(), key=lambda x: -x[1]):
        pct = (c / len(val_pairs)) * 100
        print(f"  - {p:15s}: {c:3d} pairs ({pct:5.1f}%)")

    print("\n" + "=" * 65)
    print("Hierarchical AFM-RL (Level 1 + Level 2) MegaDepth Performance:")
    print(f"  Pose AUC@5°:       {auc5 * 100:.2f}%")
    print(f"  Pose AUC@10°:      {auc10 * 100:.2f}%")
    print(f"  Pose AUC@20°:      {auc20 * 100:.2f}%")
    print(f"  Pose Accuracy@5°:  {df['pose_ok_5deg'].mean() * 100:.2f}%")
    print(f"  Pose Accuracy@10°: {df['pose_ok_10deg'].mean() * 100:.2f}%")
    print(f"  Pose Accuracy@20°: {df['pose_ok_20deg'].mean() * 100:.2f}%")
    print(f"  Average Keypoints: {df['best_n_kp'].mean():.1f} pts")
    print(f"  Average Steps:     {df['steps'].mean():.2f}")
    print("=" * 65)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-pairs", type=int, default=None, help="Cap pairs for testing (default: all 225)")
    args = parser.parse_args()
    run_evaluation(args.max_pairs)

