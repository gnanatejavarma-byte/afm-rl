"""MegaDepth-1500 Smoke Test Evaluation Script.

Evaluates fixed-keypoint baselines (500, 1000, 2000, 4000) across all 11 presets
on a small subset of 10 MegaDepth pairs.
Saves results to experiments/megadepth/results/.
"""

from pathlib import Path
import sys
import time
import pandas as pd
import numpy as np
from tqdm import tqdm

# Ensure project src/ and experiment directories are on python path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
EXPERIMENT_DIR = Path(__file__).resolve().parent

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_DIR))

import pipeline_utils as pu
from dataset import MegaDepthDataset
from pose_metrics import evaluate_pose_from_matches, compute_auc, compute_success_rates


def run_smoke_test(n_pairs: int = 10, budgets: list = [500, 1000, 2000, 4000]):
    print("=" * 75)
    print("MegaDepth-1500 Smoke Test (Fixed Keypoint Baselines)")
    print("=" * 75)
    print(f"Project root    : {PROJECT_ROOT}")
    print(f"Pairs to test   : {n_pairs}")
    print(f"Presets ({len(pu.PRESET_NAMES)}) : {', '.join(pu.PRESET_NAMES)}")
    print(f"Budgets         : {budgets}\n")

    dataset = MegaDepthDataset()
    total_available = len(dataset)
    print(f"Loaded dataset with {total_available} total calibrated pairs.")
    test_indices = list(range(min(n_pairs, total_available)))

    results_dir = EXPERIMENT_DIR / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    start_time = time.time()

    for p_idx in tqdm(test_indices, desc="Evaluating pairs"):
        item = dataset.load_images(p_idx, grayscale=True)
        img1, img2 = item["img1"], item["img2"]
        K1, K2 = item["K1"], item["K2"]
        R_gt, T_gt = item["R_gt"], item["T_gt"]
        key1 = (f"megadepth_{item['img1_rel']}", 1)
        key2 = (f"megadepth_{item['img2_rel']}", 2)

        for preset in pu.PRESET_NAMES:
            for budget in budgets:
                # 1. Run feature extraction and matching
                r = pu.run_pipeline(
                    img1, img2, preset=preset, n_kp=budget,
                    matcher="BF", ratio=0.75,
                    key1=key1, key2=key2
                )
                pts1, pts2 = r["pts1"], r["pts2"]
                effective_n_kp = r["n_kp"]

                # 2. Evaluate relative camera pose
                m = evaluate_pose_from_matches(pts1, pts2, K1, K2, R_gt, T_gt, threshold_px=1.5)

                rows.append({
                    "pair_idx": p_idx,
                    "img1": item["img1_rel"],
                    "img2": item["img2_rel"],
                    "preset": preset,
                    "budget": budget,
                    "effective_n_kp": effective_n_kp,
                    "n_matches": m["n_matches"],
                    "inliers": m["inliers"],
                    "inlier_ratio": m["inlier_ratio"],
                    "err_R_deg": m["err_R"],
                    "err_t_deg": m["err_t"],
                    "err_pose_deg": m["err_pose"],
                    "success_5": m["success_5"],
                    "success_10": m["success_10"],
                    "success_20": m["success_20"],
                })

    df = pd.DataFrame(rows)
    elapsed = time.time() - start_time
    print(f"\nCompleted {len(df)} runs in {elapsed:.2f} seconds.")

    # Save detailed CSV
    csv_path = results_dir / "smoke_test_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved raw results to: {csv_path}")

    # Generate Summary Table
    print("\n" + "=" * 85)
    print(f"{'PRESET':<12} {'BUDGET':<8} {'MATCHES':<9} {'VALID POSE':<12} {'MEAN ERR(deg)':<14} {'@5deg (%)':<10} {'@10deg (%)':<11} {'@20deg (%)':<10}")
    print("-" * 85)

    summary_rows = []
    for preset in pu.PRESET_NAMES:
        for budget in budgets:
            sub = df[(df["preset"] == preset) & (df["budget"] == budget)]
            finite_errs = sub[np.isfinite(sub["err_pose_deg"])]["err_pose_deg"]
            valid_count = len(finite_errs)
            mean_err = finite_errs.mean() if valid_count > 0 else float("nan")

            prec = compute_success_rates(sub["err_pose_deg"].values.tolist())
            aucs = compute_auc(sub["err_pose_deg"].values.tolist())

            print(f"{preset:<12} {budget:<8} {sub['n_matches'].mean():<9.1f} {f'{valid_count}/{n_pairs}':<12} "
                  f"{f'{mean_err:.2f}' if not np.isnan(mean_err) else 'N/A':<14} "
                  f"{prec['prec@5']:<10.1f} {prec['prec@10']:<11.1f} {prec['prec@20']:<10.1f}")


            summary_rows.append({
                "preset": preset,
                "budget": budget,
                "mean_matches": round(sub["n_matches"].mean(), 1),
                "valid_pose_ratio": f"{valid_count}/{n_pairs}",
                "mean_err_deg": round(mean_err, 2) if not np.isnan(mean_err) else None,
                "prec@5": prec["prec@5"],
                "prec@10": prec["prec@10"],
                "prec@20": prec["prec@20"],
                "auc@5": aucs["auc@5"],
                "auc@10": aucs["auc@10"],
                "auc@20": aucs["auc@20"],
            })

    print("=" * 85)

    summary_df = pd.DataFrame(summary_rows)
    summary_path = results_dir / "smoke_test_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"Saved summary table to: {summary_path}")


if __name__ == "__main__":
    run_smoke_test(n_pairs=10, budgets=[500, 1000, 2000, 4000])
