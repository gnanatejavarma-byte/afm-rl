"""MegaDepth-1500 Full Fixed-Baseline Evaluation.

Evaluates fixed-keypoint budgets (500, 1000, 2000, 4000) across all 11 presets
on all 1500 MegaDepth calibrated pairs.

Total configurations: 1500 pairs × 11 presets × 4 budgets = 66,000 evaluations.

CACHING STRATEGY:
  The existing pipeline_utils.py has a two-level cache:
    - Memory:  POOL_CACHE_SIZE=128 (image, preset) entries
    - Disk:    data/pool_cache/<preset>/<key>.npz
  Feature-pool construction (detect + describe) is budget-independent.
  A pool is built once at up to 5000 kp, then sliced to n_kp at match time.

  This evaluator exploits that: for each (pair, preset) the pool is built once
  (first call to run_pipeline or get_pool), then all 4 budgets reuse the same
  cached pool with only a cheap slice + match step.

Usage:
    # Quick validation (5 pairs):
    python full_fixed_eval.py --validate

    # Full run (1500 pairs):
    python full_fixed_eval.py

    # Full run, resume from a partial CSV:
    python full_fixed_eval.py --resume
"""

import argparse
import time
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm

# ---- Path setup ----
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR      = PROJECT_ROOT / "src"
EXP_DIR      = Path(__file__).resolve().parent

for p in [str(SRC_DIR), str(EXP_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

import pipeline_utils as pu
from dataset import MegaDepthDataset
from pose_metrics import evaluate_pose_from_matches, compute_auc, compute_success_rates

# ---- Constants ----
BUDGETS  = [500, 1000, 2000, 4000]
MATCHER  = "BF"
RATIO    = 0.75


def _run_one(img1, img2, preset, n_kp, key1, key2, K1, K2, R_gt, T_gt):
    """Run pipeline at one budget and return a flat result dict."""
    r = pu.run_pipeline(img1, img2, preset=preset, n_kp=n_kp,
                        matcher=MATCHER, ratio=RATIO, key1=key1, key2=key2)
    m = evaluate_pose_from_matches(r["pts1"], r["pts2"], K1, K2, R_gt, T_gt)
    return dict(
        effective_n_kp = r["n_kp"],
        n_matches      = m["n_matches"],
        inliers        = m["inliers"],
        inlier_ratio   = m["inlier_ratio"],
        err_R_deg      = m["err_R"],
        err_t_deg      = m["err_t"],
        err_pose_deg   = m["err_pose"],
        success_5      = m["success_5"],
        success_10     = m["success_10"],
        success_20     = m["success_20"],
    )


def run_evaluation(n_pairs: int | None = None,
                   results_dir: Path | None = None,
                   resume: bool = False) -> pd.DataFrame:
    """
    Main evaluation loop.

    Args:
        n_pairs:     Number of pairs to evaluate. None = all 1500.
        results_dir: Where to save CSVs.
        resume:      If True and a partial full_fixed_results.csv exists, skip
                     already-completed (pair_idx, preset, budget) triples.
    """
    dataset = MegaDepthDataset()
    total_available = len(dataset)
    n_pairs = min(n_pairs, total_available) if n_pairs is not None else total_available

    if results_dir is None:
        results_dir = EXP_DIR / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    out_csv     = results_dir / "full_fixed_results.csv"
    summary_csv = results_dir / "full_fixed_summary.csv"

    # ---- Resume: load already-done rows ----
    done_keys: set = set()
    existing_rows: list[dict] = []
    if resume and out_csv.exists():
        existing_df = pd.read_csv(out_csv)
        existing_rows = existing_df.to_dict("records")
        done_keys = {
            (r["pair_idx"], r["preset"], r["budget"])
            for r in existing_rows
        }
        print(f"[Resume] Loaded {len(existing_rows)} existing rows "
              f"({len(done_keys)} (pair,preset,budget) triples already done).")

    rows: list[dict] = list(existing_rows)
    failed_pairs: list[int] = []
    t0 = time.time()

    for p_idx in tqdm(range(n_pairs), desc="Pairs", unit="pair",
                      dynamic_ncols=True, smoothing=0.05):
        try:
            item = dataset.load_images(p_idx, grayscale=True)
        except Exception as exc:
            print(f"\n[WARN] pair {p_idx}: load_images failed: {exc}")
            failed_pairs.append(p_idx)
            continue

        img1, img2 = item["img1"], item["img2"]
        K1, K2     = item["K1"],   item["K2"]
        R_gt, T_gt = item["R_gt"], item["T_gt"]

        # Cache keys scoped to megadepth to avoid collision with hpatches pool cache
        key1 = (f"md_{item['img1_rel']}", 1)
        key2 = (f"md_{item['img2_rel']}", 2)

        for preset in pu.PRESET_NAMES:
            # Check which budgets are still needed
            needed = [b for b in BUDGETS if (p_idx, preset, b) not in done_keys]
            if not needed:
                continue  # all budgets already done for this (pair, preset)

            for budget in needed:
                try:
                    result = _run_one(img1, img2, preset, budget,
                                      key1, key2, K1, K2, R_gt, T_gt)
                    rows.append({
                        "pair_idx":       p_idx,
                        "img1":           item["img1_rel"],
                        "img2":           item["img2_rel"],
                        "preset":         preset,
                        "budget":         budget,
                        **result,
                    })
                    done_keys.add((p_idx, preset, budget))
                except Exception as exc:
                    print(f"\n[WARN] pair {p_idx} preset={preset} budget={budget}: {exc}")
                    # Record as failed entry so the pair is not silently skipped
                    rows.append({
                        "pair_idx": p_idx, "img1": item["img1_rel"],
                        "img2": item["img2_rel"], "preset": preset, "budget": budget,
                        "effective_n_kp": -1, "n_matches": 0, "inliers": 0,
                        "inlier_ratio": 0.0, "err_R_deg": float("nan"),
                        "err_t_deg": float("nan"), "err_pose_deg": float("nan"),
                        "success_5": 0, "success_10": 0, "success_20": 0,
                    })

        # Flush every 50 pairs to protect against interruption
        if (p_idx + 1) % 50 == 0:
            pd.DataFrame(rows).to_csv(out_csv, index=False)

    elapsed = time.time() - t0
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    print(f"\n[Done] {len(df)} rows | {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"       Failed pair loads: {len(failed_pairs)}")
    if failed_pairs:
        print(f"       Failed pair indices: {failed_pairs}")
    print(f"       Raw results -> {out_csv}")

    return df, elapsed, failed_pairs


def build_summary(df: pd.DataFrame, n_pairs_tested: int) -> pd.DataFrame:
    """Aggregate per (preset, budget)."""
    summary_rows = []
    for preset in pu.PRESET_NAMES:
        for budget in BUDGETS:
            sub = df[(df["preset"] == preset) & (df["budget"] == budget)]
            if sub.empty:
                continue

            errs = sub["err_pose_deg"].values.tolist()
            finite_errs = [e for e in errs if np.isfinite(e)]
            valid_n = len(finite_errs)
            mean_err   = float(np.mean(finite_errs))   if finite_errs else float("nan")
            median_err = float(np.median(finite_errs)) if finite_errs else float("nan")

            prec = compute_success_rates(errs)
            aucs = compute_auc(errs)

            summary_rows.append({
                "preset":          preset,
                "budget":          budget,
                "n_pairs":         len(sub),
                "valid_pose":      valid_n,
                "valid_ratio":     round(valid_n / len(sub), 4) if len(sub) else 0,
                "mean_err_deg":    round(mean_err, 3)   if not np.isnan(mean_err)   else None,
                "median_err_deg":  round(median_err, 3) if not np.isnan(median_err) else None,
                "prec@5":          prec["prec@5"],
                "prec@10":         prec["prec@10"],
                "prec@20":         prec["prec@20"],
                "auc@5":           aucs["auc@5"],
                "auc@10":          aucs["auc@10"],
                "auc@20":          aucs["auc@20"],
                "mean_matches":    round(sub["n_matches"].mean(), 2),
                "mean_inliers":    round(sub["inliers"].mean(), 2),
            })

    return pd.DataFrame(summary_rows)


def print_summary(summary_df: pd.DataFrame, n_pairs: int):
    print("\n" + "=" * 115)
    print(f"{'PRESET':<12} {'BUDGET':<8} {'VALID':<8} {'MEAN ERR':<11} "
          f"{'MED ERR':<10} {'@5%':<7} {'@10%':<7} {'@20%':<7} "
          f"{'AUC@5':<8} {'AUC@10':<8} {'AUC@20':<8} {'MATCHES':<10} {'INLIERS'}")
    print("-" * 115)

    for preset in pu.PRESET_NAMES:
        best_row = None
        for budget in BUDGETS:
            sub = summary_df[(summary_df["preset"] == preset) &
                             (summary_df["budget"] == budget)]
            if sub.empty:
                continue
            row = sub.iloc[0]
            me = row["mean_err_deg"]
            med = row["median_err_deg"]
            valid_label = f"{int(row['valid_pose'])}/{n_pairs}"
            me_str  = f"{me:.2f}"  if me  is not None else "N/A"
            med_str = f"{med:.2f}" if med is not None else "N/A"
            print(f"{preset:<12} {budget:<8} {valid_label:<8} "
                  f"{me_str:<11} {med_str:<10} "
                  f"{row['prec@5']:<7.1f} {row['prec@10']:<7.1f} {row['prec@20']:<7.1f} "
                  f"{row['auc@5']:<8.3f} {row['auc@10']:<8.3f} {row['auc@20']:<8.3f} "
                  f"{row['mean_matches']:<10.1f} {row['mean_inliers']:.1f}")

            if best_row is None or (row["prec@20"] > best_row["prec@20"]):
                best_row = row

        print()  # blank line between presets

    print("=" * 115)

    # Best-budget-per-preset summary
    print("\n--- Best Fixed Budget per Preset (@20° success rate) ---")
    print(f"{'PRESET':<14} {'BUDGET':<8} {'@5%':<8} {'@10%':<8} "
          f"{'@20%':<8} {'AUC@5':<8} {'AUC@10':<8} {'AUC@20':<8}")
    print("-" * 75)
    for preset in pu.PRESET_NAMES:
        sub = summary_df[summary_df["preset"] == preset].copy()
        if sub.empty:
            continue
        best = sub.loc[sub["prec@20"].idxmax()]
        print(f"{preset:<14} {int(best['budget']):<8} {best['prec@5']:<8.2f} "
              f"{best['prec@10']:<8.2f} {best['prec@20']:<8.2f} "
              f"{best['auc@5']:<8.4f} {best['auc@10']:<8.4f} {best['auc@20']:<8.4f}")


def main():
    parser = argparse.ArgumentParser(description="MegaDepth-1500 Full Fixed Baseline Eval")
    parser.add_argument("--validate", action="store_true",
                        help="Run validation on 5 pairs only, then exit.")
    parser.add_argument("--resume", action="store_true",
                        help="Skip rows already present in full_fixed_results.csv.")
    parser.add_argument("--n-pairs", type=int, default=None,
                        help="Evaluate only the first N pairs (default: all 1500).")
    args = parser.parse_args()

    results_dir = EXP_DIR / "results"

    if args.validate:
        print("=" * 75)
        print("VALIDATION RUN: 5 pairs × 11 presets × 4 budgets = 220 evaluations")
        print("=" * 75)
        df, elapsed, failed = run_evaluation(n_pairs=5, results_dir=results_dir,
                                             resume=False)
        # Save to a separate validation file so it doesn't pollute the full run
        val_path = results_dir / "validation_5pairs.csv"
        df.to_csv(val_path, index=False)
        print(f"\nValidation CSV -> {val_path}")
        summary_df = build_summary(df, n_pairs_tested=5)
        print_summary(summary_df, n_pairs=5)
        expected = 5 * 11 * 4
        if len(df) == expected:
            print(f"\n[PASS] Output has exactly {expected} rows — format matches smoke test.")
        else:
            print(f"\n[WARN] Expected {expected} rows but got {len(df)}.")
        return

    # ---- Full run ----
    n_pairs = args.n_pairs
    label   = f"{n_pairs}" if n_pairs else "1500"
    print("=" * 75)
    print(f"FULL FIXED-BASELINE EVALUATION: {label} pairs × 11 presets × 4 budgets")
    print("=" * 75)
    print(f"Caching: disk pool cache -> {PROJECT_ROOT / 'data' / 'pool_cache'}")
    print(f"         memory LRU size = {pu.POOL_CACHE_SIZE} pools")
    print(f"Budgets: {BUDGETS}")
    print(f"Presets: {', '.join(pu.PRESET_NAMES)}\n")

    df, elapsed, failed = run_evaluation(n_pairs=n_pairs,
                                         results_dir=results_dir,
                                         resume=args.resume)

    n_pairs_actual = df["pair_idx"].nunique()
    total_expected = n_pairs_actual * len(pu.PRESET_NAMES) * len(BUDGETS)
    print(f"\nUnique pairs evaluated : {n_pairs_actual}")
    print(f"Total rows             : {len(df)}  (expected {total_expected})")
    print(f"Failed pair loads      : {len(failed)}")

    summary_df = build_summary(df, n_pairs_tested=n_pairs_actual)
    summary_path = results_dir / "full_fixed_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"Summary CSV            -> {summary_path}")

    print_summary(summary_df, n_pairs=n_pairs_actual)


if __name__ == "__main__":
    main()
