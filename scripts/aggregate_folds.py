import argparse
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd

# Paths
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "results" / "phase2b"))

import stats_tools

def auc(errors, tau, n=400):
    e = np.clip(np.nan_to_num(np.asarray(errors, dtype=float), nan=1e6, posinf=1e6), 0, tau)
    ts = np.linspace(0, tau, n)
    recall = np.array([(e < t).mean() for t in ts])
    return float((np.trapezoid(recall, ts) if hasattr(np, 'trapezoid') else np.trapz(recall, ts)) / tau)

def parse_args():
    parser = argparse.ArgumentParser(description="Aggregate fold benchmark CSV into tables, Pareto plots, and bootstrap CIs")
    parser.add_argument("--csv", type=str, default="results/phase3/folds_pairs.csv", help="Input per-pair evaluations CSV")
    parser.add_argument("--out-dir", type=str, default=None, help="Output directory for aggregated summaries")
    parser.add_argument("--label", type=str, default=None, help="Optional label prefix for output files (e.g. 'dev')")
    return parser.parse_args()

def aggregate(args):
    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"Input file not found: {args.csv}")

    out_dir = Path(args.out_dir) if args.out_dir else csv_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"{args.label}_" if args.label else ""

    print("=" * 70)
    print("AGGREGATING BENCHMARK EVALUATIONS FROM CSV")
    print(f"  Input CSV:   {csv_path}")
    print(f"  Output Dir:  {out_dir}")
    print("=" * 70)

    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} total rows across {df['fold'].nunique()} fold(s): {df['fold'].unique().tolist()}")

    # 1. Summary across all evaluated methods
    methods = df["method"].unique()
    summary_rows = []

    for m in sorted(methods):
        sub = df[df["method"] == m]
        # Overall pooled metrics
        succ3 = sub["success"].mean() * 100.0
        auc1 = auc(sub["corner_err"], 1.0) * 100.0
        auc3 = auc(sub["corner_err"], 3.0) * 100.0
        auc5 = auc(sub["corner_err"], 5.0) * 100.0
        final_kp = sub["final_kp"].mean()
        cum_kp = sub["cum_kp"].mean()
        steps = sub["steps"].mean()

        # Per-fold statistics (if multiple folds)
        fold_succs = [sub[sub["fold"] == f]["success"].mean() * 100.0 for f in sub["fold"].unique()]
        succ_std = float(np.std(fold_succs)) if len(fold_succs) > 1 else 0.0

        summary_rows.append({
            "Method": m,
            "Succ@3px": succ3,
            "Succ_Std": succ_std,
            "AUC@1px": auc1,
            "AUC@3px": auc3,
            "AUC@5px": auc5,
            "Final_KP": final_kp,
            "Cum_KP": cum_kp,
            "Mean_Steps": steps,
            "N_Pairs": len(sub),
        })

    df_summary = pd.DataFrame(summary_rows)
    out_summary = out_dir / f"{prefix}folds_summary_table.csv"
    df_summary.to_csv(out_summary, index=False)
    print(f"\nSaved summary table to: {out_summary}")
    print(df_summary.round(2).to_string(index=False))

    # 2. Pareto Plot Data
    pareto_methods = [m for m in methods if any(k in m for k in ["AFM-RL", "Fixed", "ALIKED", "XFeat", "Oracle", "SIFT+L2"])]
    df_pareto = df_summary[df_summary["Method"].isin(pareto_methods)][["Method", "Final_KP", "Cum_KP", "Succ@3px", "AUC@3px"]]
    out_pareto = out_dir / f"{prefix}pareto_plot_data.csv"
    df_pareto.to_csv(out_pareto, index=False)
    print(f"\nSaved Pareto dataset to: {out_pareto}")

    # 3. Bootstrap CIs and McNemar comparisons
    ref_method = "Full AFM-RL (Strict STOP)"
    if ref_method in methods:
        print("\nComputing sequence-level bootstrap CIs against reference:", ref_method)
        stat_rows = []
        compare_targets = ["SIFT Fixed-4000", "SIFT Fixed-2000", "SIFT+L2 (Strict STOP)", "Oracle Preset Selection"]

        for target in compare_targets:
            if target not in methods:
                continue
            sub_a = df[df["method"] == ref_method].sort_values(["fold", "seq", "k"]).reset_index(drop=True)
            sub_b = df[df["method"] == target].sort_values(["fold", "seq", "k"]).reset_index(drop=True)

            merged = pd.DataFrame({
                "seq": sub_a["seq"],
                "k": sub_a["k"],
                "err_a": sub_a["corner_err"],
                "err_b": sub_b["corner_err"],
                "succ_a": sub_a["success"],
                "succ_b": sub_b["success"],
            })

            mcn = stats_tools.exact_mcnemar_test(merged["succ_a"], merged["succ_b"])
            boot = stats_tools.paired_sequence_bootstrap(merged, n_resamples=10000, seed=42)

            stat_rows.append({
                "Comparison": f"{ref_method} vs {target}",
                "Obs_Succ_Diff_%": boot["obs_succ_diff"],
                "Succ_95%_CI": f"[{boot['succ_ci'][0]:+.2f}%, {boot['succ_ci'][1]:+.2f}%]",
                "Obs_AUC3_Diff_%": boot["obs_auc_diff"],
                "AUC3_95%_CI": f"[{boot['auc_ci'][0]:+.2f}%, {boot['auc_ci'][1]:+.2f}%]",
                "McNemar_b": mcn["b_discordant"],
                "McNemar_c": mcn["c_discordant"],
                "McNemar_p_val": mcn["p_value"],
            })

        df_stats = pd.DataFrame(stat_rows)
        out_stats = out_dir / f"{prefix}statistical_comparisons.csv"
        df_stats.to_csv(out_stats, index=False)
        print(f"\nSaved statistical comparisons to: {out_stats}")
        print(df_stats.to_string(index=False))

    return df_summary

if __name__ == "__main__":
    args = parse_args()
    aggregate(args)

