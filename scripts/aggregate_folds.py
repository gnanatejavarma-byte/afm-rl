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
    pareto_methods = [m for m in methods if any(k in m for k in ["AFM-RL", "Fixed", "ALIKED", "XFeat", "Oracle", "SIFT+L2", "L1 Preset"])]
    df_pareto = df_summary[df_summary["Method"].isin(pareto_methods)][["Method", "Final_KP", "Cum_KP", "Succ@3px", "AUC@3px"]]
    out_pareto = out_dir / f"{prefix}pareto_plot_data.csv"
    df_pareto.to_csv(out_pareto, index=False)
    print(f"\nSaved Pareto dataset to: {out_pareto}")

    # 3. Primary Comparisons P1 - P5
    ref_method = "Full AFM-RL (Strict STOP)"
    if ref_method in methods:
        print("\nComputing sequence-level bootstrap CIs against reference:", ref_method)
        stat_rows = []
        p_targets = [
            ("P1 (Best Fixed on Fold VAL)", "Best Fixed (Fold VAL)"),
            ("P2 (ALIKED-500)", "ALIKED-500"),
            ("P3 (SIFT-2000)", "SIFT Fixed-2000"),
            ("P4 (SIFT-4000)", "SIFT Fixed-4000"),
            ("P5 (Oracle Preset Selection)", "Oracle Preset Selection"),
        ]

        for label, target in p_targets:
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
                "Label": label,
                "Comparison": f"{ref_method} vs {target}",
                "Full_Succ_%": sub_a["success"].mean() * 100.0,
                "Base_Succ_%": sub_b["success"].mean() * 100.0,
                "Obs_Succ_Diff_%": boot["obs_succ_diff"],
                "Succ_95%_CI": f"[{boot['succ_ci'][0]:+.2f}%, {boot['succ_ci'][1]:+.2f}%]",
                "Full_AUC3_%": auc(sub_a["corner_err"], 3.0) * 100.0,
                "Base_AUC3_%": auc(sub_b["corner_err"], 3.0) * 100.0,
                "Obs_AUC3_Diff_%": boot["obs_auc_diff"],
                "AUC3_95%_CI": f"[{boot['auc_ci'][0]:+.2f}%, {boot['auc_ci'][1]:+.2f}%]",
                "Full_Final_KP": sub_a["final_kp"].mean(),
                "Base_Final_KP": sub_b["final_kp"].mean(),
                "McNemar_b": mcn["b_discordant"],
                "McNemar_c": mcn["c_discordant"],
                "McNemar_p_val": mcn["p_value"],
            })

        df_p = pd.DataFrame(stat_rows)
        out_p = out_dir / f"{prefix}primary_comparisons_P1_P5.csv"
        df_p.to_csv(out_p, index=False)
        print(f"\nSaved primary comparisons to: {out_p}")
        print(df_p.round(2).to_string(index=False))

    # 4. Ablation Table
    # Compute 14-Preset Average L2 per pair
    l2_methods = [m for m in methods if m.endswith("+L2") and not m.startswith("Full")]
    if len(l2_methods) > 0:
        sub_l2 = df[df["method"].isin(l2_methods)]
        g_l2 = sub_l2.groupby(["fold", "seq", "k"]).agg({
            "success": "mean", "final_kp": "mean", "cum_kp": "mean", "steps": "mean", "corner_err": "mean"
        }).reset_index()

        avg_l2_row = {
            "Method": "14-Preset Average + L2",
            "Succ@3px": g_l2["success"].mean() * 100.0,
            "Succ_Std": float(np.std([g_l2[g_l2['fold']==f]['success'].mean()*100.0 for f in g_l2['fold'].unique()])),
            "AUC@1px": auc(g_l2["corner_err"], 1.0) * 100.0,
            "AUC@3px": auc(g_l2["corner_err"], 3.0) * 100.0,
            "AUC@5px": auc(g_l2["corner_err"], 5.0) * 100.0,
            "Final_KP": g_l2["final_kp"].mean(),
            "Cum_KP": g_l2["cum_kp"].mean(),
            "Mean_Steps": g_l2["steps"].mean(),
            "N_Pairs": len(g_l2),
        }
    else:
        avg_l2_row = None

    ablation_names = [
        "L1 Preset + Fixed-500",
        "L1 Preset + Fixed-1000",
        "ALIKED+L2",
        "SIFT+L2 (Strict STOP)",
        "Full AFM-RL (Strict STOP)",
    ]
    df_ablation = df_summary[df_summary["Method"].isin(ablation_names)].copy()
    if avg_l2_row is not None:
        df_ablation = pd.concat([df_ablation, pd.DataFrame([avg_l2_row])], ignore_index=True)
    out_ablation = out_dir / f"{prefix}ablation_table.csv"
    df_ablation.to_csv(out_ablation, index=False)
    print(f"\nSaved ablation table to: {out_ablation}")
    print(df_ablation.round(2).to_string(index=False))

    # 5. Level 1 & Level 2 Behavioral Distribution Analysis
    sub_full = df[df["method"] == ref_method]
    if len(sub_full) > 0:
        print("\n" + "=" * 70)
        print("LEVEL 1 & LEVEL 2 BEHAVIORAL AUDIT")
        print("=" * 70)

        # L1 Preset distribution per fold and pooled
        l1_records = []
        folds = sorted(sub_full["fold"].unique().tolist())
        for f in folds + ["Pooled"]:
            f_sub = sub_full if f == "Pooled" else sub_full[sub_full["fold"] == f]
            counts = f_sub["preset"].value_counts()
            n_tot = len(f_sub)
            probs = counts / n_tot
            entropy = float(-np.sum(probs * np.log2(probs + 1e-12)))

            # Counts for top presets
            p_aliked = int(counts.get("ALIKED", 0))
            p_sift = int(counts.get("SIFT", 0))
            p_other = n_tot - (p_aliked + p_sift)

            l1_records.append({
                "Fold": str(f),
                "Total_Pairs": n_tot,
                "ALIKED_Count": p_aliked,
                "ALIKED_%": p_aliked / n_tot * 100.0,
                "SIFT_Count": p_sift,
                "SIFT_%": p_sift / n_tot * 100.0,
                "Other_Presets_Count": p_other,
                "Other_%": p_other / n_tot * 100.0,
                "Entropy_Bits": entropy,
            })

        df_l1_stats = pd.DataFrame(l1_records)
        out_l1 = out_dir / f"{prefix}level1_selection_stats.csv"
        df_l1_stats.to_csv(out_l1, index=False)
        print("Level 1 Selection Summary:")
        print(df_l1_stats.round(2).to_string(index=False))

        # L2 Action distribution per fold and pooled
        l2_records = []
        for f in folds + ["Pooled"]:
            f_sub = sub_full if f == "Pooled" else sub_full[sub_full["fold"] == f]
            n_tot = len(f_sub)
            mean_steps = f_sub["steps"].mean()

            # Parse action strings if available
            all_actions = []
            first_actions = []
            for _, r in f_sub.iterrows():
                if "actions_str" in r and pd.notna(r["actions_str"]) and str(r["actions_str"]).strip():
                    acts = [int(a) for a in str(r["actions_str"]).split(";") if a.strip()]
                    all_actions.extend(acts)
                if "first_action" in r and pd.notna(r["first_action"]):
                    first_actions.append(int(r["first_action"]))

            tot_acts = len(all_actions)
            # Action map: 0: +500, 1: +1000, 2: -500, 3: STOP
            act_counts = pd.Series(all_actions).value_counts()
            fa_counts = pd.Series(first_actions).value_counts()

            l2_records.append({
                "Fold": str(f),
                "Total_Pairs": n_tot,
                "Mean_Steps": mean_steps,
                "Total_Actions": tot_acts,
                "+500_Count": int(act_counts.get(0, 0)),
                "+500_%": float(act_counts.get(0, 0)) / max(1, tot_acts) * 100.0,
                "+1000_Count": int(act_counts.get(1, 0)),
                "+1000_%": float(act_counts.get(1, 0)) / max(1, tot_acts) * 100.0,
                "-500_Count": int(act_counts.get(2, 0)),
                "-500_%": float(act_counts.get(2, 0)) / max(1, tot_acts) * 100.0,
                "STOP_Count": int(act_counts.get(3, 0)),
                "STOP_%": float(act_counts.get(3, 0)) / max(1, tot_acts) * 100.0,
                "FirstAct_STOP_%": float(fa_counts.get(3, 0)) / max(1, n_tot) * 100.0,
            })

        df_l2_stats = pd.DataFrame(l2_records)
        out_l2 = out_dir / f"{prefix}level2_action_stats.csv"
        df_l2_stats.to_csv(out_l2, index=False)
        print("\nLevel 2 Action Summary:")
        print(df_l2_stats.round(2).to_string(index=False))

    return df_summary

if __name__ == "__main__":
    args = parse_args()
    aggregate(args)

