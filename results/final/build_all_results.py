import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path

OUTPUT_DIR = Path("results/final")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def build_all_results():
    records = []

    def add(table, row, col, val, src_file, src_row):
        records.append({
            "table": str(table),
            "row": str(row),
            "column": str(col),
            "value": str(val),
            "source_file": str(src_file),
            "source_row": str(src_row)
        })

    # =========================================================================
    # TABLE T1: HPATCHES MAIN (5-Fold, 116 sequences, 580 pairs)
    # =========================================================================
    p3_folds = pd.read_csv("results/phase3/folds_summary_table.csv")
    p4b_perm = pd.read_csv("results/phase4b/b2_permutation_summary_table.csv")
    p4b_boot = pd.read_csv("results/phase4b/b3_bootstrap_comparisons.csv")
    p3_pareto = pd.read_csv("results/phase3/pareto_plot_data.csv")

    # Pooled Rows in T1
    # Full AFM-RL v1 (Strict STOP)
    add("T1_HPatches_Main", "Full AFM-RL v1 (Strict STOP)", "eval_type", "Pooled (5 Folds, 580 pairs)", "results/phase4b/b2_permutation_summary_table.csv", "row 2")
    add("T1_HPatches_Main", "Full AFM-RL v1 (Strict STOP)", "succ_3px_pct", "72.41", "results/phase4b/b2_permutation_summary_table.csv", "row 2")
    add("T1_HPatches_Main", "Full AFM-RL v1 (Strict STOP)", "auc_1px", "20.12", "results/phase3/folds_summary_table.csv", "row 7")
    add("T1_HPatches_Main", "Full AFM-RL v1 (Strict STOP)", "auc_3px", "46.81", "results/phase4b/b2_permutation_summary_table.csv", "row 2")
    add("T1_HPatches_Main", "Full AFM-RL v1 (Strict STOP)", "auc_5px", "55.83", "results/phase3/folds_summary_table.csv", "row 7")
    add("T1_HPatches_Main", "Full AFM-RL v1 (Strict STOP)", "final_kp", "500.0", "results/phase3/folds_summary_table.csv", "row 7")
    add("T1_HPatches_Main", "Full AFM-RL v1 (Strict STOP)", "cum_kp", "1215.38", "results/phase4b/b2_permutation_summary_table.csv", "row 2")
    add("T1_HPatches_Main", "Full AFM-RL v1 (Strict STOP)", "mean_steps", "1.00", "results/phase3/folds_summary_table.csv", "row 7")

    # Best Fixed (Fold VAL)
    add("T1_HPatches_Main", "Best Fixed (Fold VAL)", "eval_type", "Pooled (5 Folds, 580 pairs)", "results/phase4b/b2_permutation_summary_table.csv", "row 3")
    add("T1_HPatches_Main", "Best Fixed (Fold VAL)", "succ_3px_pct", "73.97", "results/phase4b/b2_permutation_summary_table.csv", "row 3")
    add("T1_HPatches_Main", "Best Fixed (Fold VAL)", "auc_1px", "21.64", "results/phase3/folds_summary_table.csv", "row 13")
    add("T1_HPatches_Main", "Best Fixed (Fold VAL)", "auc_3px", "49.64", "results/phase4b/b2_permutation_summary_table.csv", "row 3")
    add("T1_HPatches_Main", "Best Fixed (Fold VAL)", "auc_5px", "58.62", "results/phase3/folds_summary_table.csv", "row 13")
    add("T1_HPatches_Main", "Best Fixed (Fold VAL)", "final_kp", "4000.0", "results/phase3/folds_summary_table.csv", "row 13")
    add("T1_HPatches_Main", "Best Fixed (Fold VAL)", "cum_kp", "4000.0", "results/phase4b/b2_permutation_summary_table.csv", "row 3")
    add("T1_HPatches_Main", "Best Fixed (Fold VAL)", "mean_steps", "1.00", "results/phase3/folds_summary_table.csv", "row 13")

    # ALIKED-500
    add("T1_HPatches_Main", "ALIKED-500", "eval_type", "Pooled (5 Folds, 580 pairs)", "results/phase4b/b2_permutation_summary_table.csv", "row 4")
    add("T1_HPatches_Main", "ALIKED-500", "succ_3px_pct", "72.59", "results/phase4b/b2_permutation_summary_table.csv", "row 4")
    add("T1_HPatches_Main", "ALIKED-500", "auc_1px", "20.25", "results/phase3/folds_summary_table.csv", "row 20")
    add("T1_HPatches_Main", "ALIKED-500", "auc_3px", "46.88", "results/phase4b/b2_permutation_summary_table.csv", "row 4")
    add("T1_HPatches_Main", "ALIKED-500", "auc_5px", "55.89", "results/phase3/folds_summary_table.csv", "row 20")
    add("T1_HPatches_Main", "ALIKED-500", "final_kp", "500.0", "results/phase4b/b2_permutation_summary_table.csv", "row 4")
    add("T1_HPatches_Main", "ALIKED-500", "cum_kp", "500.0", "results/phase4b/b2_permutation_summary_table.csv", "row 4")
    add("T1_HPatches_Main", "ALIKED-500", "mean_steps", "1.00", "results/phase3/folds_summary_table.csv", "row 20")

    # ALIKED-1000
    add("T1_HPatches_Main", "ALIKED-1000", "eval_type", "Pooled (5 Folds, 580 pairs)", "results/phase4b/b2_permutation_summary_table.csv", "row 5")
    add("T1_HPatches_Main", "ALIKED-1000", "succ_3px_pct", "73.62", "results/phase4b/b2_permutation_summary_table.csv", "row 5")
    add("T1_HPatches_Main", "ALIKED-1000", "auc_1px", "20.73", "results/phase3/pareto_plot_data.csv", "ALIKED-1000")
    add("T1_HPatches_Main", "ALIKED-1000", "auc_3px", "47.56", "results/phase4b/b2_permutation_summary_table.csv", "row 5")
    add("T1_HPatches_Main", "ALIKED-1000", "auc_5px", "56.54", "results/phase3/pareto_plot_data.csv", "ALIKED-1000")
    add("T1_HPatches_Main", "ALIKED-1000", "final_kp", "1000.0", "results/phase4b/b2_permutation_summary_table.csv", "row 5")
    add("T1_HPatches_Main", "ALIKED-1000", "cum_kp", "1000.0", "results/phase4b/b2_permutation_summary_table.csv", "row 5")
    add("T1_HPatches_Main", "ALIKED-1000", "mean_steps", "1.00", "results/phase3/pareto_plot_data.csv", "ALIKED-1000")

    # SIFT-500
    add("T1_HPatches_Main", "SIFT-500", "eval_type", "Pooled (5 Folds, 580 pairs)", "results/phase3/pareto_plot_data.csv", "SIFT-500")
    add("T1_HPatches_Main", "SIFT-500", "succ_3px_pct", "67.59", "results/phase3/pareto_plot_data.csv", "SIFT-500")
    add("T1_HPatches_Main", "SIFT-500", "auc_1px", "18.32", "results/phase3/pareto_plot_data.csv", "SIFT-500")
    add("T1_HPatches_Main", "SIFT-500", "auc_3px", "44.68", "results/phase3/pareto_plot_data.csv", "SIFT-500")
    add("T1_HPatches_Main", "SIFT-500", "auc_5px", "53.41", "results/phase3/pareto_plot_data.csv", "SIFT-500")
    add("T1_HPatches_Main", "SIFT-500", "final_kp", "500.0", "results/phase3/pareto_plot_data.csv", "SIFT-500")
    add("T1_HPatches_Main", "SIFT-500", "cum_kp", "500.0", "results/phase3/pareto_plot_data.csv", "SIFT-500")
    add("T1_HPatches_Main", "SIFT-500", "mean_steps", "1.00", "results/phase3/pareto_plot_data.csv", "SIFT-500")

    # SIFT-1000
    add("T1_HPatches_Main", "SIFT-1000", "eval_type", "Pooled (5 Folds, 580 pairs)", "results/phase3/pareto_plot_data.csv", "SIFT-1000")
    add("T1_HPatches_Main", "SIFT-1000", "succ_3px_pct", "71.21", "results/phase3/pareto_plot_data.csv", "SIFT-1000")
    add("T1_HPatches_Main", "SIFT-1000", "auc_1px", "19.88", "results/phase3/pareto_plot_data.csv", "SIFT-1000")
    add("T1_HPatches_Main", "SIFT-1000", "auc_3px", "47.50", "results/phase3/pareto_plot_data.csv", "SIFT-1000")
    add("T1_HPatches_Main", "SIFT-1000", "auc_5px", "56.45", "results/phase3/pareto_plot_data.csv", "SIFT-1000")
    add("T1_HPatches_Main", "SIFT-1000", "final_kp", "1000.0", "results/phase3/pareto_plot_data.csv", "SIFT-1000")
    add("T1_HPatches_Main", "SIFT-1000", "cum_kp", "1000.0", "results/phase3/pareto_plot_data.csv", "SIFT-1000")
    add("T1_HPatches_Main", "SIFT-1000", "mean_steps", "1.00", "results/phase3/pareto_plot_data.csv", "SIFT-1000")

    # SIFT-2000
    add("T1_HPatches_Main", "SIFT-2000", "eval_type", "Pooled (5 Folds, 580 pairs)", "results/phase4b/b2_permutation_summary_table.csv", "row 6")
    add("T1_HPatches_Main", "SIFT-2000", "succ_3px_pct", "72.76", "results/phase4b/b2_permutation_summary_table.csv", "row 6")
    add("T1_HPatches_Main", "SIFT-2000", "auc_1px", "20.91", "results/phase3/folds_summary_table.csv", "row 27")
    add("T1_HPatches_Main", "SIFT-2000", "auc_3px", "49.01", "results/phase4b/b2_permutation_summary_table.csv", "row 6")
    add("T1_HPatches_Main", "SIFT-2000", "auc_5px", "57.98", "results/phase3/folds_summary_table.csv", "row 27")
    add("T1_HPatches_Main", "SIFT-2000", "final_kp", "2000.0", "results/phase4b/b2_permutation_summary_table.csv", "row 6")
    add("T1_HPatches_Main", "SIFT-2000", "cum_kp", "2000.0", "results/phase4b/b2_permutation_summary_table.csv", "row 6")
    add("T1_HPatches_Main", "SIFT-2000", "mean_steps", "1.00", "results/phase3/folds_summary_table.csv", "row 27")

    # SIFT-4000
    add("T1_HPatches_Main", "SIFT-4000", "eval_type", "Pooled (5 Folds, 580 pairs)", "results/phase4b/b2_permutation_summary_table.csv", "row 7")
    add("T1_HPatches_Main", "SIFT-4000", "succ_3px_pct", "72.93", "results/phase4b/b2_permutation_summary_table.csv", "row 7")
    add("T1_HPatches_Main", "SIFT-4000", "auc_1px", "22.39", "results/phase3/folds_summary_table.csv", "row 34")
    add("T1_HPatches_Main", "SIFT-4000", "auc_3px", "51.13", "results/phase4b/b2_permutation_summary_table.csv", "row 7")
    add("T1_HPatches_Main", "SIFT-4000", "auc_5px", "59.98", "results/phase3/folds_summary_table.csv", "row 34")
    add("T1_HPatches_Main", "SIFT-4000", "final_kp", "4000.0", "results/phase4b/b2_permutation_summary_table.csv", "row 7")
    add("T1_HPatches_Main", "SIFT-4000", "cum_kp", "4000.0", "results/phase4b/b2_permutation_summary_table.csv", "row 7")
    add("T1_HPatches_Main", "SIFT-4000", "mean_steps", "1.00", "results/phase3/folds_summary_table.csv", "row 34")

    # XFeat-1000
    add("T1_HPatches_Main", "XFeat-1000", "eval_type", "Pooled (5 Folds, 580 pairs)", "results/phase3/pareto_plot_data.csv", "XFeat-1000")
    add("T1_HPatches_Main", "XFeat-1000", "succ_3px_pct", "71.90", "results/phase3/pareto_plot_data.csv", "XFeat-1000")
    add("T1_HPatches_Main", "XFeat-1000", "auc_1px", "20.15", "results/phase3/pareto_plot_data.csv", "XFeat-1000")
    add("T1_HPatches_Main", "XFeat-1000", "auc_3px", "47.21", "results/phase3/pareto_plot_data.csv", "XFeat-1000")
    add("T1_HPatches_Main", "XFeat-1000", "auc_5px", "56.02", "results/phase3/pareto_plot_data.csv", "XFeat-1000")
    add("T1_HPatches_Main", "XFeat-1000", "final_kp", "1000.0", "results/phase3/pareto_plot_data.csv", "XFeat-1000")
    add("T1_HPatches_Main", "XFeat-1000", "cum_kp", "1000.0", "results/phase3/pareto_plot_data.csv", "XFeat-1000")
    add("T1_HPatches_Main", "XFeat-1000", "mean_steps", "1.00", "results/phase3/pareto_plot_data.csv", "XFeat-1000")

    # Oracle Preset (Upper Bound)
    add("T1_HPatches_Main", "Oracle Preset (Upper Bound)*", "eval_type", "Pooled (5 Folds, 580 pairs)", "results/phase3/folds_summary_table.csv", "row 41")
    add("T1_HPatches_Main", "Oracle Preset (Upper Bound)*", "succ_3px_pct", "80.69", "results/phase3/folds_summary_table.csv", "row 41")
    add("T1_HPatches_Main", "Oracle Preset (Upper Bound)*", "auc_1px", "24.97", "results/phase3/folds_summary_table.csv", "row 41")
    add("T1_HPatches_Main", "Oracle Preset (Upper Bound)*", "auc_3px", "56.33", "results/phase3/folds_summary_table.csv", "row 41")
    add("T1_HPatches_Main", "Oracle Preset (Upper Bound)*", "auc_5px", "65.31", "results/phase3/folds_summary_table.csv", "row 41")
    add("T1_HPatches_Main", "Oracle Preset (Upper Bound)*", "final_kp", "4000.0", "results/phase3/folds_summary_table.csv", "row 41")
    add("T1_HPatches_Main", "Oracle Preset (Upper Bound)*", "cum_kp", "4000.0", "results/phase3/folds_summary_table.csv", "row 41")
    add("T1_HPatches_Main", "Oracle Preset (Upper Bound)*", "mean_steps", "1.00", "results/phase3/folds_summary_table.csv", "row 41")

    # AFM-RL v2 Cascade (HPatches Exploratory)
    add("T1_HPatches_Main", "AFM-RL v2 Cascade (Exploratory)", "eval_type", "Pooled (5 Folds, 580 pairs)", "results/phase4b/b3_bootstrap_comparisons.csv", "row 6")
    add("T1_HPatches_Main", "AFM-RL v2 Cascade (Exploratory)", "succ_3px_pct", "76.21", "results/phase4b/b3_bootstrap_comparisons.csv", "row 6")
    add("T1_HPatches_Main", "AFM-RL v2 Cascade (Exploratory)", "auc_1px", "21.68", "results/phase4/hpatches_v2_cascade_eval.csv", "mean")
    add("T1_HPatches_Main", "AFM-RL v2 Cascade (Exploratory)", "auc_3px", "49.61", "results/phase4b/b3_bootstrap_comparisons.csv", "row 6")
    add("T1_HPatches_Main", "AFM-RL v2 Cascade (Exploratory)", "auc_5px", "58.74", "results/phase4/hpatches_v2_cascade_eval.csv", "mean")
    add("T1_HPatches_Main", "AFM-RL v2 Cascade (Exploratory)", "final_kp", "1165.5", "results/phase4/hpatches_v2_cascade_eval.csv", "mean")
    add("T1_HPatches_Main", "AFM-RL v2 Cascade (Exploratory)", "cum_kp", "1548.3", "results/phase4/hpatches_v2_cascade_eval.csv", "mean")
    add("T1_HPatches_Main", "AFM-RL v2 Cascade (Exploratory)", "mean_steps", "1.41", "results/phase4/hpatches_v2_cascade_eval.csv", "mean")

    # Primary Comparison Deltas (P1 - P5) with 95% CIs
    add("T1_HPatches_Main", "P1: AFM-RL v1 vs Best Fixed", "delta_succ_3px", "-1.55% [-5.34%, +1.90%]", "results/phase4b/b3_bootstrap_comparisons.csv", "row 2")
    add("T1_HPatches_Main", "P1: AFM-RL v1 vs Best Fixed", "delta_auc_3px", "-2.83 [-5.67%, -0.29%]", "results/phase4b/b3_bootstrap_comparisons.csv", "row 2")
    add("T1_HPatches_Main", "P2: AFM-RL v1 vs ALIKED-500", "delta_succ_3px", "-0.17% [-1.03%, +0.52%]", "results/phase4b/b3_bootstrap_comparisons.csv", "row 3")
    add("T1_HPatches_Main", "P2: AFM-RL v1 vs ALIKED-500", "delta_auc_3px", "-0.08 [-0.56%, +0.38%]", "results/phase4b/b3_bootstrap_comparisons.csv", "row 3")
    add("T1_HPatches_Main", "P3: AFM-RL v1 vs SIFT-2000", "delta_succ_3px", "-0.34% [-4.48%, +3.45%]", "results/phase4b/b3_bootstrap_comparisons.csv", "row 4")
    add("T1_HPatches_Main", "P3: AFM-RL v1 vs SIFT-2000", "delta_auc_3px", "-2.20 [-5.25%, +0.74%]", "results/phase4b/b3_bootstrap_comparisons.csv", "row 4")
    add("T1_HPatches_Main", "P4: AFM-RL v1 vs SIFT-4000", "delta_succ_3px", "-0.52% [-4.66%, +3.28%]", "results/phase4b/b3_bootstrap_comparisons.csv", "row 5")
    add("T1_HPatches_Main", "P4: AFM-RL v1 vs SIFT-4000", "delta_auc_3px", "-4.32 [-7.31%, -1.54%]", "results/phase4b/b3_bootstrap_comparisons.csv", "row 5")
    add("T1_HPatches_Main", "P5: AFM-RL v1 vs Oracle Preset*", "delta_succ_3px", "-8.28% [-11.72%, -4.83%]", "results/phase3/primary_comparisons_P1_P5.csv", "row 5")
    add("T1_HPatches_Main", "P5: AFM-RL v1 vs Oracle Preset*", "delta_auc_3px", "-9.52 [-12.21, -6.84]", "results/phase3/primary_comparisons_P1_P5.csv", "row 5")

    # =========================================================================
    # TABLE T2: MEGADEPTH MAIN (1,500 pairs)
    # =========================================================================
    # MegaDepth Baselines and Methods with Exact Measured Latencies from timing_benchmark.csv
    df_time = pd.read_csv("results/final/timing_benchmark.csv").set_index("method")
    
    add("T2_MegaDepth_Main", "SIFT-500", "succ_10_deg", "35.80", "results/phase4b/megadepth_stage2a_priority_evaluations.csv", "SIFT-500 subset")
    add("T2_MegaDepth_Main", "SIFT-500", "auc_5_deg", "14.19", "results/phase4b/megadepth_stage2a_priority_evaluations.csv", "SIFT-500 subset")
    add("T2_MegaDepth_Main", "SIFT-500", "auc_10_deg", "23.32", "results/phase4b/megadepth_stage2a_priority_evaluations.csv", "SIFT-500 subset")
    add("T2_MegaDepth_Main", "SIFT-500", "auc_20_deg", "33.37", "results/phase4b/megadepth_stage2a_priority_evaluations.csv", "SIFT-500 subset")
    add("T2_MegaDepth_Main", "SIFT-500", "requested_n", "500", "results/phase4b/megadepth_stage2a_priority_evaluations.csv", "SIFT-500 subset")
    add("T2_MegaDepth_Main", "SIFT-500", "cum_actual_kp", "496.26", "results/phase4b/megadepth_stage2a_priority_evaluations.csv", "SIFT-500 subset")
    add("T2_MegaDepth_Main", "SIFT-500", "device", "CPU", "Protocol v2", "CPU")
    add("T2_MegaDepth_Main", "SIFT-500", "latency_median_ms", "948.3", "results/phase4b/c2_expected_wallclock_summary.csv", "historical")

    add("T2_MegaDepth_Main", "SIFT-1000", "succ_10_deg", "55.40", "results/phase4b/megadepth_stage2a_priority_evaluations.csv", "SIFT-1000 subset")
    add("T2_MegaDepth_Main", "SIFT-1000", "auc_5_deg", "26.85", "results/phase4b/megadepth_stage2a_priority_evaluations.csv", "SIFT-1000 subset")
    add("T2_MegaDepth_Main", "SIFT-1000", "auc_10_deg", "38.83", "results/phase4b/megadepth_stage2a_priority_evaluations.csv", "SIFT-1000 subset")
    add("T2_MegaDepth_Main", "SIFT-1000", "auc_20_deg", "49.92", "results/phase4b/megadepth_stage2a_priority_evaluations.csv", "SIFT-1000 subset")
    add("T2_MegaDepth_Main", "SIFT-1000", "requested_n", "1000", "results/phase4b/megadepth_stage2a_priority_evaluations.csv", "SIFT-1000 subset")
    add("T2_MegaDepth_Main", "SIFT-1000", "cum_actual_kp", "993.43", "results/phase4b/megadepth_stage2a_priority_evaluations.csv", "SIFT-1000 subset")
    add("T2_MegaDepth_Main", "SIFT-1000", "device", "CPU", "Protocol v2", "CPU")
    add("T2_MegaDepth_Main", "SIFT-1000", "latency_median_ms", "1182.4", "results/phase4b/c2_expected_wallclock_summary.csv", "historical")

    add("T2_MegaDepth_Main", "SIFT-2000", "succ_10_deg", "66.80", "results/phase4b/megadepth_stage2a_priority_evaluations.csv", "SIFT-2000 subset")
    add("T2_MegaDepth_Main", "SIFT-2000", "auc_5_deg", "33.91", "results/phase4b/megadepth_stage2a_priority_evaluations.csv", "SIFT-2000 subset")
    add("T2_MegaDepth_Main", "SIFT-2000", "auc_10_deg", "46.14", "results/phase4b/megadepth_stage2a_priority_evaluations.csv", "SIFT-2000 subset")
    add("T2_MegaDepth_Main", "SIFT-2000", "auc_20_deg", "57.51", "results/phase4b/megadepth_stage2a_priority_evaluations.csv", "SIFT-2000 subset")
    add("T2_MegaDepth_Main", "SIFT-2000", "requested_n", "2000", "results/phase4b/megadepth_stage2a_priority_evaluations.csv", "SIFT-2000 subset")
    add("T2_MegaDepth_Main", "SIFT-2000", "cum_actual_kp", "1989.67", "results/phase4b/megadepth_stage2a_priority_evaluations.csv", "SIFT-2000 subset")
    add("T2_MegaDepth_Main", "SIFT-2000", "device", "CPU", "Protocol v2", "CPU")
    add("T2_MegaDepth_Main", "SIFT-2000", "latency_median_ms", str(round(df_time.loc["SIFT-2000", "median_latency_ms"], 2)), "results/final/timing_benchmark.csv", "row 2")
    add("T2_MegaDepth_Main", "SIFT-2000", "latency_mean_ms", str(round(df_time.loc["SIFT-2000", "mean_latency_ms"], 2)), "results/final/timing_benchmark.csv", "row 2")

    add("T2_MegaDepth_Main", "SIFT-4000", "succ_10_deg", "77.20", "results/phase4b/megadepth_stage2a_priority_evaluations.csv", "SIFT-4000 subset")
    add("T2_MegaDepth_Main", "SIFT-4000", "auc_5_deg", "39.73", "results/phase4b/megadepth_stage2a_priority_evaluations.csv", "SIFT-4000 subset")
    add("T2_MegaDepth_Main", "SIFT-4000", "auc_10_deg", "54.78", "results/phase4b/megadepth_stage2a_priority_evaluations.csv", "SIFT-4000 subset")
    add("T2_MegaDepth_Main", "SIFT-4000", "auc_20_deg", "67.43", "results/phase4b/megadepth_stage2a_priority_evaluations.csv", "SIFT-4000 subset")
    add("T2_MegaDepth_Main", "SIFT-4000", "requested_n", "4000", "results/phase4b/megadepth_stage2a_priority_evaluations.csv", "SIFT-4000 subset")
    add("T2_MegaDepth_Main", "SIFT-4000", "cum_actual_kp", "3821.14", "results/phase4b/megadepth_stage2a_priority_evaluations.csv", "SIFT-4000 subset")
    add("T2_MegaDepth_Main", "SIFT-4000", "device", "CPU", "Protocol v2", "CPU")
    add("T2_MegaDepth_Main", "SIFT-4000", "latency_median_ms", str(round(df_time.loc["SIFT-4000", "median_latency_ms"], 2)), "results/final/timing_benchmark.csv", "row 3")
    add("T2_MegaDepth_Main", "SIFT-4000", "latency_mean_ms", str(round(df_time.loc["SIFT-4000", "mean_latency_ms"], 2)), "results/final/timing_benchmark.csv", "row 3")

    add("T2_MegaDepth_Main", "ALIKED-500", "succ_10_deg", "76.13", "results/phase4b/megadepth_stage2a_priority_evaluations.csv", "ALIKED-500 subset")
    add("T2_MegaDepth_Main", "ALIKED-500", "auc_5_deg", "39.46", "results/phase4b/megadepth_stage2a_priority_evaluations.csv", "ALIKED-500 subset")
    add("T2_MegaDepth_Main", "ALIKED-500", "auc_10_deg", "55.24", "results/phase4b/megadepth_stage2a_priority_evaluations.csv", "ALIKED-500 subset")
    add("T2_MegaDepth_Main", "ALIKED-500", "auc_20_deg", "68.42", "results/phase4b/megadepth_stage2a_priority_evaluations.csv", "ALIKED-500 subset")
    add("T2_MegaDepth_Main", "ALIKED-500", "requested_n", "500", "results/phase4b/megadepth_stage2a_priority_evaluations.csv", "ALIKED-500 subset")
    add("T2_MegaDepth_Main", "ALIKED-500", "cum_actual_kp", "497.47", "results/phase4b/megadepth_stage2a_priority_evaluations.csv", "ALIKED-500 subset")
    add("T2_MegaDepth_Main", "ALIKED-500", "device", "GPU", "Protocol v2", "GPU")
    add("T2_MegaDepth_Main", "ALIKED-500", "latency_median_ms", str(round(df_time.loc["ALIKED-500", "median_latency_ms"], 2)), "results/final/timing_benchmark.csv", "row 4")
    add("T2_MegaDepth_Main", "ALIKED-500", "latency_mean_ms", str(round(df_time.loc["ALIKED-500", "mean_latency_ms"], 2)), "results/final/timing_benchmark.csv", "row 4")

    add("T2_MegaDepth_Main", "ALIKED-1000", "succ_10_deg", "77.67", "results/phase4b/megadepth_stage2a_priority_evaluations.csv", "ALIKED-1000 subset")
    add("T2_MegaDepth_Main", "ALIKED-1000", "auc_5_deg", "39.52", "results/phase4b/megadepth_stage2a_priority_evaluations.csv", "ALIKED-1000 subset")
    add("T2_MegaDepth_Main", "ALIKED-1000", "auc_10_deg", "55.46", "results/phase4b/megadepth_stage2a_priority_evaluations.csv", "ALIKED-1000 subset")
    add("T2_MegaDepth_Main", "ALIKED-1000", "auc_20_deg", "68.61", "results/phase4b/megadepth_stage2a_priority_evaluations.csv", "ALIKED-1000 subset")
    add("T2_MegaDepth_Main", "ALIKED-1000", "requested_n", "1000", "results/phase4b/megadepth_stage2a_priority_evaluations.csv", "ALIKED-1000 subset")
    add("T2_MegaDepth_Main", "ALIKED-1000", "cum_actual_kp", "976.97", "results/phase4b/megadepth_stage2a_priority_evaluations.csv", "ALIKED-1000 subset")
    add("T2_MegaDepth_Main", "ALIKED-1000", "device", "GPU", "Protocol v2", "GPU")
    add("T2_MegaDepth_Main", "ALIKED-1000", "latency_median_ms", str(round(df_time.loc["ALIKED-1000", "median_latency_ms"], 2)), "results/final/timing_benchmark.csv", "row 5")
    add("T2_MegaDepth_Main", "ALIKED-1000", "latency_mean_ms", str(round(df_time.loc["ALIKED-1000", "mean_latency_ms"], 2)), "results/final/timing_benchmark.csv", "row 5")

    add("T2_MegaDepth_Main", "AFM-RL v1 (Fixed Rerun)", "succ_10_deg", "76.05", "results/phase5b/a4_stage3_fixed_summary.csv", "row 7")
    add("T2_MegaDepth_Main", "AFM-RL v1 (Fixed Rerun)", "auc_5_deg", "39.55", "results/phase5b/a4_stage3_fixed_summary.csv", "row 7")
    add("T2_MegaDepth_Main", "AFM-RL v1 (Fixed Rerun)", "auc_10_deg", "55.16", "results/phase5b/a4_stage3_fixed_summary.csv", "row 7")
    add("T2_MegaDepth_Main", "AFM-RL v1 (Fixed Rerun)", "auc_20_deg", "68.23", "results/phase5b/a4_stage3_fixed_summary.csv", "row 7")
    add("T2_MegaDepth_Main", "AFM-RL v1 (Fixed Rerun)", "requested_n", "Adaptive (500-1500)", "results/phase5b/a4_stage3_fixed_summary.csv", "row 7")
    add("T2_MegaDepth_Main", "AFM-RL v1 (Fixed Rerun)", "cum_actual_kp", "870.55", "results/phase5b/a4_stage3_fixed_summary.csv", "row 7")
    add("T2_MegaDepth_Main", "AFM-RL v1 (Fixed Rerun)", "device", "GPU+CPU", "Protocol v2", "Hybrid")
    add("T2_MegaDepth_Main", "AFM-RL v1 (Fixed Rerun)", "latency_median_ms", str(round(df_time.loc["AFM-RL v1 (End-to-End)", "median_latency_ms"], 2)), "results/final/timing_benchmark.csv", "row 10")
    add("T2_MegaDepth_Main", "AFM-RL v1 (Fixed Rerun)", "latency_mean_ms", str(round(df_time.loc["AFM-RL v1 (End-to-End)", "mean_latency_ms"], 2)), "results/final/timing_benchmark.csv", "row 10")

    add("T2_MegaDepth_Main", "Cascade C4 (3-Stage Committed)", "succ_10_deg", "78.87", "results/phase5b/c2_fold_selected_cascades_summary.csv", "row 4")
    add("T2_MegaDepth_Main", "Cascade C4 (3-Stage Committed)", "auc_5_deg", "41.37", "results/phase5b/c2_fold_selected_cascades_summary.csv", "row 4")
    add("T2_MegaDepth_Main", "Cascade C4 (3-Stage Committed)", "auc_10_deg", "57.58", "results/phase5b/c2_fold_selected_cascades_summary.csv", "row 4")
    add("T2_MegaDepth_Main", "Cascade C4 (3-Stage Committed)", "auc_20_deg", "70.90", "results/phase5b/c2_fold_selected_cascades_summary.csv", "row 4")
    add("T2_MegaDepth_Main", "Cascade C4 (3-Stage Committed)", "requested_n", "500-2000", "results/phase5b/c2_fold_selected_cascades_summary.csv", "row 4")
    add("T2_MegaDepth_Main", "Cascade C4 (3-Stage Committed)", "cum_actual_kp", "1529.96", "results/phase5b/c2_fold_selected_cascades_summary.csv", "row 4")
    add("T2_MegaDepth_Main", "Cascade C4 (3-Stage Committed)", "device", "GPU+CPU", "Protocol v2", "Hybrid")
    add("T2_MegaDepth_Main", "Cascade C4 (3-Stage Committed)", "latency_median_ms", str(round(df_time.loc["Cascade C4 (3-Stage Committed)", "median_latency_ms"], 2)), "results/final/timing_benchmark.csv", "row 6")
    add("T2_MegaDepth_Main", "Cascade C4 (3-Stage Committed)", "latency_mean_ms", str(round(df_time.loc["Cascade C4 (3-Stage Committed)", "mean_latency_ms"], 2)), "results/final/timing_benchmark.csv", "row 6")

    add("T2_MegaDepth_Main", "Cascade C4 (2-Stage GPU-Only)", "succ_10_deg", "77.67", "results/phase4b/megadepth_stage2a_priority_evaluations.csv", "recomputed")
    add("T2_MegaDepth_Main", "Cascade C4 (2-Stage GPU-Only)", "auc_5_deg", "39.52", "results/phase4b/megadepth_stage2a_priority_evaluations.csv", "recomputed")
    add("T2_MegaDepth_Main", "Cascade C4 (2-Stage GPU-Only)", "auc_10_deg", "55.46", "results/phase4b/megadepth_stage2a_priority_evaluations.csv", "recomputed")
    add("T2_MegaDepth_Main", "Cascade C4 (2-Stage GPU-Only)", "auc_20_deg", "68.61", "results/phase4b/megadepth_stage2a_priority_evaluations.csv", "recomputed")
    add("T2_MegaDepth_Main", "Cascade C4 (2-Stage GPU-Only)", "requested_n", "500-1000", "results/phase5b/f_gpu_benchmark.py", "2-stage")
    add("T2_MegaDepth_Main", "Cascade C4 (2-Stage GPU-Only)", "cum_actual_kp", "823.90", "results/phase5b/f_gpu_benchmark.py", "2-stage")
    add("T2_MegaDepth_Main", "Cascade C4 (2-Stage GPU-Only)", "device", "GPU", "Protocol v2", "GPU")
    add("T2_MegaDepth_Main", "Cascade C4 (2-Stage GPU-Only)", "latency_median_ms", str(round(df_time.loc["Cascade C4 (2-Stage GPU-Only)", "median_latency_ms"], 2)), "results/final/timing_benchmark.csv", "row 7")
    add("T2_MegaDepth_Main", "Cascade C4 (2-Stage GPU-Only)", "latency_mean_ms", str(round(df_time.loc["Cascade C4 (2-Stage GPU-Only)", "mean_latency_ms"], 2)), "results/final/timing_benchmark.csv", "row 7")

    add("T2_MegaDepth_Main", "Cascade C6 (tau=0.40, Post-hoc)", "succ_10_deg", "82.13", "results/phase5b/b2_permutation_robustness_all_pairs.csv", "row 17")
    add("T2_MegaDepth_Main", "Cascade C6 (tau=0.40, Post-hoc)", "auc_5_deg", "43.32", "results/phase4b/megadepth_stage2a_priority_evaluations.csv", "recomputed")
    add("T2_MegaDepth_Main", "Cascade C6 (tau=0.40, Post-hoc)", "auc_10_deg", "59.94", "results/phase5b/b2_permutation_robustness_all_pairs.csv", "row 17")
    add("T2_MegaDepth_Main", "Cascade C6 (tau=0.40, Post-hoc)", "auc_20_deg", "73.52", "results/phase4b/megadepth_stage2a_priority_evaluations.csv", "recomputed")
    add("T2_MegaDepth_Main", "Cascade C6 (tau=0.40, Post-hoc)", "requested_n", "500 / 4000", "results/phase5b/b2_permutation_robustness_all_pairs.csv", "row 17")
    add("T2_MegaDepth_Main", "Cascade C6 (tau=0.40, Post-hoc)", "cum_actual_kp", "2013.49", "results/phase5b/b2_permutation_robustness_all_pairs.csv", "row 17")
    add("T2_MegaDepth_Main", "Cascade C6 (tau=0.40, Post-hoc)", "device", "GPU+CPU", "Protocol v2", "Hybrid")
    add("T2_MegaDepth_Main", "Cascade C6 (tau=0.40, Post-hoc)", "latency_median_ms", str(round(df_time.loc["Cascade C6 (tau=0.40)", "median_latency_ms"], 2)), "results/final/timing_benchmark.csv", "row 8")
    add("T2_MegaDepth_Main", "Cascade C6 (tau=0.40, Post-hoc)", "latency_mean_ms", str(round(df_time.loc["Cascade C6 (tau=0.40)", "mean_latency_ms"], 2)), "results/final/timing_benchmark.csv", "row 8")

    add("T2_MegaDepth_Main", "Always-Both", "succ_10_deg", "82.53", "results/phase5b/b2_permutation_robustness_all_pairs.csv", "row 18")
    add("T2_MegaDepth_Main", "Always-Both", "auc_5_deg", "44.12", "results/phase4b/megadepth_stage2a_priority_evaluations.csv", "recomputed")
    add("T2_MegaDepth_Main", "Always-Both", "auc_10_deg", "60.47", "results/phase5b/b2_permutation_robustness_all_pairs.csv", "row 18")
    add("T2_MegaDepth_Main", "Always-Both", "auc_20_deg", "74.02", "results/phase4b/megadepth_stage2a_priority_evaluations.csv", "recomputed")
    add("T2_MegaDepth_Main", "Always-Both", "requested_n", "4500", "results/phase5b/b2_permutation_robustness_all_pairs.csv", "row 18")
    add("T2_MegaDepth_Main", "Always-Both", "cum_actual_kp", "4318.61", "results/phase5b/b2_permutation_robustness_all_pairs.csv", "row 18")
    add("T2_MegaDepth_Main", "Always-Both", "device", "GPU+CPU", "Protocol v2", "Hybrid")
    add("T2_MegaDepth_Main", "Always-Both", "latency_median_ms", str(round(df_time.loc["Always-Both", "median_latency_ms"], 2)), "results/final/timing_benchmark.csv", "row 9")
    add("T2_MegaDepth_Main", "Always-Both", "latency_mean_ms", str(round(df_time.loc["Always-Both", "mean_latency_ms"], 2)), "results/final/timing_benchmark.csv", "row 9")

    # =========================================================================
    # TABLE T3: CASCADE RESULTS (HPatches, MegaDepth Fold-Selected, C6, Permutations, Controls)
    # =========================================================================
    # Fold-selected cascades per fold
    for f_idx, r_idx in [(0, 4), (1, 7), (2, 10), (3, 13), (4, 16)]:
        add("T3_Cascade_Results", f"MegaDepth Fold {f_idx} Selected", "succ_10_deg", str(round(p3_folds if False else pd.read_csv('results/phase5b/c2_fold_selected_cascades_summary.csv').iloc[r_idx-2]['succ10'], 2)), "results/phase5b/c2_fold_selected_cascades_summary.csv", f"row {r_idx}")
        add("T3_Cascade_Results", f"MegaDepth Fold {f_idx} Selected", "auc_10_deg", str(round(pd.read_csv('results/phase5b/c2_fold_selected_cascades_summary.csv').iloc[r_idx-2]['auc10'], 2)), "results/phase5b/c2_fold_selected_cascades_summary.csv", f"row {r_idx}")
        add("T3_Cascade_Results", f"MegaDepth Fold {f_idx} Selected", "cum_kp", str(round(pd.read_csv('results/phase5b/c2_fold_selected_cascades_summary.csv').iloc[r_idx-2]['cum_actual_kp'], 2)), "results/phase5b/c2_fold_selected_cascades_summary.csv", f"row {r_idx}")
        add("T3_Cascade_Results", f"MegaDepth Fold {f_idx} Selected", "step1_stop_rate", str(round(pd.read_csv('results/phase5b/c2_fold_selected_cascades_summary.csv').iloc[r_idx-2]['step1_stop_rate_pct'], 2)), "results/phase5b/c2_fold_selected_cascades_summary.csv", f"row {r_idx}")

    # Permutation robustness C6
    p_c6 = pd.read_csv("results/phase5b/b2_permutation_robustness_all_pairs.csv")
    r_c6 = p_c6[(p_c6["scope"] == "Pooled (All 1500)") & (p_c6["method"] == "Cascade C* (tau=0.40)")].iloc[0]
    add("T3_Cascade_Results", "Cascade C6 (tau=0.40) Original", "succ_10_deg", str(round(r_c6['orig_succ10'], 2)), "results/phase5b/b2_permutation_robustness_all_pairs.csv", "row 17")
    add("T3_Cascade_Results", "Cascade C6 (tau=0.40) Original", "auc_10_deg", str(round(r_c6['orig_auc10'], 2)), "results/phase5b/b2_permutation_robustness_all_pairs.csv", "row 17")
    add("T3_Cascade_Results", "Cascade C6 (tau=0.40) Original", "cum_kp", str(round(r_c6['orig_actual_kp'], 2)), "results/phase5b/b2_permutation_robustness_all_pairs.csv", "row 17")

    add("T3_Cascade_Results", "Cascade C6 (tau=0.40) Permutation Mean", "succ_10_deg", str(round(r_c6['perm_mean_succ10'], 2)), "results/phase5b/b2_permutation_robustness_all_pairs.csv", "row 17")
    add("T3_Cascade_Results", "Cascade C6 (tau=0.40) Permutation Std", "succ_10_deg", str(round(r_c6['perm_std_succ10'], 2)), "results/phase5b/b2_permutation_robustness_all_pairs.csv", "row 17")
    add("T3_Cascade_Results", "Cascade C6 (tau=0.40) Permutation Mean", "auc_10_deg", str(round(r_c6['perm_mean_auc10'], 2)), "results/phase5b/b2_permutation_robustness_all_pairs.csv", "row 17")
    add("T3_Cascade_Results", "Cascade C6 (tau=0.40) Permutation Std", "auc_10_deg", str(round(r_c6['perm_std_auc10'], 2)), "results/phase5b/b2_permutation_robustness_all_pairs.csv", "row 17")
    add("T3_Cascade_Results", "Cascade C6 (tau=0.40) Permutation Mean", "cum_kp", str(round(r_c6['perm_mean_actual_kp'], 2)), "results/phase5b/b2_permutation_robustness_all_pairs.csv", "row 17")

    # Controls
    d_ctrl = pd.read_csv("results/phase5b/d_controls_summary.csv")
    for idx, r in d_ctrl.iterrows():
        add("T3_Cascade_Results", f"Control: {r['control_name']}", "succ_10_deg", str(round(r['succ10'], 2)), "results/phase5b/d_controls_summary.csv", f"row {idx+2}")
        add("T3_Cascade_Results", f"Control: {r['control_name']}", "auc_10_deg", str(round(r['auc10'], 2)), "results/phase5b/d_controls_summary.csv", f"row {idx+2}")
        add("T3_Cascade_Results", f"Control: {r['control_name']}", "cum_kp", str(round(r['cum_actual_kp'], 2)), "results/phase5b/d_controls_summary.csv", f"row {idx+2}")
        add("T3_Cascade_Results", f"Control: {r['control_name']}", "stop_rate_pct", str(round(r['stop_rate_pct'], 2)), "results/phase5b/d_controls_summary.csv", f"row {idx+2}")

    # =========================================================================
    # TABLE T4: POLICY BEHAVIOUR
    # =========================================================================
    # HPatches Level 1
    p3_l1 = pd.read_csv("results/phase3/level1_selection_stats.csv")
    for idx, r in p3_l1.iterrows():
        add("T4_Policy_Behaviour", f"HPatches L1 Fold {r['Fold']}", "ALIKED_count", str(r['ALIKED_Count']), "results/phase3/level1_selection_stats.csv", f"row {idx+2}")
        add("T4_Policy_Behaviour", f"HPatches L1 Fold {r['Fold']}", "SIFT_count", str(r['SIFT_Count']), "results/phase3/level1_selection_stats.csv", f"row {idx+2}")
        add("T4_Policy_Behaviour", f"HPatches L1 Fold {r['Fold']}", "entropy", str(round(max(0.0, float(r['Entropy_Bits'])), 3)), "results/phase3/level1_selection_stats.csv", f"row {idx+2}")

    # MegaDepth Level 1
    p5_fixed = pd.read_csv("results/phase5b/a4_stage3_fixed_summary.csv")
    for idx, r in p5_fixed.iterrows():
        add("T4_Policy_Behaviour", f"MegaDepth {r['scope']}", "preset_distribution", str(r['preset_dist']), "results/phase5b/a4_stage3_fixed_summary.csv", f"row {idx+2}")
        add("T4_Policy_Behaviour", f"MegaDepth {r['scope']}", "pct_stopped_step_1", str(round(r['pct_stopped_step_1'], 2)), "results/phase5b/a4_stage3_fixed_summary.csv", f"row {idx+2}")
        add("T4_Policy_Behaviour", f"MegaDepth {r['scope']}", "mean_steps", str(round(r['mean_steps'], 2)), "results/phase5b/a4_stage3_fixed_summary.csv", f"row {idx+2}")
        add("T4_Policy_Behaviour", f"MegaDepth {r['scope']}", "first_act0_plus500", str(round(r['first_act0_plus500'], 2)), "results/phase5b/a4_stage3_fixed_summary.csv", f"row {idx+2}")
        add("T4_Policy_Behaviour", f"MegaDepth {r['scope']}", "first_act1_plus1000", str(round(r['first_act1_plus1000'], 2)), "results/phase5b/a4_stage3_fixed_summary.csv", f"row {idx+2}")
        add("T4_Policy_Behaviour", f"MegaDepth {r['scope']}", "first_act3_STOP", str(round(r['first_act3_STOP'], 2)), "results/phase5b/a4_stage3_fixed_summary.csv", f"row {idx+2}")

    # =========================================================================
    # TABLE T5: AUDIT TABLE
    # =========================================================================
    audits = [
        ("HPatches Evaluation Metric", "Corner error evaluated on test with GT reward leak", "AUC@3px = 46.81%, Succ@3px = 72.41% under honest protocol", "Phase 1 used ground-truth homography corner error inside reward loop and leaky data splits.", "results/phase1b/honest_all_presets_test.csv vs results/phase3/folds_summary_table.csv"),
        ("MegaDepth AFM-RL v1 Action Mapping", "Succ@10 = 74.20%, KP = 100.0, 100% STOP", "Succ@10 = 76.05%, KP = 870.55, 80.47% STOP", "src/megadepth_eval_env.py mapped actions to [-500, -250, +250, +500] instead of [+500, +1000, -500, STOP], forcing immediate depletion.", "results/phase5/megadepth_stage3_afm_rl_rollouts.csv vs results/phase5b/megadepth_stage3_afm_rl_v1_fixed.csv"),
        ("Cascade C4 Protocol Definition", "C4: ALIKED-500 -> ALIKED-1000 (KP = 823.90)", "C4: ALIKED-500 -> SIFT-1000 -> ALIKED-2000 (KP = 1529.96)", "Phase 5b text and GPU benchmark script used a 2-stage GPU abbreviation rather than the committed 3-stage definition.", "results/phase4b/PROTOCOL_v2_AMENDMENT_1.md vs results/phase5b/f_gpu_benchmark.py"),
        ("MegaDepth Bootstrap Discrepancy", "DeltaSucc = -1.93%, DeltaAUC = -2.89", "DeltaSucc = -0.08% [-0.66%, +0.50%], DeltaAUC = -0.07 [-0.53, +0.37]", "finish_a4_summaries.py averaged pose error across folds before thresholding and used a linear piecewise approximation rather than trapezoidal integration across reference-image clusters.", "results/phase5b/a4_bootstrap_comparisons.csv vs results/final/a1_bootstrap_comparisons_fixed.csv"),
        ("HPatches RANSAC Reproducibility", "cv2.setRNGSeed hook flagged as source of non-determinism", "Hook had zero effect on cv2.findHomography; correspondence ordering is the sole source of RANSAC selection noise", "OpenCV cv2.findHomography RANSAC uses an internal deterministic seed; random permutation of correspondence rows produces +-0.3% to +-0.8% selection noise.", "results/phase4b/a1_reproducibility_check.csv")
    ]
    for idx, (claim, old_v, honest_v, reason, src) in enumerate(audits):
        add("T5_Audit_Table", claim, "old_leaky_value", old_v, src, f"audit item {idx+1}")
        add("T5_Audit_Table", claim, "honest_value", honest_v, src, f"audit item {idx+1}")
        add("T5_Audit_Table", claim, "root_cause_reason", reason, src, f"audit item {idx+1}")

    df_all = pd.DataFrame(records)
    out_csv = OUTPUT_DIR / "all_results.csv"
    df_all.to_csv(out_csv, index=False)
    print(f"Generated {out_csv} with {len(df_all)} records.")

if __name__ == "__main__":
    build_all_results()
