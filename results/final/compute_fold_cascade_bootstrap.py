import os
import sys
import time
import numpy as np
import pandas as pd
from pathlib import Path

# Add src to path
sys.path.insert(0, 'src')
import megadepth as md

OUTPUT_DIR = Path("results/final")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CANDIDATE_CASCADES = {
    "C1": [("ALIKED", 500), ("SIFT", 2000), ("SIFT", 4000)],
    "C2": [("ALIKED", 500), ("ALIKED", 1000), ("SIFT", 4000)],
    "C3": [("SIFT", 1000), ("SIFT", 2000), ("SIFT", 4000)],
    "C4": [("ALIKED", 500), ("SIFT", 1000), ("ALIKED", 2000)],
    "C5": [("SIFT", 500), ("SIFT", 2000), ("SIFT", 4000)],
    "C6": [("ALIKED", 500), ("SIFT", 4000)],
}

FOLD_SELECTED = [
    (0, "C4", 0.40),
    (1, "C4", 0.45),
    (2, "C4", 0.40),
    (3, "C4", 0.40),
    (4, "C2", 0.40),
]

def main():
    print("=" * 70)
    print("RECOMPUTING REFERENCE-IMAGE-CLUSTER BOOTSTRAP FOR FOLD-SELECTED CASCADES")
    print("=" * 70)

    # 1. Load evaluations
    df2a = pd.read_csv("results/phase4b/megadepth_stage2a_priority_evaluations.csv")
    df2b = pd.read_csv("results/phase4b/megadepth_stage2b_secondary_evaluations.csv")
    df_all = pd.concat([df2a, df2b], ignore_index=True)
    idx_eval = df_all.set_index(["fname", "i", "j", "preset", "N"])

    pairs = md.list_pairs()
    n_pairs = len(pairs)
    print(f"Total pairs: {n_pairs}")

    def simulate_cascade(cascade_stages, tau_val):
        sim_rows = []
        for fname, i, j in pairs:
            cum_kp = 0
            best_proxy = -1.0
            best_err = 180.0
            for stage_idx, (preset, N) in enumerate(cascade_stages):
                row = idx_eval.loc[(fname, i, j, preset, N)]
                p_epi = float(row["p_epi"])
                err = float(row["pose_err"])
                kp = float(row["actual_kp"])
                cum_kp += kp
                if p_epi > best_proxy:
                    best_proxy = p_epi
                    best_err = err
                if p_epi >= tau_val:
                    break
            sim_rows.append({
                "fname": fname,
                "i": i,
                "j": j,
                "pose_err": best_err,
                "cumulative_actual_kp": cum_kp,
                "succ10": float(best_err < 10.0),
            })
        return pd.DataFrame(sim_rows)

    # 2. Simulate 5 Fold-Selected Cascades
    fold_dfs = []
    for fold, c_name, tau in FOLD_SELECTED:
        stages = CANDIDATE_CASCADES[c_name]
        df_f = simulate_cascade(stages, tau)
        df_f["fold"] = fold
        fold_dfs.append(df_f)
    df_casc_folds = pd.concat(fold_dfs, ignore_index=True)
    assert len(df_casc_folds) == 7500, f"Expected 7500 rows, got {len(df_casc_folds)}"

    # Also simulate C6 (post-hoc)
    df_c6 = simulate_cascade(CANDIDATE_CASCADES["C6"], 0.40)
    assert len(df_c6) == 1500

    # 3. Load Baselines
    baselines = {
        "ALIKED-500": df2a[(df2a["preset"] == "ALIKED") & (df2a["N"] == 500)].sort_values(["fname", "i", "j"]).copy(),
        "ALIKED-1000": df2a[(df2a["preset"] == "ALIKED") & (df2a["N"] == 1000)].sort_values(["fname", "i", "j"]).copy(),
        "SIFT-4000": df2a[(df2a["preset"] == "SIFT") & (df2a["N"] == 4000)].sort_values(["fname", "i", "j"]).copy(),
    }

    # Reference-image clustering
    df_pairs = baselines["ALIKED-500"][["fname", "i", "j"]].drop_duplicates().reset_index(drop=True)
    df_pairs["cluster_id"] = df_pairs.groupby(["fname", "i"]).ngroup()
    unique_clusters = df_pairs["cluster_id"].unique()
    n_clusters = len(unique_clusters)
    print(f"Unique reference-image clusters: {n_clusters}")

    pair_to_cluster = df_pairs.set_index(["fname", "i", "j"])["cluster_id"].to_dict()
    df_casc_folds["cluster_id"] = df_casc_folds.apply(lambda r: pair_to_cluster[(r["fname"], int(r["i"]), int(r["j"]))], axis=1)
    df_c6["cluster_id"] = df_c6.apply(lambda r: pair_to_cluster[(r["fname"], int(r["i"]), int(r["j"]))], axis=1)

    cluster_casc_indices = [np.where(df_casc_folds["cluster_id"].values == c)[0] for c in unique_clusters]
    cluster_c6_indices = [np.where(df_c6["cluster_id"].values == c)[0] for c in unique_clusters]

    cluster_base_indices = {}
    for b_name, b_df in baselines.items():
        b_df["cluster_id"] = b_df.apply(lambda r: pair_to_cluster[(r["fname"], int(r["i"]), int(r["j"]))], axis=1)
        cluster_base_indices[b_name] = [np.where(b_df["cluster_id"].values == c)[0] for c in unique_clusters]

    # Pre-extract arrays
    casc_err = df_casc_folds["pose_err"].values.astype(float)
    casc_kp = df_casc_folds["cumulative_actual_kp"].values.astype(float)
    casc_succ = df_casc_folds["succ10"].values.astype(float)

    c6_err = df_c6["pose_err"].values.astype(float)
    c6_kp = df_c6["cumulative_actual_kp"].values.astype(float)
    c6_succ = df_c6["succ10"].values.astype(float)

    casc_point_succ = np.mean(casc_succ) * 100.0
    casc_point_auc10 = md.pose_auc(casc_err, (10,))[0] * 100.0
    casc_point_kp = np.mean(casc_kp)

    c6_point_succ = np.mean(c6_succ) * 100.0
    c6_point_auc10 = md.pose_auc(c6_err, (10,))[0] * 100.0
    c6_point_kp = np.mean(c6_kp)

    print(f"\nFold-Selected Cascade 5-Fold Pooled Point Estimates:")
    print(f"  Succ@10: {casc_point_succ:.4f}%")
    print(f"  AUC@10:  {casc_point_auc10:.4f}")
    print(f"  Cum KP:  {casc_point_kp:.2f}")

    print(f"\nC6 Cascade (Post-hoc) Point Estimates:")
    print(f"  Succ@10: {c6_point_succ:.4f}%")
    print(f"  AUC@10:  {c6_point_auc10:.4f}")
    print(f"  Cum KP:  {c6_point_kp:.2f}")

    rng = np.random.RandomState(42)
    B = 10000
    print(f"\nRunning {B} reference-image-cluster bootstrap iterations...")
    sampled_cluster_indices = rng.randint(0, n_clusters, size=(B, n_clusters))

    out_rows = []

    # Run comparisons for Fold-Selected Cascade
    for b_name, b_df in baselines.items():
        b_err = b_df["pose_err"].values.astype(float)
        b_kp = b_df["actual_kp"].values.astype(float)
        b_succ = (b_err < 10.0).astype(float)

        b_point_succ = np.mean(b_succ) * 100.0
        b_point_auc10 = md.pose_auc(b_err, (10,))[0] * 100.0
        b_point_kp = np.mean(b_kp)

        delta_succ_point = casc_point_succ - b_point_succ
        delta_auc10_point = casc_point_auc10 - b_point_auc10
        delta_kp_point = casc_point_kp - b_point_kp

        b_clusters = cluster_base_indices[b_name]

        boot_delta_succ = np.zeros(B, dtype=float)
        boot_delta_auc10 = np.zeros(B, dtype=float)
        boot_delta_kp = np.zeros(B, dtype=float)

        t0 = time.time()
        for b_iter in range(B):
            c_sample = sampled_cluster_indices[b_iter]
            idx_casc = np.concatenate([cluster_casc_indices[c] for c in c_sample])
            idx_base = np.concatenate([b_clusters[c] for c in c_sample])

            boot_delta_succ[b_iter] = (np.mean(casc_succ[idx_casc]) - np.mean(b_succ[idx_base])) * 100.0
            boot_delta_auc10[b_iter] = (md.pose_auc(casc_err[idx_casc], (10,))[0] - md.pose_auc(b_err[idx_base], (10,))[0]) * 100.0
            boot_delta_kp[b_iter] = np.mean(casc_kp[idx_casc]) - np.mean(b_kp[idx_base])

        print(f"Finished Fold-Selected vs {b_name} in {time.time() - t0:.1f}s")
        ci_succ_low, ci_succ_high = np.percentile(boot_delta_succ, [2.5, 97.5])
        ci_auc_low, ci_auc_high = np.percentile(boot_delta_auc10, [2.5, 97.5])
        ci_kp_low, ci_kp_high = np.percentile(boot_delta_kp, [2.5, 97.5])

        out_rows.append({
            "target": "Fold-Selected Cascade (5-fold pooled)",
            "baseline": b_name,
            "delta_succ10_pct": float(delta_succ_point),
            "delta_succ10_ci95_low": float(ci_succ_low),
            "delta_succ10_ci95_high": float(ci_succ_high),
            "delta_auc10": float(delta_auc10_point),
            "delta_auc10_ci95_low": float(ci_auc_low),
            "delta_auc10_ci95_high": float(ci_auc_high),
            "delta_kp": float(delta_kp_point),
            "delta_kp_ci95_low": float(ci_kp_low),
            "delta_kp_ci95_high": float(ci_kp_high),
            "target_succ10_pct": float(casc_point_succ),
            "target_auc10": float(casc_point_auc10),
            "target_cum_kp": float(casc_point_kp),
            "baseline_succ10_pct": float(b_point_succ),
            "baseline_auc10": float(b_point_auc10),
            "baseline_kp": float(b_point_kp),
        })

    # Run comparisons for C6 Post-hoc Cascade
    for b_name, b_df in baselines.items():
        b_err = b_df["pose_err"].values.astype(float)
        b_kp = b_df["actual_kp"].values.astype(float)
        b_succ = (b_err < 10.0).astype(float)

        b_point_succ = np.mean(b_succ) * 100.0
        b_point_auc10 = md.pose_auc(b_err, (10,))[0] * 100.0
        b_point_kp = np.mean(b_kp)

        delta_succ_point = c6_point_succ - b_point_succ
        delta_auc10_point = c6_point_auc10 - b_point_auc10
        delta_kp_point = c6_point_kp - b_point_kp

        b_clusters = cluster_base_indices[b_name]

        boot_delta_succ = np.zeros(B, dtype=float)
        boot_delta_auc10 = np.zeros(B, dtype=float)
        boot_delta_kp = np.zeros(B, dtype=float)

        t0 = time.time()
        for b_iter in range(B):
            c_sample = sampled_cluster_indices[b_iter]
            idx_c6 = np.concatenate([cluster_c6_indices[c] for c in c_sample])
            idx_base = np.concatenate([b_clusters[c] for c in c_sample])

            boot_delta_succ[b_iter] = (np.mean(c6_succ[idx_c6]) - np.mean(b_succ[idx_base])) * 100.0
            boot_delta_auc10[b_iter] = (md.pose_auc(c6_err[idx_c6], (10,))[0] - md.pose_auc(b_err[idx_base], (10,))[0]) * 100.0
            boot_delta_kp[b_iter] = np.mean(c6_kp[idx_c6]) - np.mean(b_kp[idx_base])

        print(f"Finished C6 vs {b_name} in {time.time() - t0:.1f}s")
        ci_succ_low, ci_succ_high = np.percentile(boot_delta_succ, [2.5, 97.5])
        ci_auc_low, ci_auc_high = np.percentile(boot_delta_auc10, [2.5, 97.5])
        ci_kp_low, ci_kp_high = np.percentile(boot_delta_kp, [2.5, 97.5])

        out_rows.append({
            "target": "Cascade C6 (tau=0.40, post-hoc)",
            "baseline": b_name,
            "delta_succ10_pct": float(delta_succ_point),
            "delta_succ10_ci95_low": float(ci_succ_low),
            "delta_succ10_ci95_high": float(ci_succ_high),
            "delta_auc10": float(delta_auc10_point),
            "delta_auc10_ci95_low": float(ci_auc_low),
            "delta_auc10_ci95_high": float(ci_auc_high),
            "delta_kp": float(delta_kp_point),
            "delta_kp_ci95_low": float(ci_kp_low),
            "delta_kp_ci95_high": float(ci_kp_high),
            "target_succ10_pct": float(c6_point_succ),
            "target_auc10": float(c6_point_auc10),
            "target_cum_kp": float(c6_point_kp),
            "baseline_succ10_pct": float(b_point_succ),
            "baseline_auc10": float(b_point_auc10),
            "baseline_kp": float(b_point_kp),
        })

    df_out = pd.DataFrame(out_rows)
    out_file = OUTPUT_DIR / "fold_selected_cascade_bootstrap.csv"
    df_out.to_csv(out_file, index=False)
    print(f"\nSaved {out_file}")
    print(df_out.to_string())

if __name__ == "__main__":
    main()
