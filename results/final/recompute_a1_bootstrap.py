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

def recompute_bootstrap():
    print("=" * 70)
    print("SECTION A.1: RECOMPUTING REFERENCE-IMAGE-CLUSTER BOOTSTRAP FOR MEGADEPTH")
    print("=" * 70)
    
    # 1. Load AFM-RL v1 fixed run
    afm_file = Path("results/phase5b/megadepth_stage3_afm_rl_v1_fixed.csv")
    print(f"Reading AFM-RL v1 data from: {afm_file} (mtime: {time.ctime(afm_file.stat().st_mtime)})")
    df_afm = pd.read_csv(afm_file)
    assert len(df_afm) == 7500, f"Expected 7500 rows, got {len(df_afm)}"
    
    # 2. Load fixed baselines
    base_file = Path("results/phase4b/megadepth_stage2a_priority_evaluations.csv")
    print(f"Reading Baselines data from: {base_file} (mtime: {time.ctime(base_file.stat().st_mtime)})")
    df_base = pd.read_csv(base_file)
    
    baselines = {
        "ALIKED-500": df_base[(df_base["preset"] == "ALIKED") & (df_base["N"] == 500)].sort_values(["fname", "i", "j"]).copy(),
        "ALIKED-1000": df_base[(df_base["preset"] == "ALIKED") & (df_base["N"] == 1000)].sort_values(["fname", "i", "j"]).copy(),
        "SIFT-2000": df_base[(df_base["preset"] == "SIFT") & (df_base["N"] == 2000)].sort_values(["fname", "i", "j"]).copy(),
        "SIFT-4000": df_base[(df_base["preset"] == "SIFT") & (df_base["N"] == 4000)].sort_values(["fname", "i", "j"]).copy(),
    }
    
    for b_name, b_df in baselines.items():
        assert len(b_df) == 1500, f"Expected 1500 rows for {b_name}, got {len(b_df)}"
        
    # Pre-cluster by reference image (fname, i)
    # Get distinct pairs and clusters
    df_pairs = baselines["ALIKED-500"][["fname", "i", "j"]].drop_duplicates().reset_index(drop=True)
    df_pairs["pair_id"] = np.arange(len(df_pairs))
    
    # Create cluster assignment
    df_pairs["cluster_id"] = df_pairs.groupby(["fname", "i"]).ngroup()
    unique_clusters = df_pairs["cluster_id"].unique()
    n_clusters = len(unique_clusters)
    print(f"Total pairs: {len(df_pairs)}, Unique reference-image clusters: {n_clusters}")
    
    # Map pair_id to cluster_id
    pair_to_cluster = df_pairs.set_index(["fname", "i", "j"])["cluster_id"].to_dict()
    df_afm["cluster_id"] = df_afm.apply(lambda r: pair_to_cluster[(r["fname"], int(r["i"]), int(r["j"]))], axis=1)
    
    # Cluster index maps
    cluster_afm_indices = [np.where(df_afm["cluster_id"].values == c)[0] for c in unique_clusters]
    
    cluster_base_indices = {}
    for b_name, b_df in baselines.items():
        b_df["cluster_id"] = b_df.apply(lambda r: pair_to_cluster[(r["fname"], int(r["i"]), int(r["j"]))], axis=1)
        cluster_base_indices[b_name] = [np.where(b_df["cluster_id"].values == c)[0] for c in unique_clusters]

    # Pre-extract arrays
    afm_err = df_afm["pose_err"].values.astype(float)
    afm_kp = df_afm["cumulative_actual_kp"].values.astype(float)
    afm_succ = (afm_err < 10.0).astype(float)
    
    afm_point_succ = np.mean(afm_succ) * 100.0
    afm_point_auc10 = md.pose_auc(afm_err, (10,))[0] * 100.0
    afm_point_kp = np.mean(afm_kp)
    
    print(f"AFM-RL v1 (Fixed) Pooled Point Estimates:")
    print(f"  Succ@10: {afm_point_succ:.4f}%")
    print(f"  AUC@10:  {afm_point_auc10:.4f}")
    print(f"  Cum KP:  {afm_point_kp:.2f}")

    rng = np.random.RandomState(42)
    B = 10000
    print(f"\nRunning {B} reference-image-cluster bootstrap iterations...")
    
    # Pre-generate cluster sample indices: shape (B, n_clusters)
    sampled_cluster_indices = rng.randint(0, n_clusters, size=(B, n_clusters))
    
    out_rows = []
    
    for b_name, b_df in baselines.items():
        b_err = b_df["pose_err"].values.astype(float)
        b_kp = b_df["actual_kp"].values.astype(float)
        b_succ = (b_err < 10.0).astype(float)
        
        b_point_succ = np.mean(b_succ) * 100.0
        b_point_auc10 = md.pose_auc(b_err, (10,))[0] * 100.0
        b_point_kp = np.mean(b_kp)
        
        delta_succ_point = afm_point_succ - b_point_succ
        delta_auc10_point = afm_point_auc10 - b_point_auc10
        delta_kp_point = afm_point_kp - b_point_kp
        
        print(f"\nComparison: AFM-RL v1 vs {b_name}")
        print(f"  Baseline Point: Succ@10={b_point_succ:.4f}%, AUC@10={b_point_auc10:.4f}, KP={b_point_kp:.2f}")
        print(f"  Point Diffs: DeltaSucc={delta_succ_point:.4f}%, DeltaAUC={delta_auc10_point:.4f}, DeltaKP={delta_kp_point:.2f}")
        
        b_clusters = cluster_base_indices[b_name]
        
        boot_delta_succ = np.zeros(B, dtype=float)
        boot_delta_auc10 = np.zeros(B, dtype=float)
        boot_delta_kp = np.zeros(B, dtype=float)
        
        t0 = time.time()
        for b_iter in range(B):
            c_sample = sampled_cluster_indices[b_iter]
            
            # Gather indices
            idx_afm = np.concatenate([cluster_afm_indices[c] for c in c_sample])
            idx_base = np.concatenate([b_clusters[c] for c in c_sample])
            
            s_afm = np.mean(afm_succ[idx_afm]) * 100.0
            s_base = np.mean(b_succ[idx_base]) * 100.0
            boot_delta_succ[b_iter] = s_afm - s_base
            
            # AUC@10 via trapezoidal integration
            auc_afm = md.pose_auc(afm_err[idx_afm], (10,))[0] * 100.0
            auc_base = md.pose_auc(b_err[idx_base], (10,))[0] * 100.0
            boot_delta_auc10[b_iter] = auc_afm - auc_base
            
            kp_afm_b = np.mean(afm_kp[idx_afm])
            kp_base_b = np.mean(b_kp[idx_base])
            boot_delta_kp[b_iter] = kp_afm_b - kp_base_b
            
        print(f"  Bootstrap finished in {time.time() - t0:.1f}s")
        
        ci_succ_low, ci_succ_high = np.percentile(boot_delta_succ, [2.5, 97.5])
        ci_auc_low, ci_auc_high = np.percentile(boot_delta_auc10, [2.5, 97.5])
        ci_kp_low, ci_kp_high = np.percentile(boot_delta_kp, [2.5, 97.5])
        
        out_rows.append({
            "comparison": f"AFM-RL v1 (fixed) vs {b_name}",
            "delta_succ10_pct": float(delta_succ_point),
            "delta_succ10_ci95_low": float(ci_succ_low),
            "delta_succ10_ci95_high": float(ci_succ_high),
            "delta_auc10": float(delta_auc10_point),
            "delta_auc10_ci95_low": float(ci_auc_low),
            "delta_auc10_ci95_high": float(ci_auc_high),
            "delta_kp": float(delta_kp_point),
            "delta_kp_ci95_low": float(ci_kp_low),
            "delta_kp_ci95_high": float(ci_kp_high),
            "afm_succ10_pct": float(afm_point_succ),
            "afm_auc10": float(afm_point_auc10),
            "afm_cum_kp": float(afm_point_kp),
            "baseline_succ10_pct": float(b_point_succ),
            "baseline_auc10": float(b_point_auc10),
            "baseline_kp": float(b_point_kp)
        })
        
    df_res = pd.DataFrame(out_rows)
    out_csv = OUTPUT_DIR / "a1_bootstrap_comparisons_fixed.csv"
    df_res.to_csv(out_csv, index=False)
    print(f"\nSaved fixed bootstrap results to {out_csv}:")
    print(df_res[["comparison", "delta_succ10_pct", "delta_succ10_ci95_low", "delta_succ10_ci95_high", 
                  "delta_auc10", "delta_auc10_ci95_low", "delta_auc10_ci95_high", 
                  "delta_kp", "delta_kp_ci95_low", "delta_kp_ci95_high"]].to_string(index=False))

if __name__ == "__main__":
    recompute_bootstrap()
