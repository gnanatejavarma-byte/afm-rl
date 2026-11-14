import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path

# Add src to path
sys.path.insert(0, 'src')
import megadepth as md

OUTPUT_DIR = Path("results/phase5b")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def generate_complete_tables():
    print("=" * 70)
    print("SECTION E: GENERATING REPRODUCIBLE BENCHMARK TABLES (MEGADEPTH & HPATCHES)")
    print("=" * 70)
    
    # -------------------------------------------------------------
    # 1. MEGADEPTH BENCHMARK TABLE
    # -------------------------------------------------------------
    print("Generating MegaDepth complete results table from CSVs only...")
    df2a = pd.read_csv("results/phase4b/megadepth_stage2a_priority_evaluations.csv")
    df2b = pd.read_csv("results/phase4b/megadepth_stage2b_secondary_evaluations.csv")
    df_all_configs = pd.concat([df2a, df2b], ignore_index=True)
    
    # CPU latency lookup from Phase 5a Section F
    df_cpu_lat = pd.read_csv("results/phase5/f_wallclock_summary.csv").set_index("method")
    
    # GPU latency lookup from Phase 5b Section F.3
    f3_csv = OUTPUT_DIR / "f3_gpu_latency_summary.csv"
    gpu_lat_map = {}
    if f3_csv.exists():
        df_gpu_lat = pd.read_csv(f3_csv).set_index("method")
        if "ALIKED-500 (GPU)" in df_gpu_lat.index:
            gpu_lat_map["ALIKED-500"] = df_gpu_lat.loc["ALIKED-500 (GPU)"]["mean_latency_ms"]
        if "ALIKED-1000 (GPU)" in df_gpu_lat.index:
            gpu_lat_map["ALIKED-1000"] = df_gpu_lat.loc["ALIKED-1000 (GPU)"]["mean_latency_ms"]
        if "Cascade C6 (tau=0.40)" in df_gpu_lat.index:
            gpu_lat_map["Cascade C* (tau=0.40) [ALIKED-500 -> SIFT-4000]"] = df_gpu_lat.loc["Cascade C6 (tau=0.40)"]["mean_latency_ms"]
        if "Fold-Selected Cascade C4 (tau=0.40)" in df_gpu_lat.index:
            gpu_lat_map["Cascade (HPatches 5-Fold Inner-CV Selected Mean)"] = df_gpu_lat.loc["Fold-Selected Cascade C4 (tau=0.40)"]["mean_latency_ms"]
    
    # Target configurations to include
    target_baselines = [
        ("SIFT", 500, 500),
        ("SIFT", 1000, 1000),
        ("SIFT", 2000, 2000),
        ("SIFT", 4000, 4000),
        ("ALIKED", 500, 500),
        ("ALIKED", 1000, 1000),
        ("ALIKED", 2000, 2000),
        ("ALIKED", 4000, 4000),
        ("XFeat", 500, 500),
        ("XFeat", 1000, 1000),
        ("XFeat", 2000, 2000),
        ("XFeat", 4000, 4000),
        ("ORB", 1000, 1000),
        ("ORB", 4000, 4000),
    ]
    
    md_rows = []
    
    for preset, n_val, req_n in target_baselines:
        sub = df_all_configs[(df_all_configs["preset"] == preset) & (df_all_configs["N"] == n_val)]
        if len(sub) == 0:
            continue
        errs = sub["pose_err"].values
        succ10 = (errs < 10.0).mean() * 100.0
        auc5, auc10, auc20 = [v * 100.0 for v in md.pose_auc(errs)]
        act_kp = sub["actual_kp"].mean()
        
        m_name = f"{preset}-{n_val}"
        cpu_lat = df_cpu_lat.loc[m_name]["cold_wallclock_ms"] if m_name in df_cpu_lat.index else np.nan
        
        md_rows.append({
            "Method": m_name,
            "Category": "Fixed Baseline",
            "Succ@10_deg": succ10,
            "AUC@5_deg": auc5,
            "AUC@10_deg": auc10,
            "AUC@20_deg": auc20,
            "Requested_N": req_n,
            "Actual_Cum_KP": act_kp,
            "Wallclock_CPU_ms": cpu_lat,
            "Wallclock_GPU_ms": gpu_lat_map.get(m_name, np.nan)
        })
        
    # Add Fixed AFM-RL v1 (5-fold pooled) if available
    stage3_fixed_csv = OUTPUT_DIR / "megadepth_stage3_afm_rl_v1_fixed.csv"
    if stage3_fixed_csv.exists():
        df_v1 = pd.read_csv(stage3_fixed_csv)
        errs_v1 = df_v1["pose_err"].values
        succ10_v1 = (errs_v1 < 10.0).mean() * 100.0
        auc5_v1, auc10_v1, auc20_v1 = [v * 100.0 for v in md.pose_auc(errs_v1)]
        req_n_v1 = df_v1["final_requested_N"].mean()
        act_kp_v1 = df_v1["cumulative_actual_kp"].mean()
        
        cpu_lat_v1 = df_cpu_lat.loc["AFM-RL v1 (5-fold mean)"]["cold_wallclock_ms"] if "AFM-RL v1 (5-fold mean)" in df_cpu_lat.index else np.nan
        
        md_rows.append({
            "Method": "AFM-RL v1 (Fixed, 5-Fold Mean)",
            "Category": "Adaptive RL Agent",
            "Succ@10_deg": succ10_v1,
            "AUC@5_deg": auc5_v1,
            "AUC@10_deg": auc10_v1,
            "AUC@20_deg": auc20_v1,
            "Requested_N": req_n_v1,
            "Actual_Cum_KP": act_kp_v1,
            "Wallclock_CPU_ms": cpu_lat_v1,
            "Wallclock_GPU_ms": np.nan
        })
        
    # Add Cascade C* (tau=0.40) from Phase 5a Section C
    df_c_star = pd.read_csv("results/phase5/c_hpatches_preregistered_c_star.csv")
    row_c_star = df_c_star[df_c_star["split"] == "Pooled (All 1500)"].iloc[0]
    cpu_lat_casc = df_cpu_lat.loc["Cascade C* (tau=0.40)"]["cold_wallclock_ms"] if "Cascade C* (tau=0.40)" in df_cpu_lat.index else np.nan
    md_rows.append({
        "Method": "Cascade C* (tau=0.40) [ALIKED-500 -> SIFT-4000]",
        "Category": "Label-Free Cascade",
        "Succ@10_deg": float(row_c_star["succ10"]),
        "AUC@5_deg": float(row_c_star["auc5"]),
        "AUC@10_deg": float(row_c_star["auc10"]),
        "AUC@20_deg": float(row_c_star["auc20"]),
        "Requested_N": 2013.49,
        "Actual_Cum_KP": float(row_c_star["cum_actual_kp"]),
        "Wallclock_CPU_ms": cpu_lat_casc,
        "Wallclock_GPU_ms": gpu_lat_map.get("Cascade C* (tau=0.40) [ALIKED-500 -> SIFT-4000]", np.nan)
    })
    
    # Add Fold-Selected Cascade Mean from Section C.2
    c2_csv = OUTPUT_DIR / "c2_fold_selected_cascades_summary.csv"
    if c2_csv.exists():
        df_c2 = pd.read_csv(c2_csv)
        sub_pooled = df_c2[df_c2["scope"] == "Pooled (All 1500)"]
        md_rows.append({
            "Method": "Cascade (HPatches 5-Fold Inner-CV Selected Mean)",
            "Category": "Label-Free Cascade",
            "Succ@10_deg": float(sub_pooled["succ10"].mean()),
            "AUC@5_deg": float(sub_pooled["auc5"].mean()),
            "AUC@10_deg": float(sub_pooled["auc10"].mean()),
            "AUC@20_deg": float(sub_pooled["auc20"].mean()),
            "Requested_N": float(sub_pooled["cum_actual_kp"].mean()),
            "Actual_Cum_KP": float(sub_pooled["cum_actual_kp"].mean()),
            "Wallclock_CPU_ms": cpu_lat_casc,
            "Wallclock_GPU_ms": gpu_lat_map.get("Cascade (HPatches 5-Fold Inner-CV Selected Mean)", np.nan)
        })
        
    # Add Ablations
    df_abl = pd.read_csv("results/phase5/d_ablations_summary.csv")
    for _, r_abl in df_abl[df_abl["scene"] == "Pooled"].iterrows():
        if r_abl["method"].startswith("(i)") or r_abl["method"].startswith("(ii)"):
            md_rows.append({
                "Method": r_abl["method"],
                "Category": "Ablation Control",
                "Succ@10_deg": float(r_abl["succ10"]),
                "AUC@5_deg": float(r_abl["auc5"]),
                "AUC@10_deg": float(r_abl["auc10"]),
                "AUC@20_deg": float(r_abl["auc20"]),
                "Requested_N": float(r_abl["cum_actual_kp"]),
                "Actual_Cum_KP": float(r_abl["cum_actual_kp"]),
                "Wallclock_CPU_ms": np.nan,
                "Wallclock_GPU_ms": np.nan
            })
            
    df_md_table = pd.DataFrame(md_rows)
    df_md_table.to_csv(OUTPUT_DIR / "table_megadepth_complete.csv", index=False)
    print("Saved results/phase5b/table_megadepth_complete.csv")
    
    # -------------------------------------------------------------
    # 2. HPATCHES BENCHMARK TABLE
    # -------------------------------------------------------------
    print("Generating HPatches complete results table from CSVs only...")
    df_hp = pd.read_csv("results/phase3/folds_summary_table.csv")
    
    # Filter methods for HPatches table
    target_hp_methods = [
        "SIFT Fixed-500", "SIFT Fixed-1000", "SIFT Fixed-2000", "SIFT Fixed-4000",
        "ALIKED-500", "ALIKED+Fixed-4000",
        "XFeat-1000", "XFeat+Fixed-4000",
        "ORB+Fixed-4000",
        "Full AFM-RL (Strict STOP)",
        "Best Fixed (Fold VAL)",
        "Oracle Preset Selection"
    ]
    
    hp_rows = []
    for m in target_hp_methods:
        sub = df_hp[df_hp["Method"] == m]
        if len(sub) == 0:
            continue
        r = sub.iloc[0]
        hp_rows.append({
            "Method": m,
            "Succ@3px": float(r["Succ@3px"]),
            "AUC@1px": float(r["AUC@1px"]),
            "AUC@3px": float(r["AUC@3px"]),
            "AUC@5px": float(r["AUC@5px"]),
            "Final_KP": float(r["Final_KP"]),
            "Cum_KP": float(r["Cum_KP"]),
            "Mean_Steps": float(r["Mean_Steps"]),
            "Wallclock_CPU_ms": np.nan,
            "Wallclock_GPU_ms": np.nan
        })
        
    df_hp_table = pd.DataFrame(hp_rows)
    df_hp_table.to_csv(OUTPUT_DIR / "table_hpatches_complete.csv", index=False)
    print("Saved results/phase5b/table_hpatches_complete.csv")
    
    print("\n=== MEGADEPTH TABLE PREVIEW ===")
    print(df_md_table[["Method", "Succ@10_deg", "AUC@10_deg", "Actual_Cum_KP", "Wallclock_CPU_ms", "Wallclock_GPU_ms"]].to_string(index=False))
    
    print("\n=== HPATCHES TABLE PREVIEW ===")
    print(df_hp_table[["Method", "Succ@3px", "AUC@3px", "Cum_KP", "Mean_Steps"]].to_string(index=False))

if __name__ == "__main__":
    generate_complete_tables()
