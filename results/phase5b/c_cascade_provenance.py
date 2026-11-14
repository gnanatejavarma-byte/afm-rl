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

def run_cascade_provenance():
    print("=" * 70)
    print("SECTION C: CASCADE PROVENANCE, FOLD-SELECTED CASCADES & SENSITIVITY TABLE")
    print("=" * 70)
    
    df2a = pd.read_csv("results/phase4b/megadepth_stage2a_priority_evaluations.csv")
    df2b = pd.read_csv("results/phase4b/megadepth_stage2b_secondary_evaluations.csv")
    df_all = pd.concat([df2a, df2b], ignore_index=True)
    idx_eval = df_all.set_index(["fname", "i", "j", "preset", "N"])
    
    pairs = md.list_pairs()
    n_pairs = len(pairs)
    pair_scenes = ["0015" if p[0].startswith("0015") else "0022" for p in pairs]
    mask_0015 = np.array([s == "0015" for s in pair_scenes])
    mask_0022 = np.array([s == "0022" for s in pair_scenes])
    
    def simulate_cascade_on_pairs(cascade_stages, tau_val):
        """Simulates cascade on all 1500 pairs given stages and escalation threshold tau."""
        sim_errs = []
        sim_kps = []
        s1_stops = []
        
        for fname, i, j in pairs:
            cum_kp = 0
            best_proxy = -1.0
            best_err = 180.0
            stopped = False
            
            for stage_idx, (preset, N) in enumerate(cascade_stages):
                row = idx_eval.loc[(fname, i, j, preset, N)]
                p_epi = float(row["p_epi"])
                err = float(row["pose_err"])
                kp = float(row["actual_kp"])
                cum_kp += kp
                
                if p_epi > best_proxy:
                    best_proxy = p_epi
                    best_err = err
                    
                if stage_idx == 0 and p_epi >= tau_val:
                    s1_stops.append(True)
                    stopped = True
                    break
                elif stage_idx == 0:
                    s1_stops.append(False)
                    
                if p_epi >= tau_val:
                    stopped = True
                    break
                    
            sim_errs.append(best_err)
            sim_kps.append(cum_kp)
            
        return np.array(sim_errs), np.array(sim_kps), np.array(s1_stops)

    # -------------------------------------------------------------
    # C.2: Evaluate all 5 Fold-Selected Cascades
    # -------------------------------------------------------------
    print("\n--- C.2: Evaluating 5 Fold-Selected Cascades from HPatches Inner-CV ---")
    fold_records = []
    
    for fold, c_name, tau in FOLD_SELECTED:
        stages = CANDIDATE_CASCADES[c_name]
        errs, kps, s1 = simulate_cascade_on_pairs(stages, tau)
        
        for sc_name, msk in [("Scene 0015", mask_0015), ("Scene 0022", mask_0022), ("Pooled (All 1500)", np.ones(n_pairs, dtype=bool))]:
            e = errs[msk]
            k = kps[msk]
            s = s1[msk]
            
            succ10 = (e < 10.0).mean() * 100.0
            auc5, auc10, auc20 = [v * 100.0 for v in md.pose_auc(e)]
            mean_kp = k.mean()
            utility = (succ10 / 100.0) - (0.05 / 2000.0) * mean_kp
            stop_s1_pct = s.mean() * 100.0
            
            fold_records.append({
                "fold": fold,
                "cascade_name": c_name,
                "tau": tau,
                "cascade_definition": str([f"{p}-{n}" for p, n in stages]),
                "scope": sc_name,
                "succ10": succ10,
                "auc5": auc5,
                "auc10": auc10,
                "auc20": auc20,
                "cum_actual_kp": mean_kp,
                "step1_stop_rate_pct": stop_s1_pct,
                "utility": utility
            })
            
    df_folds = pd.DataFrame(fold_records)
    df_folds.to_csv(OUTPUT_DIR / "c2_fold_selected_cascades_summary.csv", index=False)
    print("Saved results/phase5b/c2_fold_selected_cascades_summary.csv:")
    print(df_folds[df_folds["scope"] == "Pooled (All 1500)"][["fold", "cascade_name", "tau", "succ10", "auc10", "cum_actual_kp", "utility"]].to_string(index=False))
    
    # Compute 5-fold mean +- std across folds (Pooled)
    pooled_folds = df_folds[df_folds["scope"] == "Pooled (All 1500)"]
    print(f"\n5-Fold Mean +- Std (Pooled):")
    print(f"Succ@10: {pooled_folds['succ10'].mean():.2f}% +- {pooled_folds['succ10'].std():.2f}%")
    print(f"AUC@10:  {pooled_folds['auc10'].mean():.2f} +- {pooled_folds['auc10'].std():.2f}")
    print(f"KP:      {pooled_folds['cum_actual_kp'].mean():.1f} +- {pooled_folds['cum_actual_kp'].std():.1f}")
    print(f"Utility: {pooled_folds['utility'].mean():.4f} +- {pooled_folds['utility'].std():.4f}")

    # -------------------------------------------------------------
    # C.3: Sensitivity Table: Each of C1-C6 x tau in {0.2, 0.3, 0.4, 0.5}
    # -------------------------------------------------------------
    print("\n--- C.3: Sensitivity Table: C1-C6 x tau in {0.2, 0.3, 0.4, 0.5} (All 1500 Pairs) ---")
    sens_records = []
    tau_grid = [0.2, 0.3, 0.4, 0.5]
    
    for c_name, stages in CANDIDATE_CASCADES.items():
        for tau in tau_grid:
            errs, kps, s1 = simulate_cascade_on_pairs(stages, tau)
            
            for sc_name, msk in [("Scene 0015", mask_0015), ("Scene 0022", mask_0022), ("Pooled", np.ones(n_pairs, dtype=bool))]:
                e = errs[msk]
                k = kps[msk]
                s = s1[msk]
                
                succ10 = (e < 10.0).mean() * 100.0
                auc5, auc10, auc20 = [v * 100.0 for v in md.pose_auc(e)]
                mean_kp = k.mean()
                utility = (succ10 / 100.0) - (0.05 / 2000.0) * mean_kp
                
                sens_records.append({
                    "cascade": c_name,
                    "tau": tau,
                    "stages": str([f"{p}-{n}" for p, n in stages]),
                    "scene": sc_name,
                    "succ10": succ10,
                    "auc5": auc5,
                    "auc10": auc10,
                    "auc20": auc20,
                    "cum_actual_kp": mean_kp,
                    "step1_stop_rate_pct": s.mean() * 100.0,
                    "utility": utility
                })
                
    df_sens = pd.DataFrame(sens_records)
    df_sens.to_csv(OUTPUT_DIR / "c3_sensitivity_table_all_cascades.csv", index=False)
    print("Saved results/phase5b/c3_sensitivity_table_all_cascades.csv")
    print("\nSensitivity Table (Pooled, C1-C6 x tau):")
    print(df_sens[df_sens["scene"] == "Pooled"][["cascade", "tau", "succ10", "auc10", "cum_actual_kp", "utility"]].to_string(index=False))

if __name__ == "__main__":
    run_cascade_provenance()
