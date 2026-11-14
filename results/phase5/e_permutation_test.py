import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, 'src')
import megadepth as md

OUTPUT_DIR = Path("results/phase5")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=== RUNNING PERMUTATION ROBUSTNESS TEST ===")

df2a = pd.read_csv("results/phase4b/megadepth_stage2a_priority_evaluations.csv")
df2a_lookup = df2a.set_index(["fname", "i", "j", "preset", "N"])

pairs = md.list_pairs()
perm_seeds = [42, 123, 999]
test_pairs_50 = pairs[:50]

scene_cache = {}

def get_scene_data(fname, i, j):
    if fname not in scene_cache:
        scene_cache[fname] = md._scene(fname)
    sc = scene_cache[fname]
    K0 = sc["intrinsics"][i]
    K1 = sc["intrinsics"][j]
    T_gt = sc["poses"][j] @ np.linalg.inv(sc["poses"][i])
    return K0, K1, T_gt

def run_ransac_permuted(pts0, pts1, K0, K1, T_gt, seed):
    if len(pts0) < 5:
        return 180.0
    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(pts0))
    r = md.estimate_pose(pts0[perm], pts1[perm], K0, K1)
    if r is None:
        return 180.0
    return md.pose_error(T_gt, r[0], r[1])

perm_records = []

# 1. ALIKED-500 and SIFT-4000
for method_name, preset, N in [("ALIKED-500", "ALIKED", 500), ("SIFT-4000", "SIFT", 4000)]:
    print(f"Testing {method_name} across 50 pairs and 3 seeds...")
    orig_errs = []
    perm_errs_by_seed = {s: [] for s in perm_seeds}
    
    for fname, i, j in test_pairs_50:
        K0, K1, T_gt = get_scene_data(fname, i, j)
        
        # Load saved match
        match_f = Path(f"results/phase4b/megadepth_matches/{fname}_{i}_{j}_{preset}_N{N}.npz")
        if match_f.exists():
            d = np.load(match_f)
            pts0, pts1 = d["pts0"], d["pts1"]
        else:
            pts0, pts1 = np.zeros((0, 2)), np.zeros((0, 2))
            
        orig_err = float(df2a_lookup.loc[(fname, i, j, preset, N)]["pose_err"])
        orig_errs.append(orig_err)
        
        for s in perm_seeds:
            pe = run_ransac_permuted(pts0, pts1, K0, K1, T_gt, s)
            perm_errs_by_seed[s].append(pe)
            
    orig_succ10 = float(np.mean(np.array(orig_errs) < 10.0)) * 100.0
    orig_auc10 = float(md.pose_auc(orig_errs)[1]) * 100.0
    
    perm_succ_list = [float(np.mean(np.array(perm_errs_by_seed[s]) < 10.0)) * 100.0 for s in perm_seeds]
    perm_auc_list = [float(md.pose_auc(perm_errs_by_seed[s])[1]) * 100.0 for s in perm_seeds]
    
    perm_records.append({
        "method": method_name,
        "n_pairs": 50,
        "orig_succ10": orig_succ10,
        "perm_mean_succ10": float(np.mean(perm_succ_list)),
        "perm_std_succ10": float(np.std(perm_succ_list)),
        "orig_auc10": orig_auc10,
        "perm_mean_auc10": float(np.mean(perm_auc_list)),
        "perm_std_auc10": float(np.std(perm_auc_list)),
    })

# 2. Cascade C* (tau=0.40)
print("Testing Cascade C* (tau=0.40) across 50 pairs and 3 seeds...")
casc_orig_errs = []
casc_perm_errs = {s: [] for s in perm_seeds}

# Get p_epi for ALIKED-500 and SIFT-4000 on the 50 pairs
for fname, i, j in test_pairs_50:
    K0, K1, T_gt = get_scene_data(fname, i, j)
    
    row_a = df2a_lookup.loc[(fname, i, j, "ALIKED", 500)]
    row_s = df2a_lookup.loc[(fname, i, j, "SIFT", 4000)]
    
    p1 = float(row_a["p_epi"])
    p2 = float(row_s["p_epi"])
    
    if p1 >= 0.40:
        chosen_preset, chosen_N = "ALIKED", 500
        orig_err = float(row_a["pose_err"])
    else:
        chosen_preset, chosen_N = ("ALIKED", 500) if p1 >= p2 else ("SIFT", 4000)
        orig_err = float(row_a["pose_err"]) if p1 >= p2 else float(row_s["pose_err"])
        
    casc_orig_errs.append(orig_err)
    
    match_f = Path(f"results/phase4b/megadepth_matches/{fname}_{i}_{j}_{chosen_preset}_N{chosen_N}.npz")
    if match_f.exists():
        d = np.load(match_f)
        pts0, pts1 = d["pts0"], d["pts1"]
    else:
        pts0, pts1 = np.zeros((0, 2)), np.zeros((0, 2))
        
    for s in perm_seeds:
        pe = run_ransac_permuted(pts0, pts1, K0, K1, T_gt, s)
        casc_perm_errs[s].append(pe)

c_orig_succ = float(np.mean(np.array(casc_orig_errs) < 10.0)) * 100.0
c_orig_auc = float(md.pose_auc(casc_orig_errs)[1]) * 100.0
c_perm_succ = [float(np.mean(np.array(casc_perm_errs[s]) < 10.0)) * 100.0 for s in perm_seeds]
c_perm_auc = [float(md.pose_auc(casc_perm_errs[s])[1]) * 100.0 for s in perm_seeds]

perm_records.append({
    "method": "Cascade C* (tau=0.40)",
    "n_pairs": 50,
    "orig_succ10": c_orig_succ,
    "perm_mean_succ10": float(np.mean(c_perm_succ)),
    "perm_std_succ10": float(np.std(c_perm_succ)),
    "orig_auc10": c_orig_auc,
    "perm_mean_auc10": float(np.mean(c_perm_auc)),
    "perm_std_auc10": float(np.std(c_perm_auc)),
})

df_perm = pd.DataFrame(perm_records)
df_perm.to_csv("results/phase5/e_permutation_robustness_summary.csv", index=False)
print("\nSaved results/phase5/e_permutation_robustness_summary.csv")
print(df_perm.to_string(index=False))

print("\n=== PERMUTATION ROBUSTNESS TEST COMPLETE ===")
