import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
import cv2

sys.path.insert(0, 'src')
import megadepth as md

OUTPUT_DIR = Path("results/phase5")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=== RUNNING SECTION E: RIGOROUS STATISTICS & PERMUTATIONS ===")

# 1. Load data
df2a = pd.read_csv("results/phase4b/megadepth_stage2a_priority_evaluations.csv")
df_s3 = pd.read_csv("results/phase4b/megadepth_stage3_afm_rl_rollouts.csv")

def get_scene(fname):
    return "0015" if "0015" in fname else "0022"

df2a["scene"] = df2a["fname"].apply(get_scene)
df_s3["scene"] = df_s3["fname"].apply(get_scene)

pairs = md.list_pairs()
n_pairs = len(pairs)
scenes = np.array([get_scene(f) for f, i, j in pairs])
mask_0015 = (scenes == "0015")
mask_0022 = (scenes == "0022")

# Reference image clusters (cluster = (fname_scene, i))
# In md._scene(fname), i is the reference image
cluster_ids = []
cluster_map = defaultdict(list)
for idx, (f, i, j) in enumerate(pairs):
    sc_prefix = f.split("_")[0] # '0015' or '0022'
    cid = f"{sc_prefix}_{i}"
    cluster_ids.append(cid)
    cluster_map[cid].append(idx)

cluster_ids = np.array(cluster_ids)
unique_clusters = list(cluster_map.keys())
n_clusters = len(unique_clusters)
cluster_sizes = [len(v) for v in cluster_map.values()]

print(f"Total reference-image clusters: {n_clusters}")
print(f"Cluster sizes: mean={np.mean(cluster_sizes):.2f}, min={np.min(cluster_sizes)}, max={np.max(cluster_sizes)}, median={np.median(cluster_sizes)}")

# Helper to get pair arrays
def get_arrays(preset, N):
    sub = df2a[(df2a["preset"] == preset) & (df2a["N"] == N)].set_index(["fname", "i", "j"])
    p_epi = np.array([float(sub.loc[(f, i, j)]["p_epi"]) for f, i, j in pairs])
    act_kp = np.array([float(sub.loc[(f, i, j)]["actual_kp"]) for f, i, j in pairs])
    pose_err = np.array([float(sub.loc[(f, i, j)]["pose_err"]) for f, i, j in pairs])
    return p_epi, act_kp, pose_err

p_a500, act_a500, err_a500 = get_arrays("ALIKED", 500)
p_a1000, act_a1000, err_a1000 = get_arrays("ALIKED", 1000)
p_s4000, act_s4000, err_s4000 = get_arrays("SIFT", 4000)

# AFM-RL v1 (mean across 5 folds)
s3_piv_err = df_s3.pivot(index=["fname", "i", "j"], columns="fold", values="pose_err")
s3_piv_kp = df_s3.pivot(index=["fname", "i", "j"], columns="fold", values="cumulative_kp")

err_v1_folds = [np.array([float(s3_piv_err.loc[(f, i, j)][fold]) for f, i, j in pairs]) for fold in range(5)]
kp_v1_folds = [np.array([float(s3_piv_kp.loc[(f, i, j)][fold]) for f, i, j in pairs]) for fold in range(5)]

err_v1 = np.mean(err_v1_folds, axis=0)
kp_v1 = np.mean(kp_v1_folds, axis=0)

# Cascade C* (tau=0.40)
s1_stop_c_star = (p_a500 >= 0.40)
err_c_star = np.where(s1_stop_c_star, err_a500, np.where(p_a500 >= p_s4000, err_a500, err_s4000))
kp_c_star = np.where(s1_stop_c_star, act_a500, act_a500 + act_s4000)

# Ablation (i) Always Both
err_abl1 = np.where(p_a500 >= p_s4000, err_a500, err_s4000)
kp_abl1 = act_a500 + act_s4000

# Ablation (ii) Oracle-of-Two
err_abl2 = np.where(err_a500 <= err_s4000, err_a500, err_s4000)
kp_abl2 = act_a500 + act_s4000

# Ablation (iii) Proxy-Pick ALIKED
p_a2000, act_a2000, err_a2000 = get_arrays("ALIKED", 2000)
stack_p = np.stack([p_a500, p_a1000, p_a2000], axis=1)
stack_err = np.stack([err_a500, err_a1000, err_a2000], axis=1)
best_proxy_idx = np.argmax(stack_p, axis=1)
err_abl3 = np.array([stack_err[idx, best_proxy_idx[idx]] for idx in range(len(pairs))])
kp_abl3 = act_a500 + act_a1000 + act_a2000

# Target comparisons:
comparisons = [
    ("AFM-RL v1 (5-fold)", "ALIKED-500", err_v1, kp_v1, err_a500, act_a500),
    ("AFM-RL v1 (5-fold)", "ALIKED-1000", err_v1, kp_v1, err_a1000, act_a1000),
    ("AFM-RL v1 (5-fold)", "SIFT-4000", err_v1, kp_v1, err_s4000, act_s4000),
    ("Cascade C*", "ALIKED-1000", err_c_star, kp_c_star, err_a1000, act_a1000),
    ("Cascade C*", "SIFT-4000", err_c_star, kp_c_star, err_s4000, act_s4000),
    ("Cascade C*", "Ablation (i) Always-Both", err_c_star, kp_c_star, err_abl1, kp_abl1),
    ("Cascade C*", "Ablation (ii) Oracle-of-Two", err_c_star, kp_c_star, err_abl2, kp_abl2),
    ("Cascade C*", "Ablation (iii) Proxy-Pick ALIKED", err_c_star, kp_c_star, err_abl3, kp_abl3),
]

# -------------------------------------------------------------
# 2. BOOTSTRAP HARNESS (Pair-Level and Cluster-Level)
# -------------------------------------------------------------
print("\nRunning Bootstrap Harness (1000 resamples)...")
np.random.seed(42)
N_BOOT = 1000

# Pre-generate resample indices
pair_resamples = np.random.randint(0, n_pairs, size=(N_BOOT, n_pairs))

cluster_indices_arr = np.array(unique_clusters)
cluster_resamples = np.random.choice(cluster_indices_arr, size=(N_BOOT, n_clusters), replace=True)

stat_records = []

for mA_name, mB_name, errA, kpA, errB, kpB in comparisons:
    # Full dataset point estimates
    succA = (errA < 10.0).astype(float)
    succB = (errB < 10.0).astype(float)
    d_succ = float(np.mean(succA) - np.mean(succB)) * 100.0
    
    aucA10 = md.pose_auc(errA)[1] * 100.0
    aucB10 = md.pose_auc(errB)[1] * 100.0
    d_auc10 = aucA10 - aucB10
    
    d_kp = float(np.mean(kpA) - np.mean(kpB))
    
    # Per-scene deltas
    d_succ_0015 = float(np.mean(succA[mask_0015]) - np.mean(succB[mask_0015])) * 100.0
    d_succ_0022 = float(np.mean(succA[mask_0022]) - np.mean(succB[mask_0022])) * 100.0
    
    d_auc10_0015 = (md.pose_auc(errA[mask_0015])[1] - md.pose_auc(errB[mask_0015])[1]) * 100.0
    d_auc10_0022 = (md.pose_auc(errA[mask_0022])[1] - md.pose_auc(errB[mask_0022])[1]) * 100.0
    
    d_kp_0015 = float(np.mean(kpA[mask_0015]) - np.mean(kpB[mask_0015]))
    d_kp_0022 = float(np.mean(kpA[mask_0022]) - np.mean(kpB[mask_0022]))
    
    # Pair-level bootstrap
    pair_d_succ = []
    pair_d_auc10 = []
    pair_d_kp = []
    
    for b in range(N_BOOT):
        idx = pair_resamples[b]
        eA, eB = errA[idx], errB[idx]
        kA, kB = kpA[idx], kpB[idx]
        
        pair_d_succ.append((np.mean(eA < 10.0) - np.mean(eB < 10.0)) * 100.0)
        pair_d_auc10.append((md.pose_auc(eA)[1] - md.pose_auc(eB)[1]) * 100.0)
        pair_d_kp.append(np.mean(kA) - np.mean(kB))
        
    p_succ_ci = (float(np.percentile(pair_d_succ, 2.5)), float(np.percentile(pair_d_succ, 97.5)))
    p_auc10_ci = (float(np.percentile(pair_d_auc10, 2.5)), float(np.percentile(pair_d_auc10, 97.5)))
    p_kp_ci = (float(np.percentile(pair_d_kp, 2.5)), float(np.percentile(pair_d_kp, 97.5)))
    
    # Cluster-level bootstrap
    clust_d_succ = []
    clust_d_auc10 = []
    clust_d_kp = []
    
    for b in range(N_BOOT):
        sampled_cids = cluster_resamples[b]
        # Gather all pair indices for these clusters
        idx_list = []
        for cid in sampled_cids:
            idx_list.extend(cluster_map[cid])
        idx = np.array(idx_list)
        
        eA, eB = errA[idx], errB[idx]
        kA, kB = kpA[idx], kpB[idx]
        
        clust_d_succ.append((np.mean(eA < 10.0) - np.mean(eB < 10.0)) * 100.0)
        clust_d_auc10.append((md.pose_auc(eA)[1] - md.pose_auc(eB)[1]) * 100.0)
        clust_d_kp.append(np.mean(kA) - np.mean(kB))
        
    c_succ_ci = (float(np.percentile(clust_d_succ, 2.5)), float(np.percentile(clust_d_succ, 97.5)))
    c_auc10_ci = (float(np.percentile(clust_d_auc10, 2.5)), float(np.percentile(clust_d_auc10, 97.5)))
    c_kp_ci = (float(np.percentile(clust_d_kp, 2.5)), float(np.percentile(clust_d_kp, 97.5)))
    
    stat_records.append({
        "comparison": f"{mA_name} vs {mB_name}",
        "delta_succ10": d_succ,
        "pair_ci_succ10": f"[{p_succ_ci[0]:.2f}, {p_succ_ci[1]:.2f}]",
        "clust_ci_succ10": f"[{c_succ_ci[0]:.2f}, {c_succ_ci[1]:.2f}]",
        "delta_auc10": d_auc10,
        "pair_ci_auc10": f"[{p_auc10_ci[0]:.2f}, {p_auc10_ci[1]:.2f}]",
        "clust_ci_auc10": f"[{c_auc10_ci[0]:.2f}, {c_auc10_ci[1]:.2f}]",
        "delta_kp": d_kp,
        "pair_ci_kp": f"[{p_kp_ci[0]:.1f}, {p_kp_ci[1]:.1f}]",
        "clust_ci_kp": f"[{c_kp_ci[0]:.1f}, {c_kp_ci[1]:.1f}]",
        "delta_succ10_scene0015": d_succ_0015,
        "delta_succ10_scene0022": d_succ_0022,
        "delta_auc10_scene0015": d_auc10_0015,
        "delta_auc10_scene0022": d_auc10_0022,
        "delta_kp_scene0015": d_kp_0015,
        "delta_kp_scene0022": d_kp_0022
    })

df_stat = pd.DataFrame(stat_records)
df_stat.to_csv("results/phase5/e_statistical_comparisons.csv", index=False)
print("Saved results/phase5/e_statistical_comparisons.csv")
print(df_stat[["comparison", "delta_succ10", "clust_ci_succ10", "delta_auc10", "clust_ci_auc10", "delta_kp", "clust_ci_kp"]].to_string(index=False))

# -------------------------------------------------------------
# 3. PERMUTATION ROBUSTNESS (3 Fixed Permutations on 50 Pairs)
# -------------------------------------------------------------
print("\nRunning Permutation Robustness Test (50 pairs, 3 seeds)...")
perm_seeds = [42, 123, 999]
test_pairs_50 = pairs[:50]

# Essential RANSAC function with permutation using md.estimate_pose
def run_ransac_permuted(pts0, pts1, K0, K1, T_gt, seed):
    if len(pts0) < 5:
        return 180.0
    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(pts0))
    r = md.estimate_pose(pts0[perm], pts1[perm], K0, K1)
    if r is None:
        return 180.0
    return md.pose_error(T_gt, r[0], r[1])

# Measure permutation effect on ALIKED-500, SIFT-4000, and Cascade C*
perm_records = []
scene_cache = {}

for method_name, preset, N in [("ALIKED-500", "ALIKED", 500), ("SIFT-4000", "SIFT", 4000)]:
    orig_errs = []
    perm_errs_by_seed = {s: [] for s in perm_seeds}
    
    for fname, i, j in test_pairs_50:
        if fname not in scene_cache:
            scene_cache[fname] = md._scene(fname)
        sc = scene_cache[fname]
        K0 = sc["intrinsics"][i]
        K1 = sc["intrinsics"][j]
        T_gt = sc["poses"][j] @ np.linalg.inv(sc["poses"][i])
        R_gt = T_gt[:3, :3]
        t_gt = T_gt[:3, 3]
        
        # Load saved match
        match_f = Path(f"results/phase4b/megadepth_matches/{fname}_{i}_{j}_{preset}_N{N}.npz")
        if match_f.exists():
            d = np.load(match_f)
            pts0, pts1 = d["pts0"], d["pts1"]
        else:
            pts0, pts1 = np.zeros((0, 2)), np.zeros((0, 2))
            
        # Original error from df2a
        sub = df2a[(df2a["fname"] == fname) & (df2a["i"] == i) & (df2a["j"] == j) & (df2a["preset"] == preset) & (df2a["N"] == N)]
        orig_err = float(sub["pose_err"].iloc[0])
        orig_errs.append(orig_err)
        
        # Permuted errors
        for s in perm_seeds:
            pe = run_ransac_permuted(pts0, pts1, K0, K1, T_gt, s)
            perm_errs_by_seed[s].append(pe)
            
    orig_succ10 = np.mean(np.array(orig_errs) < 10.0) * 100.0
    orig_auc10 = md.pose_auc(orig_errs)[1] * 100.0
    
    perm_succ_list = [np.mean(np.array(perm_errs_by_seed[s]) < 10.0) * 100.0 for s in perm_seeds]
    perm_auc_list = [md.pose_auc(perm_errs_by_seed[s])[1] * 100.0 for s in perm_seeds]
    
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

# Also Cascade C* on 50 pairs
casc_orig_errs = []
casc_perm_errs = {s: [] for s in perm_seeds}

for idx, (fname, i, j) in enumerate(test_pairs_50):
    sc = scene_cache[fname]
    K0 = sc["intrinsics"][i]
    K1 = sc["intrinsics"][j]
    T_gt = sc["poses"][j] @ np.linalg.inv(sc["poses"][i])
    R_gt = T_gt[:3, :3]
    t_gt = T_gt[:3, 3]
    
    p1 = p_a500[idx]
    if p1 >= 0.40:
        # ALIKED-500
        chosen_preset, chosen_N = "ALIKED", 500
    else:
        # Argmax proxy
        p2 = p_s4000[idx]
        chosen_preset, chosen_N = ("ALIKED", 500) if p1 >= p2 else ("SIFT", 4000)
        
    match_f = Path(f"results/phase4b/megadepth_matches/{fname}_{i}_{j}_{chosen_preset}_N{chosen_N}.npz")
    if match_f.exists():
        d = np.load(match_f)
        pts0, pts1 = d["pts0"], d["pts1"]
    else:
        pts0, pts1 = np.zeros((0, 2)), np.zeros((0, 2))
        
    sub = df2a[(df2a["fname"] == fname) & (df2a["i"] == i) & (df2a["j"] == j) & (df2a["preset"] == chosen_preset) & (df2a["N"] == chosen_N)]
    casc_orig_errs.append(float(sub["pose_err"].iloc[0]))
    
    for s in perm_seeds:
        pe = run_ransac_permuted(pts0, pts1, K0, K1, T_gt, s)
        casc_perm_errs[s].append(pe)

c_orig_succ = np.mean(np.array(casc_orig_errs) < 10.0) * 100.0
c_orig_auc = md.pose_auc(casc_orig_errs)[1] * 100.0
c_perm_succ = [np.mean(np.array(casc_perm_errs[s]) < 10.0) * 100.0 for s in perm_seeds]
c_perm_auc = [md.pose_auc(casc_perm_errs[s])[1] * 100.0 for s in perm_seeds]

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
print("Saved results/phase5/e_permutation_robustness_summary.csv")
print(df_perm.to_string(index=False))

print("\n=== SECTION E COMPLETE ===")
