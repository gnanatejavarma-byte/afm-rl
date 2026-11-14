import os
import sys
import json
import hashlib
import numpy as np
import pandas as pd
from pathlib import Path
from stable_baselines3 import PPO

sys.path.insert(0, 'src')
import hpatches
import megadepth as md
import pipeline_utils as pu
from level2_env import Level2Env, load_split
from megadepth_eval_env import MegaDepthEvalEnv, POOL_CACHE_DIR, STATS_CACHE_DIR
from image_stats import compute_pair_stats, normalize_stats

OUTPUT_DIR = Path("results/phase5")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=== RUNNING SECTION A DIAGNOSTICS ===")

# -------------------------------------------------------------
# A.2 OBSERVATION SHIFT ANALYSIS
# -------------------------------------------------------------
print("\n--- A.2 Observation Shift Analysis ---")

# Load L1 and L2 models for fold 0
l1_m0 = PPO.load("checkpoints_v2/fold0/level1_best/best_model.zip")
l2_m0 = PPO.load("checkpoints_v2/fold0/level2_14preset_best/best_model.zip")

# 1. HPatches Fold 0 Test Pairs Step 0 Observations
with open("configs/folds/fold0.json", "r") as f:
    fold0_split = json.load(f)

hp_test_pairs = []
for seq in fold0_split["test"]:
    for k in range(2, 7):
        hp_test_pairs.append((seq, k))

print(f"HPatches Fold 0 test pairs: {len(hp_test_pairs)}")

hp_env = Level2Env(split="val", obs_mode="proxy", eval_mode=True)
hp_obs_list = []

for seq, k in hp_test_pairs:
    img1, img2, H = hpatches.load_pair(seq, k)
    st = normalize_stats(compute_pair_stats(img1, img2))
    a_l1, _ = l1_m0.predict(st, deterministic=True)
    preset = pu.PRESET_NAMES[int(a_l1)]
    obs, _ = hp_env.reset_to(seq, k, preset)
    hp_obs_list.append(obs)

hp_obs_arr = np.array(hp_obs_list)  # (115, 34)
print(f"Computed HPatches Step 0 obs array: {hp_obs_arr.shape}")

# 2. MegaDepth Step 0 Observations across all 1500 pairs
md_pairs = md.list_pairs()
print(f"MegaDepth test pairs: {len(md_pairs)}")

# We can pre-load pool sizes for ALIKED and SIFT
def get_cached_pool_size(scene_d, img_idx, preset):
    p = str(scene_d["image_paths"][img_idx])
    param_str = f"nfeatures=4000_preset={preset}"
    key_str = f"{p}_{preset}_{param_str}"
    h = hashlib.sha256(key_str.encode("utf-8")).hexdigest()
    cache_file = Path("data/pool_cache_megadepth") / preset / f"{h}.npz"
    if cache_file.exists():
        data = np.load(cache_file)
        return len(data["pts"])
    return 4000

# Pre-load stage2a evaluations for N=500 lookups
df2a = pd.read_csv("results/phase4b/megadepth_stage2a_priority_evaluations.csv")
df2a_500 = df2a[df2a["N"] == 500].set_index(["fname", "i", "j", "preset"])

df_s3 = pd.read_csv("results/phase4b/megadepth_stage3_afm_rl_rollouts.csv")
df_s3_fold0 = df_s3[df_s3["fold"] == 0].set_index(["fname", "i", "j"])

md_obs_list = []
scene_cache = {}

for fname, i, j in md_pairs:
    if fname not in scene_cache:
        scene_cache[fname] = md._scene(fname)
    sc = scene_cache[fname]
    
    # Pair stats from disk cache
    stat_file = Path("data/megadepth/stats_cache") / f"{fname}__{i}__{j}.npy"
    if stat_file.exists():
        st = np.load(stat_file)
    else:
        img0, img1, _, _, _ = md.load_pair(fname, i, j)
        st = normalize_stats(compute_pair_stats(img0, img1))
        
    # Get chosen preset from stage 3 fold 0
    row_s3 = df_s3_fold0.loc[(fname, i, j)]
    preset = row_s3["chosen_preset"]
    preset_idx = pu.PRESET_NAMES.index(preset)
    
    # One-hot
    one_hot = np.zeros(len(pu.PRESET_NAMES), dtype=np.float32)
    one_hot[preset_idx] = 1.0
    
    # Get N=500 results
    row_eval = df2a_500.loc[(fname, i, j, preset)]
    p_epi = float(row_eval["p_epi"])
    inlier_ratio = float(row_eval["inlier_ratio"])
    n_matches = float(row_eval["n_matches"])
    actual_kp = float(row_eval["actual_kp"])
    
    # Pool size
    pool1 = get_cached_pool_size(sc, i, preset)
    pool2 = get_cached_pool_size(sc, j, preset)
    pool_size = float(min(pool1, pool2))
    
    budget_norm = float(500 / pu.N_MAX)
    effective_norm = float(actual_kp / pu.N_MAX)
    pool_norm = float(pool_size / pu.N_MAX)
    step_norm = 0.0
    delta_proxy = p_epi
    delta_ratio = inlier_ratio
    delta_matches = float(n_matches / 200.0)
    norm_matches = float(n_matches / 200.0)
    
    # As constructed by MegaDepthEvalEnv._get_obs()
    extra = np.array([
        budget_norm, effective_norm, pool_norm, step_norm,
        p_epi, delta_proxy, inlier_ratio, delta_ratio,
        norm_matches, delta_matches, 0.0
    ], dtype=np.float32)
    
    obs = np.concatenate([st, one_hot, extra]).astype(np.float32)[:34]
    md_obs_list.append(obs)

md_obs_arr = np.array(md_obs_list)  # (1500, 34)
print(f"Computed MegaDepth Step 0 obs array: {md_obs_arr.shape}")

# Dimension names matching the concatenated 34-dim vector
dim_names = [
    "stat_0_avg_texture", "stat_1_tex_change", "stat_2_avg_blur", "stat_3_blur_change",
    "stat_4_avg_corner_density", "stat_5_corner_change", "stat_6_avg_grad_mag", "stat_7_grad_spread",
    "stat_8_intensity_diff"
]
for p in pu.PRESET_NAMES:
    dim_names.append(f"onehot_{p}")
dim_names.extend([
    "env_dim_23_budget_norm", "env_dim_24_effective_norm", "env_dim_25_pool_norm",
    "env_dim_26_step_norm_vs_last_acc", "env_dim_27_last_proxy_vs_inlier_ratio",
    "env_dim_28_delta_proxy_vs_n_matches", "env_dim_29_inlier_ratio_vs_step_norm",
    "env_dim_30_delta_ratio_vs_delta_acc", "env_dim_31_n_matches_vs_delta_utility",
    "env_dim_32_delta_matches_vs_is_plateau", "env_dim_33_placeholder_vs_gap_from_best"
])

obs_records = []
for d in range(34):
    hp_col = hp_obs_arr[:, d]
    md_col = md_obs_arr[:, d]
    
    hp_m, hp_s = float(np.mean(hp_col)), float(np.std(hp_col))
    hp_q05, hp_q50, hp_q95 = float(np.percentile(hp_col, 5)), float(np.percentile(hp_col, 50)), float(np.percentile(hp_col, 95))
    
    md_m, md_s = float(np.mean(md_col)), float(np.std(md_col))
    md_q05, md_q50, md_q95 = float(np.percentile(md_col, 5)), float(np.percentile(md_col, 50)), float(np.percentile(md_col, 95))
    
    std_shift = abs(md_m - hp_m) / (hp_s + 1e-9)
    flagged = std_shift > 3.0
    
    clip_hp = float(np.mean(np.abs(hp_col) >= 4.999))
    clip_md = float(np.mean(np.abs(md_col) >= 4.999))
    
    obs_records.append({
        "dim_idx": d,
        "name": dim_names[d],
        "hp_mean": hp_m,
        "hp_std": hp_s,
        "hp_q05": hp_q05,
        "hp_q50": hp_q50,
        "hp_q95": hp_q95,
        "md_mean": md_m,
        "md_std": md_s,
        "md_q05": md_q05,
        "md_q50": md_q50,
        "md_q95": md_q95,
        "z_shift": std_shift,
        "shifted_gt_3std": flagged,
        "clip_frac_hp": clip_hp,
        "clip_frac_md": clip_md
    })

df_obs = pd.DataFrame(obs_records)
df_obs.to_csv("results/phase5/a2_observation_shift_summary.csv", index=False)
print("Saved results/phase5/a2_observation_shift_summary.csv")

print("\nDimensions shifted by > 3 HP-std:")
print(df_obs[df_obs["shifted_gt_3std"]][["dim_idx", "name", "hp_mean", "md_mean", "z_shift", "clip_frac_md"]].to_string(index=False))

# -------------------------------------------------------------
# A.3 ROLLOUT DIAGNOSTICS
# -------------------------------------------------------------
print("\n--- A.3 Rollout Diagnostics ---")
df_stage3 = pd.read_csv("results/phase4b/megadepth_stage3_afm_rl_rollouts.csv")

import ast
all_actions = []
fold_action_counts = {}

for fold, group in df_stage3.groupby("fold"):
    counts = {0: 0, 1: 0, 2: 0, 3: 0}
    for act_str in group["actions"]:
        acts = ast.literal_eval(act_str)
        all_actions.extend(acts)
        for a in acts:
            counts[a] = counts.get(a, 0) + 1
    fold_action_counts[fold] = counts

total_acts = len(all_actions)
act_hist = {a: all_actions.count(a) for a in range(4)}
act_hist_pct = {a: (all_actions.count(a) / total_acts) * 100.0 for a in range(4)}

print("\nOverall MegaDepth Stage 3 Action Distribution:")
for a in range(4):
    print(f"Action {a}: {act_hist[a]} ({act_hist_pct[a]:.2f}%)")

diag_records = []
for fold, group in df_stage3.groupby("fold"):
    c = fold_action_counts[fold]
    tot = sum(c.values())
    diag_records.append({
        "fold": fold,
        "n_rollouts": len(group),
        "mean_steps": float(group["steps"].mean()),
        "mean_final_N": float(group["final_N"].mean()),
        "mean_cum_kp": float(group["cumulative_kp"].mean()),
        "mean_final_proxy": float(group["p_epi_final"].mean()),
        "act0_count": c[0],
        "act1_count": c[1],
        "act2_count": c[2],
        "act3_count": c[3],
        "pct_action3": (c[3] / tot) * 100.0,
        "pct_action0": (c[0] / tot) * 100.0,
        "pct_terminated_early": float((group["steps"] < 5).mean()) * 100.0,
        "pct_max_steps_5": float((group["steps"] == 5).mean()) * 100.0
    })

df_diag = pd.DataFrame(diag_records)
df_diag.to_csv("results/phase5/a3_rollout_diagnostics.csv", index=False)
print("Saved results/phase5/a3_rollout_diagnostics.csv")
print(df_diag.to_string(index=False))

# Now test on 100 pairs what happens if Action 3 is interpreted as STOP (Level2Env semantics)
print("\n--- Testing Native Level2Env Semantics (Action 3 = STOP) on 100 Pairs ---")
native_records = []
for fold in range(5):
    l1_m = PPO.load(f"checkpoints_v2/fold{fold}/level1_best/best_model.zip")
    l2_m = PPO.load(f"checkpoints_v2/fold{fold}/level2_14preset_best/best_model.zip")
    
    steps_list = []
    actions_taken = []
    
    for idx, (fname, i, j) in enumerate(md_pairs[:100]):
        obs = md_obs_arr[idx] # Step 0 obs
        # Policy predicts on step 0 obs
        act, _ = l2_m.predict(obs, deterministic=True)
        act = int(act)
        actions_taken.append(act)
        if act == 3: # In Level2Env, 3 is STOP
            steps_list.append(1)
        else:
            steps_list.append(2) # Would continue
            
    c3 = actions_taken.count(3)
    tot = len(actions_taken)
    native_records.append({
        "fold": fold,
        "n_pairs": 100,
        "action3_stop_count": c3,
        "pct_action3": (c3 / tot) * 100.0,
        "pct_stopped_immediately_step1": (c3 / tot) * 100.0,
        "mean_steps_if_3_is_stop": float(np.mean(steps_list))
    })

df_native = pd.DataFrame(native_records)
df_native.to_csv("results/phase5/a3_native_semantics_test_100pairs.csv", index=False)
print("Saved results/phase5/a3_native_semantics_test_100pairs.csv")
print(df_native.to_string(index=False))

print("\n=== SECTION A DIAGNOSTICS COMPLETE ===")
