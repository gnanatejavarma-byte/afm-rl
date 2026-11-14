import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from stable_baselines3 import PPO

sys.path.insert(0, 'src')
import hpatches
import megadepth as md
import pipeline_utils as pu
from level2_env import Level2Env, load_split
from megadepth_eval_env import MegaDepthEvalEnv
from image_stats import compute_pair_stats, normalize_stats

OUTPUT_DIR = Path("results/phase5")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# -------------------------------------------------------------
# A.2 OBSERVATION SHIFT ANALYSIS
# -------------------------------------------------------------
print("Running A.2 Observation Shift Analysis...")

# Load models for fold 0
l1_m0 = PPO.load("checkpoints_v2/fold0/level1_best/best_model.zip")

# 1. HPatches Fold 0 Test Pairs Step 0 Observations
split_data = load_split(None)
hp_test_pairs = list_pairs = []
for seq, pairs in split_data["test"].items():
    for k in pairs:
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

hp_obs_arr = np.array(hp_obs_list)  # (N_hp, 34)

# 2. MegaDepth Step 0 Observations across all 1500 pairs
md_pairs = md.list_pairs()
print(f"MegaDepth test pairs: {len(md_pairs)}")

md_env = MegaDepthEvalEnv(eval_mode=True)
md_obs_list = []

for fname, i, j in md_pairs:
    img0, img1, _, _, _ = md.load_pair(fname, i, j)
    st = md_env._stats(fname, i, j, img0, img1)
    a_l1, _ = l1_m0.predict(st, deterministic=True)
    preset = pu.PRESET_NAMES[int(a_l1)]
    obs, _ = md_env.reset_to(fname, i, j, preset)
    md_obs_list.append(obs)

md_obs_arr = np.array(md_obs_list)  # (1500, 34)

# Dimension names
dim_names = [
    "stat_0_avg_texture", "stat_1_tex_change", "stat_2_avg_blur", "stat_3_blur_change",
    "stat_4_avg_corner_density", "stat_5_corner_change", "stat_6_avg_grad_mag", "stat_7_grad_spread",
    "stat_8_intensity_diff"
]
for p in pu.PRESET_NAMES:
    dim_names.append(f"onehot_{p}")
dim_names.extend([
    "feat_budget_norm", "feat_effective_norm", "feat_pool_norm", "feat_proxy_acc",
    "feat_inlier_ratio", "feat_n_matches_norm", "feat_step_norm", "feat_delta_acc",
    "feat_delta_utility", "feat_is_plateau", "feat_gap_from_best"
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
print("\nFlagged dimensions shifted by > 3 HP-std:")
print(df_obs[df_obs["shifted_gt_3std"]][["dim_idx", "name", "hp_mean", "md_mean", "z_shift"]].to_string(index=False))

# -------------------------------------------------------------
# A.3 ROLLOUT DIAGNOSTICS & ACTION MAPPING ROOT CAUSE
# -------------------------------------------------------------
print("\nRunning A.3 Rollout Diagnostics...")
df_stage3 = pd.read_csv("results/phase4b/megadepth_stage3_afm_rl_rollouts.csv")

# Parse actions from string
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

# Diagnostics summary table
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
print("\n=== A.3 ROLLOUT DIAGNOSTICS SUMMARY ===")
print(df_diag.to_string(index=False))

# Now, test the SAME 5 fold models with CORRECT Level2Env action semantics:
# In Level2Env: Action 3 is STOP, Action 0 is +500, Action 1 is +1000, Action 2 is -500.
# Let's run a test on the first 100 MegaDepth pairs using Level2Env action semantics!
print("\nTesting MegaDepth rollouts under Level2Env native action semantics (Action 3 = STOP)...")
native_records = []
for fold in range(5):
    l1_m = PPO.load(f"checkpoints_v2/fold{fold}/level1_best/best_model.zip")
    l2_m = PPO.load(f"checkpoints_v2/fold{fold}/level2_14preset_best/best_model.zip")
    
    # We will simulate rollouts where the environment obeys Level2Env semantics
    # If policy outputs action 3 -> it stops!
    # If policy outputs 0 -> +500
    # If policy outputs 1 -> +1000
    # If policy outputs 2 -> -500
    steps_list = []
    final_n_list = []
    cum_kp_list = []
    actions_taken = []
    
    for fname, i, j in md_pairs[:100]:
        img0, img1, _, _, _ = md.load_pair(fname, i, j)
        st = md_env._stats(fname, i, j, img0, img1)
        a_l1, _ = l1_m.predict(st, deterministic=True)
        preset = pu.PRESET_NAMES[int(a_l1)]
        
        obs, info = md_env.reset_to(fname, i, j, preset)
        steps = 0
        done = False
        rollout_acts = []
        
        while not done:
            action_l2, _ = l2_m.predict(obs, deterministic=True)
            act = int(action_l2)
            rollout_acts.append(act)
            steps += 1
            
            # If native Level2Env semantics:
            if act == 3: # STOP
                done = True
                break
            elif act == 0:
                md_env.n_kp = min(pu.N_MAX, md_env.n_kp + 500)
            elif act == 1:
                md_env.n_kp = min(pu.N_MAX, md_env.n_kp + 1000)
            elif act == 2:
                md_env.n_kp = max(pu.N_MIN, md_env.n_kp - 500)
                
            if steps >= 15: # Level2Env training max_steps
                done = True
                break
                
            md_env._run_and_eval()
            obs = md_env._get_obs()
            
        steps_list.append(steps)
        final_n_list.append(md_env.n_kp)
        actions_taken.extend(rollout_acts)
        
    native_records.append({
        "fold": fold,
        "mean_steps_native": float(np.mean(steps_list)),
        "mean_final_N_native": float(np.mean(final_n_list)),
        "pct_stopped_step1": float(np.mean([s == 1 for s in steps_list])) * 100.0,
        "action3_stop_count": actions_taken.count(3),
        "total_actions": len(actions_taken),
        "pct_action3": (actions_taken.count(3) / len(actions_taken)) * 100.0
    })

df_native = pd.DataFrame(native_records)
df_native.to_csv("results/phase5/a3_native_semantics_test_100pairs.csv", index=False)
print("\n=== A.3 NATIVE LEVEL2ENV ACTION SEMANTICS ON MEGADEPTH (100 PAIRS) ===")
print(df_native.to_string(index=False))

