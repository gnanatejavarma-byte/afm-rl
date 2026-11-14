import os
import sys
import ast
import json
import hashlib
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, 'src')
import megadepth as md

OUTPUT_DIR = Path("results/phase5")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=== RUNNING SECTION B: KEYPOINT ACCOUNTING ===")

# 1. Load all evaluations
df2a = pd.read_csv("results/phase4b/megadepth_stage2a_priority_evaluations.csv")
df2b = pd.read_csv("results/phase4b/megadepth_stage2b_secondary_evaluations.csv")
df_eval = pd.concat([df2a, df2b], ignore_index=True)

# Scene assignment helper
def get_scene(fname):
    return "0015" if "0015" in fname else "0022"

df_eval["scene"] = df_eval["fname"].apply(get_scene)

# Load stage3 rollouts
df_s3 = pd.read_csv("results/phase4b/megadepth_stage3_afm_rl_rollouts.csv")
df_s3["scene"] = df_s3["fname"].apply(get_scene)

# Cache pool sizes
def load_pool_sizes():
    pool_cache_dir = Path("data/pool_cache_megadepth")
    # For SIFT and ALIKED, compute per image
    import megadepth as md
    pairs = md.list_pairs()
    scene_cache = {}
    pool_sizes = {} # (fname, img_idx, preset) -> size
    for fname, i, j in pairs:
        if fname not in scene_cache:
            scene_cache[fname] = md._scene(fname)
        sc = scene_cache[fname]
        for idx in (i, j):
            p = str(sc["image_paths"][idx])
            for preset in ["SIFT", "ALIKED", "XFeat", "ORB"]:
                key = (fname, idx, preset)
                if key in pool_sizes:
                    continue
                param_str = f"nfeatures=4000_preset={preset}"
                key_str = f"{p}_{preset}_{param_str}"
                h = hashlib.sha256(key_str.encode("utf-8")).hexdigest()
                cf = pool_cache_dir / preset / f"{h}.npz"
                if cf.exists():
                    d = np.load(cf)
                    pool_sizes[key] = len(d["pts"])
                else:
                    pool_sizes[key] = 4000
    return pool_sizes

print("Loading pool sizes...")
pool_sizes_dict = load_pool_sizes()

# Pre-compute pool size per pair and preset
import megadepth as md
pairs = md.list_pairs()
pair_pool_dict = {}
for fname, i, j in pairs:
    for preset in ["SIFT", "ALIKED", "XFeat", "ORB"]:
        p1 = pool_sizes_dict.get((fname, i, preset), 4000)
        p2 = pool_sizes_dict.get((fname, j, preset), 4000)
        pair_pool_dict[(fname, i, j, preset)] = min(p1, p2)

# Summarizer helper
records = []

def analyze_subset(df_sub, method_name, scene_name):
    req = df_sub["requested_kp"].values
    act = df_sub["actual_kp"].values
    pool = df_sub["pool_size"].values
    
    n_pairs = len(df_sub)
    mean_req = float(np.mean(req))
    mean_act = float(np.mean(act))
    mean_pool = float(np.mean(pool))
    
    shortfall = np.maximum(0, req - act)
    pct_shortfall = float(np.mean(act < req)) * 100.0
    mean_short = float(np.mean(shortfall))
    max_short = float(np.max(shortfall))
    
    return {
        "method": method_name,
        "scene": scene_name,
        "n_pairs": n_pairs,
        "mean_requested_kp": mean_req,
        "mean_actual_kp": mean_act,
        "mean_pool_size": mean_pool,
        "pct_actual_lt_requested": pct_shortfall,
        "mean_shortfall_kp": mean_short,
        "max_shortfall_kp": max_short
    }

# 1. FIXED BASELINES
fixed_targets = [
    ("SIFT", 500), ("SIFT", 1000), ("SIFT", 2000), ("SIFT", 4000),
    ("ALIKED", 500), ("ALIKED", 1000), ("ALIKED", 2000), ("ALIKED", 4000),
    ("XFeat", 500), ("XFeat", 1000), ("XFeat", 2000), ("XFeat", 4000),
    ("ORB", 1000), ("ORB", 4000)
]

for preset, N in fixed_targets:
    sub = df_eval[(df_eval["preset"] == preset) & (df_eval["N"] == N)].copy()
    sub["requested_kp"] = N
    # pool_size
    sub["pool_size"] = [pair_pool_dict.get((row["fname"], row["i"], row["j"], preset), row["actual_kp"]) for _, row in sub.iterrows()]
    
    m_name = f"{preset}-{N}"
    for sc in ["0015", "0022"]:
        records.append(analyze_subset(sub[sub["scene"] == sc], m_name, sc))
    records.append(analyze_subset(sub, m_name, "Pooled"))

# 2. AFM-RL v1 (5 Folds)
for fold in range(5):
    sub = df_s3[df_s3["fold"] == fold].copy()
    # In stage 3: cumulative_kp is actual cumulative KP
    # We reconstruct requested cumulative KP from actions
    req_cum_list = []
    pool_list = []
    
    for _, row in sub.iterrows():
        acts = ast.literal_eval(row["actions"])
        # In MegaDepthEvalEnv: start at 500.
        # Action 0: STOP
        # Action 1: +500
        # Action 2: -500
        # Action 3: +1000
        curr_N = 500
        req_sum = 500
        for a in acts:
            if a == 0:
                break
            elif a == 1:
                curr_N = min(4000, curr_N + 500)
            elif a == 2:
                curr_N = max(100, curr_N - 500)
            elif a == 3:
                curr_N = min(4000, curr_N + 1000)
            req_sum += curr_N
        req_cum_list.append(req_sum)
        preset = row["chosen_preset"]
        pool_list.append(pair_pool_dict.get((row["fname"], row["i"], row["j"], preset), 4000))
        
    sub["requested_kp"] = req_cum_list
    sub["actual_kp"] = sub["cumulative_kp"]
    sub["pool_size"] = pool_list
    
    m_name = f"AFM-RL_v1_fold{fold}"
    for sc in ["0015", "0022"]:
        records.append(analyze_subset(sub[sub["scene"] == sc], m_name, sc))
    records.append(analyze_subset(sub, m_name, "Pooled"))

# 3. PRE-REGISTERED CASCADE C* (Amendment 1: [ALIKED-500 -> SIFT-4000], tau=0.40)
# Stage 1: ALIKED-500
# Stage 2: SIFT-4000
df_aliked_500 = df_eval[(df_eval["preset"] == "ALIKED") & (df_eval["N"] == 500)].set_index(["fname", "i", "j"])
df_sift_4000 = df_eval[(df_eval["preset"] == "SIFT") & (df_eval["N"] == 4000)].set_index(["fname", "i", "j"])

tau_grid = [0.40] # Pre-registered HPatches choice
for tau in [0.40]:
    casc_rows = []
    for fname, i, j in pairs:
        r1 = df_aliked_500.loc[(fname, i, j)]
        r2 = df_sift_4000.loc[(fname, i, j)]
        
        p1 = float(r1["p_epi"])
        act1 = float(r1["actual_kp"])
        req1 = 500
        pool1 = pair_pool_dict.get((fname, i, j, "ALIKED"), 4000)
        
        if p1 >= tau:
            # Stopped at Stage 1
            casc_rows.append({
                "fname": fname, "i": i, "j": j, "scene": get_scene(fname),
                "requested_kp": req1,
                "actual_kp": act1,
                "pool_size": pool1,
                "stopped_stage1": True
            })
        else:
            # Escalated to Stage 2
            act2 = float(r2["actual_kp"])
            req2 = 4000
            pool2 = pair_pool_dict.get((fname, i, j, "SIFT"), 4000)
            casc_rows.append({
                "fname": fname, "i": i, "j": j, "scene": get_scene(fname),
                "requested_kp": req1 + req2,
                "actual_kp": act1 + act2,
                "pool_size": min(pool1, pool2),
                "stopped_stage1": False
            })
            
    df_casc = pd.DataFrame(casc_rows)
    m_name = f"Cascade_C6_tau_{tau:.2f}"
    for sc in ["0015", "0022"]:
        records.append(analyze_subset(df_casc[df_casc["scene"] == sc], m_name, sc))
    records.append(analyze_subset(df_casc, m_name, "Pooled"))

df_summary = pd.DataFrame(records)
df_summary.to_csv("results/phase5/b_keypoint_accounting_summary.csv", index=False)
print("Saved results/phase5/b_keypoint_accounting_summary.csv")

# Print SIFT-4000 verification rows
print("\n=== SIFT-4000 ACCOUNTING EVIDENCE ===")
sift4_rows = df_summary[df_summary["method"] == "SIFT-4000"]
print(sift4_rows.to_string(index=False))

print("\n=== SECTION B COMPLETE ===")
