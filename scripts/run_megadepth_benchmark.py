import os
import sys
import time
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import cv2

sys.path.insert(0, 'src')
import megadepth as md
import pipeline_utils as pu
from megadepth_eval_env import get_megadepth_pool, compute_epipolar_proxy, MegaDepthEvalEnv
from stable_baselines3 import PPO

# Paths
RESULTS_DIR = Path("results/phase4b")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
MATCH_DIR = RESULTS_DIR / "megadepth_matches"
MATCH_DIR.mkdir(parents=True, exist_ok=True)

STAGE2A_CSV = RESULTS_DIR / "megadepth_stage2a_priority_evaluations.csv"
STAGE3_CSV = RESULTS_DIR / "megadepth_stage3_afm_rl_rollouts.csv"
STAGE2B_CSV = RESULTS_DIR / "megadepth_stage2b_secondary_evaluations.csv"

PRIORITY_CONFIGS = [
    ("SIFT", 500), ("SIFT", 1000), ("SIFT", 2000), ("SIFT", 4000),
    ("ALIKED", 500), ("ALIKED", 1000), ("ALIKED", 2000), ("ALIKED", 4000),
    ("XFeat", 500), ("XFeat", 1000), ("XFeat", 2000), ("XFeat", 4000),
    ("ORB", 1000), ("ORB", 4000)
]

ALL_BUDGETS = [500, 1000, 2000, 4000]

def run_stage1_pools():
    print("\n=======================================================")
    print("STAGE 1: POOL PRE-EXTRACTION (14 PRESETS x 806 IMAGES)")
    print("=======================================================")
    pool_cache_dir = Path("data/pool_cache_megadepth")
    cached_count = len(list(pool_cache_dir.glob("*/*.npz")))
    if cached_count >= 11000:
        print(f"All image pools already pre-cached on disk ({cached_count} files). Stage 1 Complete!")
        return

    pairs = md.list_pairs()
    unique_images = {}
    for fname, i, j in pairs:
        d = md._scene(fname)
        p1 = str(d["image_paths"][i])
        p2 = str(d["image_paths"][j])
        if (fname, i) not in unique_images:
            unique_images[(fname, i)] = (p1, md.load_image(fname, i))
        if (fname, j) not in unique_images:
            unique_images[(fname, j)] = (p2, md.load_image(fname, j))
            
    print(f"Total unique images to pool: {len(unique_images)}")
    t0_stage = time.perf_counter()
    
    for p_idx, preset in enumerate(pu.PRESET_NAMES, 1):
        t0 = time.perf_counter()
        print(f"[{p_idx}/14] Pre-caching pools for preset: {preset}...")
        for (fname, idx), (path, (img, K)) in unique_images.items():
            get_megadepth_pool(img, preset, path)
        print(f"[{p_idx}/14] Preset {preset} done in {time.perf_counter() - t0:.2f} s")
        
    print(f"STAGE 1 COMPLETE in {time.perf_counter() - t0_stage:.2f} s.")

def evaluate_single_config(fname, i, j, preset, N, img0, img1, K0, K1, T_gt, path0, path1):
    pool0 = get_megadepth_pool(img0, preset, path0)
    pool1 = get_megadepth_pool(img1, preset, path1)
    
    n0 = min(N, len(pool0["pts"]))
    n1 = min(N, len(pool1["pts"]))
    d0 = pool0["desc"][:n0] if n0 else None
    d1 = pool1["desc"][:n1] if n1 else None
    
    good = pu.match_ratio(d0, d1, preset, "BF", 0.75)
    if good:
        pts0 = np.float32([pool0["pts"][m.queryIdx] for m in good])
        pts1 = np.float32([pool1["pts"][m.trainIdx] for m in good])
    else:
        pts0 = pts1 = np.zeros((0, 2), dtype=np.float32)
        
    n_matches = len(pts0)
    
    # Save correspondences if not already on disk
    clean_preset = preset.replace("+", "_")
    match_file = MATCH_DIR / f"{fname}_{i}_{j}_{clean_preset}_N{N}.npz"
    if not match_file.exists():
        np.savez_compressed(match_file, pts0=pts0, pts1=pts1)
        
    proxy_res = compute_epipolar_proxy(pts0, pts1, K0, K1)
    pose_ret = md.estimate_pose(pts0, pts1, K0, K1)
    if pose_ret is None:
        pose_failure = True
        rot_err = trans_err = pose_err = 180.0
    else:
        pose_failure = False
        R_est, t_est = pose_ret
        # Compute individual rotation and translation errors
        c = np.clip((np.trace(R_est.T @ T_gt[:3, :3]) - 1) / 2, -1, 1)
        rot_err = float(np.rad2deg(np.abs(np.arccos(c))))
        v = np.clip(np.dot(t_est, T_gt[:3, 3]) / (np.linalg.norm(t_est) * np.linalg.norm(T_gt[:3, 3]) + 1e-12), -1, 1)
        t_err = float(np.rad2deg(np.arccos(v)))
        trans_err = min(t_err, 180.0 - t_err)
        pose_err = max(rot_err, trans_err)
        
    return {
        "fname": fname,
        "i": i,
        "j": j,
        "preset": preset,
        "N": N,
        "n_matches": n_matches,
        "n_inliers": proxy_res["n_inliers"],
        "inlier_ratio": proxy_res["inlier_ratio"],
        "mean_sampson": proxy_res["mean_sampson"],
        "p_epi": proxy_res["proxy"],
        "rot_err": rot_err,
        "trans_err": trans_err,
        "pose_err": pose_err,
        "actual_kp": min(n0, n1),
        "pose_failure": pose_failure
    }

def run_stage2a_priority():
    print("\n=======================================================")
    print("STAGE 2a: PRIORITY BASELINES (14 CONFIGS x 1500 PAIRS)")
    print("=======================================================")
    pairs = md.list_pairs()
    
    # Read existing records if resuming
    existing_keys = set()
    if STAGE2A_CSV.exists():
        df_ex = pd.read_csv(STAGE2A_CSV)
        if len(df_ex) >= 21000:
            print(f"All {len(df_ex)} priority baseline evaluations already complete. Stage 2a Complete!")
            return
        for _, row in df_ex.iterrows():
            existing_keys.add((row["fname"], int(row["i"]), int(row["j"]), row["preset"], int(row["N"])))
        print(f"Resuming Stage 2a: found {len(existing_keys)} existing evaluations.")
    else:
        # Write header
        pd.DataFrame(columns=[
            "fname", "i", "j", "preset", "N", "n_matches", "n_inliers",
            "inlier_ratio", "mean_sampson", "p_epi", "rot_err", "trans_err",
            "pose_err", "actual_kp", "pose_failure"
        ]).to_csv(STAGE2A_CSV, index=False)
        
    t0_stage = time.perf_counter()
    batch = []
    
    for pair_idx, (fname, i, j) in enumerate(pairs):
        if (pair_idx + 1) % 100 == 0:
            print(f"[Stage 2a] Processing pair {pair_idx + 1}/1500...")
            
        img0, img1, K0, K1, T_gt = md.load_pair(fname, i, j)
        d = md._scene(fname)
        path0 = str(d["image_paths"][i])
        path1 = str(d["image_paths"][j])
        
        for preset, N in PRIORITY_CONFIGS:
            if (fname, i, j, preset, N) in existing_keys:
                continue
            rec = evaluate_single_config(fname, i, j, preset, N, img0, img1, K0, K1, T_gt, path0, path1)
            batch.append(rec)
            existing_keys.add((fname, i, j, preset, N))
            
            if len(batch) >= 200:
                pd.DataFrame(batch).to_csv(STAGE2A_CSV, mode="a", header=False, index=False)
                batch = []
                
    if batch:
        pd.DataFrame(batch).to_csv(STAGE2A_CSV, mode="a", header=False, index=False)
        
    print(f"STAGE 2a COMPLETE in {time.perf_counter() - t0_stage:.2f} s.")

def run_stage3_afm_rl():
    print("\n=======================================================")
    print("STAGE 3: AFM-RL ROLLOUTS (5 FOLD MODELS x 1500 PAIRS)")
    print("=======================================================")
    pairs = md.list_pairs()
    
    existing_keys = set()
    if STAGE3_CSV.exists():
        df_ex = pd.read_csv(STAGE3_CSV)
        for _, row in df_ex.iterrows():
            existing_keys.add((int(row["fold"]), row["fname"], int(row["i"]), int(row["j"])))
        print(f"Resuming Stage 3: found {len(existing_keys)} existing rollouts.")
    else:
        pd.DataFrame(columns=[
            "fold", "fname", "i", "j", "chosen_preset", "steps", "actions",
            "final_N", "cumulative_kp", "p_epi_final", "rot_err", "trans_err",
            "pose_err", "pose_failure"
        ]).to_csv(STAGE3_CSV, index=False)
        
    t0_stage = time.perf_counter()
    batch = []
    
    for fold in range(5):
        print(f"\n[Stage 3] Evaluating AFM-RL Fold {fold}...")
        l1_path = f"checkpoints_v2/fold{fold}/level1_best/best_model.zip"
        l2_path = f"checkpoints_v2/fold{fold}/level2_14preset_best/best_model.zip"
        l1_model = PPO.load(l1_path)
        l2_model = PPO.load(l2_path)
        
        env = MegaDepthEvalEnv(eval_mode=True)
        
        for pair_idx, (fname, i, j) in enumerate(pairs):
            if (fold, fname, i, j) in existing_keys:
                continue
                
            if (pair_idx + 1) % 200 == 0:
                print(f"[Fold {fold}] Pair {pair_idx + 1}/1500...")
                
            img0, img1, K0, K1, T_gt = md.load_pair(fname, i, j)
            
            # Level 1 preset selection
            st = env._stats(fname, i, j, img0, img1)
            action_l1, _ = l1_model.predict(st, deterministic=True)
            preset = pu.PRESET_NAMES[int(action_l1)]
            
            # Level 2 rollout
            obs, info = env.reset_to(fname, i, j, preset)
            actions = []
            done = False
            
            while not done:
                action_l2, _ = l2_model.predict(obs, deterministic=True)
                act_int = int(action_l2)
                actions.append(act_int)
                obs, reward, terminated, truncated, info = env.step(act_int)
                done = terminated or truncated
                
            # Access GT pose post-rollout for evaluation metrics
            final_pts0 = env.last_r["pts1"]
            final_pts1 = env.last_r["pts2"]
            
            pose_ret = md.estimate_pose(final_pts0, final_pts1, K0, K1)
            if pose_ret is None:
                pose_failure = True
                rot_err = trans_err = pose_err = 180.0
            else:
                pose_failure = False
                R_est, t_est = pose_ret
                c = np.clip((np.trace(R_est.T @ T_gt[:3, :3]) - 1) / 2, -1, 1)
                rot_err = float(np.rad2deg(np.abs(np.arccos(c))))
                v = np.clip(np.dot(t_est, T_gt[:3, 3]) / (np.linalg.norm(t_est) * np.linalg.norm(T_gt[:3, 3]) + 1e-12), -1, 1)
                t_err = float(np.rad2deg(np.arccos(v)))
                trans_err = min(t_err, 180.0 - t_err)
                pose_err = max(rot_err, trans_err)
                
            batch.append({
                "fold": fold,
                "fname": fname,
                "i": i,
                "j": j,
                "chosen_preset": preset,
                "steps": info["steps"],
                "actions": str(actions),
                "final_N": info["n_kp"],
                "cumulative_kp": info["cum_kp"],
                "p_epi_final": info["last_proxy"],
                "rot_err": rot_err,
                "trans_err": trans_err,
                "pose_err": pose_err,
                "pose_failure": pose_failure
            })
            existing_keys.add((fold, fname, i, j))
            
            if len(batch) >= 100:
                pd.DataFrame(batch).to_csv(STAGE3_CSV, mode="a", header=False, index=False)
                batch = []
                
    if batch:
        pd.DataFrame(batch).to_csv(STAGE3_CSV, mode="a", header=False, index=False)
        
    print(f"STAGE 3 COMPLETE in {time.perf_counter() - t0_stage:.2f} s.")

def run_stage2b_secondary():
    print("\n=======================================================")
    print("STAGE 2b: SECONDARY BASELINES (42 CONFIGS x 1500 PAIRS)")
    print("=======================================================")
    pairs = md.list_pairs()
    
    # Secondary configs = all (preset, budget) NOT in priority configs
    secondary_configs = []
    for preset in pu.PRESET_NAMES:
        for N in ALL_BUDGETS:
            if (preset, N) not in PRIORITY_CONFIGS:
                secondary_configs.append((preset, N))
                
    print(f"Secondary configurations count: {len(secondary_configs)}")
    existing_keys = set()
    if STAGE2B_CSV.exists():
        df_ex = pd.read_csv(STAGE2B_CSV)
        for _, row in df_ex.iterrows():
            existing_keys.add((row["fname"], int(row["i"]), int(row["j"]), row["preset"], int(row["N"])))
        print(f"Resuming Stage 2b: found {len(existing_keys)} existing evaluations.")
    else:
        pd.DataFrame(columns=[
            "fname", "i", "j", "preset", "N", "n_matches", "n_inliers",
            "inlier_ratio", "mean_sampson", "p_epi", "rot_err", "trans_err",
            "pose_err", "actual_kp", "pose_failure"
        ]).to_csv(STAGE2B_CSV, index=False)
        
    t0_stage = time.perf_counter()
    batch = []
    
    for pair_idx, (fname, i, j) in enumerate(pairs):
        if (pair_idx + 1) % 100 == 0:
            print(f"[Stage 2b] Processing pair {pair_idx + 1}/1500...")
            
        img0, img1, K0, K1, T_gt = md.load_pair(fname, i, j)
        d = md._scene(fname)
        path0 = str(d["image_paths"][i])
        path1 = str(d["image_paths"][j])
        
        for preset, N in secondary_configs:
            if (fname, i, j, preset, N) in existing_keys:
                continue
            rec = evaluate_single_config(fname, i, j, preset, N, img0, img1, K0, K1, T_gt, path0, path1)
            batch.append(rec)
            existing_keys.add((fname, i, j, preset, N))
            
            if len(batch) >= 200:
                pd.DataFrame(batch).to_csv(STAGE2B_CSV, mode="a", header=False, index=False)
                batch = []
                
    if batch:
        pd.DataFrame(batch).to_csv(STAGE2B_CSV, mode="a", header=False, index=False)
        
    print(f"STAGE 2b COMPLETE in {time.perf_counter() - t0_stage:.2f} s.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["1", "2a", "3", "2b", "all", "core"], default="core")
    args = parser.parse_args()
    
    # Set OpenCV threads to full host capacity (12 threads)
    cv2.setNumThreads(12)
    
    if args.stage in ["1", "all", "core"]:
        run_stage1_pools()
    if args.stage in ["2a", "all", "core"]:
        run_stage2a_priority()
    if args.stage in ["3", "all", "core"]:
        run_stage3_afm_rl()
    if args.stage in ["2b", "all"]:
        run_stage2b_secondary()

if __name__ == "__main__":
    main()

